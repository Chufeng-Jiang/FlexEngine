"""OpenAI-style /v1/completions with SSE, cancellation and bounded admission.

Only prompt=str, one completion choice, and a subset of the Completions API
are supported. This is a local research server, not a secured public endpoint.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from queue import Empty

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from flexengine.engine.config import EngineConfig
from flexengine.engine.runtime import BusyError, Engine
from flexengine.engine.scheduler.state import TERMINAL
from flexengine.inference.models.qwen2 import Qwen2Adapter


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0, le=5)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    priority: int = Field(default=0, ge=-10, le=10)
    seed: int | None = Field(default=None, ge=0)


def api_error(message: str, status: int, code: str):
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": "invalid_request_error" if status == 400 else "server_error",
        "code": code, "param": None}}, headers={"Retry-After": "1"} if status == 429 else {})


def completion_chunk(req, model_name: str, text: str, reason=None):
    return {"id": req.request_id, "object": "text_completion.chunk", "created": req.created,
            "model": model_name, "choices": [{"text": text, "index": 0, "finish_reason": reason, "logprobs": None}]}


def create_app(config: EngineConfig | None = None, engine: Engine | None = None) -> FastAPI:
    conf = config or EngineConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = engine
        if runtime is None:
            adapter = Qwen2Adapter.from_pretrained(conf.model, conf.device, conf.dtype,
                                                   revision=conf.model_revision or None,
                                                   rmsnorm_backend=conf.rmsnorm_backend,
                                                   attention_backend=conf.attention_backend)
            runtime = Engine(conf, adapter)
        app.state.engine = runtime
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(title="FlexEngine", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz(request: Request):
        return {"status": "ok" if request.app.state.engine.fault is None else "unhealthy",
                **request.app.state.engine.snapshot()}

    @app.get("/readyz")
    def readyz(request: Request):
        runtime = request.app.state.engine
        if runtime.fault:
            raise HTTPException(status_code=503, detail=runtime.fault)
        return {"ready": True, "model": runtime.config.model}

    @app.get("/metrics")
    def metrics(request: Request):
        payload, mime = request.app.state.engine.metrics.export()
        return Response(payload, media_type=mime)

    @app.delete("/v1/requests/{request_id}")
    def cancel(request_id: str, request: Request):
        if not request.app.state.engine.cancel(request_id):
            raise HTTPException(status_code=404, detail="Unknown or finished request")
        return {"id": request_id, "cancelled": True}

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        runtime = request.app.state.engine
        if body.model and body.model != runtime.config.model:
            return api_error("Only the loaded model is available", 400, "model_not_found")
        try:
            req = runtime.submit(body.prompt, body.max_tokens, body.temperature, body.top_k, body.priority, body.seed)
        except BusyError as ex:
            return api_error(str(ex), 429, "overloaded")
        except ValueError as ex:
            return api_error(str(ex), 400, "bad_request")

        async def events():
            complete = False
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.to_thread(req.events.get, True, 0.25)
                    except Empty:
                        continue
                    if event.kind == "token":
                        yield "data: " + json.dumps(completion_chunk(req, runtime.config.model, event.text), ensure_ascii=False) + "\n\n"
                    elif event.kind == "done":
                        yield "data: " + json.dumps(completion_chunk(req, runtime.config.model, "", event.finish_reason)) + "\n\n"
                        yield "data: [DONE]\n\n"
                        complete = True
                        break
                    elif event.kind == "error":
                        yield "data: " + json.dumps({"error": {"message": event.error, "type": "server_error"}}) + "\n\n"
                        yield "data: [DONE]\n\n"
                        complete = True
                        break
            finally:
                if not complete:
                    runtime.cancel(req.request_id)

        if body.stream:
            return StreamingResponse(events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        try:
            while True:
                if await request.is_disconnected():
                    runtime.cancel(req.request_id)
                    return api_error("client disconnected", 499, "client_disconnect")
                try:
                    event = await asyncio.to_thread(req.events.get, True, 0.25)
                except Empty:
                    continue
                if event.kind == "done":
                    return {"id": req.request_id, "object": "text_completion", "created": req.created,
                            "model": runtime.config.model, "choices": [{"text": req.text, "index": 0,
                                                                            "finish_reason": event.finish_reason,
                                                                            "logprobs": None}],
                            "usage": {"prompt_tokens": len(req.prompt_ids),
                                      "completion_tokens": len(req.generated_ids),
                                      "total_tokens": len(req.prompt_ids) + len(req.generated_ids)}}
                if event.kind == "error":
                    return api_error(event.error or "model failed", 503, "runtime_failure")
        finally:
            if req.state not in TERMINAL:
                runtime.cancel(req.request_id)

    return app


# Importing this module is cheap; model loading occurs during ASGI lifespan.
app = create_app()
