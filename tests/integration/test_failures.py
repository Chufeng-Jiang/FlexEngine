import time
from dataclasses import replace

from conftest import TinyAdapter
from fastapi.testclient import TestClient

from flexengine.api.server import create_app
from flexengine.runtime import Engine
from flexengine.scheduler.state import State


class FailingAdapter(TinyAdapter):
    def forward(self, chunks, cache):
        raise RuntimeError("injected executor failure")


class SlowAdapter(TinyAdapter):
    def forward(self, chunks, cache):
        time.sleep(0.03)
        return super().forward(chunks, cache)


class CapturingEngine(Engine):
    last_request = None

    def submit(self, *args, **kwargs):
        request = super().submit(*args, **kwargs)
        self.last_request = request
        return request


def test_api_returns_429_when_waiting_admission_is_disabled(conf):
    config = replace(conf, max_waiting=0)
    engine = Engine(config, TinyAdapter())
    app = create_app(config, engine)
    with TestClient(app) as client:
        response = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 2})
        assert response.status_code == 429
        assert response.headers["retry-after"] == "1"
        assert response.json()["error"]["code"] == "overloaded"


def test_executor_failure_fails_all_requests_and_reclaims_blocks(conf):
    engine = Engine(conf, FailingAdapter())
    first = engine.submit("first", max_tokens=3)
    second = engine.submit("second", max_tokens=3)

    assert engine.step() is False

    assert first.state is State.FAILED
    assert second.state is State.FAILED
    assert "injected executor failure" in engine.fault
    assert engine.cache.used_blocks == 0
    engine.cache.assert_invariants()


def test_decode_first_schedules_prefill_after_ttft_deadline(conf):
    config = replace(
        conf,
        scheduler_policy="decode_first",
        max_num_seqs=3,
        max_num_batched_tokens=2,
        max_output_tokens=32,
        ttft_slo_ms=100,
    )
    engine = Engine(config, TinyAdapter())
    decoding_a = engine.submit("a", max_tokens=20)
    decoding_b = engine.submit("b", max_tokens=20)
    engine.step()
    assert decoding_a.generated_ids and decoding_b.generated_ids
    overdue = engine.submit("long prompt", max_tokens=2)
    overdue.arrived -= 1.0

    engine.step()

    assert overdue.cached_tokens > 0


def test_closing_sse_stream_cancels_request_and_reclaims_blocks(conf):
    config = replace(conf, max_output_tokens=32)
    engine = CapturingEngine(config, SlowAdapter())
    app = create_app(config, engine)
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "stream slowly", "max_tokens": 32, "stream": True},
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data: ") and '"text": ""' not in line:
                    break
        deadline = time.monotonic() + 2
        while engine.last_request.state not in (State.CANCELLED, State.FINISHED) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert engine.last_request.state is State.CANCELLED
        assert engine.cache.used_blocks == 0
