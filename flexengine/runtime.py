"""Thread-owned, single-GPU inference runtime.

The worker serializes GPU execution. HTTP clients interact with thread-safe
per-request event queues; each scheduling iteration runs multiple requests.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict, deque

import torch

from flexengine.config import EngineConfig
from flexengine.execution.cuda_graph import DecodeGraphManager
from flexengine.kv_cache.allocator import CacheExhausted, CacheLayout, PagedKVCache
from flexengine.kv_cache.prefix import PrefixCache
from flexengine.observability.metrics import RuntimeMetrics
from flexengine.observability.traces import TraceWriter, nvtx
from flexengine.sampling.sampler import sample
from flexengine.scheduler.policies import Scheduler
from flexengine.scheduler.state import TERMINAL, Event, Request, State


class BusyError(RuntimeError):
    pass


class Engine:
    def __init__(self, config: EngineConfig, adapter):
        self.config = config
        self.adapter = adapter
        self.tokenizer = adapter.tokenizer
        if self.tokenizer is None:
            raise ValueError("adapter.tokenizer required")
        self.cache = PagedKVCache(CacheLayout(
            layers=adapter.layers, num_blocks=config.kv_num_blocks,
            block_size=config.kv_block_size, kv_heads=adapter.nkv, head_dim=adapter.dim,
        ), device=adapter.device, dtype=next(adapter.model.parameters()).dtype)
        self.prefix_cache = PrefixCache(self.cache, config.prefix_cache_max_entries) if config.prefix_cache else None
        self.prefix_saved_tokens = 0
        self.graph_manager = None
        self.graph_scratch_blocks: list[int] = []
        if config.cuda_graph and self.cache.device.type == "cuda" and hasattr(adapter, "decode_static"):
            self.graph_manager = DecodeGraphManager(self.cache.device)
            scratch_id = "__flex_graph_scratch__"
            self.cache.reserve(scratch_id, max(self.graph_manager.buckets) * config.kv_block_size)
            self.graph_scratch_blocks = list(self.cache.tables[scratch_id])
        self.scheduler = Scheduler(
            config.scheduler_policy,
            config.max_num_batched_tokens,
            config.ttft_slo_ms,
            config.itl_slo_ms,
            config.prefill_chunk_size,
        )
        self.waiting: deque[Request] = deque()
        self.running: OrderedDict[str, Request] = OrderedDict()
        self.all_requests: dict[str, Request] = {}
        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.worker: threading.Thread | None = None
        self.fault: str | None = None
        self.step_index = 0
        self.request_sequence = 0
        self.completed_count = 0
        self.trace = TraceWriter(config.trace_dir)
        self.trace.metadata(config)
        self.metrics = RuntimeMetrics()
        eos = adapter.config.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple, set)) else ({eos} if eos is not None else set())

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.shutdown.clear()
        self.worker = threading.Thread(target=self._worker, name="flexengine-cuda-worker", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.shutdown.set()
        if self.worker:
            self.worker.join(timeout=20)
        with self.lock:
            for req in list(self.all_requests.values()):
                if req.state not in TERMINAL:
                    self._finish(req, State.CANCELLED, "cancelled")

    def submit(self, prompt: str, max_tokens: int = 128, temperature: float = 0.0, top_k: int = 0,
               priority: int = 0, seed: int | None = None) -> Request:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        if not (1 <= max_tokens <= self.config.max_output_tokens):
            raise ValueError(f"max_tokens must be between 1 and {self.config.max_output_tokens}")
        if not (0.0 <= temperature <= 5.0) or top_k < 0:
            raise ValueError("invalid sampling parameters")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            bos = getattr(self.tokenizer, "bos_token_id", None)
            if bos is None:
                raise ValueError("empty encoded prompt")
            ids = [bos]
        if len(ids) + max_tokens > self.config.max_model_len:
            raise ValueError(f"prompt tokens ({len(ids)}) + max_tokens ({max_tokens}) exceeds max_model_len ({self.config.max_model_len})")
        with self.lock:
            if self.shutdown.is_set() or self.fault:
                raise BusyError("engine is not accepting requests")
            if len(self.waiting) >= self.config.max_waiting:
                raise BusyError("bounded waiting queue is full")
            request_id = "cmpl-" + uuid.uuid4().hex
            req = Request(request_id, ids, max_tokens, temperature, top_k, priority)
            req.rng = torch.Generator(device="cpu")
            # The same trace + request order is reproducible across policies.
            # An explicit seed can override order for a specific request.
            req.rng.manual_seed(seed if seed is not None else (self.config.random_seed + self.request_sequence))
            self.request_sequence += 1
            self.waiting.append(req)
            self.all_requests[request_id] = req
            if self.metrics.registry is not None:
                self.metrics.requests.inc()
            return req

    def cancel(self, request_id: str) -> bool:
        with self.lock:
            req = self.all_requests.get(request_id)
            if req is None or req.state in TERMINAL:
                return False
            req.cancelled_requested = True
            # Worker lock protects entire forward. If in flight, cancellation
            # takes effect at the next safe boundary.
            self._finish(req, State.CANCELLED, "cancelled")
            return True

    def _finish(self, req: Request, state: State, reason: str, error: str | None = None) -> None:
        if req.state in TERMINAL:
            return
        req.transition(state)
        req.finished_at = time.monotonic()
        req.finish_reason = reason
        req.error = error
        self.cache.free_request(req.request_id)
        self.running.pop(req.request_id, None)
        self.waiting = deque(r for r in self.waiting if r.request_id != req.request_id)
        if state is State.FAILED:
            req.events.put(Event("error", error=error or reason, finish_reason=reason))
        else:
            # Emit remaining stable text on the terminal event, if needed.
            remaining = req.text[len(req.emitted_text):] if req.text.startswith(req.emitted_text) else ""
            if remaining:
                req.emitted_text += remaining
                req.events.put(Event("token", text=remaining))
            req.events.put(Event("done", finish_reason=reason))
        self.trace.write("requests", req.snapshot())
        self.all_requests.pop(req.request_id, None)
        self.completed_count += 1
        if self.metrics.registry is not None:
            self.metrics.completed.labels(reason=reason).inc()
            self.metrics.e2e.observe(req.finished_at - req.arrived)
            if req.first_token_at:
                self.metrics.ttft.observe(req.first_token_at - req.arrived)

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            req = self.waiting.popleft()
            if req.state in TERMINAL:
                continue
            if self.prefix_cache is not None and req.request_id not in self.cache.tables:
                match = self.prefix_cache.lookup(req.prompt_ids, self._prefix_identity())
                if match is not None:
                    self.cache.attach_prefix(req.request_id, match.block_ids)
                    req.cached_tokens = match.cached_tokens
                    req.prefix_cache_hit = True
                    req.saved_prefill_tokens += match.cached_tokens
                    self.prefix_saved_tokens += match.cached_tokens
                    if self.metrics.registry is not None:
                        self.metrics.prefix_hits.inc()
                        self.metrics.prefix_saved.inc(match.cached_tokens)
            req.transition(State.RUNNING)
            self.running[req.request_id] = req

    def _prefix_identity(self) -> tuple:
        layout = self.cache.layout
        return (
            self.config.model, self.config.model_revision, layout.layers, layout.kv_heads,
            layout.head_dim, layout.block_size, str(self.cache.dtype),
        )

    def _maybe_insert_prefix(self, req: Request) -> None:
        if self.prefix_cache is None or req.prefix_inserted:
            return
        complete_tokens = len(req.prompt_ids) // self.cache.layout.block_size * self.cache.layout.block_size
        if complete_tokens == 0 or req.cached_tokens < complete_tokens:
            return
        evictions_before = self.prefix_cache.evictions
        self.prefix_cache.insert(req.request_id, req.prompt_ids, self._prefix_identity())
        req.prefix_inserted = True
        evicted = self.prefix_cache.evictions - evictions_before
        if evicted and self.metrics.registry is not None:
            self.metrics.prefix_evictions.inc(evicted)

    def _graph_inputs(self, chunks, batch_size: int) -> dict[str, torch.Tensor]:
        max_blocks = (self.config.max_model_len + self.config.kv_block_size - 1) // self.config.kv_block_size
        ids = torch.zeros(batch_size, dtype=torch.long, device=self.cache.device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.cache.device)
        tables = torch.empty((batch_size, max_blocks), dtype=torch.long, device=self.cache.device)
        for row in range(batch_size):
            scratch = self.graph_scratch_blocks[row]
            tables[row].fill_(scratch)
        for row, chunk in enumerate(chunks):
            ids[row] = chunk.input_ids[0]
            positions[row] = chunk.start
            table = self.cache.tables[chunk.request_id]
            tables[row, :len(table)] = torch.tensor(table, dtype=torch.long, device=self.cache.device)
        return {"input_ids": ids, "positions": positions, "block_tables": tables}

    def _graph_forward(self, chunks) -> list[torch.Tensor] | None:
        if self.graph_manager is None or not all(len(chunk.input_ids) == 1 for chunk in chunks):
            return None
        if not all(self.running[chunk.request_id].generated_ids for chunk in chunks):
            return None
        active = len(chunks)
        bucket = self.graph_manager.bucket_for(active)
        if bucket is None:
            self.graph_manager.record_fallback()
            return None
        try:
            if not self.graph_manager.has_capture(bucket):
                examples = self._graph_inputs(chunks, bucket)
                self.graph_manager.capture(
                    bucket,
                    lambda input_ids, positions, block_tables, active_mask: self.adapter.decode_static(
                        input_ids, positions, block_tables, active_mask, self.cache
                    ),
                    examples,
                )
            output = self.graph_manager.replay(active, self._graph_inputs(chunks, active))
            return [output[row] for row in range(active)]
        except (RuntimeError, ValueError):
            self.graph_manager.record_fallback()
            return None

    def _preempt(self, victim: Request) -> None:
        self.cache.free_request(victim.request_id)
        victim.cached_tokens = 0
        victim.replaying = bool(victim.generated_ids)
        victim.preemptions += 1
        victim.transition(State.PREEMPTED)
        self.running.pop(victim.request_id, None)
        self.waiting.append(victim)
        if self.metrics.registry is not None:
            self.metrics.preemptions.inc()

    @torch.inference_mode()
    def step(self) -> bool:
        with self.lock:
            if self.fault:
                return False
            step_start = time.perf_counter()
            self._admit()
            candidates = list(self.running.values())
            with nvtx("sched/select"):
                proposed = self.scheduler.plan(candidates)
            planned = []
            # Admission checks are atomic: avoid allocating a partial step.
            for chunk in proposed:
                req = self.running.get(chunk.request_id)
                if req is None:
                    continue
                target = chunk.start + len(chunk.input_ids)
                with nvtx("kv/reserve"):
                    try:
                        self.cache.reserve(req.request_id, target)
                    except CacheExhausted:
                        # Prefer waiting to thrashing. If a non-scheduled peer
                        # holds pages, recompute-preempt it once.
                        selected = {c.request_id for c in planned}
                        victims = [v for v in self.running.values()
                                   if v.request_id != req.request_id and v.request_id not in selected
                                   and self.cache.tables.get(v.request_id)]
                        if victims:
                            victim = min(victims, key=lambda r: (r.priority, -r.cached_tokens))
                            self._preempt(victim)
                            try:
                                self.cache.reserve(req.request_id, target)
                            except CacheExhausted:
                                continue
                        else:
                            continue
                planned.append(chunk)
            shared_blocks = self.prefix_cache.shared_blocks if self.prefix_cache is not None else 0
            self.metrics.update_gauges(len(self.waiting), len(self.running), self.cache.used_blocks, shared_blocks)
            if not planned:
                return False
            scheduler_ms = (time.perf_counter() - step_start) * 1000.0
            new_tokens = sum(len(c.input_ids) for c in planned)
            prefill_tokens = sum(len(c.input_ids) for c in planned if self.running[c.request_id].in_prefill)
            cuda_start = cuda_end = None
            if torch.cuda.is_available() and self.config.device.startswith("cuda"):
                cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                cuda_start.record()
            graph_before = self.graph_manager.stats() if self.graph_manager is not None else None
            executor_start = time.perf_counter()
            try:
                with nvtx(f"executor/step batch={len(planned)} tokens={new_tokens}"):
                    outputs = self._graph_forward(planned)
                    if outputs is None:
                        outputs = self.adapter.forward(planned, self.cache)
                if len(outputs) != len(planned):
                    raise RuntimeError("wrong executor output count")
                if cuda_end is not None:
                    cuda_end.record()
                for chunk, logits in zip(planned, outputs):
                    req = self.running[chunk.request_id]
                    if req.first_scheduled_at is None:
                        req.first_scheduled_at = time.monotonic()
                    req.cached_tokens += len(chunk.input_ids)
                    self._maybe_insert_prefix(req)
                    # If this was recomputation of an already-emitted prefix,
                    # we MUST NOT generate another token from that same prefix.
                    if req.in_prefill:
                        continue
                    if getattr(req, "replaying", False):
                        req.replaying = False
                        continue
                    tid = sample(logits, req.temperature, req.top_k, req.rng)
                    req.generated_ids.append(tid)
                    now = time.monotonic()
                    if req.first_token_at is None:
                        req.first_token_at = now
                    req.last_token_at = now
                    req.token_times.append(now)
                    req.text = self.tokenizer.decode(req.generated_ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
                    # Don't emit a temporary Unicode replacement character
                    # that may be completed by following byte-level tokens.
                    stable = req.text.rstrip("\ufffd")
                    delta = stable[len(req.emitted_text):] if stable.startswith(req.emitted_text) else ""
                    if delta:
                        req.emitted_text += delta
                        req.events.put(Event("token", text=delta, token_id=tid))
                    if self.metrics.registry is not None:
                        self.metrics.tokens.inc()
                    if tid in self.eos_ids:
                        self._finish(req, State.FINISHED, "stop")
                    elif len(req.generated_ids) >= req.max_tokens:
                        self._finish(req, State.FINISHED, "length")
                if cuda_end is not None:
                    cuda_end.synchronize()
            except Exception as ex:  # noqa: BLE001 - any executor failure faults the GPU worker
                # Fail the worker and all active+queued requests; a GPU error
                # can render subsequent CUDA calls unsafe until restart.
                self.fault = f"{type(ex).__name__}: {ex}"
                for req in list(self.all_requests.values()):
                    if req.state not in TERMINAL:
                        self._finish(req, State.FAILED, "error", self.fault)
                return False
            executor_wall_ms = (time.perf_counter() - executor_start) * 1000.0
            gpu_ms = cuda_start.elapsed_time(cuda_end) if cuda_end is not None else None
            graph_delta = (
                self.graph_manager.stats_delta(graph_before)
                if self.graph_manager is not None and graph_before is not None else None
            )
            self.step_index += 1
            if self.config.trace_steps:
                self.trace.write("steps", {
                    "step": self.step_index, "batch_size": len(planned),
                    "new_tokens": new_tokens, "prefill_tokens": prefill_tokens,
                    "decode_tokens": new_tokens - prefill_tokens, "kv_used_blocks": self.cache.used_blocks,
                    "waiting": len(self.waiting), "running": len(self.running),
                    "wall_ms": (time.perf_counter() - step_start) * 1000.0,
                    "scheduler_ms": scheduler_ms, "executor_wall_ms": executor_wall_ms, "gpu_ms": gpu_ms,
                    "policy": self.config.scheduler_policy, "attention": self.config.attention_backend,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated() if cuda_end is not None else None,
                    "cuda_reserved_bytes": torch.cuda.memory_reserved() if cuda_end is not None else None,
                    "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated() if cuda_end is not None else None,
                    "prefix_cache_hits": self.prefix_cache.hits if self.prefix_cache is not None else 0,
                    "prefix_saved_tokens": self.prefix_saved_tokens,
                    "prefix_evictions": self.prefix_cache.evictions if self.prefix_cache is not None else 0,
                    "prefix_shared_blocks": self.prefix_cache.shared_blocks if self.prefix_cache is not None else 0,
                    "prefix_cache": self.prefix_cache.stats() if self.prefix_cache is not None else None,
                    "cuda_graph": self.graph_manager.stats() if self.graph_manager is not None else None,
                    "cuda_graph_delta": graph_delta,
                })
            return True

    def _worker(self):
        while not self.shutdown.is_set() and not self.fault:
            work = self.step()
            if not work:
                time.sleep(self.config.idle_sleep_seconds)

    def snapshot(self):
        with self.lock:
            prefix = self.prefix_cache
            graph = self.graph_manager.stats() if self.graph_manager is not None else None
            return {"waiting": len(self.waiting), "running": len(self.running),
                    "cache_used_blocks": self.cache.used_blocks, "cache_free_blocks": self.cache.free_blocks,
                    "steps": self.step_index, "fault": self.fault,
                    "requests_total": self.request_sequence, "requests_completed": self.completed_count,
                    "prefix_cache_hits": prefix.hits if prefix is not None else 0,
                    "prefix_saved_tokens": self.prefix_saved_tokens,
                    "prefix_evictions": prefix.evictions if prefix is not None else 0,
                    "prefix_shared_blocks": prefix.shared_blocks if prefix is not None else 0,
                    "prefix_cache": prefix.stats() if prefix is not None else None,
                    "cuda_graph": graph}
