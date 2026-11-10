"""Request lifecycle, result/events and per-request generation state."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from time import monotonic, time
from typing import Any


class State(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL = {State.FINISHED, State.CANCELLED, State.FAILED}
VALID_TRANSITIONS = {
    State.WAITING: {State.RUNNING, State.CANCELLED, State.FAILED},
    State.RUNNING: {State.PREEMPTED, State.FINISHED, State.CANCELLED, State.FAILED},
    State.PREEMPTED: {State.RUNNING, State.CANCELLED, State.FAILED},
    State.FINISHED: set(), State.CANCELLED: set(), State.FAILED: set(),
}


@dataclass
class Event:
    kind: str  # token, done, error
    text: str = ""
    token_id: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    timestamp: float = field(default_factory=monotonic)


@dataclass
class Request:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    temperature: float = 0.0
    top_k: int = 0
    priority: int = 0
    created: int = field(default_factory=lambda: int(time()))
    arrived: float = field(default_factory=monotonic)
    state: State = State.WAITING
    # How many tokens have been physically processed by the model and stored in KV.
    cached_tokens: int = 0
    generated_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""
    text: str = ""
    preemptions: int = 0
    first_token_at: float | None = None
    last_token_at: float | None = None
    finished_at: float | None = None
    first_scheduled_at: float | None = None
    token_times: list[float] = field(default_factory=list)
    error: str | None = None
    finish_reason: str | None = None
    events: Queue[Event] = field(default_factory=Queue, repr=False)
    # A distinct CPU RNG per request, invariant to other requests' batch positions.
    rng: Any = field(default=None, repr=False)
    cancelled_requested: bool = False
    prefix_cache_hit: bool = False
    prefix_inserted: bool = False
    saved_prefill_tokens: int = 0

    def transition(self, new_state: State) -> None:
        if new_state == self.state:
            return
        if new_state not in VALID_TRANSITIONS[self.state]:
            raise ValueError(f"Invalid transition {self.state} -> {new_state}")
        self.state = new_state

    @property
    def full_context(self) -> list[int]:
        # Last sampled token has not yet been forwarded into KV.
        return self.prompt_ids + self.generated_ids[:-1] if self.generated_ids else self.prompt_ids

    @property
    def in_prefill(self) -> bool:
        return self.cached_tokens < len(self.full_context)

    @property
    def next_token_count(self) -> int:
        return len(self.generated_ids)

    def next_inputs(self, count: int) -> list[int]:
        if count <= 0:
            return []
        if self.in_prefill:
            return self.full_context[self.cached_tokens : self.cached_tokens + count]
        # A fully processed prompt + generated prefix, wait for next output token.
        if not self.generated_ids:
            return []
        return [self.generated_ids[-1]]

    def snapshot(self) -> dict:
        ttft = None if self.first_token_at is None else (self.first_token_at - self.arrived) * 1000.0
        end = self.finished_at if self.finished_at is not None else monotonic()
        itls = [(b - a) * 1000 for a, b in zip(self.token_times, self.token_times[1:])]
        return {
            "request_id": self.request_id, "state": self.state.value,
            "prompt_tokens": len(self.prompt_ids), "output_tokens": len(self.generated_ids),
            "ttft_ms": ttft, "itl_ms": itls,
            "tpot_ms": ((self.token_times[-1] - self.token_times[0]) * 1000.0 / (len(self.token_times) - 1))
            if len(self.token_times) >= 2 else None,
            "e2e_ms": (end - self.arrived) * 1000.0,
            "queue_ms": None if self.first_scheduled_at is None else (self.first_scheduled_at - self.arrived) * 1000.0,
            "finish_reason": self.finish_reason, "error": self.error,
            "preemptions": self.preemptions,
            "prefix_cache_hit": self.prefix_cache_hit,
            "saved_prefill_tokens": self.saved_prefill_tokens,
        }
