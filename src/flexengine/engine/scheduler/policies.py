"""Token-budget admission decisions for three scheduling strategies.

The SLO policy uses cheap deadline proxies plus wait-ageing; it is a research
heuristic, NOT an optimal SLO controller. See benchmarks for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from flexengine.engine.scheduler.state import Request


@dataclass
class Chunk:
    request_id: str
    input_ids: list[int]
    start: int


class Scheduler:
    def __init__(
        self, policy: str, token_budget: int, ttft_ms: float, itl_ms: float, prefill_chunk_size: int = 128
    ):
        self.policy = policy
        self.token_budget = token_budget
        self.ttft_ms = ttft_ms
        self.itl_ms = itl_ms
        self.prefill_chunk_size = prefill_chunk_size

    def ordered(self, requests: list[Request]) -> list[Request]:
        now = monotonic()
        if self.policy == "fcfs":
            return sorted(requests, key=lambda r: (r.arrived, -r.priority))
        if self.policy in ("decode_first", "adaptive"):
            def rank_decode_first(r: Request):
                overdue_prefill = (
                    r.in_prefill
                    and not r.generated_ids
                    and (now - r.arrived) * 1000.0 >= self.ttft_ms
                )
                return (not overdue_prefill, r.in_prefill, -r.priority, r.arrived)

            return sorted(requests, key=rank_decode_first)
        if self.policy == "slo_aware":
            def rank(r: Request):
                if not r.generated_ids:
                    deadline = r.arrived + self.ttft_ms / 1000.0
                else:
                    deadline = (r.last_token_at or r.arrived) + self.itl_ms / 1000.0
                # Positive priority and waiting age increase urgency.
                age_bonus = min(now - r.arrived, 10.0) * 0.01
                return (deadline - now) - r.priority * 0.01 - age_bonus
            return sorted(requests, key=lambda r: (rank(r), r.arrived))
        raise ValueError("Unknown policy")

    def plan(self, requests: list[Request]) -> list[Chunk]:
        budget = self.token_budget
        chunks = []
        for req in self.ordered(requests):
            if budget <= 0:
                break
            if req.in_prefill:
                length = min(budget, len(req.full_context) - req.cached_tokens)
                if self.policy == "adaptive":
                    length = min(length, self.prefill_chunk_size)
            elif req.generated_ids:
                length = 1
            else:
                # Cached prompt with no generation is impossible outside the
                # step that processed its last prefill chunk.
                continue
            tokens = req.next_inputs(length)
            if tokens:
                chunks.append(Chunk(req.request_id, tokens, req.cached_tokens))
                budget -= len(tokens)
        assert sum(len(c.input_ids) for c in chunks) <= self.token_budget
        return chunks
