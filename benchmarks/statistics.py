"""Deterministic benchmark summary and confidence-interval helpers."""
from __future__ import annotations

import random
import statistics
from collections import Counter
from collections.abc import Callable, Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_sample_counts(warmup: int, measured: int, resamples: int) -> None:
    if warmup < 0:
        raise ValueError("warmup requests must be nonnegative")
    if measured <= 0:
        raise ValueError("measured requests must be positive")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    count = len(values)
    samples = [statistic([values[rng.randrange(count)] for _ in range(count)]) for _ in range(resamples)]
    tail = (1.0 - confidence) * 50.0
    lower = percentile(samples, tail)
    upper = percentile(samples, 100.0 - tail)
    assert lower is not None and upper is not None
    return lower, upper


def summarize_requests(
    results: list[dict],
    elapsed: float,
    *,
    warmup_results: list[dict] | None = None,
    bootstrap_resamples: int = 2_000,
    seed: int = 42,
) -> dict:
    if elapsed <= 0:
        raise ValueError("elapsed time must be positive")
    validate_sample_counts(len(warmup_results or []), len(results), bootstrap_resamples)
    warmup_results = warmup_results or []
    ttft = [item["ttft_ms"] for item in results if item.get("ttft_ms") is not None]
    tpot = [item["tpot_ms"] for item in results if item.get("tpot_ms") is not None]
    e2e = [item["e2e_ms"] for item in results if item.get("completed")]
    itls = [value for item in results for value in item.get("itl_ms", [])]
    completed = sum(bool(item.get("completed")) for item in results)
    good = sum(bool(item.get("slo_pass")) for item in results)
    throughput_samples = [1_000.0 / item["e2e_ms"] for item in results if item.get("completed") and item["e2e_ms"] > 0]
    summary = {
        "wall_seconds": elapsed,
        "warmup_requests": len(warmup_results),
        "offered_requests": len(results),
        "completed_requests": completed,
        "failures": len(results) - completed,
        "status_counts": dict(Counter(str(item.get("status")) for item in results)),
        "requests_per_sec": completed / elapsed,
        "goodput_requests_per_sec": good / elapsed,
        "slo_pass_rate": good / len(results),
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "ttft_p99_ms": percentile(ttft, 99),
        "tpot_p50_ms": percentile(tpot, 50),
        "tpot_p95_ms": percentile(tpot, 95),
        "tpot_p99_ms": percentile(tpot, 99),
        "e2e_p95_ms": percentile(e2e, 95),
        "e2e_p99_ms": percentile(e2e, 99),
        "sse_inter_event_p95_ms": percentile(itls, 95),
        "sse_inter_event_p99_ms": percentile(itls, 99),
        "cold_start_ttft_ms": next(
            (item.get("ttft_ms") for item in warmup_results if item.get("ttft_ms") is not None), None
        ),
    }
    if throughput_samples:
        low, high = bootstrap_ci(throughput_samples, confidence=0.95, resamples=bootstrap_resamples, seed=seed)
        summary["request_rate_sample_mean"] = statistics.fmean(throughput_samples)
        summary["request_rate_sample_ci95_low"] = low
        summary["request_rate_sample_ci95_high"] = high
    else:
        summary["request_rate_sample_mean"] = None
        summary["request_rate_sample_ci95_low"] = None
        summary["request_rate_sample_ci95_high"] = None
    return summary
