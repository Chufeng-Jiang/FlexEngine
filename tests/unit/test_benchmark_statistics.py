import statistics
import subprocess
import sys

import pytest

from flexengine.research.benchmark.statistics import bootstrap_ci, summarize_requests, validate_sample_counts


def _request(serial: int, *, ttft: float, tpot: float, completed: bool = True) -> dict:
    return {
        "serial": serial,
        "status": 200,
        "completed": completed,
        "slo_pass": completed,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": ttft + 100,
        "itl_ms": [tpot],
    }


def test_bootstrap_ci_is_deterministic_and_contains_sample_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    first = bootstrap_ci(values, statistics.fmean, confidence=0.95, resamples=2_000, seed=17)
    second = bootstrap_ci(values, statistics.fmean, confidence=0.95, resamples=2_000, seed=17)

    assert first == second
    assert first[0] < 3.0 < first[1]


def test_summary_excludes_warmup_and_reports_tpot_p99():
    warmup = [_request(100, ttft=9_999.0, tpot=9_999.0)]
    measured = [
        _request(0, ttft=10.0, tpot=1.0),
        _request(1, ttft=20.0, tpot=2.0),
        _request(2, ttft=30.0, tpot=100.0),
    ]

    summary = summarize_requests(measured, elapsed=2.0, warmup_results=warmup, bootstrap_resamples=200)

    assert summary["completed_requests"] == 3
    assert summary["warmup_requests"] == 1
    assert summary["requests_per_sec"] == 1.5
    assert summary["ttft_p99_ms"] == pytest.approx(29.8)
    assert summary["tpot_p99_ms"] == pytest.approx(98.04)
    assert summary["cold_start_ttft_ms"] == 9_999.0


@pytest.mark.parametrize(
    ("warmup", "measured", "resamples"),
    [(-1, 10, 100), (0, 0, 100), (0, 10, 0)],
)
def test_invalid_sample_counts_are_rejected(warmup, measured, resamples):
    with pytest.raises(ValueError):
        validate_sample_counts(warmup, measured, resamples)


def test_benchmark_module_entrypoint_loads_statistics():
    result = subprocess.run(
        [sys.executable, "-m", "flexengine.research.benchmark.runner", "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "--warmup-requests" in result.stdout
