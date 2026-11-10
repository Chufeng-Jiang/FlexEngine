import json
import math

from flexengine.research.analysis.results import collect


def write_run(root, experiment, name, repetition, requests_per_sec):
    directory = root / experiment / name
    directory.mkdir(parents=True, exist_ok=True)
    spec = {
        "experiment": experiment, "name": name, "model": "test", "workload": "w1",
        "concurrency": 1, "policy": "fcfs", "attention": "materialized",
        "token_budget": 128, "block_size": 16, "prefix_cache": False,
        "prefix_ratio": 0, "cuda_graph": False,
    }
    (directory / "spec.json").write_text(json.dumps(spec), encoding="utf8")
    run = directory / f"run-{repetition}"
    run.mkdir(exist_ok=True)
    summary = {
        "completed_requests": 2, "failures": 0, "requests_per_sec": requests_per_sec,
        "goodput_requests_per_sec": requests_per_sec, "slo_pass_rate": 1.0,
        "ttft_p50_ms": 1.0, "ttft_p95_ms": 2.0, "ttft_p99_ms": 3.0,
        "e2e_p95_ms": 4.0, "sse_inter_event_p95_ms": 5.0,
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf8")


def test_repeat_analysis_groups_sample_stdev_and_confidence(tmp_path):
    write_run(tmp_path, "E1", "complete", 1, 1.0)
    write_run(tmp_path, "E1", "complete", 2, 3.0)

    rows = collect(tmp_path, required_repetitions=2)

    assert len(rows) == 1
    assert rows[0]["requests_per_sec_mean"] == 2.0
    assert math.isclose(rows[0]["requests_per_sec_stdev"], math.sqrt(2))
    assert rows[0]["confidence"] == "mean ± sample stdev, n=2"


def test_repeat_analysis_excludes_incomplete_runs(tmp_path):
    write_run(tmp_path, "E1", "incomplete", 1, 1.0)

    assert collect(tmp_path, required_repetitions=3) == []
