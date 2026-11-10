import json
from pathlib import Path

from flexengine.research.experiments.analysis.upgrade import aggregate_runs
from flexengine.research.experiments.runners.upgrade import load_upgrade_matrix


def test_upgrade_matrix_has_unique_warm_ablation_specs():
    specs = load_upgrade_matrix(Path("data/config/upgrade_matrix.json"))

    assert len(specs) >= 15
    assert len({spec["name"] for spec in specs}) == len(specs)
    assert {"attention", "scheduler", "budget", "prefix", "graph"} <= {
        spec["study"] for spec in specs
    }
    assert all(spec["warmup_requests"] == 20 for spec in specs)
    assert all(spec["measured_requests"] == 200 for spec in specs)
    assert all(spec["repetitions"] == 1 for spec in specs)


def test_upgrade_analyzer_reports_ci_and_p99(tmp_path):
    run_root = tmp_path / "attention-materialized"
    for repetition, throughput in enumerate((10.0, 12.0, 11.0, 13.0, 9.0), 1):
        directory = run_root / f"run-{repetition}"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({
            "requests_per_sec": throughput,
            "goodput_requests_per_sec": throughput - 1,
            "ttft_p99_ms": 100.0 + repetition,
            "tpot_p99_ms": 20.0 + repetition,
            "completed_requests": 200,
            "failures": 0,
        }), encoding="utf8")

    rows = aggregate_runs(tmp_path, repetitions=5)

    assert rows[0]["requests_per_sec_mean"] == 11.0
    assert rows[0]["requests_per_sec_ci95_low"] < 11.0 < rows[0]["requests_per_sec_ci95_high"]
    assert rows[0]["ttft_p99_ms_mean"] == 103.0
    assert rows[0]["completed_requests"] == 1_000


def test_common_experiment_artifact_helpers(tmp_path):
    from flexengine.research.experiments.artifacts import add_workload_trace, runs_complete

    destination = tmp_path / "config"
    assert not runs_complete(destination, 2)

    for repetition in (1, 2):
        run = destination / f"run-{repetition}"
        run.mkdir(parents=True)
        (run / "summary.json").write_text("{}", encoding="utf8")
    assert runs_complete(destination, 2)

    trace = tmp_path / "workloads" / "trace.json"
    assert add_workload_trace(["replay"], trace_path=trace) == [
        "replay", "--save-workload-trace", str(trace),
    ]
    trace.write_text("{}", encoding="utf8")
    assert add_workload_trace(["replay"], trace_path=trace) == [
        "replay", "--workload-trace", str(trace),
    ]
