import json
from pathlib import Path

from experiments.analyze_upgrade import aggregate_runs
from experiments.run_upgrade import load_upgrade_matrix


def test_upgrade_matrix_has_unique_warm_ablation_specs():
    specs = load_upgrade_matrix(Path("experiments/config/upgrade_matrix.json"))

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
