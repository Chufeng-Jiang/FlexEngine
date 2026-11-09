import json
from pathlib import Path

import pytest

from benchmarks.replay import build_or_load_workload, load_workload_trace, save_workload_trace
from experiments.run_matrix import RunSpec, is_complete, load_matrix, mark_complete, run_id


def test_proposal_matrix_contains_controlled_e1_to_e6_subsets():
    specs = load_matrix(Path("experiments/config/proposal_matrix.json"))
    counts = {}
    for spec in specs:
        counts[spec.experiment] = counts.get(spec.experiment, 0) + 1

    assert counts == {"E1": 8, "E2": 6, "E3": 4, "E4": 9, "E5": 8, "E6": 4}
    assert len({spec.name for spec in specs}) == 39


def test_run_id_is_stable_and_repeat_specific():
    spec = RunSpec(
        experiment="E2",
        name="e2-w4-decode-first",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        workload="w4",
        concurrency=4,
        policy="decode_first",
        attention="materialized",
    )

    assert run_id(spec, 1) == "e2-w4-decode-first-r1"
    assert run_id(spec, 3) == "e2-w4-decode-first-r3"
    with pytest.raises(ValueError, match="repeat must be positive"):
        run_id(spec, 0)


def test_completion_marker_is_atomic_and_bound_to_run_identity(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert not is_complete(run_dir, "expected-r1")

    mark_complete(run_dir, "expected-r1", {"completed_requests": 24})

    assert is_complete(run_dir, "expected-r1")
    assert not is_complete(run_dir, "different-r1")
    assert json.loads((run_dir / "COMPLETE.json").read_text())["completed_requests"] == 24


def test_saved_workload_trace_round_trips_and_rejects_wrong_identity(tmp_path):
    path = tmp_path / "trace.json"
    workload = [
        {"serial": 0, "prompt": "alpha", "max_output_tokens": 8},
        {"serial": 1, "prompt": "beta", "max_output_tokens": 4},
    ]
    save_workload_trace(path, "trace-a", workload)

    assert load_workload_trace(path, "trace-a") == workload
    with pytest.raises(ValueError, match="trace identity mismatch"):
        load_workload_trace(path, "trace-b")


def test_replay_consumes_saved_workload_without_regeneration(tmp_path):
    path = tmp_path / "trace.json"
    workload = [
        {"serial": 0, "prompt": "fixed alpha", "max_output_tokens": 8},
        {"serial": 1, "prompt": "fixed beta", "max_output_tokens": 4},
    ]
    save_workload_trace(path, "matched-trace", workload)

    loaded = build_or_load_workload(
        workload="w4",
        requests=2,
        max_output_tokens=128,
        seed=999,
        input_path=path,
        output_path=None,
        trace_identity="matched-trace",
    )

    assert loaded == workload
