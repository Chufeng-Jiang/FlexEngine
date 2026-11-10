"""Resumable warm benchmark runner for the post-Proposal research upgrade."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from experiments.common import add_workload_trace, stop_process, wait_ready
except ModuleNotFoundError:
    from common import add_workload_trace, stop_process, wait_ready

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def _value(value) -> str:
    return str(value).lower().replace("_", "-")


def load_upgrade_matrix(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf8"))
    defaults = document["defaults"]
    specs = []
    for study in document["studies"]:
        keys = list(study["matrix"])
        for values in itertools.product(*(study["matrix"][key] for key in keys)):
            spec = {**defaults, **study.get("fixed", {}), **dict(zip(keys, values)), "study": study["study"]}
            suffix = "-".join(f"{key}-{_value(spec[key])}" for key in keys)
            spec["name"] = f"{study['study']}-{suffix}"
            specs.append(spec)
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("upgrade matrix names must be unique")
    return specs




def run_spec(spec: dict, output_root: Path) -> None:
    destination = output_root / spec["name"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf8")
    command = [
        sys.executable, "scripts/run_server.py", "--model", spec["model"], "--model-revision", REVISION,
        "--policy", spec["policy"], "--attention", spec["attention"],
        "--token-budget", str(spec["token_budget"]), "--prefill-chunk-size", str(spec["prefill_chunk_size"]),
        "--trace-dir", str(destination / "engine-trace"),
        "--prefix-cache" if spec["prefix_cache"] else "--no-prefix-cache",
        "--cuda-graph" if spec["cuda_graph"] else "--no-cuda-graph",
    ]
    with (destination / "server.log").open("a", encoding="utf8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            wait_ready(process)
            identity = f"upgrade-{spec['workload']}-prefix{spec['prefix_ratio']}-n{spec['measured_requests']}"
            trace = output_root / "workloads" / f"{identity}.json"
            for repetition in range(1, spec["repetitions"] + 1):
                run_dir = destination / f"run-{repetition}"
                if (run_dir / "summary.json").exists():
                    continue
                replay = [
                    sys.executable, "benchmarks/replay.py", "--model", spec["model"],
                    "--mode", spec["mode"], "--rate", str(spec["rate"]), "--concurrency", str(spec["concurrency"]),
                    "--workload", spec["workload"], "--prefix-ratio", str(spec["prefix_ratio"]),
                    "--requests", str(spec["measured_requests"]), "--warmup-requests", str(spec["warmup_requests"]),
                    "--max-output-tokens", "32", "--trace-identity", identity, "--output", str(run_dir),
                ]
                replay = add_workload_trace(replay, trace_path=trace)
                subprocess.run(replay, cwd=ROOT, check=True)
        finally:
            stop_process(process)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="experiments/config/upgrade_matrix.json")
    parser.add_argument("--output", default="experiments/results/upgrade_20261109")
    parser.add_argument("--studies", nargs="*", default=[])
    args = parser.parse_args()
    specs = load_upgrade_matrix(Path(args.matrix))
    if args.studies:
        specs = [spec for spec in specs if spec["study"] in args.studies]
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec['name']}", flush=True)
        run_spec(spec, Path(args.output))


if __name__ == "__main__":
    main()
