"""Resumable E1-E6 HTTP benchmark runner with repeated, trace-locked trials."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from flexengine.engine.config import matched_kv_blocks
from flexengine.research.experiments.artifacts import add_workload_trace, runs_complete, stop_process, wait_ready
from flexengine.research.experiments.matrix import load_matrix, spec_dict

ROOT = Path(__file__).resolve().parents[5]
MODEL_REVISIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}




def server_command(spec, trace_dir: Path) -> list[str]:
    blocks = matched_kv_blocks(1024, 16, spec.block_size)
    command = [
        sys.executable, "-m", "flexengine.serving.cli.server", "--model", spec.model,
        "--policy", spec.policy, "--attention", spec.attention,
        "--trace-dir", str(trace_dir), "--token-budget", str(spec.token_budget),
        "--block-size", str(spec.block_size), "--num-blocks", str(blocks),
        "--max-num-seqs", str(max(8, spec.concurrency)),
    ]
    if revision := MODEL_REVISIONS.get(spec.model):
        command += ["--model-revision", revision]
    command.append("--prefix-cache" if spec.prefix_cache else "--no-prefix-cache")
    command.append("--cuda-graph" if spec.cuda_graph else "--no-cuda-graph")
    return command


def run_spec(spec, output_root: Path, repetitions: int, requests: int, max_output_tokens: int) -> None:
    destination = output_root / spec.experiment / spec.name
    destination.mkdir(parents=True, exist_ok=True)
    if runs_complete(destination, repetitions):
        print(f"SKIP {spec.name}", flush=True)
        return
    trace_dir = destination / "engine-trace"
    log_path = ROOT / "logs" / "proposal" / spec.name / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["FLEX_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("a", encoding="utf8") as log:
        process = subprocess.Popen(
            server_command(spec, trace_dir), cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(process)
            identity = f"{spec.workload}-prefix{spec.prefix_ratio}-n{requests}-out{max_output_tokens}"
            workload_trace = output_root / "workloads" / f"{identity}.json"
            for repetition in range(1, repetitions + 1):
                run_dir = destination / f"run-{repetition}"
                if (run_dir / "summary.json").exists():
                    continue
                command = [
                    sys.executable, "-m", "flexengine.research.benchmark.runner", "--model", spec.model,
                    "--workload", spec.workload, "--prefix-ratio", str(spec.prefix_ratio),
                    "--concurrency", str(spec.concurrency), "--requests", str(requests),
                    "--max-output-tokens", str(max_output_tokens), "--seed", "42",
                    "--trace-identity", identity, "--output", str(run_dir),
                ]
                command = add_workload_trace(command, trace_path=workload_trace)
                subprocess.run(command, cwd=ROOT, check=True)
            (destination / "spec.json").write_text(json.dumps(spec_dict(spec), indent=2), encoding="utf8")
        finally:
            stop_process(process)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="data/config/proposal_matrix.json")
    parser.add_argument("--output", default="results/proposal_20261108")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--experiments", nargs="*", default=[])
    args = parser.parse_args()
    specs = load_matrix(Path(args.matrix))
    if args.experiments:
        specs = [spec for spec in specs if spec.experiment in args.experiments]
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec.name}", flush=True)
        run_spec(spec, Path(args.output), args.repetitions, args.requests, args.max_output_tokens)


if __name__ == "__main__":
    main()


WARM_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def load_warm_matrix(path: Path) -> list[dict]:
    """Expand the post-Proposal warm-serving study matrix."""
    import itertools

    document = json.loads(path.read_text(encoding="utf8"))
    defaults = document["defaults"]
    specs = []
    for study in document["studies"]:
        keys = list(study["matrix"])
        for values in itertools.product(*(study["matrix"][key] for key in keys)):
            spec = {**defaults, **study.get("fixed", {}), **dict(zip(keys, values)), "study": study["study"]}
            suffix = "-".join(f"{key}-{str(spec[key]).lower().replace('_', '-')}" for key in keys)
            spec["name"] = f"{study['study']}-{suffix}"
            specs.append(spec)
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("warm benchmark matrix names must be unique")
    return specs


def run_warm_spec(spec: dict, output_root: Path) -> None:
    """Run one warm-serving ablation configuration."""
    destination = output_root / spec["name"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf8")
    command = [
        sys.executable, "-m", "flexengine.serving.cli.server", "--model", spec["model"],
        "--model-revision", WARM_MODEL_REVISION, "--policy", spec["policy"],
        "--attention", spec["attention"], "--token-budget", str(spec["token_budget"]),
        "--prefill-chunk-size", str(spec["prefill_chunk_size"]),
        "--trace-dir", str(destination / "engine-trace"),
        "--prefix-cache" if spec["prefix_cache"] else "--no-prefix-cache",
        "--cuda-graph" if spec["cuda_graph"] else "--no-cuda-graph",
    ]
    log_path = ROOT / "logs" / "warm-benchmarks" / spec["name"] / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            wait_ready(process)
            identity = f"warm-{spec['workload']}-prefix{spec['prefix_ratio']}-n{spec['measured_requests']}"
            trace = output_root / "workloads" / f"{identity}.json"
            for repetition in range(1, spec["repetitions"] + 1):
                run_dir = destination / f"run-{repetition}"
                if (run_dir / "summary.json").exists():
                    continue
                replay = [
                    sys.executable, "-m", "flexengine.research.benchmark.runner", "--model", spec["model"],
                    "--mode", spec["mode"], "--rate", str(spec["rate"]), "--concurrency", str(spec["concurrency"]),
                    "--workload", spec["workload"], "--prefix-ratio", str(spec["prefix_ratio"]),
                    "--requests", str(spec["measured_requests"]), "--warmup-requests", str(spec["warmup_requests"]),
                    "--max-output-tokens", "32", "--trace-identity", identity, "--output", str(run_dir),
                ]
                subprocess.run(add_workload_trace(replay, trace_path=trace), cwd=ROOT, check=True)
        finally:
            stop_process(process)


def warm_main() -> None:
    """CLI entry point for the post-Proposal warm-serving studies."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="data/config/warm_benchmark_matrix.json")
    parser.add_argument("--output", default="results/upgrade_20261109")
    parser.add_argument("--studies", nargs="*", default=[])
    args = parser.parse_args()
    specs = load_warm_matrix(Path(args.matrix))
    if args.studies:
        specs = [spec for spec in specs if spec["study"] in args.studies]
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec['name']}", flush=True)
        run_warm_spec(spec, Path(args.output))
