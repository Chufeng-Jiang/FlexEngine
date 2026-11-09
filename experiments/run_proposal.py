"""Resumable E1-E6 HTTP benchmark runner with repeated, trace-locked trials."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from run_matrix import load_matrix, spec_dict

from flexengine.config import matched_kv_blocks

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}


def wait_ready(process: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        try:
            response = httpx.get("http://127.0.0.1:8000/healthz", timeout=1.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.25)
    raise TimeoutError("server did not become ready")


def server_command(spec, trace_dir: Path) -> list[str]:
    blocks = matched_kv_blocks(1024, 16, spec.block_size)
    command = [
        sys.executable, "scripts/run_server.py", "--model", spec.model,
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
    if all((destination / f"run-{index}" / "summary.json").exists() for index in range(1, repetitions + 1)):
        print(f"SKIP {spec.name}", flush=True)
        return
    trace_dir = destination / "engine-trace"
    log_path = destination / "server.log"
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
                    sys.executable, "benchmarks/replay.py", "--model", spec.model,
                    "--workload", spec.workload, "--prefix-ratio", str(spec.prefix_ratio),
                    "--concurrency", str(spec.concurrency), "--requests", str(requests),
                    "--max-output-tokens", str(max_output_tokens), "--seed", "42",
                    "--trace-identity", identity, "--output", str(run_dir),
                ]
                if workload_trace.exists():
                    command += ["--workload-trace", str(workload_trace)]
                else:
                    workload_trace.parent.mkdir(parents=True, exist_ok=True)
                    command += ["--save-workload-trace", str(workload_trace)]
                subprocess.run(command, cwd=ROOT, check=True)
            (destination / "spec.json").write_text(json.dumps(spec_dict(spec), indent=2), encoding="utf8")
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="experiments/config/proposal_matrix.json")
    parser.add_argument("--output", default="experiments/results/proposal_20261108")
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
