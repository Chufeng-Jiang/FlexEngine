"""Fail closed if a formal Proposal run is missing raw or metadata artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flexengine.observability.environment import validate_run_metadata


def nonempty_lines(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf8").splitlines())


def audit(root: Path, configurations: int, repetitions: int, requests: int) -> dict:
    spec_paths = sorted(root.glob("E*/**/spec.json"))
    errors = []
    if len(spec_paths) != configurations:
        errors.append(f"expected {configurations} configurations, found {len(spec_paths)}")
    total_completed = 0
    for spec_path in spec_paths:
        directory = spec_path.parent
        metadata_path = directory / "engine-trace" / "run_metadata.json"
        request_trace = directory / "engine-trace" / "requests.jsonl"
        try:
            missing = validate_run_metadata(json.loads(metadata_path.read_text(encoding="utf8")))
            if missing:
                errors.append(f"{directory}: missing engine metadata fields: {missing}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            errors.append(f"{directory}: invalid engine metadata: {error}")
        for repetition in range(1, repetitions + 1):
            run = directory / f"run-{repetition}"
            required = (run / "client_metadata.json", run / "summary.json", run / "client_requests.jsonl")
            if not all(path.exists() for path in required):
                errors.append(f"{run}: missing client artifact")
                continue
            summary = json.loads(required[1].read_text(encoding="utf8"))
            if summary["offered_requests"] != requests or summary["completed_requests"] != requests:
                errors.append(f"{run}: request count mismatch")
            if nonempty_lines(required[2]) != requests:
                errors.append(f"{run}: raw client line count mismatch")
            total_completed += summary["completed_requests"]
        if request_trace.exists() and nonempty_lines(request_trace) < repetitions * requests:
            errors.append(f"{directory}: server request trace is incomplete")
    result = {"configurations": len(spec_paths), "completed_requests": total_completed, "errors": errors}
    if errors:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/results/proposal_20261108")
    parser.add_argument("--configurations", type=int, default=39)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--requests", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.input), args.configurations, args.repetitions, args.requests), indent=2))


if __name__ == "__main__":
    main()
