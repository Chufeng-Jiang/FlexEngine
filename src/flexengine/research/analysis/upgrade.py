"""Aggregate warm upgrade runs with bootstrap confidence intervals."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from flexengine.research.benchmark.statistics import bootstrap_ci

METRICS = ("requests_per_sec", "goodput_requests_per_sec", "ttft_p99_ms", "tpot_p99_ms")


def aggregate_runs(root: Path, repetitions: int = 1) -> list[dict]:
    rows = []
    for config_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if "-aborted" in config_dir.name:
            continue
        summaries = []
        for repetition in range(1, repetitions + 1):
            path = config_dir / f"run-{repetition}" / "summary.json"
            if path.exists():
                summaries.append(json.loads(path.read_text(encoding="utf8")))
        if len(summaries) != repetitions:
            continue
        spec_path = config_dir / "spec.json"
        row = json.loads(spec_path.read_text(encoding="utf8")) if spec_path.exists() else {"name": config_dir.name}
        row["repetitions"] = repetitions
        row["completed_requests"] = sum(item["completed_requests"] for item in summaries)
        row["failures"] = sum(item["failures"] for item in summaries)
        for metric in METRICS:
            values = [float(item[metric]) for item in summaries if item.get(metric) is not None]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            if values:
                low, high = bootstrap_ci(values, confidence=0.95, resamples=10_000, seed=42)
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/upgrade_20261109")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.input)
    rows = aggregate_runs(root, args.repetitions)
    if not rows:
        raise SystemExit("no complete upgrade configurations")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "aggregate.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"configurations": len(rows), "output": str(root / "aggregate.csv")}, indent=2))


if __name__ == "__main__":
    main()
