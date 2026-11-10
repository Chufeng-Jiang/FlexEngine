"""Aggregate repeated E1-E6 runs into reproducible CSV and Markdown."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = (
    "requests_per_sec", "goodput_requests_per_sec", "slo_pass_rate",
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "e2e_p95_ms",
    "sse_inter_event_p95_ms",
)


def collect(root: Path, required_repetitions: int | None = None) -> list[dict]:
    rows = []
    for spec_path in sorted(root.glob("E*/**/spec.json")):
        spec = json.loads(spec_path.read_text(encoding="utf8"))
        summaries = []
        for repetition in range(1, (required_repetitions or 999) + 1):
            path = spec_path.parent / f"run-{repetition}" / "summary.json"
            if not path.exists():
                if required_repetitions is None:
                    break
                continue
            summaries.append(json.loads(path.read_text(encoding="utf8")))
        if required_repetitions is not None and len(summaries) != required_repetitions:
            continue
        row = {**spec, "repetitions": len(summaries)}
        row["confidence"] = f"mean ± sample stdev, n={len(summaries)}"
        row["completed_requests"] = sum(item["completed_requests"] for item in summaries)
        row["failures"] = sum(item["failures"] for item in summaries)
        for metric in METRICS:
            values = [item[metric] for item in summaries if item.get(metric) is not None]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            row[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], expected_repetitions: int) -> None:
    lines = [
        "# Proposal E1-E6 repeated benchmark",
        "",
        f"Configurations: {len(rows)}. Expected repetitions/configuration: {expected_repetitions}.",
        "All values are arithmetic means; `±` is the sample standard deviation.",
        "",
        "| Exp | Configuration | Reps | req/s | TTFT p95 ms | ITL event p95 ms | failures |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['name']} | {row['repetitions']} | "
            f"{row['requests_per_sec_mean']:.3f} ± {row['requests_per_sec_stdev']:.3f} | "
            f"{row['ttft_p95_ms_mean']:.1f} ± {row['ttft_p95_ms_stdev']:.1f} | "
            f"{row['sse_inter_event_p95_ms_mean']:.1f} ± {row['sse_inter_event_p95_ms_stdev']:.1f} | "
            f"{row['failures']} |"
        )
    incomplete = [row["name"] for row in rows if row["repetitions"] != expected_repetitions]
    lines += ["", "## Completeness", ""]
    lines.append("All discovered configurations are complete." if not incomplete else "Incomplete: " + ", ".join(incomplete))
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/proposal_20261108")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.input)
    rows = collect(root, args.repetitions)
    if not rows:
        raise SystemExit(f"no completed specs found under {root}")
    write_csv(root / "aggregate.csv", rows)
    write_report(root / "REPORT.md", rows, args.repetitions)
    print(json.dumps({"configurations": len(rows), "aggregate": str(root / "aggregate.csv")}, indent=2))


if __name__ == "__main__":
    main()


WARM_METRICS = ("requests_per_sec", "goodput_requests_per_sec", "ttft_p99_ms", "tpot_p99_ms")


def collect_warm(root: Path, repetitions: int = 1) -> list[dict]:
    """Aggregate complete warm-serving runs with bootstrap confidence intervals."""
    from flexengine.research.benchmark.statistics import bootstrap_ci

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
        for metric in WARM_METRICS:
            values = [float(item[metric]) for item in summaries if item.get(metric) is not None]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            if values:
                low, high = bootstrap_ci(values, confidence=0.95, resamples=10_000, seed=42)
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return rows


def warm_main() -> None:
    """CLI entry point for warm-serving result aggregation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/upgrade_20261109")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.input)
    rows = collect_warm(root, args.repetitions)
    if not rows:
        raise SystemExit("no complete warm benchmark configurations")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(root / "aggregate.csv", [{field: row.get(field) for field in fields} for row in rows])
    print(json.dumps({"configurations": len(rows), "output": str(root / "aggregate.csv")}, indent=2))
