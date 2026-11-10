"""Reconstruct summary tables/figures from one or more benchmark folders.

Pass folders produced by `flexengine.benchmarking.runner`. Requires matplotlib.
Can also aggregate server-side `requests.jsonl` when --server-traces is set.
Never inserts fabricated experimental data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def pct(vals, percent):
    if not vals:
        return None
    sorted_vals = sorted(vals)
    idx = (len(sorted_vals) - 1) * percent / 100
    a, b = int(idx), min(len(vals)-1, int(idx)+1)
    return sorted_vals[a] + (sorted_vals[b]-sorted_vals[a]) * (idx-a)


def load_jsonl(file):
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf8").splitlines() if line.strip()]


def analyze(run_dir: Path):
    client_path = run_dir / "client_requests.jsonl"
    server_path = run_dir / "requests.jsonl"
    client = load_jsonl(client_path)
    server = load_jsonl(server_path)
    info = json.loads((run_dir / "client_metadata.json").read_text()) if (run_dir / "client_metadata.json").exists() else {}
    good = [r for r in client if r.get("slo_pass")]
    completed = [r for r in client if r.get("completed")]
    duration = (max(r["end_t"] for r in client)-min(r["sent_t"] for r in client)) if client else None
    rows = {
        "run_dir": str(run_dir), "model": info.get("args",{}).get("model"),
        "mode": info.get("args",{}).get("mode"), "workload": info.get("args",{}).get("workload"),
        "concurrency": info.get("args",{}).get("concurrency"),
        "offered_rate": info.get("args",{}).get("rate"),
        "offered":len(client), "completed":len(completed),"failed":len(client)-len(completed),
        "slo_pass":len(good),
        "request_throughput":len(completed)/duration if duration else None,
        "slo_goodput":len(good)/duration if duration else None,
        "ttft_p95_ms":pct([r["ttft_ms"] for r in client if r.get("ttft_ms") is not None],95),
        "itl_event_p95_ms":pct([t for r in client for t in r.get("itl_ms",[])],95),
        "e2e_p99_ms":pct([r["e2e_ms"] for r in completed],99),
        "server_output_tokens": sum(r.get("output_tokens", 0) for r in server),
        "server_token_itl_p95_ms":pct([t for r in server for t in r.get("itl_ms",[])],95),
        "server_token_throughput":sum(r.get("output_tokens",0) for r in server)/duration if duration and server else None,
    }
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("runs",nargs="+",help="One or more benchmark run folders")
    p.add_argument("--output",default="results/analysis")
    p.add_argument("--plots", action="store_true")
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=[analyze(Path(r)) for r in args.runs]
    with (out/"aggregate.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    md=["# FlexEngine Benchmark Results", "", "**Generated from recorded files; no synthetic measurements inserted.**", "",
        "| Run | Completed/Offered | TTFT P95 (ms) | Server ITL P95 (ms) | SLO Goodput (req/s) |",
        "|---|---:|---:|---:|---:|"]
    def show(x): return "—" if x is None else f"{x:.2f}" if isinstance(x,float) else str(x)
    for row in rows:
        md.append(f"| {Path(row['run_dir']).name} | {row['completed']}/{row['offered']} | {show(row['ttft_p95_ms'])} | {show(row['server_token_itl_p95_ms'])} | {show(row['slo_goodput'])} |")
    md += ["", "## Cautions", "", "- Compare only under matched model/precision/trace/seed/hardware.",
           "- `server_token_itl_p95_ms` requires server requests.jsonl copied into the run folder.",
           "- SSE event timing can differ from per-token latency; see README.",
           "- 3–5 repeats are recommended for defensible statistics."]
    (out/"RESULTS.md").write_text("\n".join(md)+"\n",encoding="utf8")
    if args.plots:
        import matplotlib.pyplot as plt
        labels=[Path(r["run_dir"]).name for r in rows]
        fig,ax=plt.subplots(figsize=(max(7,len(rows)*1.2),4.2))
        ax.bar(labels,[r["slo_goodput"] or 0 for r in rows])
        ax.set_ylabel("SLO goodput (requests/sec)")
        ax.set_title("FlexEngine: SLO Goodput by Run")
        ax.tick_params(axis="x",rotation=35)
        fig.tight_layout(); fig.savefig(out/"slo_goodput.png",dpi=170); plt.close(fig)
        fig,ax=plt.subplots(figsize=(max(7,len(rows)*1.2),4.2))
        ax.bar(labels,[r["ttft_p95_ms"] or 0 for r in rows])
        ax.set_ylabel("TTFT P95 (ms)")
        ax.set_title("FlexEngine: Time to First Token (P95)")
        ax.tick_params(axis="x",rotation=35)
        fig.tight_layout(); fig.savefig(out/"ttft_p95.png",dpi=170); plt.close(fig)
    print("Created",out/"aggregate.csv", "and",out/"RESULTS.md")


if __name__=="__main__":
    main()
