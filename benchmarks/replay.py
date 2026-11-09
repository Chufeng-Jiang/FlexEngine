"""Client-side benchmark: real HTTP/SSE, closed-loop and Poisson arrivals.

Emits raw client observations (including 429 failures), run metadata, and
summary. Does not pretend SSE chunks always equal individual model tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import httpx


def save_workload_trace(path: Path, identity: str, workload: list[dict]) -> None:
    path.write_text(json.dumps({"trace_identity": identity, "requests": workload}, indent=2), encoding="utf8")


def load_workload_trace(path: Path, identity: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf8"))
    if data.get("trace_identity") != identity:
        raise ValueError("trace identity mismatch")
    requests = data.get("requests")
    if not isinstance(requests, list):
        raise TypeError("invalid workload trace")
    return requests


def build_or_load_workload(
    workload: str,
    requests: int,
    max_output_tokens: int,
    seed: int,
    input_path: Path | None,
    output_path: Path | None,
    trace_identity: str,
    prefix_ratio: int = 0,
) -> list[dict]:
    if input_path is not None:
        loaded = load_workload_trace(input_path, trace_identity)
        if len(loaded) != requests:
            raise ValueError(f"workload trace has {len(loaded)} requests, expected {requests}")
        return loaded
    rng = random.Random(seed)
    generated = []
    for serial in range(requests):
        prompt_tokens, requested_output_tokens = make_workload(workload, rng)
        prompt = (
            build_shared_prompt(prompt_tokens, prefix_ratio, serial, rng)
            if workload == "w5" else build_prompt(prompt_tokens, rng)
        )
        generated.append(
            {
                "serial": serial,
                "prompt": prompt,
                "max_output_tokens": min(requested_output_tokens, max_output_tokens),
            }
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_workload_trace(output_path, trace_identity, generated)
    return generated


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def build_prompt(target_tokens, rng):
    # Approximate only; actual tokens must be measured server-side.
    sentences = ["The quick brown fox jumps over the lazy dog.",
                 "Please explain the practical trade-offs of inference scheduling.",
                 "A GPU runs many small matrix operations during text generation."]
    return " ".join(rng.choice(sentences) for _ in range(max(1, target_tokens // 11)))


def build_shared_prompt(target_tokens, prefix_ratio, serial, rng):
    if prefix_ratio not in (0, 25, 50, 75):
        raise ValueError("prefix_ratio must be 0, 25, 50 or 75")
    units = max(4, target_tokens // 8)
    shared = units * prefix_ratio // 100
    common = ["Shared inference systems context sentence."] * shared
    unique = [f"Request {serial} unique suffix {rng.randrange(1_000_000)}."]
    unique.extend("Independent scheduling payload sentence." for _ in range(units - shared - 1))
    return " ".join(common + unique)


def make_workload(name, rng):
    if name == "w2":
        return 1024, 32
    if name == "w3":
        return 256, 512
    if name == "w4":
        p = rng.random()
        return ((128,128) if p < .5 else (1024,32) if p < .8 else (256,512))
    if name == "w5":
        return 128, 128
    return 128, 128


async def send(client, url, model, serial, prompt, max_tokens, seed, ttft_slo_ms, itl_slo_ms):
    start = time.perf_counter()
    token_timestamps = []
    total_chars = 0
    status = None
    reason = None
    error = None
    try:
        async with client.stream("POST", url, json={"model":model, "prompt":prompt,
                                                    "max_tokens":max_tokens,"temperature":0.0,"seed": seed + serial,
                                                    "stream":True}) as resp:
            status = resp.status_code
            if status != 200:
                error = (await resp.aread()).decode("utf8")[:500]
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    data = json.loads(raw)
                    if "error" in data:
                        error = str(data["error"])
                        break
                    choice = data["choices"][0]
                    if choice["text"]:
                        token_timestamps.append(time.perf_counter())
                        total_chars += len(choice["text"])
                    if choice.get("finish_reason"):
                        reason = choice["finish_reason"]
    except Exception as ex:  # noqa: BLE001 - every per-request failure must be recorded in benchmark output
        error = repr(ex)
    end = time.perf_counter()
    itls = [(b-a)*1000 for a,b in pairwise(token_timestamps)]
    ttft = (token_timestamps[0] - start)*1000 if token_timestamps else None
    completed = status == 200 and reason in ("stop", "length") and error is None
    pass_slo = completed and ttft is not None and ttft <= ttft_slo_ms and all(x <= itl_slo_ms for x in itls)
    return {"serial":serial, "status":status, "completed":completed, "finish_reason":reason,
            "error":error, "sent_t":start, "end_t":end, "ttft_ms":ttft, "itl_ms":itls,
            "tpot_ms":(token_timestamps[-1]-token_timestamps[0])*1000/(len(token_timestamps)-1) if len(token_timestamps)>1 else None,
            "e2e_ms":(end-start)*1000, "sse_text_events":len(token_timestamps),
            "text_chars":total_chars, "slo_pass":bool(pass_slo), "max_tokens_requested":max_tokens}


async def run(args):
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"created_utc":datetime.now(timezone.utc).isoformat(), "args": vars(args),
                "timing": "client monotonic / first nonempty SSE text event; output token count server-side only",
                "seed": args.seed, "trace_identity": args.trace_identity}
    (out / "client_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    # Precompute workload before launching tasks: request i receives identical
    # prompt/output targets under every server scheduling strategy.
    workload = build_or_load_workload(
        workload=args.workload,
        requests=args.requests,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        input_path=Path(args.workload_trace) if args.workload_trace else None,
        output_path=Path(args.save_workload_trace) if args.save_workload_trace else None,
        trace_identity=args.trace_identity,
        prefix_ratio=args.prefix_ratio,
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        async def job(i):
            item = workload[i]
            async with sem:
                r = await send(client, args.url, args.model, i, item["prompt"], item["max_output_tokens"],
                               args.seed, args.ttft_slo_ms, args.itl_slo_ms)
            results.append(r)

        started = time.perf_counter()
        if args.mode == "poisson":
            tasks = []
            for i in range(args.requests):
                if i:
                    await asyncio.sleep(rng.expovariate(args.rate))
                tasks.append(asyncio.create_task(job(i)))
            await asyncio.gather(*tasks)
        else:
            await asyncio.gather(*(job(i) for i in range(args.requests)))
        elapsed = time.perf_counter() - started
    with (out / "client_requests.jsonl").open("w", encoding="utf8") as f:
        for r in sorted(results, key=lambda x:x["serial"]):
            f.write(json.dumps(r) + "\n")
    ttft = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    e2e = [r["e2e_ms"] for r in results if r["completed"]]
    itls = [x for r in results for x in r["itl_ms"]]
    finished = sum(r["completed"] for r in results)
    good = sum(r["slo_pass"] for r in results)
    summary = {"wall_seconds":elapsed, "offered_requests":len(results), "completed_requests":finished,
               "failures":len(results)-finished, "status_counts": dict(Counter(str(r["status"]) for r in results)),
               "requests_per_sec": finished/elapsed,
               "goodput_requests_per_sec": good/elapsed, "slo_pass_rate":good/len(results) if results else 0,
               "ttft_p50_ms":percentile(ttft, 50), "ttft_p95_ms":percentile(ttft, 95),
               "ttft_p99_ms":percentile(ttft, 99), "e2e_p95_ms":percentile(e2e, 95),
               "sse_inter_event_p95_ms":percentile(itls, 95),
               "caveat": "Use engine requests.jsonl for exact output tokens/s and per-token ITL. Client SSE inter-event gaps may skip empty-token events."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--mode", choices=["concurrency", "poisson"], default="concurrency")
    p.add_argument("--workload", choices=["w1", "w2", "w3", "w4", "w5"], default="w1")
    p.add_argument("--prefix-ratio", type=int, choices=[0, 25, 50, 75], default=0)
    p.add_argument("--requests", type=int, default=24)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trace-identity", default="default-trace")
    p.add_argument("--workload-trace", default=None)
    p.add_argument("--save-workload-trace", default=None)
    p.add_argument("--max-output-tokens", type=int, default=128)
    p.add_argument("--ttft-slo-ms", type=float, default=2000)
    p.add_argument("--itl-slo-ms", type=float, default=200)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--output", default="experiments/results/latest")
    args = p.parse_args()
    if min(args.requests, args.concurrency, args.rate, args.timeout) <= 0:
        p.error("requests, concurrency, rate, timeout must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
