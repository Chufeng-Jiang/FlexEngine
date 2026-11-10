# FlexEngine RTX 3080 Ti Benchmark Report

**Date:** 2026-11-08  
**Status:** Qwen2.5-0.5B official parity passed; W1 benchmark completed with three independent runs per configuration

## Test environment

- GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16 GiB, compute capability 8.6
- Host: Windows + WSL2 Ubuntu 22.04
- PyTorch: 2.15.0+cu132
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, FP16
- Runtime: FlexEngine, `decode_first`, 512 KV blocks, maximum model length 2048
- Workload: W1, 24 requests/run, 128 requested output tokens, seed 42
- SLO: TTFT <= 2000 ms and every client-observed inter-event interval <= 200 ms
- Repeats: 3 independent runs/configuration; mean and sample standard deviation are reported

The official Qwen2.5-0.5B logits parity test and both tiny-random attention parity tests passed before benchmarking.

## Mean results (three runs)

Values are mean +/- sample standard deviation across runs.

| Backend | Concurrency | Completed | SLO pass | Requests/s | Server tokens/s | TTFT p95 (ms) | Server ITL p95 (ms) | SLO goodput (req/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Materialized SDPA | 1 | 72/72 | 72/72 | 0.1587 +/- 0.0039 | 20.31 +/- 0.50 | 74.43 +/- 7.64 | 61.66 +/- 1.77 | 0.1587 +/- 0.0039 |
| Materialized SDPA | 2 | 72/72 | 72/72 | 0.2240 +/- 0.0056 | 28.67 +/- 0.72 | 191.59 +/- 12.80 | 85.99 +/- 1.49 | 0.2240 +/- 0.0056 |
| Materialized SDPA | 4 | 72/72 | 61/72 | 0.2744 +/- 0.0115 | 35.12 +/- 1.47 | 430.55 +/- 40.18 | 149.65 +/- 24.05 | 0.2334 +/- 0.0589 |
| Materialized SDPA | 8 | 72/72 | 0/72 | 0.3335 +/- 0.0035 | 42.69 +/- 0.44 | 969.00 +/- 82.70 | 212.43 +/- 3.17 | 0.0000 +/- 0.0000 |
| Blockwise PyTorch | 4 | 72/72 | 0/72 | 0.0304 +/- 0.0021 | 3.89 +/- 0.27 | 2384.18 +/- 474.88 | 1404.28 +/- 140.01 | 0.0000 +/- 0.0000 |

## Findings

1. **Stable SLO operating point:** Materialized concurrency 2 completed all 72 requests and passed the configured SLO for all 72. It provides 28.67 +/- 0.72 server tokens/s.
2. **Best measured average goodput:** Materialized concurrency 4 reaches 0.2334 +/- 0.0589 good requests/s, but only 61/72 requests pass the SLO. Its tail latency is materially less stable than concurrency 2.
3. **Maximum raw throughput:** Materialized concurrency 8 reaches 42.69 +/- 0.44 tokens/s, but server ITL p95 averages 212.43 ms and no request passes the strict all-interval SLO.
4. **Blockwise baseline bottleneck:** At concurrency 4, the page-by-page PyTorch implementation reaches only 3.89 +/- 0.27 tokens/s. Materialized SDPA is about 9.02x faster by mean server token throughput at the matched concurrency.
5. **Reliability:** All 360 benchmark requests completed; there were no HTTP failures, admission failures, CUDA faults or incomplete requests.

## Interpretation

For latency-sensitive use under the configured SLO, concurrency 2 is the defensible default. Concurrency 4 is useful when higher aggregate goodput is preferred and occasional ITL violations are acceptable. Concurrency 8 maximizes raw throughput but is outside the current ITL objective. The current Blockwise backend is a correctness and architecture baseline, not a competitive performance backend; the result supports prioritizing a fused Triton/CUDA paged-attention kernel.

## Validity and limitations

- These measurements cover W1 and Qwen2.5-0.5B only; W4, larger models and policy comparisons remain pending.
- Runs used the same seed and request order. They were executed sequentially rather than randomized ABBA order, so thermal or clock drift may affect later runs.
- The RTX 3080 Ti is a laptop GPU with a reported 16 GiB; results should not be generalized to desktop 3080 Ti hardware.
- Client SSE inter-event timing and server sampled-token ITL are both retained. Reported table ITL uses the server trace.
- The all-interval SLO is intentionally strict: one interval above 200 ms makes the request fail the SLO.
- No claim is made that FlexEngine outperforms vLLM, SGLang or another production runtime; no matched external-runtime comparison was run.

## Artifacts

- Per-run client data: `results/w1_*`
- Per-run server traces: each run directory's `requests.jsonl`
- Fifteen-run raw table: `results/analysis_3runs_20261108/aggregate.csv`
- Three-run mean/standard-deviation table: `results/analysis_3runs_20261108/average_summary.csv`
- Generated plots and per-run Markdown: `results/analysis_3runs_20261108/`
