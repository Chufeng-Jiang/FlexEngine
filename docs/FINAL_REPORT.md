# FlexEngine 2.0 — Final Proposal Report

**Completed:** 2026-11-09  
**Target:** single NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16 GiB, WSL2  
**Status:** Proposal completed; code, correctness, profiling, E1-E6 and matched baseline evidence delivered.

## Delivered system

FlexEngine now includes a custom official-weight Qwen2.5 0.5B/1.5B decoder,
continuous batching, chunked prefill, FCFS/decode-first/SLO-aware scheduling,
reference-counted paged KV storage, exact-token shared prefix reuse, recompute
preemption, materialized/blockwise/fused-Triton attention, engine-integrated
1/2/4/8 CUDA Graph decode, OpenAI-style streaming completions, cancellation,
Prometheus/JSONL/NVTX telemetry, deterministic benchmark traces and strict
artifact auditing.

## Correctness and reliability

- CPU/integration: **75 passed, 11 skipped**.
- Complete official 0.5B GPU suite: **14 passed**.
- Final official 1.5B logits parity plus generation/KV reclamation: **2 passed**.
- Complete generation peaks: 0.5B 1.018 GB allocated; 1.5B 3.127 GB allocated.
- Failure coverage includes invalid input, bounded admission, KV exhaustion,
  preemption, disconnect/explicit cancellation, executor failure, overdue
  prefill fairness and shared-prefix cancellation ownership.
- Repository-wide Ruff: **all checks passed**.

## Formal experiment audit

The formal `proposal_20261108` artifact contains **39 configurations × 3
successful repetitions × 8 requests = 936/936 completed requests**. The audit
reported zero errors. Every accepted run has client raw data, client metadata,
summary, server request/step trace and environment metadata. Official model
snapshot SHAs are recorded.

Representative mean request throughput:

| Result | req/s |
|---|---:|
| 0.5B c1 / c4 / c8 | 0.671 / 1.054 / 1.112 |
| 1.5B c4 | 0.925 |
| W4 FCFS / decode-first / SLO-aware | 0.892 / 0.898 / 0.965 |
| token budget 64 / 512 | 0.801 / 1.034 |
| block-16 blockwise / materialized / Triton | 0.180 / 1.016 / 0.946 |
| 75% prefix off / on | 1.021 / 1.052 |
| materialized graph off / on | 1.108 / 3.034 |
| Triton graph off / on | 1.132 / 2.645 |

The materialized CUDA Graph comparison produced a **2.74×** throughput increase.
Prefix caching at 75% sharing reduced mean TTFT p95 from 948.8 ms to 790.5 ms.
The full table with sample standard deviations is
`experiments/results/proposal_20261108/REPORT.md` and `aggregate.csv`.

## Profiling and optimization

Nsight Systems recorded 24,318 launches in the short Python-blockwise baseline.
At 512 context tokens the microbenchmark measured:

- Python blockwise: 33.662 ms/call.
- Materialized SDPA: 0.540 ms/call.
- Fused Triton paged decode: 0.537 ms/call.

The fused kernel is about **62.7×** faster than the blockwise research baseline
in this microbenchmark. Nsight raw reports are retained under
`experiments/profiling/`, including the final `_paged_decode_kernel` NCU capture.

## Matched vLLM comparison

An isolated Python 3.12 environment with vLLM 0.33.0 reused the exact W1 saved
trace and matching model, FP16, max length, greedy parameters, concurrency,
request count, output cap and seed. Across three repetitions:

- vLLM: **14.779 ± 1.209 req/s**, TTFT p95 131.2 ± 111.9 ms.
- matched FlexEngine E1 c4: **1.054 ± 0.058 req/s**, TTFT p95 743.3 ms.

This honestly quantifies the remaining production-performance gap.

## Reproduction and environment

The complete English guide is `README.md`; Chinese quick start is
`docs/QUICK_START_CN.md`. Frozen package exports and machine/model metadata are
under `environment/`. Reproduction commands cover installation, Docker, parity,
generation, service smoke, E1-E6, artifact audit, Nsight and vLLM.

## Threats to validity

The formal matrix is intentionally small (eight requests per repetition and a
32-token output cap); SSE inter-event latency is not exact token ITL; cold
Triton compilation contributes to first-trial variance; only one GPU and two
small dense Qwen2.5 models were evaluated. The raw artifacts are supplied so
larger experiments can extend these conclusions without replacing them.

## Conclusion

All mandatory Proposal software and evidence items are complete. FlexEngine is
a correct, auditable research runtime with measured advanced optimizations, not
a production replacement for vLLM.
## Post-Proposal warm serving upgrade (2026-11-09)

The research follow-up is complete. It adds adaptive decode reservation and
bounded prefill, phase-level scheduler/executor/GPU/CUDA-Graph telemetry,
prefix lookup/hit/reused-token/eviction counters, and an optional FlashInfer
Tensor-Core paged-decode backend with Qwen2.5 GQA support.

The warm matrix used 20 warm-up plus 200 measured requests for each of 25
configurations. All 5,000 measured requests completed successfully. Each
configuration was run once at the user's direction, so reported values are
single-run point estimates rather than cross-run confidence intervals.

Key findings:

- Fair warm attention throughput was effectively tied: materialized 0.992,
  Triton 1.000, and FlashInfer 0.998 req/s. Triton delivered the strongest
  Goodput (0.620 req/s).
- CUDA Graph with Triton raised SLO pass rate from 64.5% to 98.5% and reduced
  TTFT P99 from 1,682 ms to 824 ms, while throughput stayed near 1 req/s.
- Adaptive scheduling achieved 91% SLO pass at arrival rate 0.5, 26% at 1.0,
  and 0% at 1.5. This identifies the service-quality saturation boundary.
- Token budget 256 was the best tested operating point for the rate-1 W4 load,
  with 27.5% SLO pass versus 18.5%, 20.0%, and 26.5% for 64, 128, and 512.
- Prefix reuse at 75% sharing did not improve throughput in this single run.
  TPOT P99 improved from 207 ms to 188 ms, while SLO pass changed 48% to 43%.

The earlier 62.7x Triton figure remains valid only for the isolated kernel
microbenchmark against the Python blockwise reference. It is not presented as
an end-to-end engine speedup. Cold compilation, dispatch, synchronization,
prefill, GEMM and scheduler work explain why kernel and system measurements
differ; the fair experiment uses SDPA prefill for both fused decode backends.

Final verification after the upgrade: **102 passed, 4 skipped**, with repository
Ruff clean. Environment exports include FlashInfer 0.7.3 and the validated
PyTorch/CUDA stack. The formal upgrade aggregate is
`experiments/results/upgrade_20261109/aggregate.csv`; raw traces and diagnostic
aborted runs are retained alongside it for auditability.
