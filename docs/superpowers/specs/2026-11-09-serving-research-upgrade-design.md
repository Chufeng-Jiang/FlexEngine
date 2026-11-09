# Serving Research Upgrade Design

## Intent

Turn the E2-E6 observations into defensible systems conclusions by separating cold start from steady state, measuring service-quality goodput, exposing scheduler/cache/graph costs, and comparing the custom paged-attention kernel with an industry implementation.

## Goals

- Produce warm, trace-locked benchmarks with warmup exclusion, 95% confidence intervals, and P99 metrics.
- Measure dual TTFT/TPOT SLO goodput over offered load.
- Explain E4 and E6 with per-step phase and CUDA timing rather than throughput alone.
- Measure prefix-cache request hits, token reuse, lookup cost, and estimated prefill time saved.
- Add adaptive prefill budgeting without starving decode or overdue prefill.
- Add an optional FlashInfer backend when the installed GPU/software combination supports it; absence must be an explicit skip, never a silent fallback.

## Non-goals

- Multi-GPU prefill/decode disaggregation.
- Hopper-only FlashAttention-3 or FP8 claims on the Ampere RTX 3080 Ti.
- Claiming kernel microbenchmark speedup as end-to-end engine speedup.
- Replacing the existing reproducible E1-E6 artifact.

## Architecture

The HTTP benchmark owns workload warmup, steady-state sampling, bootstrap confidence intervals, and dual-SLO client observations. The runtime emits cumulative and per-step telemetry for scheduler planning, model execution, cache lookup, CUDA graph use, and prefix reuse. An adaptive scheduler reserves decode work first and fills the remaining measured token budget with bounded prefill chunks. Attention backends remain pluggable, with FlashInfer optional and fail-closed.

## Approaches considered

1. **Kernel-only optimization.** Lowest implementation cost, but cannot explain the end-to-end discrepancy and risks optimizing a non-dominant operator.
2. **Replace the engine with vLLM/FlashInfer.** Produces performance quickly but defeats the research goal of understanding the custom engine.
3. **Instrument, establish warm baselines, then optimize one factor at a time.** Selected because it preserves FlexEngine and produces causal evidence.

## Validation

Every new metric has a deterministic unit test. Scheduler changes are tested for budget bounds, decode reservation, prefill progress, and overload. Optional FlashInfer tests skip with an explicit reason when unavailable. GPU experiments use a fixed saved trace, 200 measured requests, 20 warmup requests, five repetitions, and preserve raw outputs.
