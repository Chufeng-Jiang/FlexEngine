# Serving Research Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade FlexEngine with warm steady-state benchmarking, causal telemetry, adaptive token budgeting, deep prefix metrics, and an optional FlashInfer comparison.

**Architecture:** Benchmark measurement stays client-side while engine phase telemetry stays server-side. Scheduler policy remains deterministic and bounded; optional third-party kernels are isolated behind the existing attention backend interface and never silently substituted.

**Tech Stack:** Python 3.13, PyTorch CUDA, Triton, optional FlashInfer, FastAPI/httpx, pytest, Nsight Systems/Compute.

**Spec:** `docs/superpowers/specs/2026-11-09-serving-research-upgrade-design.md`

## Global Constraints

- Target is one RTX 3080 Ti Laptop GPU under WSL2.
- Existing E1-E6 artifacts remain immutable.
- Warmup observations cannot enter measured throughput or latency distributions.
- Optional backends fail closed and report availability explicitly.
- All comparisons reuse identical saved request traces and model revisions.

## Review Focus

- Zero measured requests or warmup greater than trace length must fail validation.
- Poisson warmup must not distort measured arrival timing.
- Adaptive scheduling must not starve decode or overdue first-token work.
- Cache hit counters must distinguish request hits from reused tokens.
- Missing FlashInfer must skip explicitly rather than fall back to materialized attention.

---

### Task 1: Warm benchmark statistics

**Files:**
- Modify: `src/benchmarks/replay.py`
- Create: `src/benchmarks/statistics.py`
- Test: `tests/unit/test_benchmark_statistics.py`

**Interfaces:**
- Produces: `bootstrap_ci(values, statistic, confidence, resamples, seed)` and warmup-aware benchmark summaries.

- [ ] Write failing tests for deterministic bootstrap intervals, warmup exclusion, TPOT P99, and invalid counts.
- [ ] Run `pytest tests/unit/test_benchmark_statistics.py -v` and confirm failure.
- [ ] Implement warmup requests, separate cold/warm timing, P99 metrics, and deterministic 95% bootstrap CIs.
- [ ] Run the focused tests and confirm pass.

### Task 2: Adaptive decode-reserved scheduler

**Files:**
- Modify: `src/flexengine/config.py`
- Modify: `src/flexengine/scheduler/policies.py`
- Modify: `src/scripts/run_server.py`
- Test: `tests/unit/test_scheduler.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `adaptive` policy with bounded `prefill_chunk_size`, decode reservation, and observed-step feedback.

- [ ] Write failing tests for decode reservation, chunk bounds, prefill progress, and environment parsing.
- [ ] Run focused scheduler/config tests and confirm failure.
- [ ] Implement the minimal adaptive policy and configuration surface.
- [ ] Run focused tests and confirm pass.

### Task 3: Phase and graph telemetry

**Files:**
- Modify: `src/flexengine/runtime.py`
- Modify: `src/flexengine/execution/cuda_graph.py`
- Modify: `src/flexengine/observability/metrics.py`
- Test: `tests/integration/test_runtime.py`
- Test: `tests/gpu/test_engine_cuda_graph.py`

**Interfaces:**
- Produces: per-step `scheduler_ms`, `executor_wall_ms`, `gpu_ms`, graph bucket hit/fallback/padded-slot deltas, and synchronization accounting.

- [ ] Write failing trace assertions for phase timings and graph deltas.
- [ ] Run focused tests and confirm failure.
- [ ] Instrument without adding forced synchronization outside profiling mode.
- [ ] Run focused CPU/GPU tests and confirm pass.

### Task 4: Prefix-cache effectiveness telemetry

**Files:**
- Modify: `src/flexengine/kv_cache/prefix.py`
- Modify: `src/flexengine/runtime.py`
- Modify: `src/flexengine/observability/metrics.py`
- Test: `tests/unit/test_prefix_cache.py`
- Test: `tests/integration/test_prefix_runtime.py`

**Interfaces:**
- Produces: lookup count/time, request hit rate inputs, reused tokens/blocks/bytes, insertions, and evictions.

- [ ] Write failing tests for hit/miss lookup accounting and reused block/token totals.
- [ ] Run focused tests and confirm failure.
- [ ] Implement cumulative counters and trace fields.
- [ ] Run focused tests and confirm pass.

### Task 5: Optional FlashInfer backend

**Files:**
- Create: `src/flexengine/attention/flashinfer_backend.py`
- Modify: `src/flexengine/config.py`
- Modify: `src/flexengine/models/qwen2.py`
- Modify: `pyproject.toml`
- Create: `tests/gpu/test_flashinfer_attention.py`

**Interfaces:**
- Produces: `flashinfer_available()` and graph-compatible batch paged decode adapter.

- [ ] Write parity tests that explicitly skip when FlashInfer is unavailable.
- [ ] Verify the unsupported path fails closed.
- [ ] Implement the optional backend using caller-owned stable workspace and paged metadata.
- [ ] Run parity at lengths 1, 17, 129, and 513 with non-contiguous pages and GQA.

### Task 6: Research matrix, profiling, and report

**Files:**
- Create: `data/config/upgrade_matrix.json`
- Create: `src/experiments/run_upgrade.py`
- Create: `src/experiments/analyze_upgrade.py`
- Create: `results/profiling/profile_upgrade.sh`
- Modify: `README.md`
- Modify: `docs/FINAL_REPORT.md`

**Interfaces:**
- Consumes: metrics from Tasks 1-5.
- Produces: five-repetition warm artifact, confidence intervals, SLO load curves, attention/graph breakdown, and ablation report.

- [ ] Test matrix uniqueness, saved-trace identity, result completeness, and analyzer statistics.
- [ ] Run the warm matrix with 20 warmup plus 200 measured requests and one repetition (user override on 2026-11-09).
- [ ] Capture Nsight Systems for materialized/Triton/FlashInfer with graph off/on and Nsight Compute for attention kernels.
- [ ] Generate tables/plots and document cold start separately from steady state.
- [ ] Run full pytest, Ruff, artifact audit, and reproduction smoke test.

## Self-review

All design goals map to Tasks 1-6. Interface names are unique and later tasks consume only declared outputs. Failure cases in Review Focus are assigned to Tasks 1, 2, 4, and 5. The plan deliberately retains the original E1-E6 evidence and treats FlashInfer as optional because binary support on Python 3.13/Ampere is not guaranteed.
