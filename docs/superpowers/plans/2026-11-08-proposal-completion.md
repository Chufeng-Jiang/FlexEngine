# FlexEngine Proposal Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every mandatory FlexEngine Proposal DoD item and E1-E6 with engine-integrated Prefix Cache, CUDA Graph and Triton Paged Attention plus reproducible measurements.

**Architecture:** Preserve the single serialized GPU worker and existing request/KV abstractions. Extend physical KV ownership with shared immutable blocks, add graph and optimized-attention backends behind explicit fallbacks, and drive all experiments from deterministic recorded configurations.

**Tech Stack:** Python 3.13, PyTorch 2.15.0+cu132, Transformers 5.22.2, Triton, FastAPI, pytest, Nsight Systems/Compute, WSL2 CUDA.

**Spec:** `docs/superpowers/specs/2026-11-08-proposal-completion-design.md`

## Global Constraints

- Target one RTX 3080 Ti Laptop GPU, 16 GiB, compute capability 8.6.
- Lock the verified FlexEngine environment; isolate vLLM dependencies separately.
- Serialize GPU experiments and retain raw traces.
- Use TDD for every production change and run the complete CPU/GPU suites at phase gates.
- Report infeasible hardware points and negative optimizations; never fabricate or silently omit results.
- Git metadata is unavailable; replace commit steps with filesystem/status checkpoints until it is restored.

## Review Focus

- Shared-prefix cancellation must never free a block still referenced by another request or cache entry (Tasks 6-7).
- Graph padding must not leak inactive-slot data or change logits/state for active requests (Tasks 8-9).
- Prefix lookup must reject model/layout/revision mismatches and partial-block false hits (Task 6).
- Triton must fall back safely for unsupported dtype, shape, block size or excessive context (Task 11).
- Experiment resumption must not mix traces from different configurations or count partial runs as complete (Tasks 4, 12, 14).

---

### Task 1: Reproduction metadata and DoD ledger

**Files:**
- Create: `src/flexengine/observability/environment.py`
- Create: `src/flexengine/tooling/environment.py`
- Create: `docs/DOD_STATUS.md`
- Modify: `src/flexengine/observability/traces.py`
- Test: `tests/unit/test_environment.py`

**Interfaces:**
- Produces: `collect_environment() -> dict`, `validate_run_metadata(data: dict) -> list[str]`.

- [ ] Write failing tests requiring driver, CUDA, torch, transformers, GPU, model revision, policy, backend, seed and trace identity fields.
- [ ] Run `CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/unit/test_environment.py`; expect missing-interface failure.
- [ ] Implement metadata collection without initializing CUDA when the configured device is CPU.
- [ ] Add a line-by-line Proposal DoD ledger with evidence paths and `pending` status.
- [ ] Run the focused test and full CPU suite; expect pass.
- [ ] Record a checkpoint listing created/modified files.

### Task 2: Qwen2.5-1.5B correctness and complete generation

**Files:**
- Modify: `tests/gpu/test_hf_parity.py`
- Create: `tests/gpu/test_real_generation.py`
- Modify: `src/flexengine/models/qwen2.py`
- Modify: `docs/QUICK_START_CN.md`

**Interfaces:**
- Consumes: `Qwen2Adapter.from_pretrained(...)`.
- Produces: parametrized official-model parity for 0.5B and 1.5B; deterministic complete-text generation test.

- [ ] Add failing parametrized tests for both official model IDs, incremental logits and EOS/length completion.
- [ ] Run the 1.5B test; expect failure only from an identified compatibility/resource gap.
- [ ] Make the minimal adapter compatibility fix if required; do not loosen numerical tolerances without measured error evidence.
- [ ] Run both official parity and generation tests on CUDA; expect pass.
- [ ] Record peak allocated/reserved GPU memory for both models.
- [ ] Update reproduction commands and DoD evidence.

### Task 3: Reliability and failure matrix

**Files:**
- Create: `tests/integration/test_failures.py`
- Modify: `tests/integration/test_runtime.py`
- Modify: `tests/integration/test_api.py`
- Modify: `src/flexengine/runtime.py`
- Modify: `src/flexengine/api/server.py`

**Interfaces:**
- Produces tested behavior for disconnect, cancellation, 429 admission, KV exhaustion/preemption, cleanup, starvation bounds and worker fail-fast.

- [ ] Add one failing test per Proposal F01-F07 behavior, including invariant checks after each terminal state.
- [ ] Run the new file and confirm each new case fails for the intended missing behavior.
- [ ] Implement only the missing runtime/API behavior and trace fields.
- [ ] Run integration plus allocator/scheduler suites; expect pass and no leaked blocks/requests.
- [ ] Update DoD ledger with test node IDs.

### Task 4: Deterministic experiment manifests and resumable runner

**Files:**
- Create: `data/config/proposal_matrix.json`
- Create: `src/flexengine/experiments/matrix.py`
- Create: `tests/unit/test_experiment_matrix.py`
- Modify: `src/flexengine/benchmarking/runner.py`

**Interfaces:**
- Produces: `load_matrix(path) -> list[RunSpec]`, `run_id(spec, repeat) -> str`, atomic completion marker and matched request-trace export/import.

- [ ] Add failing tests for exact E1-E6 controlled subsets, unique run IDs, resume behavior and mismatched-trace rejection.
- [ ] Run focused tests; expect missing-module failure.
- [ ] Implement immutable `RunSpec` and atomic result-directory lifecycle.
- [ ] Extend replay to consume a saved workload trace rather than regenerating it per invocation.
- [ ] Verify focused tests and a two-request dry run.

### Task 5: E1-E4 configuration controls

**Files:**
- Modify: `src/flexengine/config.py`
- Modify: `src/flexengine/cli/server.py`
- Modify: `src/flexengine/experiments/matrix.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_experiment_matrix.py`

**Interfaces:**
- Produces CLI/environment controls for token budget, KV block size/count, model, scheduler and attention backend; E4 helper matching total KV bytes.

- [ ] Add failing tests for CLI overrides and `matched_kv_blocks(base_blocks, base_size, candidate_size) -> int`.
- [ ] Implement validated controls and persist them in metadata.
- [ ] Verify equal KV-pool byte calculations for block sizes 8/16/32.
- [ ] Run a small E1-E4 sanity matrix before long measurements.

### Task 6: Prefix Cache allocator and index

**Files:**
- Create: `src/flexengine/kv_cache/prefix.py`
- Modify: `src/flexengine/kv_cache/allocator.py`
- Create: `tests/unit/test_prefix_cache.py`
- Modify: `tests/unit/test_allocator.py`

**Interfaces:**
- Produces: `PrefixKey`, `PrefixMatch`, `PrefixCache.lookup(tokens, identity)`, `insert(...)`, `release_request(...)`, `evict(...)`; allocator `retain_block`, `release_block`, `attach_prefix`.

- [ ] Add failing tests for longest complete-block match, model/layout mismatch, refcounts, divergent tails, LRU eviction and cancellation ownership.
- [ ] Run focused tests; expect missing interfaces.
- [ ] Implement true physical-block reference counting and immutable shared-block rules.
- [ ] Implement the exact-token longest-prefix index and bounded LRU ownership.
- [ ] Run allocator/prefix tests plus the full CPU suite.

### Task 7: Prefix Cache runtime integration and E5 telemetry

**Files:**
- Modify: `src/flexengine/runtime.py`
- Modify: `src/flexengine/scheduler/state.py`
- Modify: `src/flexengine/models/qwen2.py`
- Modify: `src/flexengine/observability/metrics.py`
- Modify: `src/flexengine/observability/traces.py`
- Create: `tests/integration/test_prefix_runtime.py`

**Interfaces:**
- Consumes Task 6 prefix APIs.
- Produces prompt admission with cached-token offset, saved-prefill accounting, F08-safe cancellation and prefix metrics.

- [ ] Add failing integration tests for hit/miss, shared-prefix divergence, concurrent cancellation, eviction and output parity with cache disabled.
- [ ] Implement lookup/attach before scheduling and insertion after complete-block prefill.
- [ ] Add Prometheus/JSONL fields for hits, saved tokens, evictions and shared blocks.
- [ ] Run focused, full CPU and CUDA 0.5B parity tests.

### Task 8: CUDA Graph bucket manager

**Files:**
- Create: `src/flexengine/execution/cuda_graph.py`
- Create: `src/flexengine/execution/__init__.py`
- Create: `tests/gpu/test_cuda_graph.py`

**Interfaces:**
- Produces: `DecodeGraphManager(buckets=(1,2,4,8))`, `eligible(...)`, `capture(...)`, `replay(...)`, `stats()`.

- [ ] Add failing CUDA tests for bucket selection, padding masks, stable staging addresses, replay parity and unsupported eager fallback.
- [ ] Implement static staging buffers and graph lifecycle for a minimal decode callable.
- [ ] Verify repeated replay does not reuse stale inactive-slot values.
- [ ] Record graph-pool memory and replay counters.

### Task 9: Engine-integrated decode CUDA Graph

**Files:**
- Modify: `src/flexengine/models/qwen2.py`
- Modify: `src/flexengine/runtime.py`
- Modify: `src/flexengine/config.py`
- Modify: `src/flexengine/observability/traces.py`
- Create: `tests/gpu/test_engine_cuda_graph.py`

**Interfaces:**
- Consumes Task 8 manager.
- Produces eager/graph decode switch, automatic padded bucket replay and eager fallback without request delay.

- [ ] Add failing end-to-end parity tests for active counts 1/2/3/4/5/8, non-contiguous block tables and fallback.
- [ ] Refactor decode inputs into stable graph-compatible tensors without changing eager semantics.
- [ ] Integrate capture/replay after warmup and emit graph hit/fallback metadata.
- [ ] Run GPU parity, generation and a short server sanity test for graph on/off.

### Task 10: Nsight baseline and bottleneck report

**Files:**
- Modify: `src/flexengine/benchmarking/profiling/engine.py`
- Modify: `src/flexengine/benchmarking/profiling/kernel.py`
- Create: `docs/NSIGHT_BASELINE.md`
- Create: `results/profiling/README.md`

**Interfaces:**
- Produces reproducible `nsys`/`ncu` commands and captured report paths tied to run metadata.

- [ ] Verify Nsight Systems and Compute availability/permissions with minimal probes.
- [ ] Capture materialized and blockwise decode timelines with NVTX ranges.
- [ ] Quantify CPU launch gaps, attention time, memory traffic and synchronization.
- [ ] Select and document the Triton kernel target with measured evidence.

### Task 11: Fused Triton Paged Attention

**Files:**
- Create: `src/flexengine/kernels/paged_attention.py`
- Modify: `src/flexengine/attention/paged.py`
- Modify: `src/flexengine/config.py`
- Create: `tests/gpu/test_triton_paged_attention.py`

**Interfaces:**
- Produces `triton_paged_attention(q, k_pages, v_pages, block_table, length, scale) -> Tensor` and backend `triton` with explicit capability/fallback checks.

- [ ] Add failing tests for FP16 GQA, block sizes 8/16/32, non-contiguous pages, context boundaries and unsupported fallback.
- [ ] Implement a fused decode kernel consuming block tables directly.
- [ ] Compare output/top-1/logit errors with materialized and blockwise references.
- [ ] Run NCU on the kernel and matched E2E before/after sanity measurements.
- [ ] Keep or report the optimization based on correctness; report negative E2E results honestly.

### Task 12: E5-E6 and complete E1-E6 execution

**Files:**
- Modify: `data/config/proposal_matrix.json`
- Modify: `src/flexengine/experiments/matrix.py`
- Modify: `src/flexengine/experiments/analyze.py`
- Create: `src/flexengine/experiments/analyze_repeats.py`
- Test: `tests/unit/test_analysis.py`

**Interfaces:**
- Consumes all runtime features and RunSpec.
- Produces raw/aggregate/mean-standard-deviation artifacts for E1-E6.

- [ ] Add failing analysis tests for grouping, sample standard deviation, incomplete-run exclusion and confidence metadata.
- [ ] Implement repeat aggregation and plots without synthetic values.
- [ ] Run the controlled E1-E6 matrix with at least three repetitions per final configuration.
- [ ] Verify every completed run has client/server traces, metadata and expected request counts.

### Task 13: Isolated vLLM comparison

**Files:**
- Create: `src/flexengine/benchmarking/baselines/vllm/README.md`
- Create: `src/flexengine/benchmarking/baselines/vllm/run_server.sh`
- Create: `data/config/vllm_matched.json`

**Interfaces:**
- Produces a separate-environment matched server command and reuses the saved FlexEngine request trace.

- [ ] Determine an installable vLLM/Python/PyTorch/CUDA combination without modifying the FlexEngine environment.
- [ ] Validate model, precision, max length, sampling and streaming equivalence.
- [ ] Run three matched repetitions at selected stable concurrency.
- [ ] If externally blocked, preserve exact installation/runtime evidence and mark only this comparison blocked, not FlexEngine features.

### Task 14: Final report and DoD audit

**Files:**
- Modify: `docs/FINAL_REPORT.md`
- Modify: `docs/DOD_STATUS.md`
- Modify: `docs/QUICK_START_CN.md`
- Modify: `README.md`
- Create: `docs/PROPOSAL_COMPLETION_REPORT.md`

**Interfaces:**
- Consumes all test, profiling and experiment artifacts.
- Produces the final claim ledger with direct evidence links and explicit limitations.

- [ ] Re-run full CPU suite, full GPU parity/generation suite and feature-specific CUDA tests.
- [ ] Verify all E1-E6 artifact counts and regenerate aggregate tables/plots from raw data.
- [ ] Fill every Proposal DoD line with `complete`, `infeasible`, or `externally blocked` plus evidence; strict completion requires no implementation item left pending.
- [ ] Document measured wins, negative results, hardware limits, version differences and threats to validity.
- [ ] Run every reproduction command in the final quick-start path that is feasible locally.
- [ ] Record final filesystem/status checkpoint and stop all benchmark services.
