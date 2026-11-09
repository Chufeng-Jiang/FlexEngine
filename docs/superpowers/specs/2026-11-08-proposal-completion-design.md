# FlexEngine Proposal Completion Design

## Intent

Complete `docs/PROPOSAL_2_0.md` honestly and reproducibly on the available RTX 3080 Ti Laptop GPU. Completion means satisfying every mandatory Definition of Done item, implementing the components required by E1-E6, producing real measurements, and retaining evidence for every claim.

## Current baseline

- Qwen2.5-0.5B official logits parity passes on CUDA.
- Scheduler, continuous batching, chunked prefill, paged KV allocation, block-table attention, sampling, SSE, cancellation, metrics and benchmark tooling exist.
- W1 materialized concurrency 1/2/4/8 and blockwise concurrency 4 have three-run measurements.
- Prefix sharing/COW, decoder CUDA Graph replay and fused Triton Paged Attention do not exist.
- Qwen2.5-1.5B, full failure matrix, scheduler experiments, W2-W5, E2-E6 and Nsight evidence are incomplete.

## Goals

1. Pass official single-GPU generation/parity for Qwen2.5-0.5B and 1.5B.
2. Complete correctness and reliability matrices, including cancellation, bounded admission, KV exhaustion, recompute preemption, shared-prefix cancellation and cleanup.
3. Implement exact-token Prefix Cache with immutable shared full blocks, reference counting, copy-on-write for divergent tails, bounded eviction and traceable hit/saved-token metrics.
4. Implement static-bucket CUDA Graph replay for decode with eager fallback, parity tests, replay coverage and graph-pool memory accounting.
5. Implement a fused Triton Paged Attention kernel selected from profiling evidence, with reference parity and fallback for unsupported shapes.
6. Run E1-E6 with controlled traces and at least three repetitions for reportable configurations.
7. Produce Nsight Systems and Nsight Compute evidence linking a measured bottleneck to the selected optimization and its end-to-end effect.
8. Run a matched vLLM comparison if the current Python/CUDA environment supports an installable version; otherwise document the exact external compatibility blocker separately from FlexEngine's DoD.
9. Deliver raw artifacts, aggregate tables, plots, reproduction commands and a final DoD audit.

## Non-goals and anti-goals

- Excluded: multi-GPU/NCCL, MoE, speculative decoding, quantization, production authentication/TLS and autoscaling.
- Do not claim superiority over vLLM or generalize laptop-GPU results to desktop/server GPUs.
- Do not treat microbenchmark-only Graph/Triton probes as engine optimizations.
- Do not fabricate results, silently skip configurations or relabel incomplete work as complete.
- No performance change without correctness parity and matched before/after measurements.

## Constraints

- One RTX 3080 Ti Laptop GPU, 16 GiB, compute capability 8.6.
- WSL2 Ubuntu 22.04, PyTorch 2.15.0+cu132, Transformers 5.22.2.
- The workspace has no usable Git metadata, so changes cannot be committed until repository metadata is restored; filesystem artifacts remain authoritative.
- Long-running GPU experiments are serialized to avoid cross-run interference.
- Each run records model/revision, precision, seed, policy, cache/graph/kernel settings, GPU/driver/software versions and timestamps.

## Approaches considered

### Strict Definition of Done — selected

Implement all missing runtime features and execute E1-E6 before declaring completion. This is the only approach compatible with the request to finish the entire Proposal.

### Core-system completion only — rejected

Finish P0, profiling and scheduler/KV experiments while leaving Prefix, Graph and Triton as future work. This would not satisfy the original Proposal.

### Experiment-first with one advanced optimization — rejected

Profile the existing engine and implement only the highest-value optimization. This matches the risk-control fallback but not strict completion.

## Architecture changes

### Prefix Cache

- Key by exact token IDs plus model/revision and KV-layout identity.
- Lookup returns the longest cached prefix ending at a complete-block boundary; an entire cached sequence need not equal the request.
- Share only complete immutable KV blocks; partial tails remain request-owned.
- Physical blocks have true reference counts; divergence allocates private blocks.
- LRU eviction removes cache ownership and frees storage only after the final reference drops.
- Cancellation, preemption and completion release request references independently.
- Record lookup, hit, saved-prefill tokens, evictions and shared-block occupancy.

### Decode CUDA Graph

- Separate prefill and decode paths.
- Capture fixed decode buckets with stable input/output/KV addresses.
- Capture buckets 1/2/4/8. Non-exact active counts are padded to the next bucket with masked inactive slots; requests are never delayed solely to fill a bucket.
- Copy dynamic IDs, positions and block tables into static staging buffers before replay.
- Eager fallback covers unsupported buckets, backends and errors.
- Record eligibility, captures, replays, fallbacks, graph-pool bytes and parity.

### Triton Paged Attention

- Implement fused decode Paged Attention as a strict completion item. Nsight determines the expected end-to-end value; a small or negative E2E result is reported rather than used to skip implementation.
- Consume block tables directly without materializing contiguous K/V.
- Support Qwen2.5 GQA, FP16, causal decode and configured block sizes; unsupported cases fall back.
- Compare against materialized SDPA and blockwise reference across lengths, non-contiguous pages and GQA ratios.

## Execution phases

1. Correctness foundation: 1.5B parity/generation, metadata completeness, reliability gap audit and tests.
2. Existing-feature experiments: E1-E4 across model, scheduler, budget and KV variables; establish profiler baseline.
3. Prefix Cache: TDD implementation, C07/F08, W5 and E5.
4. CUDA Graph: parity/fallback implementation and E6 graph ablation.
5. Profile-guided Triton: Nsight-guided implementation, kernel parity, NCU evidence and E6 kernel ablation.
6. External comparison and closure: matched vLLM when feasible, three-run final matrix, plots, report and line-by-line DoD audit.

## Testing strategy

- Every production behavior begins with a focused failing test.
- CPU tests cover scheduler, allocation/refcounts, prefix lookup/eviction/COW and state transitions.
- CUDA tests cover 0.5B/1.5B parity, graph eager/replay parity and Triton/reference parity.
- Integration tests cover SSE disconnect/cancellation, admission, preemption, shared-prefix cancellation and worker failure.
- Full CPU suite uses `CUDA_VISIBLE_DEVICES=""`; GPU tests run separately.

## Experiment design

- E1: 0.5B/1.5B, concurrency 1/2/4/8, matched trace.
- E2: FCFS/decode-first/SLO-aware on W2/W4, matched arrivals.
- E3: token budgets 64/128/256/512.
- E4: block sizes 8/16/32 and materialized/reference/optimized paged backends under matched KV capacity.
- E5: Prefix off/on with shared-prefix ratios 0/25/50/75%.
- E6: eager/graph and Triton off/on, including parity, memory and end-to-end measurements.
- Use at least three repetitions for final configurations; randomize or ABBA order where practical and report mean plus sample standard deviation.
- The matrix is a controlled one-factor subset, not a full Cartesian product: E1 fixes decode-first/materialized/budget 128/block 16; E2 fixes 0.5B/budget 128/block 16 and varies scheduler; E3 fixes 0.5B/decode-first/materialized/block 16 and varies budget; E4 fixes 0.5B/decode-first/budget 128 and varies block/backend; E5 fixes the selected stable baseline and varies prefix ratio/cache; E6 fixes the selected stable baseline and varies eager/graph and materialized/Triton.
- E4 matches total configured KV bytes. Derived block counts vary with block size so the physical K/V pool byte capacity stays constant.

## Compatibility policy

- FlexEngine and its Docker/reproduction contract lock the currently verified Python, PyTorch, CUDA and Transformers versions unless a required tool has a demonstrated incompatibility.
- vLLM comparison uses a separate isolated environment with the closest supported CUDA/PyTorch stack and identical model, precision, prompt trace and API semantics; version differences are disclosed.

## Completion criteria

- Every mandatory DoD line has evidence or an explicitly documented external blocker outside the FlexEngine implementation.
- E1-E6 have raw records, aggregates and interpretation; hardware-infeasible points are reported rather than omitted.
- Prefix, Graph and Triton are engine-integrated and tested, not standalone probes.
- The final report distinguishes implemented, measured, infeasible and externally blocked work.
