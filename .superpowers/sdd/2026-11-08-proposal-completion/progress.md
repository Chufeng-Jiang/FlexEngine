# SDD ledger — plan: docs/superpowers/plans/2026-11-08-proposal-completion.md

Setup: Ruling: repository has no `.git` metadata, so no isolated worktree or commits can be created — execute in the user-designated workspace and preserve filesystem checkpoints — cost if wrong: changes lack Git rollback/history until repository metadata is restored.

Pre-flight: Task 4 produces RunSpec/trace identity consumed by Tasks 5 and 12; signatures are fixed in the plan and no conflict found.
Pre-flight: Task 6 produces PrefixCache/refcount APIs consumed by Task 7; longest complete-block semantics match the spec.
Pre-flight: Task 8 produces DecodeGraphManager consumed by Task 9; 1/2/4/8 padded-bucket semantics match the spec.
Pre-flight: Tasks 7, 9 and 11 produce runtime modes consumed by Task 12; metadata must uniquely identify prefix/graph/backend settings.
Pre-flight: Tasks 1-13 produce evidence consumed by Task 14; every artifact must remain regenerable from raw traces.

Task 1: Ruling: `collect_environment` accepts an optional `device` argument rather than the plan's zero-argument signature — CPU metadata must not initialize a broken/unwanted CUDA runtime — cost if wrong: callers that assumed unconditional GPU probing must pass their configured device.
Task 1: complete (filesystem checkpoint; tests: `CUDA_VISIBLE_DEVICES="" python -m pytest -q` -> 49 passed, 1 skipped; CUDA environment script identified RTX 3080 Ti and driver 610.60; touched-file Ruff clean after safe fixes).

Task 2: Ruling: the existing adapter already supported 1.5B, so no artificial production change was made merely to force a RED test — added an end-to-end characterization test and verified both official models — cost if wrong: model-version regressions are caught by optional GPU tests rather than a new adapter branch.
Task 2: complete (filesystem checkpoint; official 0.5B and 1.5B logits parity each 1 passed; complete generation/KV reclamation each 1 passed; peak allocated bytes 1,017,782,272 and 3,126,618,112 respectively; new test Ruff clean).

Task 3: Ruling: decode-first promotes a prefill only after it exceeds the configured TTFT SLO, preserving decode priority before the deadline — prevents starvation without redefining normal decode-first ordering — cost if wrong: overloaded decode latency can rise when overdue prefills are admitted.
Task 3: complete (filesystem checkpoint; new F02/F04/F05/F07 evidence plus existing F01/F03/F06 tests; full CPU suite 53 passed, 2 skipped; touched-file Ruff clean).

Task 4: Ruling: Proposal matrix is expanded from declarative fixed/matrix dimensions into 39 explicit RunSpec objects — avoids hand-maintained duplicate JSON while preserving a non-Cartesian controlled subset — cost if wrong: a loader regression can alter experiment counts, guarded by an exact-count test.
Task 4: complete (filesystem checkpoint; matrix/identity/atomic marker/saved-trace tests 5 passed; two-request generate and replay dry runs each 2/2 completed; full CPU suite 58 passed, 2 skipped; touched-file Ruff clean).

Task 5: Ruling: E3 changes only the scheduler token budget, while E4 keeps total KV token capacity constant via `matched_kv_blocks` — isolates the intended independent variable — cost if wrong: allocator metadata would expose a mismatch, and the controlled matrix tests pin all values.
Task 5: complete (filesystem checkpoint; CLI/config control tests 2 passed; full CPU suite 60 passed, 2 skipped; touched-file Ruff clean after mechanical import/annotation fixes).

Task 6: Ruling: request-table references and non-request cache references are counted separately, and writes are rejected whenever a physical block has multiple owners — makes cancellation and LRU release independently auditable — cost if wrong: overly broad ref retention would be detected as a free-state/refcount invariant failure.
Task 6: complete (filesystem checkpoint; longest exact full-block match, identity mismatch, divergent private tail, LRU, cancellation and retained-owner tests 10 passed; full CPU suite 66 passed, 2 skipped; touched-file Ruff clean).

Task 7: Ruling: prefix lookup occurs at admission and insertion occurs as soon as all complete prompt blocks have been populated; partial prompt tails and generated blocks remain request-private — preserves output parity while maximizing reusable prefill — cost if wrong: the enabled/disabled deterministic parity test fails.
Task 7: complete (filesystem checkpoint; runtime hit/miss, saved-prefill, output parity, concurrent cancellation and eviction tests 4 passed; focused Task 5-7 suite 16 passed; full CPU suite 70 passed, 2 skipped; official Qwen2.5-0.5B CUDA parity 1 passed in 4.00s; touched-file Ruff clean).

Task 8: Ruling: capture owns stable tensor staging for each 1/2/4/8 bucket and copies active rows before every replay; inactive rows are zeroed and masked — prevents stale padded-slot state without delaying requests — cost if wrong: repeated replay parity and pointer-stability tests fail.
Task 8: complete (filesystem checkpoint; RTX 3080 Ti bucket/padding/stable-address/replay/fallback tests 3 passed in 0.67s; touched-file Ruff clean).

Task 9: Ruling: graph decode uses a fixed tensor block table and a full masked context gather, while unsupported counts or capture errors fall back to the existing eager path — prioritizes graph safety and parity before the fused Triton replacement — cost if wrong: graph memory/work grows with max context, measured in E6 and replaced by Task 11's fused path.
Task 9: complete (filesystem checkpoint; official Qwen2.5-0.5B Engine graph/eager parity for active counts 1/2/3/4/5/8 plus 9-request fallback passed in 11.58s; CPU suite 71 passed, 4 skipped; touched-file Ruff clean).

Task 10: complete (Nsight Systems 2024.6.2 and Nsight Compute 2025.1.1 verified; raw Systems, blockwise NCU and Triton NCU reports captured; bottleneck report records 24,318 launches in the short baseline).

Task 11: Ruling: Triton decode specializes power-of-two context buckets and uses the actual length as a runtime mask — avoids per-token recompilation while preserving exact paged lookup — cost if wrong: non-contiguous 1/17/129/513-token parity tests fail.
Task 11: complete (fused single/batched paged decode, graph integration, five Triton correctness tests, official graph+Triton parity; 512-token 0.537 ms vs blockwise 33.662 ms).

Task 12: complete (39 controlled E1-E6 configurations × 3 successful repetitions; 936/936 requests; strict client/server trace and metadata audit returned zero errors; aggregate mean/sample-stdev CSV and Markdown generated).

Task 13: complete (isolated Python 3.12 `.venv-vllm`, vLLM 0.33.0/PyTorch 2.14+cu132, neutral generation config, exact saved trace, 3 repetitions and 24/24 requests; mean 14.779 ± 1.209 req/s).

Task 14: complete (English README, final report, DoD audit, environment/system and dual requirements exports, Docker/reproduction/profiling guides; final gates: CPU 75 passed/11 skipped, full 0.5B GPU 14 passed, 1.5B parity/generation 2 passed, repository-wide Ruff clean).
