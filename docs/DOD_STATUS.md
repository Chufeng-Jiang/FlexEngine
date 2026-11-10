# FlexEngine Proposal Definition of Done

| Requirement | Status | Evidence |
|---|---|---|
| Two official Qwen2.5 dense models generate on one GPU | Complete | 0.5B/1.5B parity: `tests/gpu/test_hf_parity.py`; complete generation and KV reclamation: `tests/gpu/test_real_generation.py` |
| Own Scheduler, KV Cache and Model Executor | Complete | `src/flexengine/scheduler/`, `src/flexengine/kv_cache/`, `src/flexengine/models/qwen2.py` |
| Continuous batching and chunked prefill | Complete | `src/flexengine/runtime.py`, integration tests |
| Block Table consumed by Attention | Complete | `src/flexengine/attention/paged.py`, attention tests |
| SSE, cancellation, backpressure, preemption and cleanup tests | Complete | `tests/integration/test_failures.py`, `test_runtime.py`, `test_api.py`, `tests/unit/test_allocator.py` cover F01-F07 and block invariants |
| FCFS/decode-first/SLO-aware controlled comparison | Complete | E2 in `proposal_20261108/aggregate.csv` |
| TTFT, ITL, TPOT, P99, Goodput and VRAM artifacts | Complete | E1-E6 raw traces, aggregate report, generation peak memory |
| Shared Prefix Cache | Complete | Exact full-block lookup, refcounts, LRU, runtime telemetry and E5 |
| Engine CUDA Graph | Complete | 1/2/4/8 capture/replay parity, fallback telemetry and E6 |
| Fused Triton Paged Attention | Complete | Non-contiguous page correctness, NCU capture, E4/E6 |
| Auditable Nsight bottleneck analysis | Complete | `docs/NSIGHT_BASELINE.md`, raw `.nsys-rep`/`.ncu-rep` |
| Isolated matched vLLM baseline | Complete | vLLM 0.33.0, three exact-trace repetitions |
| Metadata, raw data, reproduction, Docker and final report | Complete | README, environment exports, 39×3 audit, Dockerfile, final report |

Formal audit result: 39 configurations, 936 completed requests, zero errors. CPU/integration: 75 passed; complete 0.5B GPU suite: 14 passed; final 1.5B parity/generation: 2 passed.
