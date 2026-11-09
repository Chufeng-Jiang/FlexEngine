# Experiment protocol

| Experiment | Strategy | Metrics | Required runs |
|---|---|---|---|
| E1 | FCFS, `0.5B/1.5B`, C=1/2/4/8 | TTFT p50/p95/p99; token throughput; QPS | ≥3 repeats/config |
| E2 | FCFS / decode_first / slo_aware on W4 | goodput–arrival-rate, p99, queue time | 3–5 repeats; seed constant |
| E3 | budget 64/128/256 | TTFT–ITL Pareto, step composition | ≥3 repeats/config |
| E4 | block sizes 8/16/32; materialized vs blockwise | peak VRAM, used blocks, latency | ≥3 repeats/config |
| E5 | prefix shared vs random | pending prefix cache implementation | N/A |
| E6 | CUDA Graph / Triton RMSNorm | independent micro probe only; decoder Graph pending | N/A |

Use one RTX 3080 Ti, one loaded checkpoint and tokenizer revision, constant prompt data/order, same EOS suppression policy (default none), identical seeds, warmed GPU and no competing workloads. Run ABBA or randomized order to mitigate clock drift. For statistical claims, bootstrap across **independent runs/requests**, not adjacent tokens. Select SLO thresholds from preflight FCFS (e.g. its p75/p90) and keep them unchanged across strategies. Keep rejected requests in arrival/goodput denominators.

## Core metrics

TTFT = first nonempty token text received - send time; server TTFT counts first **sampled** token (including zero-length text EOS, if any). The distinction must be explained in reported results. TPOT = `(last_token_time - first_token_time)/(output_count-1)` for ≥2 generated tokens. Server ITL is between **sampled tokens**, client SSE inter-event interval is between emitted **text chunks**, which may differ. Throughput from server `requests.jsonl` is sampled generated tokens / measurement wall seconds. Goodput requires completion plus fixed TTFT/ITL SLO, divided by measurement wall seconds.

## Failure tests

- F01 mid-generation cancellation, F02 broken SSE stream, F03 KV exhaustion/preemption, F04 429 on admission, F05 starvation under continuous small requests, F06 repeated block reclamation, F07 GPU exception fail-fast, F08 sharing eviction (pending shared prefix).

## Profiling questions

- `nsys`: CPU dispatch gap, kernel density, excess synchronization, prefill/decode interference.
- `ncu`: warp occupancy, DRAM/L2 bandwidth, stalls and roofline for hotspots.
- CUDA Events: kernel execution subset; NOT a substitute for client E2E latency.
- Record memory_allocated, reserved and peak for PyTorch; correlate with `nvidia-smi` process/device memory (not identical counters).
