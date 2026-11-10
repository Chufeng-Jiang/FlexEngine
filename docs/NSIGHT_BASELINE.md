# Nsight Baseline — RTX 3080 Ti / WSL2

Captured on 2026-11-08 with Qwen2.5-0.5B-Instruct, blockwise paged attention,
decode-first scheduling and batch size 2.

## Reproducible captures

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  --stats=true -o results/profiling/nsys_baseline \
  python -m flexengine.research.benchmarking.profiling.engine \
  --steps 8 --batch-size 2 --attention blockwise \
  --trace-dir results/profiling/trace_baseline

/usr/local/cuda/bin/ncu --set basic --launch-count 1 --force-overwrite \
  -o results/profiling/ncu_blockwise \
  python -m flexengine.research.benchmarking.profiling.kernel \
  --backend blockwise --seq-len 128 --iters 1

/usr/local/cuda/bin/ncu --set basic --nvtx --nvtx-include attention_profile/ \
  --force-overwrite -o results/profiling/ncu_triton \
  python -m flexengine.research.benchmarking.profiling.kernel \
  --backend triton --seq-len 512 --iters 1
```

Raw reports are `results/profiling/nsys_baseline.nsys-rep`,
`results/profiling/ncu_blockwise.ncu-rep` and
`results/profiling/ncu_triton.ncu-rep`.

## Observed baseline

- Eight engine steps took 3.395 seconds under profiler instrumentation.
- The 20-token prefill step took 1.579 seconds in the NVTX summary.
- Seven two-token decode steps averaged 254.0 ms each.
- The CUDA API summary recorded 24,318 kernel launches and 2,252 asynchronous
  copies. Kernel-launch overhead and Python-per-block attention orchestration are
  therefore first-order bottlenecks.
- The isolated 128-token blockwise attention target measured 27.81 ms/call
  under the one-kernel NCU capture command. This is profiler-instrumented and is
  not used as an end-to-end latency claim.

These observations motivate both mandatory optimizations in the Proposal:
CUDA Graph replay reduces CPU launch overhead, while fused Triton paged
attention removes the per-block Python/kernel-launch loop.

The final 512-token microbenchmark measured 33.662 ms/call for blockwise,
0.540 ms/call for materialized SDPA and 0.537 ms/call for fused Triton. The
final NCU report contains `_paged_decode_kernel`; NCU-instrumented wall time is
not used as the microbenchmark latency.
