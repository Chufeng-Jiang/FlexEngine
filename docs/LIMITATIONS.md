# Limitations / truthful implementation ledger

**Completed, CPU tested:** token-budget scheduler, variable-size chunked prefill, request isolation, physical KV block allocator/free list, blockwise online-softmax attention across physical pages, materialized SDPA comparison backend, cancellation, HTTP SSE, bounded waiting, serial GPU-worker design, OpenAI-style one-choice completions, prompt/output admission checks, JSONL metrics, synthetic Qwen-style decoder parity.

**Implemented but not verified on a real model/GPU in this delivery environment:** Qwen2.5 checkpoint loading, GPU execution, NVTX and Nsight traces, CUDA Event accuracy, optional Triton RMSNorm, real W1–W4 performance at target hardware, Hugging Face Qwen logits parity, OOM recovery under injected CUDA failures.

**Not implemented:** accelerated fused Paged Attention kernel, CUDA Graph capture/replay of the actual decoder, shared-prefix cache or radix tree, copy-on-write reference counting, speculative decoding, TP/PP/EP or NCCL, quantized weights, exact OpenAI API parity, server-side request timeout, distributed backpressure and autoscaling, production authentication/TLS.

## Engineering next steps

1. Run `tests/gpu/test_hf_parity.py` with installed transformers and real weights; adjust architecture compatibility if Transformer version changes.
2. Run 0.5B with `materialized` then `blockwise`; capture correct generation and compare step traces. Evaluate whether large per-block Python overhead warrants fused CUDA/Triton paged attention.
3. Profile single-step decode with Nsight Systems: assess GPU idle intervals, kernel launch overhead, memory traffic and dynamic batching efficiency.
4. Implement static-bucket, memory-address-stable CUDA Graph replay for an optimized decode backend only after establishing a reliable fixed-shape prefill/decode split.
5. Add exact full-block prefix cache and COW, including tests for shared-prefix cancellation and eviction.
6. Perform multiple repeated matched benchmarks; record measured metrics, errors and confidence intervals. Do not invent speedups.

## Model-specific caveats

Qwen2.5-0.5B/1.5B/3B standard dense checkpoints use the Qwen2 model architecture. Non-default model revisions, RoPE scaling or architecture additions may require changes. The implemented position computation matches basic half-split rotary embedding but must be compared against the installed Transformers revision.
