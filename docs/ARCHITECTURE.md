# FlexEngine Architecture and invariants

```text
HTTP clients                     HTTP replay / Nsight
    |                                     |
FastAPI /v1/completions       Benchmark clients
    |
Engine.submit() ---> waiting deque [bounded] <-- request IDs
    |
single serialized GPU worker thread
    |
SchedulerPolicy.order(req) + token budget -> variable chunks
    |
PagedKVCache.reserve() -> block table + [L,blocks,T,KVheads,D] GPU tensors
    |
Qwen2Adapter.forward(chunks) -> embeddings, QKV projection, RoPE
    |
request-specific attention_for_request()
       +-- blockwise direct page views, online log-sum-exp
       +-- materialized gather + PyTorch SDPA (comparison baseline)
    |
MLP + final norm + head -> last logits per request
    |
Greedy/top-k sample -> tokenizer.decode -> per-request event Queue
    |
SSE / non-stream completion
```

## Step semantics

1. Each request starts with `cached_tokens=0`, `generated_ids=[]`.
2. Scheduler first slices an unprocessed prompt prefix (`prefill`) into available step-token budget. All allocated K/V slots are reserved **before** the forward.
3. Forward stores each layer's K/V for the newly scheduled tokens at `cached_tokens ... cached_tokens+chunk_len`. Each query can only attend to `key_position <= query_position`, so chunked prefill is correct.
4. When the final prompt slice is done, sample the first output token from its final logits. The sampled token has been emitted but is **not** yet stored in KV.
5. The next decode step forwards the previously sampled token, then samples another output token. Thus `cached_tokens` trails the emitted output by one token.
6. Finishing after EOS/length frees all physical blocks, and the terminal event gives finish_reason. Cancellation occurs at the next serialized safe point.
7. On KV pressure, at most one non-selected peer is recompute-preempted per candidate. Its prior generated text remains; the cached prefix is recomputed before the next decode token. This is a basic policy, not a sophisticated fair/optimal preemption system.

## KV & memory invariants

- Fixed block shape `[layers, blocks, block_size, kv_heads, head_dim]` for both K and V.
- `tables[request_id][logical_block] = physical_block` maps logical pages. Two requests cannot own the same block in the current implementation.
- `refs[physical]` is 1 if allocated and 0 otherwise. References to shared blocks are reserved for future prefix caching; **current runtime does not share physical KV blocks**.
- At every safe point `len(free)+sum(len(table) for table in tables)=num_blocks`.
- Cancellation, error and completion always free per-request pages, and request metadata is retired from the active dictionary to avoid retaining complete histories forever.
- Current blockwise attention performs stable softmax across physical blocks, without requiring full K/V materialization. It does, however, launch separate PyTorch ops per page; it is slow and not a fused CUDA kernel.

## Numerical / backend scope

- Models: dense Qwen2/Qwen2.5 causal decoder weights; no sliding windows, RoPE scaling, Qwen3 attention norms or experts.
- GQA is implemented by repeating KV heads. This is functionally correct but can use extra memory traffic.
- FP16 numerical agreement is assessed with logits error and top-1 agreement against official Hugging Face. CPU synthetic tests exercise correctness without model downloads. Actual target GPU verification is a pending user-side step.
- CUDA Event times are explicitly recorded separately from `time.perf_counter` wall times. NVTX marks are only active for CUDA.

## Threading / failure scope

- One dedicated worker thread owns model and CUDA execution; API submits and cancels under one reentrant lock.
- Waiting requests are bounded; slow SSE consumers are **not** fully bounded/flow-controlled, so this version should only be used locally with moderate token limits.
- Any executor exception fails the current active queue and makes the instance unhealthy. Restart the service on GPU errors; do not claim it self-recovers.
