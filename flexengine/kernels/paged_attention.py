"""Fused Triton paged-attention kernel for one-token decode."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    table_ptr,
    out_ptr,
    stride_qh: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_oh: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LENGTH,
    SEQ_BUCKET: tl.constexpr,
):
    head = tl.program_id(0)
    kv_head = head // GROUP_SIZE
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    q = tl.load(q_ptr + head * stride_qh + dims, mask=dim_mask, other=0.0).to(tl.float32)
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    for start in tl.static_range(0, SEQ_BUCKET, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        valid = offsets < LENGTH
        block = tl.load(table_ptr + start // BLOCK_SIZE)
        k_offsets = (
            block * stride_kb
            + (offsets[:, None] % BLOCK_SIZE) * stride_kt
            + kv_head * stride_kh
            + dims[None, :]
        )
        keys = tl.load(k_ptr + k_offsets, mask=valid[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        probabilities = tl.where(valid, probabilities, 0.0)
        v_offsets = (
            block * stride_vb
            + (offsets[:, None] % BLOCK_SIZE) * stride_vt
            + kv_head * stride_vh
            + dims[None, :]
        )
        values = tl.load(v_ptr + v_offsets, mask=valid[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        accumulator = accumulator * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=0)
        running_max = new_max
    output = accumulator / running_sum
    tl.store(out_ptr + head * stride_oh + dims, output, mask=dim_mask)


def triton_paged_decode(q, cache, layer: int, request_id: str, length: int) -> torch.Tensor:
    if not q.is_cuda or q.shape[0] != 1:
        raise ValueError("Triton paged decode requires one CUDA query token")
    heads, head_dim = q.shape[1:]
    if heads % cache.layout.kv_heads:
        raise ValueError("query/KV head ratio is invalid")
    sequence_bucket = max(cache.layout.block_size, triton.next_power_of_2(length))
    table_blocks = sequence_bucket // cache.layout.block_size
    request_table = cache.tables[request_id]
    table = torch.full((table_blocks,), request_table[-1], device=q.device, dtype=torch.int32)
    table[:len(request_table)] = torch.tensor(request_table, device=q.device, dtype=torch.int32)
    keys = cache.k[layer]
    values = cache.v[layer]
    output = torch.empty((heads, head_dim), device=q.device, dtype=q.dtype)
    block_d = triton.next_power_of_2(head_dim)
    _paged_decode_kernel[(heads,)](
        q[0], keys, values, table, output,
        q.stride(1), keys.stride(0), keys.stride(1), keys.stride(2),
        values.stride(0), values.stride(1), values.stride(2), output.stride(0),
        cache.layout.kv_heads, heads // cache.layout.kv_heads, head_dim, block_d,
        cache.layout.block_size, length, sequence_bucket,
        num_warps=4,
    )
    return output.unsqueeze(0)


@triton.jit
def _paged_decode_batch_kernel(
    q_ptr, k_ptr, v_ptr, table_ptr, positions_ptr, active_ptr, out_ptr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kb: tl.constexpr, stride_kt: tl.constexpr, stride_kh: tl.constexpr,
    stride_vb: tl.constexpr, stride_vt: tl.constexpr, stride_vh: tl.constexpr,
    stride_tb: tl.constexpr, stride_ob: tl.constexpr, stride_oh: tl.constexpr,
    GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MAX_SEQ_LEN: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP_SIZE
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    active = tl.load(active_ptr + batch).to(tl.int1)
    length = tl.load(positions_ptr + batch) + 1
    q = tl.load(q_ptr + batch * stride_qb + head * stride_qh + dims, mask=dim_mask, other=0.0).to(tl.float32)
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    for start in tl.static_range(0, MAX_SEQ_LEN, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        valid = (offsets < length) & active
        block = tl.load(table_ptr + batch * stride_tb + start // BLOCK_SIZE)
        k_offsets = block * stride_kb + (offsets[:, None] % BLOCK_SIZE) * stride_kt + kv_head * stride_kh + dims[None, :]
        keys = tl.load(k_ptr + k_offsets, mask=valid[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.where(active, tl.exp(running_max - new_max), 0.0)
        probabilities = tl.where(valid, tl.exp(scores - new_max), 0.0)
        v_offsets = block * stride_vb + (offsets[:, None] % BLOCK_SIZE) * stride_vt + kv_head * stride_vh + dims[None, :]
        values = tl.load(v_ptr + v_offsets, mask=valid[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        accumulator = accumulator * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=0)
        running_max = new_max
    output = tl.where(active, accumulator / running_sum, 0.0)
    tl.store(out_ptr + batch * stride_ob + head * stride_oh + dims, output, mask=dim_mask)


def triton_paged_decode_batch(q, cache, layer, block_tables, positions, active_mask) -> torch.Tensor:
    batch, heads, head_dim = q.shape
    output = torch.empty_like(q)
    keys = cache.k[layer]
    values = cache.v[layer]
    _paged_decode_batch_kernel[(batch, heads)](
        q, keys, values, block_tables, positions, active_mask, output,
        q.stride(0), q.stride(1), keys.stride(0), keys.stride(1), keys.stride(2),
        values.stride(0), values.stride(1), values.stride(2), block_tables.stride(0),
        output.stride(0), output.stride(1), heads // cache.layout.kv_heads, head_dim,
        triton.next_power_of_2(head_dim), cache.layout.block_size,
        block_tables.shape[1] * cache.layout.block_size, num_warps=4,
    )
    return output
