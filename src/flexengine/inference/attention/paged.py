"""Two Qwen GQA attention backends for controlled baseline comparison.

blockwise: block-table-directed *online softmax*, no whole-sequence K/V
materialization. Designed for correctness/understanding; Python loop and
per-block kernel launches are slow. For production performance a fused Triton
paged attention kernel is a future optimization, NOT claimed implemented.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from flexengine.engine.kv_cache.allocator import PagedKVCache


def attention_for_request(q: torch.Tensor, cache: PagedKVCache, layer: int, request_id: str,
                          length: int, positions: torch.Tensor, backend: str = "blockwise") -> torch.Tensor:
    """q:[Q,H,D], positions:[Q], returns [Q,H,D], causal per request."""
    if q.ndim != 3 or positions.ndim != 1 or len(q) != len(positions):
        raise ValueError("invalid query shape")
    if q.shape[1] % cache.layout.kv_heads:
        raise ValueError("Q heads must be a multiple of KV heads")
    ratio = q.shape[1] // cache.layout.kv_heads
    scale = 1.0 / math.sqrt(q.shape[-1])
    if backend == "materialized":
        k, v = cache.materialize(layer, request_id, length)
        k = k.repeat_interleave(ratio, dim=1).transpose(0, 1).unsqueeze(0)
        v = v.repeat_interleave(ratio, dim=1).transpose(0, 1).unsqueeze(0)
        query = q.transpose(0, 1).unsqueeze(0)
        key_pos = torch.arange(length, device=q.device)
        mask = (key_pos.unsqueeze(0) <= positions.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        return F.scaled_dot_product_attention(query, k, v, attn_mask=mask, dropout_p=0.0,
                                               is_causal=False).squeeze(0).transpose(0, 1)
    if backend == "triton":
        if len(q) == 1 and q.is_cuda:
            from flexengine.inference.kernels.paged_attention import triton_paged_decode
            return triton_paged_decode(q, cache, layer, request_id, length)
        return attention_for_request(q, cache, layer, request_id, length, positions, "materialized")
    if backend == "flashinfer":
        if len(q) == 1 and q.is_cuda:
            from flexengine.inference.attention.flashinfer_backend import flashinfer_paged_decode_batch

            table = torch.tensor([cache.tables[request_id]], device=q.device, dtype=torch.long)
            return flashinfer_paged_decode_batch(q, cache, layer, table, positions, torch.ones(1, device=q.device))
        return attention_for_request(q, cache, layer, request_id, length, positions, "materialized")
    if backend != "blockwise":
        raise ValueError("Unknown attention backend")
    q32 = q.float()
    running_max = torch.full((q.shape[0], q.shape[1]), -float("inf"), device=q.device)
    running_sum = torch.zeros_like(running_max)
    running_value = torch.zeros((*running_max.shape, q.shape[-1]), device=q.device, dtype=torch.float32)
    for start, kb, vb in cache.iter_blocks(layer, request_id, length):
        kh = kb.float().repeat_interleave(ratio, dim=1)
        vh = vb.float().repeat_interleave(ratio, dim=1)
        scores = torch.einsum("qhd,thd->qht", q32, kh) * scale
        valid = (torch.arange(kb.shape[0], device=q.device) + start).unsqueeze(0) <= positions.unsqueeze(1)
        scores = scores.masked_fill(~valid.unsqueeze(1), -float("inf"))
        bmax = scores.max(dim=-1).values
        newmax = torch.maximum(running_max, bmax)
        # All query positions are >=0; first page always contains token 0.
        alpha = torch.exp(torch.where(torch.isfinite(running_max), running_max - newmax,
                                      torch.full_like(newmax, -float("inf"))))
        weights = torch.exp(scores - newmax.unsqueeze(-1))
        weights = torch.where(valid.unsqueeze(1), weights, torch.zeros_like(weights))
        running_value = running_value * alpha.unsqueeze(-1) + torch.einsum("qht,thd->qhd", weights, vh)
        running_sum = running_sum * alpha + weights.sum(-1)
        running_max = newmax
    return (running_value / running_sum.clamp_min(1e-12).unsqueeze(-1)).to(q.dtype)
