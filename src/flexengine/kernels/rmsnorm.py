"""Optional Triton RMSNorm for contiguous 2D activations.

This file is optional and imported only for --rmsnorm-backend triton.
Use accuracy benchmarks before substituting it for the native Qwen RMSNorm.
"""
from __future__ import annotations


def triton_rmsnorm(x, weight, eps: float):
    try:
        import triton
        import triton.language as tl
    except ImportError as ex:
        raise RuntimeError("Install flexengine[triton] to enable Triton RMSNorm") from ex
    import torch

    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("Triton RMSNorm requires contiguous [tokens,hidden] CUDA tensor")
    out = torch.empty_like(x)
    n, hidden = x.shape
    if hidden > 8192:
        raise ValueError("RMSNorm Triton width >8192 not supported")
    # Define kernel on demand: no triton import overhead when using torch backend.
    @triton.jit
    def _norm(X, W, Y, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        values = tl.load(X + row * H + cols, mask=cols < H, other=0).to(tl.float32)
        weights = tl.load(W + cols, mask=cols < H, other=0).to(tl.float32)
        variance = tl.sum(values * values, 0) / H
        result = values * tl.rsqrt(variance + EPS) * weights
        tl.store(Y + row * H + cols, result, mask=cols < H)

    _norm[(n,)](x, weight, out, hidden, eps, triton.next_power_of_2(hidden))
    return out
