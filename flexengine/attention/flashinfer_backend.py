"""Optional FlashInfer paged-decode adapter; never silently falls back."""
from __future__ import annotations

import torch

try:
    import flashinfer as _FLASHINFER
except (ImportError, OSError):
    _FLASHINFER = None

_WORKSPACES: dict[torch.device, torch.Tensor] = {}
_WRAPPERS: dict[tuple, object] = {}
_SIGNATURES: dict[tuple, tuple] = {}
_PLANS = 0


def flashinfer_available() -> bool:
    return _FLASHINFER is not None and torch.cuda.is_available()


def flashinfer_workspace(device: torch.device) -> torch.Tensor:
    workspace = _WORKSPACES.get(device)
    if workspace is None:
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        _WORKSPACES[device] = workspace
    return workspace


def flashinfer_backend_stats() -> dict[str, int]:
    return {"workspaces": len(_WORKSPACES), "wrappers": len(_WRAPPERS), "plans": _PLANS}


def flashinfer_paged_decode_batch(q, cache, layer, block_tables, positions, active_mask):
    global _PLANS
    if _FLASHINFER is None:
        raise RuntimeError(
            "FlashInfer backend is unavailable; install the optional flashinfer dependency "
            "for a supported Python/CUDA build"
        )
    if q.device.type != "cuda":
        raise RuntimeError("FlashInfer backend requires CUDA tensors")
    active = active_mask.bool()
    page_counts = torch.where(active, positions // cache.layout.block_size + 1, 0).to(torch.int32)
    indptr = torch.cat((
        torch.zeros(1, device=q.device, dtype=torch.int32),
        page_counts.cumsum(0, dtype=torch.int32),
    ))
    indices = torch.cat([
        block_tables[row, :count] for row, count in enumerate(page_counts.tolist()) if count
    ]).to(torch.int32)
    last_page_len = torch.where(
        active, positions % cache.layout.block_size + 1, torch.zeros_like(positions)
    ).to(torch.int32)
    workspace = flashinfer_workspace(q.device)
    key = (
        q.device, q.shape[0], q.shape[1], cache.layout.kv_heads,
        cache.layout.head_dim, cache.layout.block_size, cache.dtype,
    )
    wrapper = _WRAPPERS.get(key)
    if wrapper is None:
        wrapper = _FLASHINFER.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        _WRAPPERS[key] = wrapper
    signature = (tuple(page_counts.tolist()), tuple(indices.tolist()), tuple(last_page_len.tolist()))
    if _SIGNATURES.get(key) != signature:
        wrapper.plan(
            indptr, indices, last_page_len, q.shape[1], cache.layout.kv_heads,
            cache.layout.head_dim, cache.layout.block_size, data_type=cache.dtype,
        )
        _SIGNATURES[key] = signature
        _PLANS += 1
    output = wrapper.run(q, (cache.k[layer], cache.v[layer]))
    return output * active[:, None, None]
