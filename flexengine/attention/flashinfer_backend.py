"""Optional FlashInfer paged-decode adapter; never silently falls back."""
from __future__ import annotations

import torch

try:
    import flashinfer as _FLASHINFER
except (ImportError, OSError):
    _FLASHINFER = None

_WORKSPACES: dict[torch.device, torch.Tensor] = {}


def flashinfer_available() -> bool:
    return _FLASHINFER is not None and torch.cuda.is_available()


def flashinfer_workspace(device: torch.device) -> torch.Tensor:
    workspace = _WORKSPACES.get(device)
    if workspace is None:
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        _WORKSPACES[device] = workspace
    return workspace


def flashinfer_paged_decode_batch(q, cache, layer, block_tables, positions, active_mask):
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
    wrapper = _FLASHINFER.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        indptr, indices, last_page_len, q.shape[1], cache.layout.kv_heads,
        cache.layout.head_dim, cache.layout.block_size, data_type=cache.dtype,
    )
    output = wrapper.run(q, (cache.k[layer], cache.v[layer]))
    return output * active[:, None, None]
