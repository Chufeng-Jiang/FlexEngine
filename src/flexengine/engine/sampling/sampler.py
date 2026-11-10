"""Per-request RNG; greedy and temperature/top-k multinomial sampling."""
from __future__ import annotations

import torch


def sample(logits: torch.Tensor, temperature: float, top_k: int, generator: torch.Generator | None = None) -> int:
    if logits.ndim != 1:
        raise ValueError("Expected vocabulary logits [vocab]")
    if temperature < 0 or top_k < 0:
        raise ValueError("temperature and top_k must be >= 0")
    if temperature == 0.0:
        return int(logits.argmax().item())
    # Sampling on CPU with a separate per-request RNG keeps results stable
    # across scheduling strategies and batch positions (at a performance cost).
    x = logits.detach().to(device="cpu", dtype=torch.float32) / temperature
    if top_k > 0 and top_k < x.numel():
        vals, idxs = torch.topk(x, top_k)
        probs = torch.softmax(vals, dim=-1)
        return int(idxs[torch.multinomial(probs, 1, generator=generator)].item())
    probs = torch.softmax(x, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())
