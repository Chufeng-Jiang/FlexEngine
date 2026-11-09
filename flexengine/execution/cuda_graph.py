"""Fixed-bucket CUDA Graph capture for decoder-only steps."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch


@dataclass
class _CapturedGraph:
    graph: torch.cuda.CUDAGraph
    inputs: dict[str, torch.Tensor]
    active_mask: torch.Tensor
    output: torch.Tensor


class DecodeGraphManager:
    def __init__(self, device: str | torch.device = "cuda:0", buckets=(1, 2, 4, 8)):
        self.device = torch.device(device)
        self.buckets = tuple(sorted(set(buckets)))
        if not self.buckets or self.buckets[0] <= 0:
            raise ValueError("buckets must be positive")
        self._graphs: dict[int, _CapturedGraph] = {}
        self._captures = 0
        self._replays = 0
        self._fallbacks = 0
        self._padded_slots = 0
        self._graph_pool_bytes = 0

    def bucket_for(self, active_count: int) -> int | None:
        if active_count <= 0:
            return None
        return next((bucket for bucket in self.buckets if active_count <= bucket), None)

    def eligible(self, active_count: int) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available() and self.bucket_for(active_count) is not None

    def capture(
        self,
        bucket: int,
        decode: Callable[..., torch.Tensor],
        example_inputs: Mapping[str, torch.Tensor],
    ) -> None:
        if bucket not in self.buckets or not self.eligible(bucket):
            raise ValueError("bucket is not eligible for CUDA Graph capture")
        if not example_inputs:
            raise ValueError("capture requires inputs")
        if any(tensor.device != self.device or tensor.shape[0] != bucket for tensor in example_inputs.values()):
            raise ValueError("capture inputs must use the configured device and bucket batch size")
        static_inputs = {name: tensor.clone() for name, tensor in example_inputs.items()}
        mask_shape = (bucket,) + (1,) * (next(iter(static_inputs.values())).ndim - 1)
        active_mask = torch.ones(mask_shape, device=self.device)
        warmup = torch.cuda.Stream(device=self.device)
        warmup.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup):
            decode(**static_inputs, active_mask=active_mask)
        torch.cuda.current_stream(self.device).wait_stream(warmup)
        torch.cuda.synchronize(self.device)
        before = torch.cuda.memory_reserved(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = decode(**static_inputs, active_mask=active_mask)
        self._graphs[bucket] = _CapturedGraph(graph, static_inputs, active_mask, output)
        self._captures += 1
        self._graph_pool_bytes += max(0, torch.cuda.memory_reserved(self.device) - before)

    def staging_addresses(self, bucket: int) -> dict[str, int]:
        captured = self._graphs[bucket]
        return {
            **{name: tensor.data_ptr() for name, tensor in captured.inputs.items()},
            "active_mask": captured.active_mask.data_ptr(),
            "output": captured.output.data_ptr(),
        }

    def has_capture(self, bucket: int) -> bool:
        return bucket in self._graphs

    def replay(self, active_count: int, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        bucket = self.bucket_for(active_count)
        if bucket is None or bucket not in self._graphs:
            raise ValueError("no captured graph for active batch")
        captured = self._graphs[bucket]
        if set(inputs) != set(captured.inputs):
            raise ValueError("replay input names differ from capture")
        for name, source in inputs.items():
            target = captured.inputs[name]
            if source.shape[0] != active_count or source.shape[1:] != target.shape[1:]:
                raise ValueError("replay input shape differs from capture")
            target.zero_()
            target[:active_count].copy_(source)
        captured.active_mask.zero_()
        captured.active_mask[:active_count] = 1
        captured.graph.replay()
        self._replays += 1
        self._padded_slots += bucket - active_count
        return captured.output[:active_count].clone()

    def run(self, decode: Callable[..., torch.Tensor], inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not inputs:
            raise ValueError("decode requires inputs")
        active_count = next(iter(inputs.values())).shape[0]
        bucket = self.bucket_for(active_count)
        if bucket in self._graphs:
            return self.replay(active_count, inputs)
        self._fallbacks += 1
        return decode(**inputs)

    def record_fallback(self) -> None:
        self._fallbacks += 1

    def stats(self) -> dict[str, int]:
        return {
            "captures": self._captures,
            "replays": self._replays,
            "fallbacks": self._fallbacks,
            "padded_slots": self._padded_slots,
            "graph_pool_bytes": self._graph_pool_bytes,
        }
