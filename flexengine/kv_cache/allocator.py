"""Paged KV pool with logical block tables and reference-counted storage."""
from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch


class CacheExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheLayout:
    layers: int
    num_blocks: int
    block_size: int
    kv_heads: int
    head_dim: int


class PagedKVCache:
    def __init__(self, layout: CacheLayout, device="cpu", dtype=torch.float32):
        self.layout = layout
        self.device = torch.device(device)
        self.dtype = dtype
        shape = (layout.layers, layout.num_blocks, layout.block_size, layout.kv_heads, layout.head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=self.device)
        self.v = torch.empty(shape, dtype=dtype, device=self.device)
        self.free: deque[int] = deque(range(layout.num_blocks))
        self.tables: dict[str, list[int]] = {}
        self.refs = [0] * layout.num_blocks
        self.retained_refs = [0] * layout.num_blocks

    @property
    def free_blocks(self) -> int:
        return len(self.free)

    @property
    def used_blocks(self) -> int:
        return self.layout.num_blocks - len(self.free)

    def capacity_tokens(self) -> int:
        return self.layout.num_blocks * self.layout.block_size

    def available_slots(self, request_id: str) -> int:
        return len(self.tables.get(request_id, [])) * self.layout.block_size

    def missing_blocks(self, request_id: str, target_tokens: int) -> int:
        need = math.ceil(target_tokens / self.layout.block_size)
        return max(0, need - len(self.tables.get(request_id, [])))

    def reserve(self, request_id: str, target_tokens: int) -> None:
        if target_tokens < 0:
            raise ValueError("negative target_tokens")
        needed = self.missing_blocks(request_id, target_tokens)
        if needed > len(self.free):
            raise CacheExhausted(f"Need {needed} blocks; only {len(self.free)} free")
        table = self.tables.setdefault(request_id, [])
        for _ in range(needed):
            block = self.free.popleft()
            assert self.refs[block] == 0
            self.refs[block] = 1
            table.append(block)

    def retain_block(self, block: int) -> None:
        """Add a non-request owner, such as the prefix-cache index."""
        if block < 0 or block >= self.layout.num_blocks or self.refs[block] <= 0:
            raise ValueError("cannot retain an unallocated block")
        self.refs[block] += 1
        self.retained_refs[block] += 1

    def release_block(self, block: int) -> None:
        """Release one non-request owner without affecting request tables."""
        if block < 0 or block >= self.layout.num_blocks or self.retained_refs[block] <= 0:
            raise ValueError("block has no retained owner")
        self.retained_refs[block] -= 1
        self._release_ref(block)

    def attach_prefix(self, request_id: str, blocks: Sequence[int]) -> None:
        """Attach already-populated complete blocks to a new request."""
        if request_id in self.tables:
            raise ValueError("request already has a block table")
        block_ids = list(blocks)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("prefix contains duplicate blocks")
        if any(block < 0 or block >= self.layout.num_blocks or self.refs[block] <= 0 for block in block_ids):
            raise ValueError("prefix contains an unallocated block")
        for block in block_ids:
            self.refs[block] += 1
        self.tables[request_id] = block_ids

    def _release_ref(self, block: int) -> None:
        if self.refs[block] <= 0:
            raise RuntimeError("corrupt block reference count")
        self.refs[block] -= 1
        if self.refs[block] == 0:
            self.free.append(block)

    def free_request(self, request_id: str) -> None:
        for block in self.tables.pop(request_id, []):
            self._release_ref(block)

    def _locators(self, request_id: str, start: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if start < 0 or count < 0 or start + count > self.available_slots(request_id):
            raise IndexError("KV slots not reserved")
        table = self.tables[request_id]
        slots = range(start, start + count)
        ids = torch.tensor([table[i // self.layout.block_size] for i in slots], dtype=torch.long, device=self.device)
        offs = torch.tensor([i % self.layout.block_size for i in slots], dtype=torch.long, device=self.device)
        return ids, offs

    def write(self, layer: int, request_id: str, start: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.shape != values.shape or keys.ndim != 3:
            raise ValueError("K/V shapes must be [tokens, kv_heads, head_dim]")
        if tuple(keys.shape[1:]) != (self.layout.kv_heads, self.layout.head_dim):
            raise ValueError("wrong KV shape")
        if keys.dtype != self.dtype or values.dtype != self.dtype:
            raise ValueError("wrong KV dtype")
        ids, offsets = self._locators(request_id, start, keys.shape[0])
        if any(self.refs[block] > 1 for block in ids.unique().tolist()):
            raise RuntimeError("cannot write a shared immutable KV block")
        self.k[layer, ids, offsets] = keys
        self.v[layer, ids, offsets] = values

    def materialize(self, layer: int, request_id: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Comparison/reference backend ONLY; gathers pages into contiguous memory."""
        if length <= 0:
            raise ValueError("length must be >0")
        ids, offsets = self._locators(request_id, 0, length)
        return self.k[layer, ids, offsets], self.v[layer, ids, offsets]

    def iter_blocks(self, layer: int, request_id: str, length: int):
        """Yield views of non-contiguous physical pages without a full KV copy."""
        if length <= 0 or length > self.available_slots(request_id):
            raise ValueError("requested length is invalid/unreserved")
        for j, bid in enumerate(self.tables[request_id]):
            start = j * self.layout.block_size
            if start >= length:
                break
            n = min(self.layout.block_size, length - start)
            yield start, self.k[layer, bid, :n], self.v[layer, bid, :n]

    def assert_invariants(self) -> None:
        used = [item for table in self.tables.values() for item in table]
        free = list(self.free)
        if len(free) != len(set(free)) or set(free) & set(used):
            raise AssertionError("free/used blocks overlap or duplicate")
        request_refs = Counter(used)
        for block in range(self.layout.num_blocks):
            expected = request_refs[block] + self.retained_refs[block]
            if self.refs[block] != expected:
                raise AssertionError("ref count mismatch")
            if (self.refs[block] == 0) != (block in self.free):
                raise AssertionError("free/allocated state mismatch")
