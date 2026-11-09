"""Exact-token, complete-block shared-prefix cache."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from flexengine.kv_cache.allocator import PagedKVCache


@dataclass(frozen=True)
class PrefixKey:
    identity: Hashable
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class PrefixMatch:
    key: PrefixKey
    block_ids: tuple[int, ...]

    @property
    def cached_tokens(self) -> int:
        return len(self.key.tokens)


class PrefixCache:
    """Bounded LRU whose entries own one reference to every cached block."""

    def __init__(self, allocator: PagedKVCache, max_entries: int = 128):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.allocator = allocator
        self.max_entries = max_entries
        self._entries: OrderedDict[PrefixKey, tuple[int, ...]] = OrderedDict()
        self.lookups = 0
        self.hits = 0
        self.evictions = 0

    def lookup(self, tokens: Sequence[int], identity: Hashable) -> PrefixMatch | None:
        self.lookups += 1
        block_size = self.allocator.layout.block_size
        token_tuple = tuple(tokens)
        # KV pages do not contain the logits after the final prompt token.
        # Always leave at least one token for the executor to produce them.
        reusable_limit = (max(0, len(token_tuple) - 1) // block_size) * block_size
        best = None
        for entry_key, blocks in self._entries.items():
            if entry_key.identity != identity:
                continue
            common = 0
            limit = min(len(token_tuple), len(entry_key.tokens))
            while common < limit and token_tuple[common] == entry_key.tokens[common]:
                common += 1
            common = min(common // block_size * block_size, reusable_limit)
            if common and (best is None or common > best[0]):
                best = (common, entry_key, blocks[: common // block_size])
        if best is None:
            return None
        length, entry_key, blocks = best
        self._entries.move_to_end(entry_key)
        self.hits += 1
        return PrefixMatch(PrefixKey(identity, token_tuple[:length]), blocks)

    def insert(self, request_id: str, tokens: Sequence[int], identity: Hashable) -> PrefixMatch:
        block_size = self.allocator.layout.block_size
        cached_tokens = len(tokens) // block_size * block_size
        if cached_tokens == 0:
            raise ValueError("prefix has no complete block")
        block_count = cached_tokens // block_size
        table = self.allocator.tables.get(request_id, [])
        if len(table) < block_count:
            raise ValueError("request does not own the prefix blocks")
        key = PrefixKey(identity, tuple(tokens[:cached_tokens]))
        existing = self._entries.get(key)
        if existing is not None:
            self._entries.move_to_end(key)
            return PrefixMatch(key, existing)
        blocks = tuple(table[:block_count])
        for block in blocks:
            self.allocator.retain_block(block)
        self._entries[key] = blocks
        while len(self._entries) > self.max_entries:
            self.evict()
        return PrefixMatch(key, blocks)

    def release_request(self, request_id: str) -> None:
        self.allocator.free_request(request_id)

    def evict(self, count: int = 1) -> int:
        if count < 0:
            raise ValueError("count must be nonnegative")
        removed = 0
        while self._entries and removed < count:
            _, blocks = self._entries.popitem(last=False)
            for block in blocks:
                self.allocator.release_block(block)
            removed += 1
            self.evictions += 1
        return removed

    @property
    def shared_blocks(self) -> int:
        return sum(len(blocks) for blocks in self._entries.values())
