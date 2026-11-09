import pytest
import torch

from flexengine.kv_cache.allocator import CacheLayout, PagedKVCache
from flexengine.kv_cache.prefix import PrefixCache

IDENTITY = ("model", "revision", 1, 1, 1, 4, "float32")


def make_cache(num_blocks: int = 8) -> PagedKVCache:
    return PagedKVCache(CacheLayout(1, num_blocks, 4, 1, 1))


def seed_prefix(cache: PagedKVCache, request_id: str, tokens: list[int]) -> None:
    cache.reserve(request_id, len(tokens))
    values = torch.arange(len(tokens), dtype=torch.float32).reshape(-1, 1, 1)
    cache.write(0, request_id, 0, values, values)


def test_lookup_returns_longest_complete_block_prefix_and_rejects_partial_tail():
    allocator = make_cache()
    prefixes = PrefixCache(allocator, max_entries=4)
    seed_prefix(allocator, "source", list(range(10)))
    prefixes.insert("source", list(range(10)), IDENTITY)

    match = prefixes.lookup(list(range(8)) + [99, 100], IDENTITY)

    assert match is not None
    assert match.cached_tokens == 8
    assert len(match.block_ids) == 2
    partial = prefixes.lookup(list(range(7)), IDENTITY)
    assert partial is not None
    assert partial.cached_tokens == 4


def test_lookup_reuses_partial_prefix_of_a_longer_cached_prompt():
    allocator = make_cache()
    prefixes = PrefixCache(allocator, max_entries=4)
    seed_prefix(allocator, "source", list(range(16)))
    prefixes.insert("source", list(range(16)), IDENTITY)

    match = prefixes.lookup(list(range(10)) + [99, 100, 101, 102], IDENTITY)

    assert match is not None
    assert match.cached_tokens == 8
    assert len(match.block_ids) == 2


def test_exact_prompt_hit_leaves_last_block_when_logits_are_not_cached():
    allocator = make_cache()
    prefixes = PrefixCache(allocator, max_entries=4)
    seed_prefix(allocator, "source", list(range(8)))
    prefixes.insert("source", list(range(8)), IDENTITY)

    match = prefixes.lookup(list(range(8)), IDENTITY)

    assert match is not None
    assert match.cached_tokens == 4


def test_lookup_isolated_by_model_revision_and_layout_identity():
    allocator = make_cache()
    prefixes = PrefixCache(allocator, max_entries=4)
    seed_prefix(allocator, "source", list(range(8)))
    prefixes.insert("source", list(range(8)), IDENTITY)

    assert prefixes.lookup(list(range(8)), ("other", *IDENTITY[1:])) is None
    assert prefixes.lookup(list(range(8)), (IDENTITY[0], "other", *IDENTITY[2:])) is None
    assert prefixes.lookup(list(range(8)), (*IDENTITY[:-2], 8, IDENTITY[-1])) is None


def test_attach_prefix_refcounts_and_divergent_tail_is_private():
    allocator = make_cache()
    prefixes = PrefixCache(allocator, max_entries=4)
    seed_prefix(allocator, "source", list(range(8)))
    inserted = prefixes.insert("source", list(range(8)), IDENTITY)
    source_blocks = tuple(allocator.tables["source"])

    allocator.attach_prefix("child", inserted.block_ids)
    allocator.reserve("child", 9)

    assert tuple(allocator.tables["child"][:2]) == source_blocks
    assert allocator.tables["child"][2] not in source_blocks
    assert [allocator.refs[block] for block in source_blocks] == [3, 3]
    with pytest.raises(RuntimeError, match="shared immutable"):
        allocator.write(0, "child", 0, torch.ones(1, 1, 1), torch.ones(1, 1, 1))


def test_lru_eviction_releases_only_cache_ownership():
    allocator = make_cache()
    prefixes = PrefixCache(allocator, max_entries=1)
    seed_prefix(allocator, "first", list(range(4)))
    first = prefixes.insert("first", list(range(4)), IDENTITY)
    allocator.attach_prefix("consumer", first.block_ids)
    seed_prefix(allocator, "second", list(range(10, 14)))

    prefixes.insert("second", list(range(10, 14)), IDENTITY)

    block = first.block_ids[0]
    assert allocator.refs[block] == 2
    assert prefixes.evictions == 1
    allocator.free_request("first")
    assert allocator.refs[block] == 1
    allocator.free_request("consumer")
    assert allocator.refs[block] == 0


def test_cancellation_releases_request_without_releasing_cache_or_peer():
    allocator = make_cache()
    prefixes = PrefixCache(allocator, max_entries=2)
    seed_prefix(allocator, "source", list(range(8)))
    match = prefixes.insert("source", list(range(8)), IDENTITY)
    allocator.attach_prefix("a", match.block_ids)
    allocator.attach_prefix("b", match.block_ids)

    prefixes.release_request("a")

    assert all(allocator.refs[block] == 3 for block in match.block_ids)
    assert allocator.materialize(0, "b", 8)[0].shape[0] == 8
    allocator.assert_invariants()


def test_lookup_accounts_hits_misses_reused_tokens_blocks_and_time():
    allocator = make_cache()
    prefixes = PrefixCache(allocator, max_entries=2)
    seed_prefix(allocator, "source", list(range(8)))
    prefixes.insert("source", list(range(8)), IDENTITY)

    assert prefixes.lookup(list(range(8)) + [9], IDENTITY) is not None
    assert prefixes.lookup([99, 100, 101, 102, 103], IDENTITY) is None

    stats = prefixes.stats()
    assert stats["lookups"] == 2
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["reused_tokens"] == 8
    assert stats["reused_blocks"] == 2
    assert stats["lookup_ns"] > 0
