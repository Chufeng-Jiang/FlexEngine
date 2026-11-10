import pytest
import torch

from flexengine.engine.kv_cache.allocator import CacheExhausted, CacheLayout, PagedKVCache


def make_cache(n=10, block=4):
    return PagedKVCache(CacheLayout(2, n, block, 2, 3))


def test_allocate_cross_boundaries_and_readback():
    c = make_cache(10, 4)
    c.reserve("a", 3)
    c.reserve("a", 5)
    c.reserve("b", 7)
    assert len(c.tables["a"]) == 2
    assert len(c.tables["b"]) == 2
    a = torch.arange(5 * 2 * 3).reshape(5, 2, 3).float()
    c.write(0, "a", 0, a, a + 100)
    ka, va = c.materialize(0, "a", 5)
    torch.testing.assert_close(ka, a)
    torch.testing.assert_close(va, a + 100)
    blocks = list(c.iter_blocks(0, "a", 5))
    assert [len(k) for _, k, _ in blocks] == [4, 1]
    c.assert_invariants()
    c.free_request("a")
    c.free_request("b")
    c.assert_invariants()
    assert c.free_blocks == 10


def test_reserve_is_atomic_and_oom_handled():
    c = make_cache(3, 2)
    c.reserve("a", 4)
    with pytest.raises(CacheExhausted):
        c.reserve("b", 4)
    assert "b" not in c.tables
    c.free_request("a")
    c.reserve("b", 4)
    c.assert_invariants()


def test_block_reuse_no_leak():
    c = make_cache(7, 4)
    for _ in range(500):
        c.reserve("x", 17)
        c.free_request("x")
    assert c.free_blocks == 7
    c.assert_invariants()


def test_invalid_slot():
    c = make_cache()
    c.reserve("a", 4)
    with pytest.raises(IndexError):
        c.write(0, "a", 4, torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))


def test_retain_and_release_block_tracks_non_request_owner():
    c = make_cache()
    c.reserve("a", 4)
    block = c.tables["a"][0]

    c.retain_block(block)
    c.free_request("a")

    assert c.refs[block] == 1
    assert block not in c.free
    c.release_block(block)
    assert c.refs[block] == 0
    assert block in c.free
    c.assert_invariants()
