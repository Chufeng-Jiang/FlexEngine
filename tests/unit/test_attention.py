import pytest
import torch

from flexengine.engine.kv_cache.allocator import CacheLayout, PagedKVCache
from flexengine.inference.attention.paged import attention_for_request


@pytest.mark.parametrize("length", [1, 3, 4, 5, 15, 16, 17, 34])
@pytest.mark.parametrize("qcount", [1, 2, 3])
def test_direct_paged_attention_matches_materialized(length, qcount):
    if length < qcount:
        return
    torch.manual_seed(length + 17 * qcount)
    cache = PagedKVCache(CacheLayout(2, 20, 4, 2, 8))
    cache.reserve("r", length)
    keys, vals = torch.randn(length, 2, 8), torch.randn(length, 2, 8)
    cache.write(1, "r", 0, keys, vals)
    query = torch.randn(qcount, 4, 8)
    positions = torch.arange(length - qcount, length)
    paged = attention_for_request(query, cache, 1, "r", length, positions, "blockwise")
    baseline = attention_for_request(query, cache, 1, "r", length, positions, "materialized")
    torch.testing.assert_close(paged, baseline, atol=1e-5, rtol=1e-5)


def test_block_table_noncontiguous_physical_pages():
    torch.manual_seed(0)
    c = PagedKVCache(CacheLayout(1, 8, 2, 1, 4))
    c.reserve("a", 2)
    c.reserve("other", 4)
    c.reserve("a", 5)
    assert c.tables["a"] == [0, 3, 4]
    c.write(0, "a", 0, torch.randn(5, 1, 4), torch.randn(5, 1, 4))
    q = torch.randn(2, 2, 4)
    p = torch.tensor([3, 4])
    torch.testing.assert_close(attention_for_request(q,c,0,"a",5,p,"blockwise"),
                               attention_for_request(q,c,0,"a",5,p,"materialized"), atol=1e-5, rtol=1e-5)
