import pytest
import torch

from flexengine.attention.paged import attention_for_request
from flexengine.kv_cache.allocator import CacheLayout, PagedKVCache

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("length", [1, 17, 129, 513])
def test_triton_paged_decode_matches_sdpa_with_noncontiguous_pages(length):
    torch.manual_seed(7 + length)
    cache = PagedKVCache(CacheLayout(1, 80, 16, 2, 64), device="cuda:0", dtype=torch.float16)
    cache.reserve("r", length)
    cache.tables["r"] = list(reversed(cache.tables["r"]))
    assert any(b - a != 1 for a, b in zip(cache.tables["r"], cache.tables["r"][1:])) or length <= 16
    keys = torch.randn(length, 2, 64, device="cuda:0", dtype=torch.float16)
    values = torch.randn_like(keys)
    cache.write(0, "r", 0, keys, values)
    query = torch.randn(1, 8, 64, device="cuda:0", dtype=torch.float16)
    position = torch.tensor([length - 1], device="cuda:0")

    expected = attention_for_request(query, cache, 0, "r", length, position, "materialized")
    actual = attention_for_request(query, cache, 0, "r", length, position, "triton")

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_triton_backend_uses_materialized_sdpa_for_multi_token_prefill(monkeypatch):
    cache = PagedKVCache(CacheLayout(1, 8, 4, 1, 16), device="cuda:0", dtype=torch.float16)
    cache.reserve("r", 4)
    keys = torch.randn(4, 1, 16, device="cuda:0", dtype=torch.float16)
    cache.write(0, "r", 0, keys, keys)
    query = torch.randn(4, 2, 16, device="cuda:0", dtype=torch.float16)
    positions = torch.arange(4, device="cuda:0")

    expected = attention_for_request(query, cache, 0, "r", 4, positions, "materialized")
    monkeypatch.setattr(cache, "iter_blocks", lambda *args: (_ for _ in ()).throw(AssertionError("blockwise")))
    actual = attention_for_request(query, cache, 0, "r", 4, positions, "triton")

    torch.testing.assert_close(actual, expected)
