import pytest
import torch

from flexengine.attention.flashinfer_backend import (
    flashinfer_available,
    flashinfer_paged_decode_batch,
    flashinfer_workspace,
)
from flexengine.attention.paged import attention_for_request
from flexengine.kv_cache.allocator import CacheLayout, PagedKVCache


def test_flashinfer_unavailable_fails_closed(monkeypatch):
    monkeypatch.setattr("flexengine.attention.flashinfer_backend._FLASHINFER", None)
    q = torch.zeros(1, 2, 4)
    cache = PagedKVCache(CacheLayout(1, 2, 4, 1, 4))

    with pytest.raises(RuntimeError, match="FlashInfer backend is unavailable"):
        flashinfer_paged_decode_batch(q, cache, 0, torch.zeros(1, 1, dtype=torch.long),
                                     torch.zeros(1, dtype=torch.long), torch.ones(1))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flashinfer_workspace_is_reused_per_device():
    first = flashinfer_workspace(torch.device("cuda:0"))
    second = flashinfer_workspace(torch.device("cuda:0"))

    assert first.data_ptr() == second.data_ptr()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not flashinfer_available(), reason="FlashInfer is not installed for this Python/CUDA build")
@pytest.mark.parametrize("length", [1, 17, 129, 513])
def test_flashinfer_paged_decode_matches_sdpa(length):
    torch.manual_seed(31 + length)
    cache = PagedKVCache(CacheLayout(1, 80, 16, 2, 64), device="cuda:0", dtype=torch.float16)
    cache.reserve("r", length)
    cache.tables["r"] = list(reversed(cache.tables["r"]))
    keys = torch.randn(length, 2, 64, device="cuda:0", dtype=torch.float16)
    values = torch.randn_like(keys)
    cache.write(0, "r", 0, keys, values)
    query = torch.randn(1, 8, 64, device="cuda:0", dtype=torch.float16)
    position = torch.tensor([length - 1], device="cuda:0")
    table = torch.tensor([cache.tables["r"]], device="cuda:0", dtype=torch.long)

    expected = attention_for_request(query, cache, 0, "r", length, position, "materialized")
    actual = flashinfer_paged_decode_batch(query, cache, 0, table, position, torch.ones(1, device="cuda:0"))

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
