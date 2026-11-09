"""Optional correctness test: tiny random Qwen2 weights, no Hub download.

Run `pytest -q tests/gpu` after installing transformers. This test works on
CPU too; mark GPU tests for CI separation because HF version matrix differs.
"""
import pytest
import torch

from flexengine.kv_cache.allocator import CacheLayout, PagedKVCache
from flexengine.models.qwen2 import Qwen2Adapter
from flexengine.scheduler.policies import Chunk


@pytest.mark.parametrize("backend", ["blockwise", "materialized"])
def test_tiny_random_qwen2_logits_match_full_and_incremental(backend):
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(123)
    cfg = Qwen2Config(vocab_size=50, hidden_size=64, intermediate_size=96,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64,
                      hidden_act="silu", rope_theta=1000000.0,
                      attention_dropout=0.0)
    base = Qwen2ForCausalLM(cfg).eval()
    adapter = Qwen2Adapter(base, "cpu", attention_backend=backend)
    cache = PagedKVCache(CacheLayout(2, 12, 4, 2, 16), device="cpu", dtype=torch.float32)
    ids = [3, 7, 13, 8, 2, 4, 5]
    cache.reserve("r", len(ids))
    got = adapter.forward([Chunk("r", ids, 0)], cache)[0]
    with torch.no_grad():
        expected = base(torch.tensor([ids])).logits[0, -1]
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)
    cache.free_request("r")
    cache.reserve("r", len(ids))
    adapter.forward([Chunk("r", ids[:3], 0)], cache)
    adapter.forward([Chunk("r", ids[3:6], 3)], cache)
    incremental = adapter.forward([Chunk("r", [ids[6]], 6)], cache)[0]
    torch.testing.assert_close(incremental, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.gpu
def test_real_qwen_optional():
    import os
    if os.getenv("FLEX_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("Set FLEX_RUN_REAL_MODEL_TESTS=1 to load real Qwen2.5 weights")
    from flexengine.models.qwen2 import Qwen2Adapter
    model_id = os.getenv("FLEX_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    adapter = Qwen2Adapter.from_pretrained(model_id, device="cuda:0", dtype="float16")
    ids = adapter.tokenizer.encode("What is 2 + 2?", add_special_tokens=False)
    cache = PagedKVCache(CacheLayout(adapter.layers, 200, 16, adapter.nkv, adapter.dim),
                         device="cuda:0", dtype=torch.float16)
    cache.reserve("r", len(ids))
    got = adapter.forward([Chunk("r", ids, 0)], cache)[0]
    with torch.no_grad():
        expected = adapter.model(torch.tensor([ids], device="cuda:0")).logits[0, -1]
    torch.testing.assert_close(got, expected, atol=0.08, rtol=0.05)
