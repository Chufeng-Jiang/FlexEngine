"""Optional complete generation test for official Qwen2.5 checkpoints."""
import os
from queue import Empty

import pytest
import torch

from flexengine.config import EngineConfig
from flexengine.models.qwen2 import Qwen2Adapter
from flexengine.runtime import Engine


@pytest.mark.gpu
def test_official_qwen_completes_text_generation_and_reclaims_kv():
    if os.getenv("FLEX_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("Set FLEX_RUN_REAL_MODEL_TESTS=1 to load official Qwen2.5 weights")
    model_id = os.getenv("FLEX_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    torch.cuda.reset_peak_memory_stats()
    adapter = Qwen2Adapter.from_pretrained(
        model_id, device="cuda:0", dtype="float16", attention_backend="materialized"
    )
    config = EngineConfig(
        model=model_id,
        device="cuda:0",
        dtype="float16",
        max_model_len=256,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        kv_block_size=16,
        kv_num_blocks=64,
        attention_backend="materialized",
        trace_steps=False,
        max_output_tokens=16,
    )
    engine = Engine(config, adapter)
    engine.start()
    try:
        request = engine.submit("The capital of France is", max_tokens=8, temperature=0.0)
        token_events = []
        while True:
            try:
                event = request.events.get(timeout=120)
            except Empty:
                pytest.fail("official model generation did not finish within 120 seconds")
            if event.kind == "token":
                token_events.append(event)
            elif event.kind == "error":
                pytest.fail(event.error)
            elif event.kind == "done":
                assert event.finish_reason in ("stop", "length")
                break
        assert token_events
        assert request.generated_ids
        assert request.text.strip()
        assert engine.cache.used_blocks == 0
        print(
            f"model={model_id} output={request.text!r} "
            f"peak_allocated={torch.cuda.max_memory_allocated()} "
            f"peak_reserved={torch.cuda.max_memory_reserved()}"
        )
    finally:
        engine.stop()
