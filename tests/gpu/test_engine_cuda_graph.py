import os

import pytest
import torch

from flexengine.config import EngineConfig
from flexengine.models.qwen2 import Qwen2Adapter
from flexengine.runtime import Engine
from flexengine.scheduler.state import State

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(os.getenv("FLEX_RUN_REAL_MODEL_TESTS") != "1", reason="real model test disabled"),
]


def run_batch(adapter, active_count: int, graph: bool):
    config = EngineConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda:0",
        dtype="float16",
        max_model_len=32,
        max_num_seqs=max(8, active_count),
        max_num_batched_tokens=32,
        kv_block_size=4,
        kv_num_blocks=96,
        max_output_tokens=4,
        cuda_graph=graph,
        trace_steps=False,
    )
    engine = Engine(config, adapter)
    try:
        requests = [engine.submit(f"graph parity {index}", max_tokens=3) for index in range(active_count)]
        for _ in range(20):
            engine.step()
            if all(request.state is State.FINISHED for request in requests):
                break
        assert all(request.state is State.FINISHED for request in requests)
        return [request.generated_ids for request in requests], engine.snapshot()
    finally:
        engine.stop()


def test_engine_graph_matches_eager_for_all_buckets_and_fallback():
    adapter = Qwen2Adapter.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", device="cuda:0", dtype="float16", attention_backend="blockwise"
    )
    for active_count in (1, 2, 3, 4, 5, 8):
        eager, _ = run_batch(adapter, active_count, graph=False)
        replayed, snapshot = run_batch(adapter, active_count, graph=True)
        assert replayed == eager
        assert snapshot["cuda_graph"]["replays"] >= 1
        assert snapshot["cuda_graph"]["padded_slots"] >= (1 if active_count in (3, 5) else 0)

    _, snapshot = run_batch(adapter, 9, graph=True)
    assert snapshot["cuda_graph"]["fallbacks"] >= 1


def test_engine_graph_with_fused_triton_attention_matches_eager():
    adapter = Qwen2Adapter.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", device="cuda:0", dtype="float16", attention_backend="triton"
    )

    eager, _ = run_batch(adapter, 3, graph=False)
    replayed, snapshot = run_batch(adapter, 3, graph=True)

    assert replayed == eager
    assert snapshot["cuda_graph"]["replays"] >= 1
