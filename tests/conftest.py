"""Small CPU-only stand-in model for deterministic runtime/API tests.

No network/model download needed; these are NOT Qwen correctness tests.
"""
import types

import pytest
import torch

from flexengine.engine.config import EngineConfig
from flexengine.engine.runtime import Engine


class TinyTokenizer:
    bos_token_id = 1

    def encode(self, text: str, add_special_tokens=False):
        return [ord(c) % 20 + 1 for c in text]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(97 + (i % 26)) for i in ids if i != 0)


class TinyAdapter:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = torch.nn.Embedding(32, 4)
        self.layers = 2
        self.nkv = 1
        self.dim = 4
        self.config = types.SimpleNamespace(eos_token_id=0)
        self.tokenizer = TinyTokenizer()

    def forward(self, chunks, cache):
        output = []
        for c in chunks:
            # Write real per-request KV pages for both layers, so resource and
            # block-boundary lifecycle is exercised in CPU-only tests.
            for layer in range(self.layers):
                k = torch.stack([torch.full((1, 4), float(t)) for t in c.input_ids])
                v = k + 1
                cache.write(layer, c.request_id, c.start, k, v)
            logits = torch.full((32,), -100.0)
            logits[(c.input_ids[-1] + 1) % 32] = 10.0
            output.append(logits)
        return output


@pytest.fixture
def conf():
    return EngineConfig(model="test", device="cpu", dtype="float32", max_model_len=64,
                        max_num_seqs=4, max_num_batched_tokens=8, max_waiting=8,
                        kv_block_size=4, kv_num_blocks=20, max_output_tokens=16,
                        idle_sleep_seconds=0.001, trace_steps=False)


@pytest.fixture
def engine(conf):
    runtime = Engine(conf, TinyAdapter())
    yield runtime
    runtime.stop()
