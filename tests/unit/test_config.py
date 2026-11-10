import os

from flexengine.cli.server import apply_args, build_parser
from flexengine.config import matched_kv_blocks


def test_matched_kv_blocks_preserves_pool_bytes():
    assert matched_kv_blocks(512, 16, 8) == 1024
    assert matched_kv_blocks(512, 16, 16) == 512
    assert matched_kv_blocks(512, 16, 32) == 256


def test_run_server_experiment_flags_map_to_environment(monkeypatch):
    for key in (
        "FLEX_MAX_NUM_BATCHED_TOKENS",
        "FLEX_KV_BLOCK_SIZE",
        "FLEX_KV_NUM_BLOCKS",
        "FLEX_MAX_MODEL_LEN",
        "FLEX_MAX_NUM_SEQS",
        "FLEX_PREFIX_CACHE",
        "FLEX_PREFIX_CACHE_MAX_ENTRIES",
        "FLEX_PREFILL_CHUNK_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)
    args = build_parser().parse_args(
        [
            "--token-budget", "256",
            "--block-size", "8",
            "--num-blocks", "1024",
            "--max-model-len", "4096",
            "--max-num-seqs", "16",
            "--prefix-cache",
            "--prefix-cache-max-entries", "64",
            "--prefill-chunk-size", "96",
            "--policy", "adaptive",
        ]
    )

    apply_args(args)

    assert os.environ["FLEX_MAX_NUM_BATCHED_TOKENS"] == "256"
    assert os.environ["FLEX_KV_BLOCK_SIZE"] == "8"
    assert os.environ["FLEX_KV_NUM_BLOCKS"] == "1024"
    assert os.environ["FLEX_MAX_MODEL_LEN"] == "4096"
    assert os.environ["FLEX_MAX_NUM_SEQS"] == "16"
    assert os.environ["FLEX_PREFIX_CACHE"] == "true"
    assert os.environ["FLEX_PREFIX_CACHE_MAX_ENTRIES"] == "64"
    assert os.environ["FLEX_PREFILL_CHUNK_SIZE"] == "96"
    assert os.environ["FLEX_SCHEDULER_POLICY"] == "adaptive"
