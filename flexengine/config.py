"""Configuration and environment parsing for the single-GPU engine."""
from __future__ import annotations

import os
from dataclasses import dataclass


def matched_kv_blocks(base_blocks: int, base_block_size: int, candidate_block_size: int) -> int:
    if min(base_blocks, base_block_size, candidate_block_size) <= 0:
        raise ValueError("KV block counts and sizes must be positive")
    total_tokens = base_blocks * base_block_size
    if total_tokens % candidate_block_size:
        raise ValueError("candidate block size must divide total KV token capacity")
    return total_tokens // candidate_block_size


@dataclass(frozen=True)
class EngineConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = ""  # optional immutable HF commit SHA
    device: str = "cuda:0"
    dtype: str = "float16"
    max_model_len: int = 2048
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 128
    prefill_chunk_size: int = 128
    max_waiting: int = 64
    kv_block_size: int = 16
    kv_num_blocks: int = 1024
    prefix_cache: bool = False
    prefix_cache_max_entries: int = 128
    cuda_graph: bool = False
    scheduler_policy: str = "fcfs"  # fcfs, decode_first, slo_aware
    attention_backend: str = "blockwise"  # blockwise, materialized, triton
    rmsnorm_backend: str = "torch"  # torch, triton
    ttft_slo_ms: float = 2000.0
    itl_slo_ms: float = 200.0
    idle_sleep_seconds: float = 0.003
    trace_dir: str = ""
    trace_steps: bool = True
    max_output_tokens: int = 512
    random_seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "max_model_len", "max_num_seqs", "max_num_batched_tokens", "prefill_chunk_size", "kv_block_size",
            "kv_num_blocks", "prefix_cache_max_entries", "max_output_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_waiting < 0:
            raise ValueError("max_waiting must be nonnegative")
        if self.scheduler_policy not in ("fcfs", "decode_first", "slo_aware", "adaptive"):
            raise ValueError("Unknown scheduler_policy")
        if self.attention_backend not in ("blockwise", "materialized", "triton"):
            raise ValueError("Unknown attention_backend")
        if self.rmsnorm_backend not in ("torch", "triton"):
            raise ValueError("Unknown rmsnorm_backend")
        if self.kv_num_blocks * self.kv_block_size < self.max_model_len:
            raise ValueError("KV pool cannot contain one max-length request")
        if self.max_num_batched_tokens < 1:
            raise ValueError("token budget must be positive")

    @classmethod
    def from_env(cls) -> EngineConfig:
        def env(key: str, cast, default):
            value = os.getenv(f"FLEX_{key.upper()}")
            return cast(value) if value is not None else default

        defaults = cls()
        kwargs = {}
        for key, field_type in (
            ("model", str), ("model_revision", str), ("device", str), ("dtype", str), ("max_model_len", int),
            ("max_num_seqs", int), ("max_num_batched_tokens", int), ("prefill_chunk_size", int),
            ("max_waiting", int),
            ("kv_block_size", int), ("kv_num_blocks", int), ("scheduler_policy", str),
            ("prefix_cache_max_entries", int),
            ("attention_backend", str), ("rmsnorm_backend", str),
            ("ttft_slo_ms", float), ("itl_slo_ms", float), ("idle_sleep_seconds", float),
            ("trace_dir", str), ("max_output_tokens", int), ("random_seed", int),
        ):
            kwargs[key] = env(key, field_type, getattr(defaults, key))
        kwargs["trace_steps"] = env("trace_steps", lambda v: v.lower() in ("true", "yes", "1"), True)
        kwargs["prefix_cache"] = env("prefix_cache", lambda v: v.lower() in ("true", "yes", "1"), False)
        kwargs["cuda_graph"] = env("cuda_graph", lambda v: v.lower() in ("true", "yes", "1"), False)
        return cls(**kwargs)
