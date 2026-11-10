import argparse
import os

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch FlexEngine as a local service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=None)
    p.add_argument("--model-revision", default=None)
    p.add_argument("--policy", choices=["fcfs", "decode_first", "slo_aware", "adaptive"], default=None)
    p.add_argument("--attention", choices=["blockwise", "materialized", "triton", "flashinfer"], default=None)
    p.add_argument("--trace-dir", default=None)
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--prefill-chunk-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prefix-cache-max-entries", type=int, default=None)
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None)
    return p


def apply_args(args: argparse.Namespace) -> None:
    if args.model:
        os.environ["FLEX_MODEL"] = args.model
    if args.model_revision:
        os.environ["FLEX_MODEL_REVISION"] = args.model_revision
    if args.policy:
        os.environ["FLEX_SCHEDULER_POLICY"] = args.policy
    if args.attention:
        os.environ["FLEX_ATTENTION_BACKEND"] = args.attention
    if args.trace_dir:
        os.environ["FLEX_TRACE_DIR"] = args.trace_dir
    for argument, environment in (
        ("token_budget", "FLEX_MAX_NUM_BATCHED_TOKENS"),
        ("prefill_chunk_size", "FLEX_PREFILL_CHUNK_SIZE"),
        ("block_size", "FLEX_KV_BLOCK_SIZE"),
        ("num_blocks", "FLEX_KV_NUM_BLOCKS"),
        ("max_model_len", "FLEX_MAX_MODEL_LEN"),
        ("max_num_seqs", "FLEX_MAX_NUM_SEQS"),
        ("prefix_cache_max_entries", "FLEX_PREFIX_CACHE_MAX_ENTRIES"),
    ):
        value = getattr(args, argument)
        if value is not None:
            os.environ[environment] = str(value)
    if args.prefix_cache is not None:
        os.environ["FLEX_PREFIX_CACHE"] = str(args.prefix_cache).lower()
    if args.cuda_graph is not None:
        os.environ["FLEX_CUDA_GRAPH"] = str(args.cuda_graph).lower()


def main():
    args = build_parser().parse_args()
    apply_args(args)
    uvicorn.run("flexengine.api.server:app", host=args.host, port=args.port, reload=False, workers=1)


if __name__ == "__main__":
    main()
