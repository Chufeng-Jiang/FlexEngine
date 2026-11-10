"""Run a deterministic in-process workload for NVTX/CUDA-event profiling.

nsys profile --trace=cuda,nvtx,osrt --sample=none -o /tmp/flex_profile \\
  python -m flexengine.benchmarking.profiling.engine --steps 100 --batch-size 4
"""
import argparse
import json
import time

from flexengine.config import EngineConfig
from flexengine.models.qwen2 import Qwen2Adapter
from flexengine.runtime import Engine
from flexengine.scheduler.state import TERMINAL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--policy", choices=["fcfs", "decode_first", "slo_aware"], default="decode_first")
    p.add_argument("--attention", choices=["blockwise", "materialized", "triton"], default="blockwise")
    p.add_argument("--trace-dir", default="results/profile")
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    conf = EngineConfig(model=args.model, max_num_seqs=args.batch_size, scheduler_policy=args.policy,
                        attention_backend=args.attention, trace_dir=args.trace_dir,
                        max_output_tokens=512, max_model_len=2048,
                        cuda_graph=args.cuda_graph, prefix_cache=args.prefix_cache)
    adapter = Qwen2Adapter.from_pretrained(conf.model, conf.device, conf.dtype,
                                          revision=conf.model_revision or None,
                                          attention_backend=conf.attention_backend)
    runtime = Engine(conf, adapter)
    reqs = [runtime.submit(f"Request {i}: briefly explain GPU inference optimization.", max_tokens=256)
            for i in range(args.batch_size)]
    try:
        t0 = time.perf_counter()
        for step in range(args.steps):
            runtime.step()
            if all(r.state in TERMINAL for r in reqs):
                break
        print(json.dumps({"steps":runtime.step_index, "elapsed_sec":time.perf_counter()-t0,
                          "engine":runtime.snapshot(), "requests":[r.snapshot() for r in reqs]}, indent=2))
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
