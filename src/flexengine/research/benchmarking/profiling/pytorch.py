"""PyTorch profiler alternative for CPU/GPU operator diagnosis.

Export Chrome trace. Run without profiler for final wall time/QPS.
"""
import argparse
from pathlib import Path

import torch

from flexengine.engine.config import EngineConfig
from flexengine.engine.runtime import Engine
from flexengine.inference.models.qwen2 import Qwen2Adapter


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--steps",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--output",default="results/pytorch_trace.json")
    args=p.parse_args()
    conf=EngineConfig(model=args.model,max_num_seqs=args.batch_size,
                      max_num_batched_tokens=128,trace_steps=False)
    adapter=Qwen2Adapter.from_pretrained(conf.model,conf.device,conf.dtype,
                                       revision=conf.model_revision or None)
    engine=Engine(conf,adapter)
    try:
        for i in range(args.batch_size):
            engine.submit(f"Question {i}: Explain the LLM KV cache.", max_tokens=64)
        activities=[torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=True,with_stack=False) as prof:
            for _ in range(args.steps):
                engine.step()
                prof.step()
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        prof.export_chrome_trace(str(path))
        print(prof.key_averages().table(sort_by="self_cuda_time_total" if torch.cuda.is_available()
                                        else "self_cpu_time_total",row_limit=20))
        print("Saved trace to",path)
    finally:
        engine.stop()


if __name__=="__main__":
    main()
