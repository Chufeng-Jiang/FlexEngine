"""Microbenchmark/NCU target: paged blockwise PyTorch attention vs gathered SDPA.

This times a research baseline, NOT a fused Triton paged kernel.
"""
import argparse
import time

import torch

from flexengine.attention.paged import attention_for_request
from flexengine.kv_cache.allocator import CacheLayout, PagedKVCache


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backend", choices=["blockwise", "materialized", "triton"], default="blockwise")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--nheads", type=int, default=12)
    p.add_argument("--kvheads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--iters", type=int, default=50)
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for kernel profiling")
    cache=PagedKVCache(CacheLayout(1, (args.seq_len+15)//16+1,16,args.kvheads,args.head_dim),
                       device="cuda",dtype=torch.float16)
    cache.reserve("r",args.seq_len)
    cache.write(0,"r",0,torch.randn(args.seq_len,args.kvheads,args.head_dim,device="cuda",dtype=torch.float16),
                torch.randn(args.seq_len,args.kvheads,args.head_dim,device="cuda",dtype=torch.float16))
    q=torch.randn(1,args.nheads,args.head_dim,device="cuda",dtype=torch.float16)
    pos=torch.tensor([args.seq_len-1],device="cuda")
    for _ in range(10):
        attention_for_request(q,cache,0,"r",args.seq_len,pos,args.backend)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    torch.cuda.nvtx.range_push("attention_profile")
    for _ in range(args.iters):
        attention_for_request(q,cache,0,"r",args.seq_len,pos,args.backend)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print({"backend":args.backend,"seq_len":args.seq_len,"ms_per_call":(time.perf_counter()-t0)*1000/args.iters})


if __name__ == "__main__":
    main()
