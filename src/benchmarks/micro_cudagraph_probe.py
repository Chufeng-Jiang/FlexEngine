"""Static-shape CUDA Graph feasibility probe, NOT the engine graph backend.

Captures one Qwen-style RMSNorm + linear + activation block to measure replay
vs eager launch time. Not a decode graph, so do not claim graph mode in runtime.
"""
import argparse
import statistics

import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=896)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--iters", type=int, default=100)
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    x=torch.randn(args.batch,args.hidden,device="cuda",dtype=torch.float16)
    norm=torch.nn.RMSNorm(args.hidden,eps=1e-6).cuda().half()
    proj=torch.nn.Linear(args.hidden,args.hidden,bias=False).cuda().half()
    def f():
        return torch.nn.functional.silu(proj(norm(x)))
    for _ in range(8):
        f()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured=f()
    torch.cuda.synchronize()
    def measure(fn):
        times=[]
        for _ in range(args.iters):
            t0=torch.cuda.Event(enable_timing=True); t1=torch.cuda.Event(enable_timing=True)
            t0.record(); fn(); t1.record(); t1.synchronize()
            times.append(t0.elapsed_time(t1))
        return statistics.median(times)
    eager_ms=measure(f)
    graph_ms=measure(g.replay)
    torch.testing.assert_close(captured,f(),rtol=2e-2,atol=2e-2)
    print({"eager_gpu_event_ms":eager_ms,"graph_gpu_event_ms":graph_ms,
           "scope":"micro-probe only, not full engine decode graph"})


if __name__ == "__main__":
    main()
