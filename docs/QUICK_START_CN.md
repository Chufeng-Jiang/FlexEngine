# FlexEngine 本地快速复现（RTX 3080 Ti / WSL2）

## 第一步：解压并安装

```bash
unzip FlexEngine_2_0_Source.zip
cd FlexEngine
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 首先确认当前 torch 是 CUDA 版，若不是从 https://pytorch.org/get-started/locally/ 安装。
python -m pip install -e '.[dev]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## 第二步：先跑测试（无需下载模型）

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/unit tests/integration
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/gpu/test_hf_parity.py -k tiny_random
```

第一条测试应该无需 GPU，也不下载权重；第二条需要 `transformers`，但使用随机小模型，不下载权重。显式清空 `CUDA_VISIBLE_DEVICES` 可让 CPU 基线不受本机 CUDA 驱动状态影响。所有报告中的官方模型 parity 和 GPU benchmark 都要在你的设备上确认。

## 第三步：下载官方权重并运行

```bash
FLEX_RUN_REAL_MODEL_TESTS=1 python -m pytest -q tests/gpu/test_hf_parity.py -k real_qwen

# 两个 Proposal 目标模型的完整生成与 KV 回收
FLEX_RUN_REAL_MODEL_TESTS=1 FLEX_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  python -m pytest -q tests/gpu/test_real_generation.py -s
FLEX_RUN_REAL_MODEL_TESTS=1 FLEX_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  python -m pytest -q tests/gpu/test_real_generation.py -s

# Terminal A：启动服务器
FLEX_KV_NUM_BLOCKS=512 FLEX_MAX_MODEL_LEN=2048 \
  python scripts/run_server.py --policy decode_first --attention blockwise \
  --trace-dir experiments/results/local_3080ti

# Terminal B：测试 HTTP SSE
python scripts/smoke_client.py
```

如果 GPU 显存不足，先将 `FLEX_KV_NUM_BLOCKS=512` 降为 `256`；如果速度非常慢（blockwise PyTorch 按页运行很多小 Kernel），换成 `--attention materialized` 运行对照。

## 第四步：测性能并生成报告

```bash
python benchmarks/replay.py --workload w1 --concurrency 2 --requests 20 \
  --output experiments/results/w1_c2
cp experiments/results/local_3080ti/requests.jsonl experiments/results/w1_c2/requests.jsonl
python experiments/analyze.py experiments/results/w1_c2 \
  --output experiments/results/analysis --plots
```

从 `experiments/results/analysis/RESULTS.md` 和 CSV 获取汇总结果。正式论文/面试不要用唯一一轮数据下性能结论，至少做 3–5 次重复，并准备 Nsight 截图。

## 第五步：Profiling

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none -o flex_nsys \
  python benchmarks/profile_engine.py --steps 80 --batch-size 4

python benchmarks/profile_pytorch.py --steps 12 --batch-size 2

ncu --set full --target-processes all -o flex_ncu \
  python benchmarks/profile_kernel.py --backend blockwise --seq-len 512
```

## 完整实现范围说明

当前实现了完整的自研 Qwen2.5 eager forward 路径、Scheduler、分页 KV 读写、SSE 和 benchmark，但 **完整 decoder CUDA Graph、Prefix Caching、真正 fused Triton Paged Attention** 仍是后续研究工作。初版优先追求可复现的 correctness 和模块透明性，不承诺超越 vLLM。
