# FlexEngine 2.0

FlexEngine is a from-scratch, single-GPU LLM inference runtime for studying scheduling, paged KV memory, continuous batching, shared-prefix reuse, CUDA Graph replay, and fused paged attention. It uses official Qwen2.5 weights, but does **not** call Hugging Face `generate()`, vLLM, or SGLang for FlexEngine execution.

The complete Proposal was implemented and validated on an NVIDIA GeForce RTX 3080 Ti Laptop GPU with 16 GiB under WSL2. This is research software: it is auditable and reproducible, but it is not a hardened public inference service.

## Project status

| Area | Implemented evidence |
|---|---|
| Models | Custom Qwen2.5 0.5B/1.5B decoder forward, RoPE, GQA, FP16 official-weight parity |
| Scheduling | FCFS, decode-first with overdue-prefill fairness, heuristic SLO-aware, token-budget chunked prefill |
| Batching | Continuous multi-request batching with per-request state and RNG |
| KV memory | Physical paged K/V tensors, block tables, reclamation, recompute preemption, true reference counts |
| Prefix cache | Exact-token longest complete-block match, immutable shared pages, divergent private tails, bounded LRU |
| Attention | Materialized SDPA reference, blockwise online softmax, fused Triton single-token paged decode |
| CUDA Graph | Engine-integrated 1/2/4/8 decode buckets, upward padding, masks, stable staging, eager fallback |
| Generation | Greedy, temperature, top-k, streaming text, deterministic per-request seeds |
| API | FastAPI, OpenAI-style `/v1/completions`, SSE, disconnect/explicit cancellation, 429 backpressure |
| Observability | JSONL, CUDA Events, NVTX, Prometheus, Nsight Systems and Nsight Compute artifacts |
| Experiments | Controlled E1-E6 matrix, 39 configurations × 3 successful repetitions, raw and aggregate reports |
| Baseline | Isolated vLLM 0.33.0 comparison using the exact saved request trace |

## Architecture

```text
HTTP/SSE clients
      |
      v
FastAPI admission and cancellation
      |
      v
Request state machine ----> FCFS / Decode-first / SLO-aware scheduler
      |                                  |
      |                         token-budget step plan
      v                                  v
Prefix lookup/attach ----------> single GPU worker
                                      |
                    +-----------------+-----------------+
                    |                                   |
              chunked prefill                    one-token decode
                    |                                   |
             eager custom Qwen2            eager or CUDA Graph 1/2/4/8
                    |                                   |
                    +---------------+-------------------+
                                    |
                       Paged KV allocator/block tables
                                    |
             materialized SDPA / blockwise / fused Triton attention
                                    |
                         sampler -> SSE token events
                                    |
                  JSONL + Prometheus + NVTX + CUDA Events
```

The GPU worker is the sole owner of model execution. HTTP threads communicate through thread-safe request queues. Every physical KV block has an exact reference count composed of request-table and prefix-index ownership. Cache eviction and request cancellation release only their own references.

### Source map

```text
flexengine/
  api/server.py                    FastAPI, SSE, cancellation, health, metrics
  runtime.py                       admission, batching, preemption, graph dispatch
  scheduler/{state,policies}.py    lifecycle and three scheduling policies
  kv_cache/{allocator,prefix}.py   pages, block tables, refcounts, prefix LRU
  models/qwen2.py                  custom Qwen2.5 eager/static decoder paths
  attention/paged.py               backend selection and reference attention
  kernels/paged_attention.py       fused Triton paged-decode kernels
  execution/cuda_graph.py          fixed-bucket capture/replay manager
  sampling/sampler.py              greedy/top-k/temperature sampling
  observability/                   environment, JSONL, NVTX, Prometheus
benchmarks/                         HTTP, engine and kernel benchmarks
experiments/                        matrix runner, aggregation and strict audit
tests/                              CPU, integration and real-GPU correctness
environment/                        frozen package exports and system metadata
```

## Validated environment

Exact exports: [`environment/system.json`](environment/system.json), [`environment/requirements-flexengine.txt`](environment/requirements-flexengine.txt), and [`environment/requirements-vllm.txt`](environment/requirements-vllm.txt).

| Component | Validated value |
|---|---|
| OS | WSL2 Ubuntu, Linux 6.18.33.2, glibc 2.35 |
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16,384 MiB, compute capability 8.6 |
| Driver | 610.60 |
| Python | 3.13.5 (FlexEngine); 3.12.2 (isolated vLLM) |
| PyTorch | 2.15.0+cu132 |
| CUDA / cuDNN | 13.2 / 9.26.0 |
| Transformers / Triton | 5.22.2 / 3.9.0 |
| vLLM baseline | 0.33.0 with PyTorch 2.14.0+cu132 |

Pinned model snapshots:

- Qwen2.5-0.5B-Instruct: `7ae557604adf67be50417f59c2c2f167def9a775`
- Qwen2.5-1.5B-Instruct: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

Regenerate the exports:

```bash
.venv/bin/python scripts/export_environment.py
```

## Installation

Run inside WSL2 Ubuntu or Linux with a compatible NVIDIA driver:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python scripts/check_environment.py --device cuda:0
```

For the exact validated package set, install `environment/requirements-flexengine.txt` into a fresh environment, then install this project editable.

### Docker

```bash
docker build -t flexengine:2.0 .
docker run --rm --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  flexengine:2.0
```

## Correctness and test reproduction

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q
# Validated: 75 passed, 11 skipped

python -m ruff check flexengine scripts benchmarks experiments tests
# Validated: All checks passed
```

Official 0.5B full GPU suite:

```bash
FLEX_RUN_REAL_MODEL_TESTS=1 \
FLEX_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
python -m pytest -q tests/gpu -s
# Validated: 14 passed
```

Official 1.5B parity and complete generation/KV reclamation:

```bash
FLEX_RUN_REAL_MODEL_TESTS=1 \
FLEX_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
python -m pytest -q \
  tests/gpu/test_hf_parity.py::test_real_qwen_optional \
  tests/gpu/test_real_generation.py::test_official_qwen_completes_text_generation_and_reclaims_kv -s
# Validated: 2 passed
```

| Model | Generated continuation | Peak allocated | Peak reserved |
|---|---|---:|---:|
| 0.5B | ` Paris. It is the largest city in` | 1,017,782,272 B | 1,128,267,776 B |
| 1.5B | ` Paris. The capital of France is also` | 3,126,618,112 B | 3,401,580,544 B |

## Start the service and smoke test

Terminal A:

```bash
source .venv/bin/activate
PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/run_server.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --policy decode_first --attention triton \
  --prefix-cache --cuda-graph \
  --trace-dir experiments/results/local
```

Terminal B:

```bash
python scripts/smoke_client.py
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/metrics
```

Streaming request:

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Explain continuous batching.","max_tokens":32,"temperature":0,"stream":true}'
```

The server listens on localhost and has no authentication or TLS. Do not expose it directly to the public Internet.

## Configuration

Use `python scripts/run_server.py --help`. Important environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `FLEX_MODEL` | Qwen2.5-0.5B-Instruct | official model ID |
| `FLEX_MODEL_REVISION` | empty | immutable HF commit SHA; pin formal runs |
| `FLEX_DEVICE` / `FLEX_DTYPE` | `cuda:0` / `float16` | device and dtype |
| `FLEX_MAX_MODEL_LEN` | 2048 | prompt plus output limit |
| `FLEX_MAX_NUM_SEQS` | 8 | active sequence limit |
| `FLEX_MAX_NUM_BATCHED_TOKENS` | 128 | scheduler step budget |
| `FLEX_KV_BLOCK_SIZE` / `FLEX_KV_NUM_BLOCKS` | 16 / 1024 | page geometry |
| `FLEX_SCHEDULER_POLICY` | `fcfs` | `fcfs`, `decode_first`, `slo_aware` |
| `FLEX_ATTENTION_BACKEND` | `blockwise` | `materialized`, `blockwise`, `triton` |
| `FLEX_PREFIX_CACHE` | false | shared-prefix KV reuse |
| `FLEX_PREFIX_CACHE_MAX_ENTRIES` | 128 | prefix LRU bound |
| `FLEX_CUDA_GRAPH` | false | engine decode graph replay |
| `FLEX_TRACE_DIR` | empty | raw server trace directory |

KV bytes = `2 × layers × blocks × block_size × kv_heads × head_dim × dtype_bytes`.

## Formal benchmark methodology

The audited matrix contains 39 controlled E1-E6 configurations. Each has three successful repetitions; every repetition uses eight requests, fixed seed, saved request trace, matrix-defined concurrency, and at most 32 output tokens. Total: 936/936 completed formal requests.

Artifacts:

- `experiments/results/proposal_20261108/E*/...`
- `experiments/results/proposal_20261108/aggregate.csv`
- `experiments/results/proposal_20261108/REPORT.md`
- `experiments/config/proposal_matrix.json`

Reproduce and audit:

```bash
python experiments/run_proposal.py --repetitions 3 --requests 8 \
  --max-output-tokens 32 --output experiments/results/proposal_20261108
python experiments/analyze_proposal.py \
  --input experiments/results/proposal_20261108 --repetitions 3
python experiments/audit_proposal.py \
  --input experiments/results/proposal_20261108 \
  --configurations 39 --repetitions 3 --requests 8
# Validated: 39 configurations, 936 completed requests, errors: []
```

TTFT is the first non-empty SSE text event. SSE inter-event p95 is not exact token-level ITL because events can merge or omit token pieces; exact server token timestamps remain in `requests.jsonl`. Values below are mean ± sample standard deviation where shown.

## Selected benchmark results

| Experiment | Configuration | Requests/s | TTFT p95 ms | SSE event p95 ms |
|---|---|---:|---:|---:|
| E1 | 0.5B, concurrency 1 | 0.671 ± 0.050 | 188.5 | 58.0 |
| E1 | 0.5B, concurrency 4 | 1.054 ± 0.058 | 743.3 | 137.0 |
| E1 | 0.5B, concurrency 8 | 1.112 ± 0.006 | 1423.9 | 242.4 |
| E1 | 1.5B, concurrency 4 | 0.925 ± 0.013 | 682.8 | 145.6 |
| E2 | W4 FCFS | 0.892 ± 0.039 | 1661.4 | 157.9 |
| E2 | W4 decode-first | 0.898 ± 0.019 | 1779.0 | 148.1 |
| E2 | W4 SLO-aware | 0.965 ± 0.015 | 1585.5 | 128.3 |
| E3 | token budget 64 | 0.801 ± 0.029 | 2578.4 | 146.7 |
| E3 | token budget 512 | 1.034 ± 0.024 | 763.7 | 136.6 |
| E4 | block 16, blockwise | 0.180 ± 0.010 | 3369.8 | 825.4 |
| E4 | block 16, materialized | 1.016 ± 0.045 | 572.4 | 157.0 |
| E4 | block 16, Triton | 0.946 ± 0.326 | 1497.3 | 350.3 |
| E5 | 75% shared prefix, cache off | 1.021 ± 0.040 | 948.8 | 135.0 |
| E5 | 75% shared prefix, cache on | 1.052 ± 0.021 | 790.5 | 140.5 |
| E6 | materialized, graph off | 1.108 ± 0.030 | 540.0 | 123.5 |
| E6 | materialized, graph on | 3.034 ± 0.600 | 667.5 | 85.6 |
| E6 | Triton, graph off | 1.132 ± 0.071 | 1225.2 | 175.3 |
| E6 | Triton, graph on | 2.645 ± 0.640 | 1260.4 | 120.3 |

CUDA Graph improved materialized end-to-end request throughput by 2.74× in E6. The fused Triton microbenchmark at 512 context tokens measured 0.537 ms/call versus 33.662 ms/call for Python blockwise (about 62.7×). End-to-end Triton includes cold compilation and other engine costs; the first cold-compile trial is retained in mean/stdev.

## vLLM matched baseline

vLLM runs in `.venv-vllm`, never in FlexEngine's environment. It uses vLLM 0.33.0, FP16, max length 2048, greedy parameters, concurrency 4, eight requests, 32 output tokens, seed 42, neutral `--generation-config vllm`, and the exact saved W1 trace.

| Engine | Requests/s | TTFT p95 ms | SSE event p95 ms | Completed |
|---|---:|---:|---:|---:|
| FlexEngine E1 0.5B c4 | 1.054 ± 0.058 | 743.3 | 137.0 | 24/24 |
| vLLM 0.33.0 | 14.779 ± 1.209 | 131.2 ± 111.9 | 14.6 ± 7.9 | 24/24 |

See `benchmarks/vllm/README.md`; raw summaries are in `experiments/results/vllm_20261109/`. This quantifies the gap to a production engine; it is not a claim that FlexEngine outperforms vLLM.

## Profiling reproduction

See `docs/NSIGHT_BASELINE.md` for the full report and commands.

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  --stats=true -o experiments/profiling/nsys_baseline \
  .venv/bin/python benchmarks/profile_engine.py --steps 8 --batch-size 2 \
  --attention blockwise --trace-dir experiments/profiling/trace_baseline

/usr/local/cuda/bin/ncu --set basic --nvtx \
  --nvtx-include attention_profile/ --force-overwrite \
  -o experiments/profiling/ncu_triton \
  .venv/bin/python benchmarks/profile_kernel.py \
  --backend triton --seq-len 512 --iters 1
```

Captured artifacts include `nsys_baseline.nsys-rep`, `ncu_blockwise.ncu-rep`, and `ncu_triton.ncu-rep`. Nsight recorded 24,318 kernel launches in the short blockwise baseline, motivating graph replay and fused attention.

## Guides

- `docs/QUICK_START_CN.md` — Chinese quick start
- `docs/ARCHITECTURE.md` — architecture notes
- `docs/NSIGHT_BASELINE.md` — profiling report
- `docs/FINAL_REPORT.md` — completed Proposal report
- `docs/DOD_STATUS.md` — requirement audit
- `docs/PROPOSAL_2_0.md` — original Proposal

## Limitations

- Single GPU only; no tensor/pipeline parallelism, NCCL, speculative decoding, quantization, or production security layer.
- SLO-aware is a research heuristic, not an optimal controller.
- The Triton kernel specializes power-of-two sequence buckets; first use pays compilation cost.
- Prefix reuse shares complete immutable blocks and re-executes at least the final prompt token because cached KV does not contain prompt-final logits.
- CUDA Graph uses a masked full-context static path; graph memory/context trade-offs remain visible in telemetry.
- The formal matrix is deliberately small and should be extended before deployment-level capacity conclusions.

## Intended use

FlexEngine is an educational/research runtime and portfolio-quality systems artifact. Review security, licensing, model terms, and operational safeguards before adapting it for real users.
