# Isolated matched vLLM baseline

This comparison uses a separate Python 3.12 environment, `.venv-vllm`; it does
not modify FlexEngine's environment. The installation follows the current
[official vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html?device=cuda):

```bash
uv venv .venv-vllm --python /home/chufeng/.local/bin/python3.12 --seed
uv pip install --python .venv-vllm/bin/python vllm --torch-backend=auto
```

The comparison pins the same official model, FP16 precision, maximum context,
greedy sampling, request count, concurrency, seed and saved request trace. See
`data/config/vllm_matched.json`.

Start the isolated server:

```bash
bash src/flexengine/benchmarking/baselines/vllm/run_server.sh
```

For each of three repetitions run:

```bash
flexengine benchmark \
  --url http://127.0.0.1:8001/v1/completions \
  --model Qwen/Qwen2.5-0.5B-Instruct --workload w1 \
  --concurrency 4 --requests 8 --max-output-tokens 32 --seed 42 \
  --trace-identity w1-prefix0-n8-out32 \
  --workload-trace results/proposal_20261108/workloads/w1-prefix0-n8-out32.json \
  --output results/vllm_20261109/run-N
```
