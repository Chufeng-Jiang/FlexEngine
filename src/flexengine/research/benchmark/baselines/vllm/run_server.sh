#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/.venv-vllm/bin/vllm" serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --generation-config vllm \
  --seed 42
