#!/usr/bin/env bash
set -euo pipefail
mkdir -p experiments/profiling/upgrade
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
  -o experiments/profiling/upgrade/steady_state "$@"
