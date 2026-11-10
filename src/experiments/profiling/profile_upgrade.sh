#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/profiling/upgrade
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
  -o results/profiling/upgrade/steady_state "$@"
