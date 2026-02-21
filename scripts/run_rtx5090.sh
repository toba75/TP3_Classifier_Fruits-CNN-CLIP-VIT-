#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

docker build -t tp3-experiments:rtx5090 -f Dockerfile.rtx5090 .

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$ROOT_DIR:/workspace" \
  tp3-experiments:rtx5090 bash /workspace/run_experiments.sh