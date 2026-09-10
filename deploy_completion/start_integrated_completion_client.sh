#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${RA_PYTHON:-/home/geekplus/miniforge3/envs/piper_ros/bin/python}"

exec "$PYTHON_BIN" deploy_completion/integrated_completion_online_inference_execution.py \
  --config deploy_completion/integrated_completion_online_inference.yaml \
  "$@"
