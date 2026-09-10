#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_NAME="agilex_make_breakfast_generalize_720_subtasks_1500_relative_balanced_ttrtc"
CHECKPOINT_DIR="/home/geekplus/develop/ra_ttrtc/openpi/checkpoints/agilex_make_breakfast_generalize_720_subtasks_1500_relative_balanced_ttrtc/39999"
COMPLETION_HEAD_DIR="/home/geekplus/develop/ra_ttrtc/openpi/checkpoints/done_head_h768"
PORT="8001"
DEFAULT_PROMPT=""
ACTION_HORIZON="50"
CONTROL_HZ="30.0"
NUM_DENOISING_STEPS="5"
EXECUTION_HORIZON="10"
TRAINED_SIMULATED_DELAY="6"
MAX_CONDITIONED_PREFIX_STEPS=""
DELAY_BUFFER_SIZE="20"
DELAY_PERCENTILE="1.0"
INITIAL_DELAY_S="0.10"
MIN_DELAY_STEPS="1"
MAX_DELAY_STEPS=""
REPLAN_MARGIN_STEPS="1"
FILTER_ALPHA="1.0"
HOLD_LAST_ON_UNDERRUN="true"
HISTORY_TOLERANCE_S="0.2"
HISTORY_WINDOW_S="2.0"
RECORD_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --config) CONFIG_NAME="$2"; shift 2 ;;
    --completion-head-dir) COMPLETION_HEAD_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --default-prompt) DEFAULT_PROMPT="$2"; shift 2 ;;
    --action-horizon) ACTION_HORIZON="$2"; shift 2 ;;
    --control-hz) CONTROL_HZ="$2"; shift 2 ;;
    --num-denoising-steps) NUM_DENOISING_STEPS="$2"; shift 2 ;;
    --execution-horizon) EXECUTION_HORIZON="$2"; shift 2 ;;
    --trained-simulated-delay) TRAINED_SIMULATED_DELAY="$2"; shift 2 ;;
    --max-conditioned-prefix-steps) MAX_CONDITIONED_PREFIX_STEPS="$2"; shift 2 ;;
    --delay-buffer-size) DELAY_BUFFER_SIZE="$2"; shift 2 ;;
    --delay-percentile) DELAY_PERCENTILE="$2"; shift 2 ;;
    --initial-delay-s) INITIAL_DELAY_S="$2"; shift 2 ;;
    --min-delay-steps) MIN_DELAY_STEPS="$2"; shift 2 ;;
    --max-delay-steps) MAX_DELAY_STEPS="$2"; shift 2 ;;
    --replan-margin-steps) REPLAN_MARGIN_STEPS="$2"; shift 2 ;;
    --filter-alpha) FILTER_ALPHA="$2"; shift 2 ;;
    --hold-last-on-underrun) HOLD_LAST_ON_UNDERRUN="$2"; shift 2 ;;
    --completion-history-tolerance-s) HISTORY_TOLERANCE_S="$2"; shift 2 ;;
    --completion-history-window-s) HISTORY_WINDOW_S="$2"; shift 2 ;;
    --record) RECORD_FLAG="--record"; shift ;;
    -h|--help)
      sed -n '1,95p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_triton_gemm=false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
RUNTIME_PYTHON="${OPENPI_RUNTIME_PYTHON:-/home/geekplus/develop/ra_ttrtc/openpi/.venv/bin/python}"

cmd=(
  "$RUNTIME_PYTHON" deploy_completion/serve_integrated_completion_policy.py
  --port "$PORT"
  --action-horizon "$ACTION_HORIZON"
  --control-hz "$CONTROL_HZ"
  --num-denoising-steps "$NUM_DENOISING_STEPS"
  --execution-horizon "$EXECUTION_HORIZON"
  --trained-simulated-delay "$TRAINED_SIMULATED_DELAY"
  --delay-buffer-size "$DELAY_BUFFER_SIZE"
  --delay-percentile "$DELAY_PERCENTILE"
  --initial-delay-s "$INITIAL_DELAY_S"
  --min-delay-steps "$MIN_DELAY_STEPS"
  --replan-margin-steps "$REPLAN_MARGIN_STEPS"
  --filter-alpha "$FILTER_ALPHA"
  --completion-head-dir "$COMPLETION_HEAD_DIR"
  --completion-history-tolerance-s "$HISTORY_TOLERANCE_S"
  --completion-history-window-s "$HISTORY_WINDOW_S"
)

if [[ -n "$DEFAULT_PROMPT" ]]; then cmd+=(--default-prompt "$DEFAULT_PROMPT"); fi
if [[ -n "$MAX_CONDITIONED_PREFIX_STEPS" ]]; then cmd+=(--max-conditioned-prefix-steps "$MAX_CONDITIONED_PREFIX_STEPS"); fi
if [[ -n "$MAX_DELAY_STEPS" ]]; then cmd+=(--max-delay-steps "$MAX_DELAY_STEPS"); fi
if [[ "$HOLD_LAST_ON_UNDERRUN" == "true" || "$HOLD_LAST_ON_UNDERRUN" == "1" || "$HOLD_LAST_ON_UNDERRUN" == "yes" ]]; then
  cmd+=(--hold-last-on-underrun)
else
  cmd+=(--no-hold-last-on-underrun)
fi
if [[ -n "$RECORD_FLAG" ]]; then cmd+=("$RECORD_FLAG"); fi

cmd+=(
  policy:checkpoint
  --policy.config "$CONFIG_NAME"
  --policy.dir "$CHECKPOINT_DIR"
)

exec "${cmd[@]}"
