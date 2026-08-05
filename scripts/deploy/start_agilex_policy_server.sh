#!/usr/bin/env bash
set -euo pipefail

# Start openpi policy server for AgileX using a local or remote checkpoint.
# Usage:
#   scripts/deploy/start_agilex_policy_server.sh \
#     --checkpoint-dir /path/to/pi05_base \
#     --port 8000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_NAME="pi05_agilex"
CHECKPOINT_DIR=""
PORT="8000"
DEFAULT_PROMPT=""
RECORD_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --default-prompt)
      DEFAULT_PROMPT="$2"
      shift 2
      ;;
    --record)
      RECORD_FLAG="--record"
      shift 1
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  start_agilex_policy_server.sh --checkpoint-dir <path_or_gs_uri> [options]

Required:
  --checkpoint-dir PATH    Checkpoint dir, e.g. gs://openpi-assets/checkpoints/pi05_base

Optional:
  --config NAME            Train config name (default: pi05_agilex)
  --port PORT              Serving port (default: 8000)
  --default-prompt TEXT    Fallback prompt when request has no prompt key
  --record                 Enable PolicyRecorder
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "Error: --checkpoint-dir is required" >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "[openpi] Starting AgileX policy server"
echo "  config: $CONFIG_NAME"
echo "  checkpoint: $CHECKPOINT_DIR"
echo "  port: $PORT"

cmd=(
  uv run scripts/serve_policy.py
  --port "$PORT"
  policy:checkpoint
  --policy.config "$CONFIG_NAME"
  --policy.dir "$CHECKPOINT_DIR"
)

if [[ -n "$DEFAULT_PROMPT" ]]; then
  cmd+=(--default-prompt "$DEFAULT_PROMPT")
fi

if [[ -n "$RECORD_FLAG" ]]; then
  cmd+=("$RECORD_FLAG")
fi

exec "${cmd[@]}"
