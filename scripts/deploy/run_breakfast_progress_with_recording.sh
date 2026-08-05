#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/scripts/deploy/breakfast_progress_online_inference.yaml"
LOG_ROOT="${ROOT_DIR}/logs/breakfast_progress_online"
EXPORT_MP4=0
CLIENT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --export-mp4)
      EXPORT_MP4=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  run_breakfast_progress_with_recording.sh [--export-mp4] [client options]

Examples:
  bash scripts/deploy/run_breakfast_progress_with_recording.sh --speed_percent 20
  bash scripts/deploy/run_breakfast_progress_with_recording.sh --export-mp4

The policy server must already be running. Run this script inside the Python
environment that provides ROS2, cv_bridge, Piper SDK, and openpi_client.
EOF
      exit 0
      ;;
    *)
      CLIENT_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$ROOT_DIR"
mkdir -p "$LOG_ROOT"
RUN_ID="$(date -u +"%Y%m%dT%H%M%S_%6NZ")"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "$RUN_DIR"

RECORDER_PID=""
stop_recorder() {
  if [[ -n "$RECORDER_PID" ]] && kill -0 "$RECORDER_PID" 2>/dev/null; then
    kill -TERM "$RECORDER_PID" 2>/dev/null || true
    wait "$RECORDER_PID" 2>/dev/null || true
  fi
  RECORDER_PID=""
}
trap stop_recorder EXIT INT TERM

echo "[run] Output directory: $RUN_DIR"
python scripts/deploy/record_top_camera.py \
  --output-dir "$RUN_DIR" \
  >"$RUN_DIR/top_recorder.log" 2>&1 &
RECORDER_PID=$!

for _ in $(seq 1 150); do
  if [[ -f "$RUN_DIR/top_recorder.ready" ]]; then
    break
  fi
  if ! kill -0 "$RECORDER_PID" 2>/dev/null; then
    echo "ERROR: Top-camera recorder exited before receiving its first frame." >&2
    cat "$RUN_DIR/top_recorder.log" >&2 || true
    exit 1
  fi
  sleep 0.1
done

if [[ ! -f "$RUN_DIR/top_recorder.ready" ]]; then
  echo "ERROR: Timed out waiting for the first Top-camera frame." >&2
  cat "$RUN_DIR/top_recorder.log" >&2 || true
  exit 1
fi

echo "[run] Top-camera recorder is ready. Starting robot client."
set +e
BREAKFAST_RUN_DIR="$RUN_DIR" python scripts/deploy/online_inference_execution.py \
  --config "$CONFIG_PATH" \
  "${CLIENT_ARGS[@]}"
CLIENT_STATUS=$?
set -e

stop_recorder
trap - EXIT INT TERM

if [[ -s "$RUN_DIR/events.jsonl" && -s "$RUN_DIR/top.mp4" ]]; then
  RENDER_ARGS=(--run-dir "$RUN_DIR")
  if [[ "$EXPORT_MP4" -eq 1 ]]; then
    RENDER_ARGS+=(--export-mp4)
  fi
  python scripts/deploy/render_online_progress.py "${RENDER_ARGS[@]}"
else
  echo "WARNING: Skipping report generation because events.jsonl or top.mp4 is missing." >&2
fi

echo "[run] Finished. Results: $RUN_DIR"
exit "$CLIENT_STATUS"
