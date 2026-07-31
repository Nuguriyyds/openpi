#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG_NAME="pi05_agilex_breakfast_progress"
readonly EXP_NAME="breakfast_progress_v1"
readonly GPU_DEVICES="4,5,6,7"
readonly FSDP_DEVICES=4
readonly HF_LEROBOT_HOME_PATH="/mnt/data/dataset/ei/huggingface"
readonly DATASET_PATH="${HF_LEROBOT_HOME_PATH}/wyt/agilex_make_breakfast_380_subtask_furniturevla_progress"
readonly WEIGHTS_PATH="/mnt/data/models/openpi/openpi-assets/checkpoints/pi05_base/params"
readonly NORM_STATS_PATH="/mnt/data/models/openpi/assets/${CONFIG_NAME}/wyt/agilex_make_breakfast_380_subtask_furniturevla_progress/norm_stats.json"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    echo "Set it without exposing the key:" >&2
    echo '  read -rsp "W&B API key: " WANDB_API_KEY; echo; export WANDB_API_KEY' >&2
    exit 1
fi

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "ERROR: Breakfast progress dataset not found: ${DATASET_PATH}" >&2
    exit 1
fi

if [[ ! -d "${WEIGHTS_PATH}" ]]; then
    echo "ERROR: pi0.5 base weights not found: ${WEIGHTS_PATH}" >&2
    exit 1
fi

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME_PATH}"

echo "Computing training-only normalization statistics."
echo "Expected subset: 1,500 train episodes; 20 held-out test episodes."

# Norm stats do not need a GPU. Keeping GPUs hidden prevents this process from
# reserving memory immediately before the training process starts.
CUDA_VISIBLE_DEVICES="" \
JAX_PLATFORMS=cpu \
uv run scripts/compute_norm_stats.py \
    --config-name "${CONFIG_NAME}"

if [[ ! -s "${NORM_STATS_PATH}" ]]; then
    echo "ERROR: Norm stats were not written to: ${NORM_STATS_PATH}" >&2
    exit 1
fi

echo "Normalization statistics completed: ${NORM_STATS_PATH}"
echo "Starting ${CONFIG_NAME}/${EXP_NAME} on GPUs ${GPU_DEVICES}."
echo "The existing experiment directory will be overwritten."

CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --fsdp-devices="${FSDP_DEVICES}" \
    --overwrite
