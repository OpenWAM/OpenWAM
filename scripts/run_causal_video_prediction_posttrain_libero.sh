#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_fixed128_rollout_context_defaults.sh"

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
LOG_RANK=${LOG_RANK:-"0"}
CONFIG_NAME=${CONFIG_NAME:-"causal_video_prediction_libero_latent_local"}
open_wam_reject_cli_config_override_args "$@"
OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_ARGS=()
open_wam_append_fixed128_rollout_context_args OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_ARGS "${CONFIG_NAME}"

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-causal-video-libero"}
export WANDB_MODE=${WANDB_MODE:-"online"}

if [ "${NGPU}" -gt 1 ]; then
  uv run python -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --local-ranks-filter="${LOG_RANK}" \
    --master_port "${MASTER_PORT}" \
    --tee 3 \
    -m open_wam.training.train \
    --config-name "${CONFIG_NAME}" \
    --devices "${NGPU}" \
    "${OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_ARGS[@]}" \
    "$@"
else
  uv run python -m open_wam.training.train \
    --config-name "${CONFIG_NAME}" \
    --devices "${NGPU}" \
    "${OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_ARGS[@]}" \
    "$@"
fi
