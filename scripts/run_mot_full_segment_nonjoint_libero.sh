#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
LOG_RANK=${LOG_RANK:-"0"}
# Defaults to the method-1-non-joint-aligned two-stream MoT config. Override
# with `CONFIG_NAME=mot_libero_latent_local_full_segment` for the action-only
# debug retrain (frozen video backbone).
CONFIG_NAME=${CONFIG_NAME:-"mot_libero_latent_local_full_segment_non_joint_aligned"}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-libero-policy-train"}
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
    "$@"
else
  uv run python -m open_wam.training.train \
    --config-name "${CONFIG_NAME}" \
    --devices "${NGPU}" \
    "$@"
fi
