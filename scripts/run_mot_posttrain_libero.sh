#!/usr/bin/env bash
set -euo pipefail
set -x

umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
LOG_RANK=${LOG_RANK:-"0"}
CONFIG_NAME=${CONFIG_NAME:-"mot_libero_latent_local_joint"}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-method5-libero"}
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
