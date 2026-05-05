#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

echo "ERROR: Traditional Method 2 register_attached is obsolete and intentionally disabled." >&2
echo "Use the maintained Method 2 parallel_stream lingbot_exact_action_conditioned / joint-denoise path instead." >&2
exit 1

# Historical launcher body retained for reference only.
umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
LOG_RANK=${LOG_RANK:-"0"}
CONFIG_NAME=${CONFIG_NAME:-"register_attached_libero_latent_local"}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-method2-libero"}
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
