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
# Defaults to the method-1-non-joint-aligned two-stream MoT config. Override
# with `CONFIG_NAME=mot_libero_latent_local_full_segment_non_joint_action_only`
# for the action-only retrain path (frozen video backbone).
CONFIG_NAME=${CONFIG_NAME:-"mot_libero_latent_local_full_segment_non_joint_aligned"}
open_wam_reject_cli_config_override_args "$@"
OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_ARGS=()
open_wam_append_fixed128_rollout_context_args OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_ARGS "${CONFIG_NAME}"

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-libero-policy-train"}
export WANDB_MODE=${WANDB_MODE:-"online"}
# M5 packed-coupling configs jointly train video DiT (~5B) + action expert
# (~2B). On 4×L40S that 7.66B trainable footprint exceeds GPU memory; FSDP2
# CPU offload is needed to fit. Override with OPEN_WAM_FSDP_CPU_OFFLOAD=0
# when running on hardware with enough VRAM to skip the offload.
export OPEN_WAM_FSDP_CPU_OFFLOAD=${OPEN_WAM_FSDP_CPU_OFFLOAD:-1}

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
