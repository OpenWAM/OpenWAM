#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_fixed128_rollout_context_defaults.sh"

# Default to the maintained strict fixed-128 M5 video-then-action config.
# Historical config names are permanently rejected; use a maintained config.
CONFIG_NAME=${CONFIG_NAME:-"mot_libero_video_then_action"}

export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-libero-policy-train"}
# M5 packed-coupling configs jointly train video DiT (~5B) + action expert
# (~2B). On 4×L40S that 7.66B trainable footprint exceeds GPU memory; FSDP2
# CPU offload is needed to fit. Override with OPEN_WAM_FSDP_CPU_OFFLOAD=0
# when running on hardware with enough VRAM to skip the offload.
export OPEN_WAM_FSDP_CPU_OFFLOAD=${OPEN_WAM_FSDP_CPU_OFFLOAD:-1}

open_wam_launch_training "${CONFIG_NAME}" "$@"
