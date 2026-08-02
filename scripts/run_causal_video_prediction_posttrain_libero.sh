#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_fixed128_rollout_context_defaults.sh"

CONFIG_NAME=${CONFIG_NAME:-"causal_video_prediction_libero_latent_local"}

export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-causal-video-libero"}

open_wam_launch_training "${CONFIG_NAME}" "$@"
