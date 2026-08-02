#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_fixed128_rollout_context_defaults.sh"

# Default to the maintained strict fixed-128 M5 joint config.
CONFIG_NAME=${CONFIG_NAME:-"mot_libero_latent_local_joint_heng_compatible"}

export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-method5-libero"}

open_wam_launch_training "${CONFIG_NAME}" "$@"
