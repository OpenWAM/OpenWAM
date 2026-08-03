#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_fixed128_rollout_context_defaults.sh"

CONFIG_NAME=${CONFIG_NAME:-"parallel_stream_libero_lingbot_exact"}

export WANDB_PROJECT=${WANDB_PROJECT:-"lingbot-va-posttrain-libero"}

open_wam_launch_training "${CONFIG_NAME}" "$@"
