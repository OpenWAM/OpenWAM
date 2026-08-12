#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/training_launcher_common.sh"
source "${SCRIPT_DIR}/libero_legacy_compatibility.sh"

# Architecture launchers select a policy program by default. The lower-level
# lingbot_exact profile remains available explicitly for checkpoint archaeology.
CONFIG_NAME=${CONFIG_NAME:-"parallel_stream_libero_joint"}

export WANDB_PROJECT=${WANDB_PROJECT:-"lingbot-va-posttrain-libero"}

open_wam_reject_removed_libero_policy_config "${CONFIG_NAME}"
open_wam_launch_training "${CONFIG_NAME}" "$@"
