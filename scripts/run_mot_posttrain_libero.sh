#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[deprecated] use scripts/run_dual_expert_posttrain_libero.sh" >&2
export CONFIG_NAME=${CONFIG_NAME:-"dual_expert_libero_joint"}
export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-method5-libero"}
exec bash "${SCRIPT_DIR}/run_dual_expert_posttrain_libero.sh" "$@"
