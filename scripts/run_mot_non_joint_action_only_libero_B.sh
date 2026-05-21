#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_fixed128_rollout_context_defaults.sh"
open_wam_reject_deprecated_libero_launcher "scripts/run_mot_non_joint_action_only_libero_B.sh"

exec bash "${SCRIPT_DIR}/deprecated/run_mot_non_joint_action_only_libero_B.sh" "$@"
