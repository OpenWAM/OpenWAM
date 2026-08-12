#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_legacy_compatibility.sh"
open_wam_reject_removed_libero_launcher "scripts/run_mot_non_joint_aligned_libero_A.sh" \
  "scripts/run_dual_expert_posttrain_libero.sh with a maintained CONFIG_NAME"
