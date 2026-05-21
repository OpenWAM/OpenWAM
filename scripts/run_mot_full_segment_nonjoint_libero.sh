#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_fixed128_rollout_context_defaults.sh"
open_wam_reject_deprecated_libero_launcher "scripts/run_mot_full_segment_nonjoint_libero.sh" \
  "scripts/run_mot_nonjoint_posttrain_libero.sh"

exec bash "${SCRIPT_DIR}/run_mot_nonjoint_posttrain_libero.sh" "$@"
