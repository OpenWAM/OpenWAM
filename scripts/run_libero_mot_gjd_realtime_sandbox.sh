#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. New GJD rollout work should use
# `scripts/run_gjd_libero.sh rollout --architecture ... --ablation ...` directly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION="${M5_GJD_ABLATION:-${GJD_ABLATION:-vanilla}}"

exec bash "${SCRIPT_DIR}/run_gjd_libero.sh" rollout --architecture dual_expert --ablation "${ABLATION}" "$@"
