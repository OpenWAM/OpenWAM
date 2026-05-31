#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. New M1/M5 GJD work should use
# `scripts/run_gjd_libero.sh train --method ... --ablation ...` directly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION="${M5_GJD_ABLATION:-${GJD_ABLATION:-vanilla}}"

exec bash "${SCRIPT_DIR}/run_gjd_libero.sh" train --method m5 --ablation "${ABLATION}" "$@"
