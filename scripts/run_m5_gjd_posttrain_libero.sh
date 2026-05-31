#!/usr/bin/env bash
set -euo pipefail

if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  M5_GJD_VARIANT=vanilla    bash scripts/run_m5_gjd_posttrain_libero.sh [train args...]
  M5_GJD_VARIANT=pure_joint bash scripts/run_m5_gjd_posttrain_libero.sh [train args...]
  M5_GJD_VARIANT=mode_token bash scripts/run_m5_gjd_posttrain_libero.sh [train args...]

You may also pass the variant as the first positional argument:
  bash scripts/run_m5_gjd_posttrain_libero.sh mode_token --save-root ...

Supported variants:
  vanilla     Standard M5 GJD 0.6/0.2/0.2 joint/FDM/IDM mixture.
  pure_joint  GJD path with joint=1.0 and conditional modes disabled.
  mode_token  Standard M5 GJD mixture plus one learned text-space mode token.

The GJD config defaults to the current full-segment W64 sampler. Fixed-128 GJD
training is deprecated. All remaining args are forwarded through
scripts/run_gjd_libero.sh.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

variant="${M5_GJD_VARIANT:-${GJD_VARIANT:-vanilla}}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  variant="$1"
  shift
fi
variant="${variant//-/_}"

case "${variant}" in
  vanilla|standard)
    variant="vanilla"
    ;;
  pure_joint|joint_only|joint_1_0|joint_1p0)
    variant="pure_joint"
    ;;
  mode_token|mode_text_token|token)
    variant="mode_token"
    ;;
  *)
    echo "Unknown M5 GJD variant '${variant}'. Run with --help for choices." >&2
    exit 2
    ;;
esac

exec bash "${SCRIPT_DIR}/run_gjd_libero.sh" train --method m5 --ablation "${variant}" "$@"
