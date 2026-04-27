#!/usr/bin/env bash
# Scheme A: method-1-non-joint-aligned MoT. Both video and action trained.
#
# Override any of these by exporting the env var before running, e.g.:
#   NGPU=2 bash scripts/run_mot_non_joint_aligned_libero_A.sh
set -euo pipefail

NGPU=${NGPU:-4}
MASTER_PORT=${MASTER_PORT:-29511}
SAVE_ROOT=${SAVE_ROOT:-runs/method5_mot_full_segment_non_joint_aligned_libero}

if [ -z "${TRANSFORMER_SUBDIR:-}" ]; then
  echo "TRANSFORMER_SUBDIR must point to an exported video-only runtime transformer." >&2
  exit 2
fi

NGPU="${NGPU}" MASTER_PORT="${MASTER_PORT}" bash scripts/run_mot_full_segment_nonjoint_libero.sh --save-root "${SAVE_ROOT}" --transformer-subdir "${TRANSFORMER_SUBDIR}" "$@"
