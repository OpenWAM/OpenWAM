#!/usr/bin/env bash
# Scheme B: non-joint two-stream MoT, action-only trainable (video frozen).
# Action expert sees the noisy-video K/V distribution it will see at
# inference, but the video backbone is not updated.
#
# Override any of these by exporting the env var before running, e.g.:
#   NGPU=2 bash scripts/run_mot_non_joint_action_only_libero_B.sh
set -euo pipefail

NGPU=${NGPU:-4}
MASTER_PORT=${MASTER_PORT:-29511}
SAVE_ROOT=${SAVE_ROOT:-runs/method5_mot_full_segment_non_joint_action_only_libero}

if [ -z "${TRANSFORMER_SUBDIR:-}" ]; then
  echo "TRANSFORMER_SUBDIR must point to an exported video-only runtime transformer." >&2
  exit 2
fi

NGPU="${NGPU}" MASTER_PORT="${MASTER_PORT}" CONFIG_NAME=mot_libero_latent_local_full_segment_non_joint_action_only \
  bash scripts/run_mot_full_segment_nonjoint_libero.sh \
    --save-root "${SAVE_ROOT}" \
    --transformer-subdir "${TRANSFORMER_SUBDIR}" \
    "$@"
