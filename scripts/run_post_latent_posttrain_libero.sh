#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/training_launcher_common.sh"

# Maintained method-4 launcher defaults to the video-conditioned config.
CONFIG_NAME=${CONFIG_NAME:-"post_latent_libero_latent_local_video_conditioned"}

export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-method4-post-latent-libero-video-conditioned"}

open_wam_launch_training "${CONFIG_NAME}" "$@"
