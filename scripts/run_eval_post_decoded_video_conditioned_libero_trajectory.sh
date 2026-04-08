#!/usr/bin/env bash
set -euo pipefail

uv run python -m open_wam.evals.evaluate --cfg configs/evals/post_decoded_libero_latent_local_video_conditioned_trajectory.yaml "$@"
