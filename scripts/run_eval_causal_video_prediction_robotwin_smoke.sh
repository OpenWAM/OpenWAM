#!/usr/bin/env bash
set -euo pipefail

uv run python -m open_wam.evals.evaluate --cfg configs/evals/causal_video_prediction_robotwin_smoke.yaml "$@"
