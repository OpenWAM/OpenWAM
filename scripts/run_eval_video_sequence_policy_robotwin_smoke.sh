#!/usr/bin/env bash
set -euo pipefail

uv run python -m open_wam.evals.evaluate --cfg configs/evals/video_sequence_policy_robotwin_smoke.yaml "$@"
