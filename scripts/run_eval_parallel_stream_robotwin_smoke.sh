#!/usr/bin/env bash
set -euo pipefail

uv run python -m open_wam.evals.evaluate --cfg configs/evals/parallel_stream_robotwin_smoke.yaml "$@"
