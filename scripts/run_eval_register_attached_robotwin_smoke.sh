#!/usr/bin/env bash
set -euo pipefail

uv run python -m open_wam.evals.evaluate --cfg configs/evals/register_attached_robotwin_smoke.yaml "$@"
