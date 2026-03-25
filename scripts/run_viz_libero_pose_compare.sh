#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv run mjpython scripts/visualize_libero_pose_compare.py \
  --cfg configs/experiments/contract_only_libero.yaml \
  "$@"
