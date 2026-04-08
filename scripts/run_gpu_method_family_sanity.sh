#!/usr/bin/env bash
set -euo pipefail

# Dedicated 1-GPU sanity suite for methods 1 through 5.
# The tests themselves are CUDA-gated and will exit early unless this opt-in
# env var is set, so keep the runner explicit.
export OPEN_WAM_RUN_GPU_SANITY="${OPEN_WAM_RUN_GPU_SANITY:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv run pytest -q tests/test_gpu_method_family_sanity.py "$@"
