#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/evaluate_libero_env_tracking.py "$@"
