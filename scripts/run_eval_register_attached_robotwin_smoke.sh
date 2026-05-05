#!/usr/bin/env bash
set -euo pipefail

echo "ERROR: Traditional Method 2 register_attached is obsolete and intentionally disabled." >&2
echo "Use the maintained Method 2 parallel_stream lingbot_exact_action_conditioned / joint-denoise path instead." >&2
exit 1

# Historical command retained for reference only.
uv run python -m open_wam.evals.evaluate --cfg configs/evals/register_attached_robotwin_smoke.yaml "$@"
