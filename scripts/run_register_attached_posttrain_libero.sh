#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

echo "ERROR: Traditional Method 2 register_attached is obsolete and intentionally disabled." >&2
echo "Use the maintained Method 2 parallel_stream lingbot_exact_action_conditioned / joint-denoise path instead." >&2
exit 1
