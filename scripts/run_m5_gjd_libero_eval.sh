#!/usr/bin/env bash
set -euo pipefail

if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_m5_gjd_libero_eval.sh

This entrypoint is deprecated and fails closed. It used the generic sampled
eval path, which does not preserve the current dual-expert GJD rollout contract.

Use the unified GJD rollout path instead:

  bash scripts/run_gjd_libero.sh rollout --architecture dual_expert --ablation vanilla \
    --checkpoint /path/to/checkpoint_step_N ...

Valid ablations are vanilla, pure_joint, and mode_token.

For multi-episode dual-expert GJD eval, use a loop or a future GJD-aware batch wrapper
that delegates to scripts/run_gjd_libero.sh rollout. Do not use
scripts/run_libero_sampled_eval.py for GJD.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cat >&2 <<'EOF'
scripts/run_m5_gjd_libero_eval.sh is deprecated and now fails closed.
It bypassed the unified GJD rollout contract by calling generic sampled eval.
Use scripts/run_gjd_libero.sh rollout --architecture dual_expert --ablation <vanilla|pure_joint|mode_token> instead.
EOF
exit 2
