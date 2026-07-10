#!/usr/bin/env bash
set -euo pipefail

# Candidate-set oracle coverage grid for the Task-8 subtask benchmark.
# This intentionally uses dummy dynamics + constant scoring: Open-WAM/Gemini is
# evaluated separately after the candidate set has recoverable alternatives.

MANIFEST=${MANIFEST:-outputs/fdm_guided_planning_bootstrap/pi0fast_task8_subtask_manifest_8states_20260622/captured_states.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/fdm_guided_planning_bootstrap}
RUN_SUFFIX=${RUN_SUFFIX:-metricfix_20260622}
POLICY=${POLICY:-pi0fast}
PI0FAST_MODEL=${PI0FAST_MODEL:-lerobot/pi0fast-libero}
POLICY_DEVICE=${POLICY_DEVICE:-cuda:0}
CUDA_DEVICE=${CUDA_DEVICE:-0}

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing capture manifest: ${MANIFEST}" >&2
  exit 1
fi

specs=(
  "n8_t06:8:0.6"
  "n16_t06:16:0.6"
  "n32_t06:32:0.6"
  "n8_t04:8:0.4"
  "n8_t08:8:0.8"
  "n8_t10:8:1.0"
)

for spec in "${specs[@]}"; do
  IFS=: read -r label sample_count temperature <<< "${spec}"
  run_id="pi0fast_task8_stall_diversity_${label}_${RUN_SUFFIX}"
  echo "RUN ${run_id}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    uv run python scripts/run_libero_task8_offline_reranking.py \
      --policy "${POLICY}" \
      --pi0fast-model "${PI0FAST_MODEL}" \
      --policy-device "${POLICY_DEVICE}" \
      --replay-capture-manifest "${MANIFEST}" \
      --capture-mode no_progress_stall \
      --episode-indices 0,1,2,3 \
      --stages pick_up_pot_1 \
      --max-states 8 \
      --max-env-steps 520 \
      --num-policy-samples "${sample_count}" \
      --no-include-baseline-candidate \
      --planner-dynamics dummy \
      --evaluator constant \
      --baseline-action-selection-mode select_action \
      --baseline-policy-temperature 0.0 \
      --planner-temperature "${temperature}" \
      --chunk-action-steps 10 \
      --execute-action-steps 10 \
      --run-id "${run_id}" \
      --output-dir "${OUTPUT_DIR}" \
      2>&1 | tee "${OUTPUT_DIR}/${run_id}.log"
done
