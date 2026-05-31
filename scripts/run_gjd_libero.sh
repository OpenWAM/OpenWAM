#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/libero_fixed128_rollout_context_defaults.sh"

GJD_STAGE=${GJD_STAGE:-train}
GJD_METHOD=${GJD_METHOD:-m5}
GJD_ABLATION=${GJD_ABLATION:-${M5_GJD_ABLATION:-vanilla}}
PASSTHROUGH_ARGS=()

if [[ $# -gt 0 ]]; then
  case "$1" in
    train|rollout)
      GJD_STAGE="$1"
      shift
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      GJD_STAGE="${2:?--stage requires train or rollout}"
      shift 2
      ;;
    --stage=*)
      GJD_STAGE="${1#--stage=}"
      shift
      ;;
    --method)
      GJD_METHOD="${2:?--method requires m1 or m5}"
      shift 2
      ;;
    --method=*)
      GJD_METHOD="${1#--method=}"
      shift
      ;;
    --ablation)
      GJD_ABLATION="${2:?--ablation requires vanilla, pure_joint, or mode_token}"
      shift 2
      ;;
    --ablation=*)
      GJD_ABLATION="${1#--ablation=}"
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage:
  bash scripts/run_gjd_libero.sh [train|rollout] [--method m1|m5] [--ablation vanilla|pure_joint|mode_token] [args...]

Defaults:
  stage=train, method=m5, ablation=vanilla

Examples:
  bash scripts/run_gjd_libero.sh train --method m1 --ablation pure_joint
  bash scripts/run_gjd_libero.sh train --method m5 --ablation mode_token
  bash scripts/run_gjd_libero.sh rollout --method m5 --ablation pure_joint --checkpoint /path/to/checkpoint_step_N

Contracts:
  - M1 and M5 GJD use the current full-segment W64 training setup.
  - Fixed-128 GJD training is deprecated; use this launcher for GJD comparisons.
  - M5 GJD uses the legacy-prefix per-chunk proprio contract. M1 GJD keeps
    full clean modality slots for conditional modes while sharing the W64 sampler.
  - M5 GJD rollout uses the standard MoT visualization path with the same
    compatibility opt-in as maintained M5 joint rollout: LingBot streaming VAE,
    MoT inference window 30, one startup model frame, five env init steps, max
    timestep 800, and max chunks 50 unless explicitly overridden.
  - Defaults to the maintained video-only LIBERO step-3500 transformer in the config.
  - Auto-creates a method+ablation-specific save root unless --save-root/--run-name is supplied.
  - Auto-creates a method+ablation-specific rollout suffix unless --suffix is supplied.
  - `pure_joint` stays on the GJD code path; it is not the ordinary M1/M5 joint config.
EOF
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH_ARGS+=("$@")
      break
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

normalize_choice() {
  local value="$1"
  value="${value,,}"
  value="${value//-/_}"
  printf '%s\n' "${value}"
}

GJD_STAGE="$(normalize_choice "${GJD_STAGE}")"
GJD_METHOD="$(normalize_choice "${GJD_METHOD}")"
GJD_ABLATION="$(normalize_choice "${GJD_ABLATION}")"

case "${GJD_STAGE}" in
  train|rollout) ;;
  *)
    echo "Unknown GJD stage '${GJD_STAGE}'. Expected train or rollout." >&2
    exit 2
    ;;
esac

case "${GJD_METHOD}" in
  m1|method1|method_1) GJD_METHOD="m1" ;;
  m5|method5|method_5) GJD_METHOD="m5" ;;
  *)
    echo "Unknown GJD method '${GJD_METHOD}'. Expected m1 or m5." >&2
    exit 2
    ;;
esac

case "${GJD_ABLATION}" in
  vanilla|pure_joint|mode_token) ;;
  *)
    echo "Unknown GJD ablation '${GJD_ABLATION}'. Expected vanilla, pure_joint, or mode_token." >&2
    exit 2
    ;;
esac

if [[ "${GJD_METHOD}" == "m1" ]]; then
  GJD_CONFIG_NAME="parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible"
  GJD_TRAIN_LAUNCHER="${SCRIPT_DIR}/run_parallel_stream_posttrain_libero.sh"
  GJD_PROB_PREFIX="policy_variant.joint_denoise_training_mode_probs"
else
  GJD_CONFIG_NAME="mot_libero_latent_local_generalist_joint_denoising_heng_compatible"
  GJD_TRAIN_LAUNCHER="${SCRIPT_DIR}/run_mot_nonjoint_posttrain_libero.sh"
  GJD_PROB_PREFIX="policy_variant.mot_generalist_training_mode_probs"
fi
GJD_VANILLA_PROB_MAP='{"joint": 0.6, "action_conditioned_video": 0.2, "video_conditioned_action": 0.2}'
GJD_PURE_JOINT_PROB_MAP='{"joint": 1.0, "action_conditioned_video": 0.0, "video_conditioned_action": 0.0}'
GJD_CFG_PATH="configs/experiments/${GJD_CONFIG_NAME}.yaml"
GJD_M5_CURRENT_FRONTEND_ENCODE_MODE="lingbot_streaming_vae"

gjd_has_explicit_train_output_identity() {
  local previous_was_set=0
  local arg
  for arg in "$@"; do
    if [[ "${previous_was_set}" == "1" ]]; then
      case "${arg}" in
        trainer.run_name=*) return 0 ;;
      esac
      previous_was_set=0
      continue
    fi
    case "${arg}" in
      --save-root|--save-root=*|--run-name|--run-name=*) return 0 ;;
      --set) previous_was_set=1 ;;
      --set=trainer.run_name=*) return 0 ;;
      trainer.run_name=*) return 0 ;;
    esac
  done
  return 1
}

gjd_has_explicit_parent_or_checkpoint_root() {
  local previous_was_set=0
  local arg
  for arg in "$@"; do
    if [[ "${previous_was_set}" == "1" ]]; then
      case "${arg}" in
        trainer.default_root_dir=*|trainer.checkpoint_dir=*) return 0 ;;
      esac
      previous_was_set=0
      continue
    fi
    case "${arg}" in
      --checkpoint-dir|--checkpoint-dir=*) return 0 ;;
      --set) previous_was_set=1 ;;
      --set=trainer.default_root_dir=*|--set=trainer.checkpoint_dir=*) return 0 ;;
      trainer.default_root_dir=*|trainer.checkpoint_dir=*) return 0 ;;
    esac
  done
  return 1
}

build_default_train_identity_args() {
  local -n target_args="$1"
  if gjd_has_explicit_train_output_identity "${PASSTHROUGH_ARGS[@]}"; then
    return 0
  fi
  local run_id
  local save_root
  run_id="${GJD_RUN_ID:-gjd_libero_${GJD_METHOD}_${GJD_ABLATION}_$(date +%Y%m%d_%H%M%S)_$$}"
  if [[ -n "${GJD_SAVE_ROOT:-}" ]]; then
    target_args+=(--save-root "${GJD_SAVE_ROOT}")
  elif gjd_has_explicit_parent_or_checkpoint_root "${PASSTHROUGH_ARGS[@]}"; then
    target_args+=(--run-name "${run_id}")
  else
    save_root="${GJD_RUNS_ROOT:-runs}/${run_id}"
    target_args+=(--save-root "${save_root}")
  fi
}

gjd_has_explicit_train_wandb_project() {
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    return 0
  fi
  local previous_was_set=0
  local arg
  for arg in "$@"; do
    if [[ "${previous_was_set}" == "1" ]]; then
      case "${arg}" in
        trainer.wandb_project=*) return 0 ;;
      esac
      previous_was_set=0
      continue
    fi
    case "${arg}" in
      --wandb-project|--wandb-project=*) return 0 ;;
      --set) previous_was_set=1 ;;
      --set=trainer.wandb_project=*|trainer.wandb_project=*) return 0 ;;
    esac
  done
  return 1
}

build_default_train_tracking_args() {
  local -n target_args="$1"
  if gjd_has_explicit_train_wandb_project "${PASSTHROUGH_ARGS[@]}"; then
    return 0
  fi
  target_args+=(--wandb-project "${GJD_WANDB_PROJECT:-openwam-gjd-libero}")
}

gjd_has_explicit_rollout_suffix() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --suffix|--suffix=*) return 0 ;;
    esac
  done
  return 1
}

build_default_rollout_identity_args() {
  local -n target_args="$1"
  if gjd_has_explicit_rollout_suffix "${PASSTHROUGH_ARGS[@]}"; then
    return 0
  fi
  target_args+=(--suffix "${GJD_ROLLOUT_SUFFIX:-gjd_libero_${GJD_METHOD}_${GJD_ABLATION}}")
}

gjd_has_cli_arg() {
  local flag="$1"
  shift
  local arg
  for arg in "$@"; do
    case "${arg}" in
      "${flag}"|"${flag}"=*) return 0 ;;
    esac
  done
  return 1
}

gjd_cli_arg_value() {
  local flag="$1"
  shift
  local previous_was_flag=0
  local arg
  for arg in "$@"; do
    if [[ "${previous_was_flag}" == "1" ]]; then
      printf '%s\n' "${arg}"
      return 0
    fi
    case "${arg}" in
      "${flag}") previous_was_flag=1 ;;
      "${flag}"=*)
        printf '%s\n' "${arg#*=}"
        return 0
        ;;
    esac
  done
  return 1
}

gjd_reject_deprecated_m5_frontend_encode_mode() {
  local requested="${GJD_M5_FRONTEND_ENCODE_MODE:-}"
  local explicit_requested
  if explicit_requested="$(gjd_cli_arg_value --frontend-encode-mode "${PASSTHROUGH_ARGS[@]}")"; then
    requested="${explicit_requested}"
  fi
  if [[ -n "${requested}" && "${requested}" != "${GJD_M5_CURRENT_FRONTEND_ENCODE_MODE}" ]]; then
    echo "GJD M5 rollout requires --frontend-encode-mode ${GJD_M5_CURRENT_FRONTEND_ENCODE_MODE}; '${requested}' is deprecated." >&2
    exit 2
  fi
}

gjd_append_default_arg() {
  local -n default_args_ref="$1"
  local flag="$2"
  local value="$3"
  if ! gjd_has_cli_arg "${flag}" "${PASSTHROUGH_ARGS[@]}"; then
    default_args_ref+=("${flag}" "${value}")
  fi
}

gjd_append_default_flag() {
  local -n default_args_ref="$1"
  local flag="$2"
  if ! gjd_has_cli_arg "${flag}" "${PASSTHROUGH_ARGS[@]}"; then
    default_args_ref+=("${flag}")
  fi
}

build_default_m5_rollout_semantic_args() {
  local -n rollout_semantic_args_ref="$1"
  gjd_reject_deprecated_m5_frontend_encode_mode
  gjd_append_default_arg rollout_semantic_args_ref --frontend-encode-mode "${GJD_M5_CURRENT_FRONTEND_ENCODE_MODE}"
  gjd_append_default_arg rollout_semantic_args_ref --mot-inference-window-size "${GJD_M5_MOT_INFERENCE_WINDOW_SIZE:-30}"
  gjd_append_default_arg rollout_semantic_args_ref --startup-model-obs-frames "${GJD_M5_STARTUP_MODEL_OBS_FRAMES:-1}"
  gjd_append_default_arg rollout_semantic_args_ref --startup-env-init-steps "${GJD_M5_STARTUP_ENV_INIT_STEPS:-5}"
  gjd_append_default_arg rollout_semantic_args_ref --max-timestep "${GJD_M5_MAX_TIMESTEP:-800}"
  gjd_append_default_arg rollout_semantic_args_ref --max-chunks "${GJD_M5_MAX_CHUNKS:-50}"
  gjd_append_default_flag rollout_semantic_args_ref --allow-deprecated-libero-config
}

build_ablation_args() {
  local -n target_args="$1"
  case "${GJD_ABLATION}" in
    vanilla)
      target_args+=(
        --set "${GJD_PROB_PREFIX}=${GJD_VANILLA_PROB_MAP}"
        --set policy_variant.generalist_mode_text_token=false
      )
      ;;
    pure_joint)
      target_args+=(
        --set "${GJD_PROB_PREFIX}=${GJD_PURE_JOINT_PROB_MAP}"
        --set policy_variant.generalist_mode_text_token=false
      )
      ;;
    mode_token)
      target_args+=(
        --set "${GJD_PROB_PREFIX}=${GJD_VANILLA_PROB_MAP}"
        --set policy_variant.generalist_mode_text_token=true
      )
      ;;
  esac
}

GJD_ABLATION_ARGS=()
build_ablation_args GJD_ABLATION_ARGS

if [[ "${GJD_STAGE}" == "train" ]]; then
  open_wam_reject_cli_config_override_args "${PASSTHROUGH_ARGS[@]}"
  if [[ -n "${CONFIG_NAME:-}" ]]; then
    CONFIG_NAME_NORMALIZED="$(open_wam_normalize_config_name "${CONFIG_NAME}")"
    if [[ "${CONFIG_NAME_NORMALIZED}" != "${GJD_CONFIG_NAME}" ]]; then
      echo "GJD ${GJD_METHOD} train requires CONFIG_NAME=${GJD_CONFIG_NAME}; got ${CONFIG_NAME}." >&2
      exit 2
    fi
  fi
  export CONFIG_NAME="${GJD_CONFIG_NAME}"
  GJD_DEFAULT_TRAIN_IDENTITY_ARGS=()
  build_default_train_identity_args GJD_DEFAULT_TRAIN_IDENTITY_ARGS
  GJD_DEFAULT_TRAIN_TRACKING_ARGS=()
  build_default_train_tracking_args GJD_DEFAULT_TRAIN_TRACKING_ARGS
  echo "[run_gjd_libero] stage=train method=${GJD_METHOD} ablation=${GJD_ABLATION} config=${GJD_CONFIG_NAME}" >&2
  exec bash "${GJD_TRAIN_LAUNCHER}" \
    "${GJD_DEFAULT_TRAIN_IDENTITY_ARGS[@]}" \
    "${GJD_DEFAULT_TRAIN_TRACKING_ARGS[@]}" \
    "${PASSTHROUGH_ARGS[@]}" \
    "${GJD_ABLATION_ARGS[@]}"
fi

open_wam_reject_cli_config_override_args "${PASSTHROUGH_ARGS[@]}"
if [[ -n "${CFG:-}" ]]; then
  CFG_NORMALIZED="$(open_wam_normalize_config_name "${CFG}")"
  if [[ "${CFG_NORMALIZED}" != "${GJD_CONFIG_NAME}" ]]; then
    echo "GJD ${GJD_METHOD} rollout requires CFG=${GJD_CFG_PATH}; got ${CFG}." >&2
    exit 2
  fi
  GJD_CFG_PATH="${CFG}"
fi
GJD_ROLLOUT_CONTEXT_ARGS=()
open_wam_append_fixed128_rollout_context_args GJD_ROLLOUT_CONTEXT_ARGS "${GJD_CONFIG_NAME}"
GJD_DEFAULT_ROLLOUT_IDENTITY_ARGS=()
build_default_rollout_identity_args GJD_DEFAULT_ROLLOUT_IDENTITY_ARGS
if [[ "${GJD_METHOD}" == "m5" ]]; then
  GJD_M5_ROLLOUT_SEMANTIC_ARGS=()
  build_default_m5_rollout_semantic_args GJD_M5_ROLLOUT_SEMANTIC_ARGS
  GJD_REALTIME_ARGS=(
    "${REPO_ROOT}/scripts/run_libero_mot_visualization.py"
    --cfg "${GJD_CFG_PATH}"
    "${GJD_ROLLOUT_CONTEXT_ARGS[@]}"
    "${GJD_DEFAULT_ROLLOUT_IDENTITY_ARGS[@]}"
    "${GJD_M5_ROLLOUT_SEMANTIC_ARGS[@]}"
    "${PASSTHROUGH_ARGS[@]}"
    "${GJD_ABLATION_ARGS[@]}"
  )
else
  GJD_REALTIME_ARGS=(
    "${REPO_ROOT}/scripts/run_libero_realtime_sandbox.py"
    --cfg "${GJD_CFG_PATH}"
    "${GJD_ROLLOUT_CONTEXT_ARGS[@]}"
    "${GJD_DEFAULT_ROLLOUT_IDENTITY_ARGS[@]}"
    "${PASSTHROUGH_ARGS[@]}"
    "${GJD_ABLATION_ARGS[@]}"
  )
fi

echo "[run_gjd_libero] stage=rollout method=${GJD_METHOD} ablation=${GJD_ABLATION} cfg=${GJD_CFG_PATH}" >&2
if [[ "${OPEN_WAM_PRINT_REALTIME_ARGV:-0}" == "1" ]]; then
  open_wam_print_train_argv_json "${GJD_REALTIME_ARGS[@]}"
  exit 0
fi

exec uv run python "${GJD_REALTIME_ARGS[@]}"
