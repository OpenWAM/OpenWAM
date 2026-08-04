#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/libero_fixed128_rollout_context_defaults.sh"

GJD_STAGE=${GJD_STAGE:-train}
GJD_ARCHITECTURE=${GJD_ARCHITECTURE:-${GJD_METHOD:-dual_expert}}
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
    --architecture)
      GJD_ARCHITECTURE="${2:?--architecture requires parallel_stream or dual_expert}"
      shift 2
      ;;
    --architecture=*)
      GJD_ARCHITECTURE="${1#--architecture=}"
      shift
      ;;
    --method)
      GJD_ARCHITECTURE="${2:?--method requires m1 or m5}"
      shift 2
      ;;
    --method=*)
      GJD_ARCHITECTURE="${1#--method=}"
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
  bash scripts/run_gjd_libero.sh [train|rollout] [--architecture parallel_stream|dual_expert] [--ablation vanilla|pure_joint|mode_token] [args...]

Defaults:
  stage=train, architecture=dual_expert, ablation=vanilla

Examples:
  bash scripts/run_gjd_libero.sh train --architecture parallel_stream --ablation pure_joint
  bash scripts/run_gjd_libero.sh train --architecture dual_expert --ablation mode_token
  bash scripts/run_gjd_libero.sh rollout --architecture dual_expert --ablation pure_joint --checkpoint /path/to/checkpoint_step_N

Compatibility:
  --method m1|m5 and GJD_METHOD remain accepted aliases.

Contracts:
  - dual_expert is the maintained standard GJD architecture. Treat
    parallel_stream GJD as a compatibility/diagnostic path until its known
    context contract issue is resolved.
  - Both architectures use the current full-segment W64 training setup.
  - Vanilla and mode-token training default to the mixed real/counterfactual
    dynamics source mixer; configure the counterfactual latent roots through
    configs/local_paths.yaml or explicit --set overrides.
  - pure_joint disables the mixed source mixer and stays demo-only.
  - Fixed-128 GJD training is deprecated; use this launcher for GJD comparisons.
  - dual_expert GJD uses the legacy-prefix per-chunk proprio contract and requires
    single-frame condition latents:
      policy_variant.sequence_contract=legacy_prefix_single_frame_perchunk_proprio
      policy_variant.context_condition_latent_source=single_frame_condition_latent
      policy_variant.use_condition_latents=true
      policy_variant.require_condition_latents=true
      data.sample_construction.condition_source_frame_offset=-1
      data.sample_construction.start_padding_frames=0
      data.sample_construction.target_alignment=legacy
  - Known parallel_stream GJD issue: parallel_stream does not use the dual_expert
    legacy-prefix/single-frame
    condition-latent default. It keeps full clean modality slots so conditional
    FDM/IDM modes are representable. Forcing the dual_expert single-frame condition
    source on current parallel_stream legacy full-segment samples would require a pre-target
    context/loss frame; the current legacy uniform segment path has
    loss_frame_start=0, and parallel_stream's legacy-prefix path supports only
    pure joint. Current parallel_stream differences from dual_expert:
      policy_variant.sequence_contract=default
      policy_variant.context_condition_latent_source=video_latents
      policy_variant.require_condition_latents=false
      data.sample_construction.condition_source_frame_offset=0
      rollout uses run_libero_realtime_sandbox.py, not the dual-expert visualization path
  - dual_expert GJD rollout uses the standard dual-expert visualization path with
    the same compatibility opt-in as maintained dual-expert joint rollout: LingBot
    streaming VAE, inference window 30, one startup model frame, five env init steps, max
    timestep 1500, and max chunks 100 unless explicitly overridden.
  - Defaults to the maintained video-only LIBERO step-3500 transformer in the config.
  - Auto-creates an architecture+ablation-specific save root unless --save-root/--run-name is supplied.
  - Auto-creates an architecture+ablation-specific rollout suffix unless --suffix is supplied.
  - `pure_joint` stays on the GJD code path; it is not the ordinary joint program.
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
GJD_ARCHITECTURE="$(normalize_choice "${GJD_ARCHITECTURE}")"
GJD_ABLATION="$(normalize_choice "${GJD_ABLATION}")"

case "${GJD_STAGE}" in
  train|rollout) ;;
  *)
    echo "Unknown GJD stage '${GJD_STAGE}'. Expected train or rollout." >&2
    exit 2
    ;;
esac

case "${GJD_ARCHITECTURE}" in
  parallel_stream|m1|method1|method_1)
    GJD_ARCHITECTURE="parallel_stream"
    ;;
  dual_expert|m5|method5|method_5|mot)
    GJD_ARCHITECTURE="dual_expert"
    ;;
  *)
    echo "Unknown GJD architecture '${GJD_ARCHITECTURE}'. Expected parallel_stream or dual_expert." >&2
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

if [[ "${GJD_ARCHITECTURE}" == "parallel_stream" ]]; then
  GJD_CONFIG_NAME="parallel_stream_libero_generalist_joint_denoising"
  GJD_TRAIN_LAUNCHER="${SCRIPT_DIR}/run_parallel_stream_posttrain_libero.sh"
  GJD_PROB_PREFIX="policy_variant.generalist_denoising_mode_probs"
else
  GJD_CONFIG_NAME="dual_expert_libero_generalist_joint_denoising"
  GJD_TRAIN_LAUNCHER="${SCRIPT_DIR}/run_dual_expert_posttrain_libero.sh"
  GJD_PROB_PREFIX="policy_variant.generalist_denoising_mode_probs"
fi
GJD_VANILLA_PROB_MAP='{"joint": 0.6, "action_conditioned_video": 0.2, "video_conditioned_action": 0.2}'
GJD_PURE_JOINT_PROB_MAP='{"joint": 1.0, "action_conditioned_video": 0.0, "video_conditioned_action": 0.0}'
GJD_CFG_PATH="configs/experiments/${GJD_CONFIG_NAME}.yaml"
GJD_DUAL_EXPERT_CURRENT_FRONTEND_ENCODE_MODE="lingbot_streaming_vae"

print_gjd_architecture_contract_notice() {
  if [[ "${GJD_ARCHITECTURE}" != "parallel_stream" ]]; then
    return 0
  fi
  cat >&2 <<'EOF'
[run_gjd_libero] known parallel_stream GJD issue (historical M1 path):
[run_gjd_libero]   dual_expert is the maintained standard GJD architecture. It uses legacy-prefix
[run_gjd_libero]   single-frame condition latents with condition_source_frame_offset=-1,
[run_gjd_libero]   start_padding_frames=0, and target_alignment=legacy.
[run_gjd_libero]   parallel_stream GJD is a compatibility/diagnostic path. It does not
[run_gjd_libero]   default to the dual-expert single-frame condition-latent contract
[run_gjd_libero]   because its conditional FDM/IDM needs full clean modality slots, while its
[run_gjd_libero]   legacy-prefix only supports pure joint and the current legacy
[run_gjd_libero]   full-segment samples have loss_frame_start=0.
[run_gjd_libero]   Current parallel_stream differences: sequence_contract=default,
[run_gjd_libero]   context_condition_latent_source=video_latents,
[run_gjd_libero]   require_condition_latents=false, condition_source_frame_offset=0,
[run_gjd_libero]   and rollout uses run_libero_realtime_sandbox.py instead of the
[run_gjd_libero]   dual-expert visualization path.
EOF
}

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
  run_id="${GJD_RUN_ID:-gjd_libero_${GJD_ARCHITECTURE}_${GJD_ABLATION}_$(date +%Y%m%d_%H%M%S)_$$}"
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
  target_args+=(--suffix "${GJD_ROLLOUT_SUFFIX:-gjd_libero_${GJD_ARCHITECTURE}_${GJD_ABLATION}}")
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

gjd_reject_deprecated_dual_expert_frontend_encode_mode() {
  local requested="${GJD_DUAL_EXPERT_FRONTEND_ENCODE_MODE:-${GJD_M5_FRONTEND_ENCODE_MODE:-}}"
  local explicit_requested
  if explicit_requested="$(gjd_cli_arg_value --frontend-encode-mode "${PASSTHROUGH_ARGS[@]}")"; then
    requested="${explicit_requested}"
  fi
  if [[ -n "${requested}" && "${requested}" != "${GJD_DUAL_EXPERT_CURRENT_FRONTEND_ENCODE_MODE}" ]]; then
    echo "dual_expert GJD rollout requires --frontend-encode-mode ${GJD_DUAL_EXPERT_CURRENT_FRONTEND_ENCODE_MODE}; '${requested}' is deprecated." >&2
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

build_default_dual_expert_rollout_semantic_args() {
  local -n rollout_semantic_args_ref="$1"
  gjd_reject_deprecated_dual_expert_frontend_encode_mode
  gjd_append_default_arg rollout_semantic_args_ref --frontend-encode-mode "${GJD_DUAL_EXPERT_CURRENT_FRONTEND_ENCODE_MODE}"
  gjd_append_default_arg rollout_semantic_args_ref --dual-expert-inference-window-size "${GJD_DUAL_EXPERT_INFERENCE_WINDOW_SIZE:-${GJD_M5_DUAL_EXPERT_INFERENCE_WINDOW_SIZE:-30}}"
  gjd_append_default_arg rollout_semantic_args_ref --startup-model-obs-frames "${GJD_DUAL_EXPERT_STARTUP_MODEL_OBS_FRAMES:-${GJD_M5_STARTUP_MODEL_OBS_FRAMES:-1}}"
  gjd_append_default_arg rollout_semantic_args_ref --startup-env-init-steps "${GJD_DUAL_EXPERT_STARTUP_ENV_INIT_STEPS:-${GJD_M5_STARTUP_ENV_INIT_STEPS:-5}}"
  gjd_append_default_arg rollout_semantic_args_ref --max-timestep "${GJD_DUAL_EXPERT_MAX_TIMESTEP:-${GJD_M5_MAX_TIMESTEP:-1500}}"
  gjd_append_default_arg rollout_semantic_args_ref --max-chunks "${GJD_DUAL_EXPERT_MAX_CHUNKS:-${GJD_M5_MAX_CHUNKS:-100}}"
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
        --set policy_variant.generalist_training_paradigm=demo_only
        --set data.sample_construction.sample_order_mode=replacement
        --set data.generalist_dynamics_mixture.train_latent_root=null
        --set data.generalist_dynamics_mixture.val_latent_root=null
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
      echo "GJD ${GJD_ARCHITECTURE} train requires CONFIG_NAME=${GJD_CONFIG_NAME}; got ${CONFIG_NAME}." >&2
      exit 2
    fi
  fi
  export CONFIG_NAME="${GJD_CONFIG_NAME}"
  GJD_DEFAULT_TRAIN_IDENTITY_ARGS=()
  build_default_train_identity_args GJD_DEFAULT_TRAIN_IDENTITY_ARGS
  GJD_DEFAULT_TRAIN_TRACKING_ARGS=()
  build_default_train_tracking_args GJD_DEFAULT_TRAIN_TRACKING_ARGS
  echo "[run_gjd_libero] stage=train architecture=${GJD_ARCHITECTURE} ablation=${GJD_ABLATION} config=${GJD_CONFIG_NAME}" >&2
  print_gjd_architecture_contract_notice
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
    echo "GJD ${GJD_ARCHITECTURE} rollout requires CFG=${GJD_CFG_PATH}; got ${CFG}." >&2
    exit 2
  fi
  GJD_CFG_PATH="${CFG}"
fi
GJD_ROLLOUT_CONTEXT_ARGS=()
open_wam_append_fixed128_rollout_context_args GJD_ROLLOUT_CONTEXT_ARGS "${GJD_CONFIG_NAME}"
GJD_DEFAULT_ROLLOUT_IDENTITY_ARGS=()
build_default_rollout_identity_args GJD_DEFAULT_ROLLOUT_IDENTITY_ARGS
if [[ "${GJD_ARCHITECTURE}" == "dual_expert" ]]; then
  GJD_DUAL_EXPERT_ROLLOUT_SEMANTIC_ARGS=()
  build_default_dual_expert_rollout_semantic_args GJD_DUAL_EXPERT_ROLLOUT_SEMANTIC_ARGS
  GJD_REALTIME_ARGS=(
    "${REPO_ROOT}/scripts/run_libero_dual_expert_visualization.py"
    --cfg "${GJD_CFG_PATH}"
    "${GJD_ROLLOUT_CONTEXT_ARGS[@]}"
    "${GJD_DEFAULT_ROLLOUT_IDENTITY_ARGS[@]}"
    "${GJD_DUAL_EXPERT_ROLLOUT_SEMANTIC_ARGS[@]}"
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

echo "[run_gjd_libero] stage=rollout architecture=${GJD_ARCHITECTURE} ablation=${GJD_ABLATION} cfg=${GJD_CFG_PATH}" >&2
print_gjd_architecture_contract_notice
if [[ "${OPEN_WAM_PRINT_REALTIME_ARGV:-0}" == "1" ]]; then
  open_wam_print_train_argv_json "${GJD_REALTIME_ARGS[@]}"
  exit 0
fi

exec uv run python "${GJD_REALTIME_ARGS[@]}"
