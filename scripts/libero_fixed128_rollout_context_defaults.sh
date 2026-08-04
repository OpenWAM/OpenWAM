#!/usr/bin/env bash
# Shared LIBERO training-launcher contracts. Fixed-128 defaults are applied
# before user-provided CLI args so explicit `--set` values still win; the
# process owner at the end keeps all maintained presets on one launch path.

OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_DEFAULT_ARGS=(
  --set data.sample_construction.mode=hierarchical_fixed_segment
  --set data.sample_construction.segment_frames=128
  --set data.sample_construction.chunk_size=4
  --set data.sample_construction.window_size=30
  --set data.sample_construction.randomize_geometry=false
  --set data.sample_construction.start_padding_frames=0
  --set data.sample_construction.target_alignment=next_after_context
  --set data.sample_construction.rollout_context_policy=one_frame
  --set data.sample_construction.tail_padding_policy=zero_order_hold
  --set data.sample_construction.padded_target_policy=mask_loss
  --set data.sample_construction.task_start_power=0.5
  --set data.sample_construction.demo_count_power=0.0
  --set data.sample_construction.trajectory_start_power=1.0
  --set policy_variant.proprio_context_mode=per_chunk_additive
)

open_wam_normalize_config_name() {
  local config_name="${1:-}"
  config_name="${config_name##*/}"
  config_name="${config_name%.yaml}"
  config_name="${config_name%.yml}"
  case "${config_name}" in
    mot_libero_latent_local_*_heng_compatible)
      config_name="${config_name#mot_libero_latent_local_}"
      config_name="dual_expert_libero_${config_name%_heng_compatible}"
      ;;
    dual_expert_libero_latent_local_*_heng_compatible)
      config_name="${config_name#dual_expert_libero_latent_local_}"
      config_name="dual_expert_libero_${config_name%_heng_compatible}"
      ;;
    mot_libero_latent_local*)
      config_name="dual_expert_libero_${config_name#mot_libero_}"
      ;;
    mot_libero_*)
      config_name="dual_expert_libero_${config_name#mot_libero_}"
      ;;
    mot_robotwin_smoke)
      config_name="dual_expert_robotwin_smoke"
      ;;
    parallel_stream_libero_lingbot_m1_*)
      config_name="${config_name#parallel_stream_libero_lingbot_m1_}"
      config_name="parallel_stream_libero_${config_name%_heng_compatible}"
      ;;
    parallel_stream_robotwin_lingbot_m1_*)
      config_name="${config_name#parallel_stream_robotwin_lingbot_m1_}"
      config_name="parallel_stream_robotwin_${config_name%_heng_compatible}"
      ;;
    parallel_stream_libero_lingbot_joint_denoise|parallel_stream_libero_lingbot_joint_denoise_heng_compatible)
      config_name="parallel_stream_libero_joint_denoise"
      ;;
    parallel_stream_libero_lingbot_exact_heng_compatible)
      config_name="parallel_stream_libero_lingbot_exact"
      ;;
  esac
  printf '%s\n' "${config_name}"
}

open_wam_reject_cli_config_override_args() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --config-name|--config-name=*|--cfg|--cfg=*|--config|--config=*)
        echo "Do not pass ${arg} to this launcher; set CONFIG_NAME=... instead." >&2
        return 2
        ;;
    esac
  done
}

open_wam_should_apply_fixed128_rollout_context() {
  local config_name
  config_name="$(open_wam_normalize_config_name "${1:-}")"
  if [[ "${OPEN_WAM_ENABLE_FIXED128_ROLLOUT_CONTEXT:-1}" != "1" ]]; then
    return 1
  fi
  case "${config_name}" in
    *generalist_joint_denoising*) return 1 ;;
    parallel_stream_libero_lingbot_exact) return 0 ;;
    parallel_stream_libero_joint_denoise) return 0 ;;
    parallel_stream_libero_action_noisy_to_video) return 0 ;;
    parallel_stream_libero_action_then_video) return 0 ;;
    parallel_stream_libero_current_frame_action_chunk) return 0 ;;
    parallel_stream_libero_decoupled_same_step) return 0 ;;
    parallel_stream_libero_fastwam_first_frame) return 0 ;;
    parallel_stream_libero_joint) return 0 ;;
    parallel_stream_libero_video_noisy_to_action) return 0 ;;
    parallel_stream_libero_video_then_action) return 0 ;;
    dual_expert_libero_latent_local_full_segment_non_joint_action_only) return 0 ;;
    dual_expert_libero_action_noisy_to_video) return 0 ;;
    dual_expert_libero_action_then_video) return 0 ;;
    dual_expert_libero_decoupled_same_step) return 0 ;;
    dual_expert_libero_joint) return 0 ;;
    dual_expert_libero_video_noisy_to_action) return 0 ;;
    dual_expert_libero_video_then_action) return 0 ;;
    *) return 1 ;;
  esac
}

open_wam_removed_libero_policy_config_reason() {
  local config_name
  config_name="$(open_wam_normalize_config_name "${1:-}")"
  case "${config_name}" in
    dual_expert_libero_latent_local) echo "legacy M5 local config without strict one-frame fixed-128 rollout parity" ;;
    dual_expert_libero_latent_local_idm) echo "legacy M5 IDM config without strict one-frame fixed-128 rollout parity" ;;
    dual_expert_libero_latent_local_joint) echo "legacy M5 joint config without strict one-frame fixed-128 rollout parity" ;;
    dual_expert_libero_latent_local_joint_full_segment) echo "legacy M5 full-segment config" ;;
    dual_expert_libero_latent_local_full_segment) echo "legacy M5 full-segment config" ;;
    dual_expert_libero_latent_local_full_segment_non_joint_aligned) echo "legacy M5 aligned full-segment config" ;;
    dual_expert_libero_latent_local_full_segment_with_latent) echo "legacy M5 full-segment latent config" ;;
    parallel_stream_libero_lingbot_exact_local) echo "legacy local M1 exact config" ;;
    parallel_stream_libero_joint_denoise_heng_compatible_contextual_fixed_geometry) echo "legacy contextual-subwindow parallel-stream joint config" ;;
    parallel_stream_libero_joint_denoise_heng_compatible_contextual_subwindow) echo "legacy contextual-subwindow parallel-stream joint config" ;;
    parallel_stream_libero_joint_denoise_heng_compatible_random_subwindow) echo "legacy random-subwindow parallel-stream joint config" ;;
    *) return 1 ;;
  esac
}

open_wam_removed_libero_launcher_replacement() {
  local launcher_name="${1:-}"
  launcher_name="${launcher_name##*/}"
  case "${launcher_name}" in
    run_mot_non_joint_aligned_libero_A.sh)
      echo "scripts/run_dual_expert_posttrain_libero.sh with a current canonical CONFIG_NAME"
      ;;
    run_mot_non_joint_action_only_libero_B.sh)
      echo "scripts/run_dual_expert_posttrain_libero.sh with a current canonical CONFIG_NAME"
      ;;
    run_mot_full_segment_nonjoint_libero.sh)
      echo "scripts/run_dual_expert_posttrain_libero.sh"
      ;;
    *) return 1 ;;
  esac
}

open_wam_reject_removed_libero_launcher() {
  local launcher_name="${1:-}"
  local replacement="${2:-}"
  if [ -z "${replacement}" ]; then
    replacement="$(open_wam_removed_libero_launcher_replacement "${launcher_name}")" || return 0
  fi
  echo "Removed LIBERO launcher '${launcher_name}'." >&2
  echo "Use ${replacement}." >&2
  echo "Git history retains the historical implementation; there is no runtime opt-in." >&2
  return 2
}

open_wam_reject_removed_libero_policy_config() {
  local config_name="${1:-}"
  local reason
  if reason="$(open_wam_removed_libero_policy_config_reason "${config_name}")"; then
    echo "Removed LIBERO M1/M5 config '${config_name}': ${reason}." >&2
    echo "Current non-GJD launchers require strict fixed-128 rollout parity; current GJD launchers require full-segment W64 sampling." >&2
    echo "Use a current canonical config. Git history retains the historical YAML." >&2
    return 2
  fi
}

open_wam_append_fixed128_rollout_context_args() {
  local -n target_args="$1"
  local config_name="${2:-}"
  if open_wam_should_apply_fixed128_rollout_context "${config_name}"; then
    target_args+=("${OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_DEFAULT_ARGS[@]}")
  fi
}

open_wam_print_train_argv_json() {
  local argv_python="${OPEN_WAM_TRAIN_ARGV_PYTHON:-python3}"
  "${argv_python}" - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
}

open_wam_maybe_print_train_argv() {
  if [[ "${OPEN_WAM_PRINT_TRAIN_ARGV:-0}" == "1" ]]; then
    open_wam_print_train_argv_json "$@"
    exit 0
  fi
}

open_wam_launch_training() {
  local config_name="${1:?Pass the experiment config name as the first argument.}"
  shift
  local ngpu="${NGPU:-1}"
  local master_port="${MASTER_PORT:-29501}"
  local log_rank="${LOG_RANK:-0}"
  local -a rollout_context_args=()
  local -a train_args=()

  export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
  export WANDB_MODE=${WANDB_MODE:-"online"}

  open_wam_reject_cli_config_override_args "$@"
  open_wam_reject_removed_libero_policy_config "${config_name}"
  open_wam_append_fixed128_rollout_context_args rollout_context_args "${config_name}"
  train_args=(
    --config-name "${config_name}"
    --devices "${ngpu}"
    "${rollout_context_args[@]}"
    "$@"
  )
  open_wam_maybe_print_train_argv "${train_args[@]}"

  if [ "${ngpu}" -gt 1 ]; then
    uv run python -m torch.distributed.run \
      --nproc_per_node="${ngpu}" \
      --local-ranks-filter="${log_rank}" \
      --master_port "${master_port}" \
      --tee 3 \
      -m open_wam.cli.train \
      "${train_args[@]}"
  else
    uv run python -m open_wam.cli.train \
      "${train_args[@]}"
  fi
}
