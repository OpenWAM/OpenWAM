#!/usr/bin/env bash
# Shared LIBERO policy-training defaults. These are applied by the maintained
# training launchers before user-provided CLI args, so explicit `--set` values
# passed by the caller still win.

OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_DEFAULT_ARGS=(
  --set data.sample_construction.mode=hierarchical_fixed_segment
  --set data.sample_construction.segment_frames=128
  --set data.sample_construction.chunk_size=4
  --set data.sample_construction.window_size=30
  --set data.sample_construction.randomize_geometry=true
  --set data.sample_construction.start_padding_frames=3
  --set data.sample_construction.context_prefix_policy=rollout_history
  --set data.sample_construction.tail_padding_policy=zero_order_hold
  --set data.sample_construction.padded_target_policy=mask_loss
  --set data.sample_construction.task_start_power=0.5
  --set data.sample_construction.demo_count_power=0.0
  --set data.sample_construction.trajectory_start_power=1.0
)

open_wam_normalize_config_name() {
  local config_name="${1:-}"
  config_name="${config_name##*/}"
  config_name="${config_name%.yaml}"
  config_name="${config_name%.yml}"
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
    parallel_stream_libero_lingbot_exact_heng_compatible) return 0 ;;
    parallel_stream_libero_lingbot_joint_denoise_heng_compatible) return 0 ;;
    parallel_stream_libero_lingbot_m1_*_heng_compatible) return 0 ;;
    mot_libero_latent_local_full_segment_non_joint_action_only) return 0 ;;
    mot_libero_latent_local_action_noisy_to_video_heng_compatible) return 0 ;;
    mot_libero_latent_local_action_then_video_heng_compatible) return 0 ;;
    mot_libero_latent_local_decoupled_same_step_heng_compatible) return 0 ;;
    mot_libero_latent_local_generalist_joint_denoising_heng_compatible) return 0 ;;
    mot_libero_latent_local_joint_heng_compatible) return 0 ;;
    mot_libero_latent_local_video_noisy_to_action_heng_compatible) return 0 ;;
    mot_libero_latent_local_video_then_action_heng_compatible) return 0 ;;
    *) return 1 ;;
  esac
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
