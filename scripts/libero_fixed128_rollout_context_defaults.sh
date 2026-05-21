#!/usr/bin/env bash
# Shared LIBERO fixed-128 policy-training defaults. These are applied by the
# maintained training launchers before user-provided CLI args, so explicit
# `--set` values passed by the caller still win.

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
  --set policy_variant.proprio_context_mode=text_context_token
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

open_wam_deprecated_libero_policy_config_reason() {
  local config_name
  config_name="$(open_wam_normalize_config_name "${1:-}")"
  case "${config_name}" in
    mot_libero_latent_local) echo "legacy M5 local config without strict one-frame fixed-128 rollout parity" ;;
    mot_libero_latent_local_idm) echo "legacy M5 IDM config without strict one-frame fixed-128 rollout parity" ;;
    mot_libero_latent_local_joint) echo "legacy M5 joint config without strict one-frame fixed-128 rollout parity" ;;
    mot_libero_latent_local_joint_full_segment) echo "legacy M5 full-segment config" ;;
    mot_libero_latent_local_full_segment) echo "legacy M5 full-segment config" ;;
    mot_libero_latent_local_full_segment_non_joint_aligned) echo "legacy M5 aligned full-segment config" ;;
    mot_libero_latent_local_full_segment_with_latent) echo "legacy M5 full-segment latent config" ;;
    parallel_stream_libero_lingbot_exact_local) echo "legacy local M1 exact config" ;;
    parallel_stream_libero_lingbot_joint_denoise_heng_compatible_contextual_fixed_geometry) echo "legacy contextual-subwindow M1 joint config" ;;
    parallel_stream_libero_lingbot_joint_denoise_heng_compatible_contextual_subwindow) echo "legacy contextual-subwindow M1 joint config" ;;
    parallel_stream_libero_lingbot_joint_denoise_heng_compatible_random_subwindow) echo "legacy random-subwindow M1 joint config" ;;
    *) return 1 ;;
  esac
}

open_wam_allows_deprecated_libero_config() {
  case "${OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG:-0}" in
    1|true|yes) return 0 ;;
    *) return 1 ;;
  esac
}

open_wam_resolve_libero_policy_config_name() {
  local config_name="${1:-}"
  local normalized
  normalized="$(open_wam_normalize_config_name "${config_name}")"
  if open_wam_deprecated_libero_policy_config_reason "${config_name}" >/dev/null 2>&1 \
    && open_wam_allows_deprecated_libero_config; then
    printf 'configs/experiments/deprecated/%s.yaml\n' "${normalized}"
    return 0
  fi
  printf '%s\n' "${config_name}"
}

open_wam_deprecated_libero_launcher_replacement() {
  local launcher_name="${1:-}"
  launcher_name="${launcher_name##*/}"
  case "${launcher_name}" in
    run_mot_non_joint_aligned_libero_A.sh)
      echo "scripts/run_mot_nonjoint_posttrain_libero.sh with a current *_heng_compatible CONFIG_NAME"
      ;;
    run_mot_non_joint_action_only_libero_B.sh)
      echo "scripts/run_mot_nonjoint_posttrain_libero.sh with a current *_heng_compatible CONFIG_NAME"
      ;;
    run_mot_full_segment_nonjoint_libero.sh)
      echo "scripts/run_mot_nonjoint_posttrain_libero.sh"
      ;;
    *) return 1 ;;
  esac
}

open_wam_reject_deprecated_libero_launcher() {
  local launcher_name="${1:-}"
  local replacement="${2:-}"
  if open_wam_allows_deprecated_libero_config; then
    return 0
  fi
  if [ -z "${replacement}" ]; then
    replacement="$(open_wam_deprecated_libero_launcher_replacement "${launcher_name}")" || return 0
  fi
  echo "Refusing deprecated LIBERO launcher '${launcher_name}'." >&2
  echo "Use ${replacement}." >&2
  echo "Set OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG=1 only for historical debugging." >&2
  return 2
}

open_wam_reject_deprecated_libero_policy_config() {
  local config_name="${1:-}"
  local reason
  if open_wam_allows_deprecated_libero_config; then
    return 0
  fi
  if reason="$(open_wam_deprecated_libero_policy_config_reason "${config_name}")"; then
    echo "Refusing deprecated LIBERO M1/M5 config '${config_name}': ${reason}." >&2
    echo "Current launchers require strict fixed-128, one-frame rollout conditioning, fixed 4-frame chunks, no head padding, and proprio text-context tokens." >&2
    echo "Use a current *_heng_compatible config, or set OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG=1 only for historical debugging." >&2
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
