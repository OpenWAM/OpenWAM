#!/usr/bin/env bash
# Historical LIBERO aliases and removed-entrypoint diagnostics.

open_wam_normalize_config_name() {
  local config_name="${1:-}"
  config_name="${config_name##*/}"
  config_name="${config_name%.yaml}"
  config_name="${config_name%.yml}"
  case "${config_name}" in
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
      config_name="parallel_stream_libero_${config_name}"
      ;;
    parallel_stream_robotwin_lingbot_m1_*)
      config_name="${config_name#parallel_stream_robotwin_lingbot_m1_}"
      config_name="parallel_stream_robotwin_${config_name}"
      ;;
    parallel_stream_libero_lingbot_joint_denoise)
      config_name="parallel_stream_libero_joint"
      ;;
  esac
  printf '%s\n' "${config_name}"
}

open_wam_removed_libero_policy_config_reason() {
  local config_name
  config_name="$(open_wam_normalize_config_name "${1:-}")"
  case "${config_name}" in
    dual_expert_libero_latent_local) echo "retired dual-expert local config" ;;
    dual_expert_libero_latent_local_idm) echo "retired dual-expert IDM config" ;;
    dual_expert_libero_latent_local_joint) echo "retired dual-expert joint config" ;;
    dual_expert_libero_latent_local_joint_full_segment) echo "retired dual-expert full-segment config" ;;
    dual_expert_libero_latent_local_full_segment) echo "retired dual-expert full-segment config" ;;
    dual_expert_libero_latent_local_full_segment_non_joint_aligned) echo "retired aligned dual-expert config" ;;
    dual_expert_libero_latent_local_full_segment_with_latent) echo "retired latent dual-expert config" ;;
    parallel_stream_libero_lingbot_exact_local) echo "retired local parallel-stream config" ;;
    parallel_stream_libero_current_frame_action_chunk) echo "retired current-frame action-chunk experiment" ;;
    parallel_stream_libero_fastwam_first_frame) echo "retired first-frame FastWAM experiment" ;;
    *) return 1 ;;
  esac
}

open_wam_reject_removed_libero_policy_config() {
  local config_name="${1:-}"
  local reason
  if reason="$(open_wam_removed_libero_policy_config_reason "${config_name}")"; then
    echo "Removed LIBERO policy config '${config_name}': ${reason}." >&2
    echo "Use a maintained experiment config. Git history retains the historical YAML." >&2
    return 2
  fi
}

open_wam_removed_libero_launcher_replacement() {
  local launcher_name="${1:-}"
  launcher_name="${launcher_name##*/}"
  case "${launcher_name}" in
    run_mot_non_joint_aligned_libero_A.sh)
      echo "scripts/run_dual_expert_posttrain_libero.sh with a maintained CONFIG_NAME"
      ;;
    run_mot_non_joint_action_only_libero_B.sh)
      echo "scripts/run_dual_expert_posttrain_libero.sh with a maintained CONFIG_NAME"
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
