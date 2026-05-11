from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts/libero_fixed128_rollout_context_defaults.sh"
POSTTRAIN_LAUNCHERS = (
    "scripts/run_causal_video_prediction_posttrain_libero.sh",
    "scripts/run_mot_full_segment_nonjoint_libero.sh",
    "scripts/run_mot_posttrain_libero.sh",
    "scripts/run_parallel_stream_posttrain_libero.sh",
    "scripts/run_post_decoded_posttrain_libero.sh",
    "scripts/run_post_latent_posttrain_libero.sh",
    "scripts/run_video_sequence_policy_posttrain_libero.sh",
)
FIXED_128_VALUES = (
    "data.sample_construction.mode=hierarchical_fixed_segment",
    "data.sample_construction.segment_frames=128",
    "data.sample_construction.chunk_size=4",
    "data.sample_construction.window_size=30",
    "data.sample_construction.randomize_geometry=true",
    "data.sample_construction.start_padding_frames=3",
    "data.sample_construction.context_prefix_policy=rollout_history",
    "data.sample_construction.tail_padding_policy=zero_order_hold",
    "data.sample_construction.padded_target_policy=mask_loss",
    "data.sample_construction.task_start_power=0.5",
    "data.sample_construction.demo_count_power=0.0",
    "data.sample_construction.trajectory_start_power=1.0",
)


def _default_args_for(config_name: str, *, enabled: bool = True) -> list[str]:
    env = os.environ.copy()
    env["OPEN_WAM_ENABLE_FIXED128_ROLLOUT_CONTEXT"] = "1" if enabled else "0"
    command = f"""
set -euo pipefail
source {str(HELPER_PATH)!r}
args=()
open_wam_append_fixed128_rollout_context_args args {config_name!r}
printf '%s\\n' "${{args[@]}}"
"""
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return [line for line in result.stdout.splitlines() if line]


def _reject_config_override_result(*args: str) -> subprocess.CompletedProcess[str]:
    command = f"""
set -euo pipefail
source {str(HELPER_PATH)!r}
open_wam_reject_cli_config_override_args "$@"
"""
    return subprocess.run(
        ["bash", "-lc", command, "bash", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_fixed128_rollout_context_defaults_apply_to_supported_policy_training_configs() -> None:
    for config_name in (
        "parallel_stream_libero_lingbot_exact_heng_compatible",
        "parallel_stream_libero_lingbot_m1_joint_heng_compatible",
        "mot_libero_latent_local_full_segment_non_joint_action_only",
        "mot_libero_latent_local_generalist_joint_denoising_heng_compatible",
    ):
        args = _default_args_for(config_name)

        for value in FIXED_128_VALUES:
            assert value in args


def test_fixed128_rollout_context_defaults_normalize_config_paths() -> None:
    for config_name in (
        "parallel_stream_libero_lingbot_exact_heng_compatible.yaml",
        "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml",
        "/tmp/configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yml",
    ):
        args = _default_args_for(config_name)

        for value in FIXED_128_VALUES:
            assert value in args


def test_fixed128_rollout_context_defaults_are_gated_and_disableable() -> None:
    assert _default_args_for("causal_video_prediction_libero_latent_local") == []
    assert _default_args_for("mot_libero_latent_local_joint") == []
    assert _default_args_for("mot_libero_latent_local_full_segment") == []
    assert _default_args_for("mot_libero_latent_local_full_segment_non_joint_aligned") == []
    assert _default_args_for("parallel_stream_libero_lingbot_exact_local") == []
    assert _default_args_for("parallel_stream_libero_lingbot_joint_denoise_heng_compatible_contextual_subwindow") == []
    assert _default_args_for("parallel_stream_libero_lingbot_joint_denoise_heng_compatible_random_subwindow") == []
    assert _default_args_for("parallel_stream_libero_lingbot_exact_heng_compatible", enabled=False) == []


def test_launchers_reject_late_cli_config_overrides() -> None:
    for args in (
        ("--config-name", "mot_libero_latent_local_joint"),
        ("--config-name=mot_libero_latent_local_joint",),
        ("--cfg", "configs/experiments/mot_libero_latent_local_joint.yaml"),
        ("--config=configs/experiments/mot_libero_latent_local_joint.yaml",),
    ):
        result = _reject_config_override_result(*args)

        assert result.returncode == 2
        assert "set CONFIG_NAME=... instead" in result.stderr

    assert _reject_config_override_result("--set", "training.num_steps=1").returncode == 0


def test_libero_posttrain_launchers_inherit_shared_fixed128_defaults() -> None:
    for relative_path in POSTTRAIN_LAUNCHERS:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

        assert "libero_fixed128_rollout_context_defaults.sh" in text
        assert "open_wam_reject_cli_config_override_args" in text
        assert "open_wam_append_fixed128_rollout_context_args" in text
        assert '"${OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_ARGS[@]}"' in text
