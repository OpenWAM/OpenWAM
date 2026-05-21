from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts/libero_fixed128_rollout_context_defaults.sh"
POSTTRAIN_LAUNCHERS = (
    "scripts/run_causal_video_prediction_posttrain_libero.sh",
    "scripts/run_mot_nonjoint_posttrain_libero.sh",
    "scripts/run_mot_posttrain_libero.sh",
    "scripts/run_parallel_stream_posttrain_libero.sh",
    "scripts/run_post_decoded_posttrain_libero.sh",
    "scripts/run_post_latent_posttrain_libero.sh",
    "scripts/run_video_sequence_policy_posttrain_libero.sh",
)
DEPRECATED_MOT_WRAPPERS = (
    "scripts/deprecated/run_mot_non_joint_aligned_libero_A.sh",
    "scripts/deprecated/run_mot_non_joint_action_only_libero_B.sh",
)
TOP_LEVEL_DEPRECATED_MOT_WRAPPER_STUBS = (
    "scripts/run_mot_full_segment_nonjoint_libero.sh",
    "scripts/run_mot_non_joint_aligned_libero_A.sh",
    "scripts/run_mot_non_joint_action_only_libero_B.sh",
)
FIXED_128_VALUES = (
    "data.sample_construction.mode=hierarchical_fixed_segment",
    "data.sample_construction.segment_frames=128",
    "data.sample_construction.chunk_size=4",
    "data.sample_construction.window_size=30",
    "data.sample_construction.randomize_geometry=false",
    "data.sample_construction.start_padding_frames=0",
    "data.sample_construction.target_alignment=next_after_context",
    "data.sample_construction.rollout_context_policy=one_frame",
    "data.sample_construction.tail_padding_policy=zero_order_hold",
    "data.sample_construction.padded_target_policy=mask_loss",
    "data.sample_construction.task_start_power=0.5",
    "data.sample_construction.demo_count_power=0.0",
    "data.sample_construction.trajectory_start_power=1.0",
    "policy_variant.proprio_context_mode=text_context_token",
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


def _launcher_train_argv(relative_path: str, *, env_overrides: dict[str, str] | None = None) -> list[str]:
    env = os.environ.copy()
    env.update(
        {
            "OPEN_WAM_PRINT_TRAIN_ARGV": "1",
            "NGPU": "1",
        }
    )
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / relative_path)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def _deprecated_launcher_result(relative_path: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG", None)
    env["NGPU"] = "1"
    return subprocess.run(
        ["bash", str(REPO_ROOT / relative_path)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_fixed128_rollout_context_defaults_apply_to_supported_policy_training_configs() -> None:
    for config_name in (
        "parallel_stream_libero_lingbot_exact_heng_compatible",
        "parallel_stream_libero_lingbot_m1_joint_heng_compatible",
        "mot_libero_latent_local_full_segment_non_joint_action_only",
        "mot_libero_latent_local_video_then_action_heng_compatible",
        "mot_libero_latent_local_joint_heng_compatible",
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
        ("--cfg", "configs/experiments/deprecated/mot_libero_latent_local_joint.yaml"),
        ("--config=configs/experiments/deprecated/mot_libero_latent_local_joint.yaml",),
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
        assert "open_wam_resolve_libero_policy_config_name" in text
        assert "open_wam_append_fixed128_rollout_context_args" in text
        assert "open_wam_maybe_print_train_argv" in text
        assert '"${OPEN_WAM_FIXED128_ROLLOUT_CONTEXT_ARGS[@]}"' in text


def test_mot_posttrain_launcher_defaults_to_strict_fixed128_joint_config() -> None:
    argv = _launcher_train_argv("scripts/run_mot_posttrain_libero.sh")

    assert argv[:4] == [
        "--config-name",
        "mot_libero_latent_local_joint_heng_compatible",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value in argv


def test_mot_nonjoint_launcher_defaults_to_strict_fixed128_video_then_action_config() -> None:
    argv = _launcher_train_argv("scripts/run_mot_nonjoint_posttrain_libero.sh")

    assert argv[:4] == [
        "--config-name",
        "mot_libero_latent_local_video_then_action_heng_compatible",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value in argv


def test_mot_launchers_reject_legacy_configs_by_default() -> None:
    for relative_path, config_name in (
        ("scripts/run_mot_posttrain_libero.sh", "mot_libero_latent_local_joint"),
        (
            "scripts/run_mot_nonjoint_posttrain_libero.sh",
            "mot_libero_latent_local_full_segment_non_joint_aligned",
        ),
    ):
        env = os.environ.copy()
        env.update(
            {
                "OPEN_WAM_PRINT_TRAIN_ARGV": "1",
                "CONFIG_NAME": config_name,
                "NGPU": "1",
            }
        )
        result = subprocess.run(
            ["bash", str(REPO_ROOT / relative_path)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 2
        assert "Refusing deprecated LIBERO M1/M5 config" in result.stderr


def test_deprecated_mot_wrapper_scripts_fail_closed_by_default() -> None:
    for relative_path in (*DEPRECATED_MOT_WRAPPERS, *TOP_LEVEL_DEPRECATED_MOT_WRAPPER_STUBS):
        result = _deprecated_launcher_result(relative_path)

        assert result.returncode == 2
        assert "Refusing deprecated LIBERO launcher" in result.stderr
        assert "run_mot_nonjoint_posttrain_libero.sh" in result.stderr


def test_mot_launchers_allow_legacy_configs_only_with_explicit_opt_in() -> None:
    legacy_joint_argv = _launcher_train_argv(
        "scripts/run_mot_posttrain_libero.sh",
        env_overrides={
            "CONFIG_NAME": "mot_libero_latent_local_joint",
            "OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG": "true",
        },
    )

    assert legacy_joint_argv[:2] == [
        "--config-name",
        "configs/experiments/deprecated/mot_libero_latent_local_joint.yaml",
    ]
    for value in FIXED_128_VALUES:
        assert value not in legacy_joint_argv


def test_deprecated_aligned_wrapper_preserves_legacy_config_when_allowed() -> None:
    argv = _launcher_train_argv(
        "scripts/deprecated/run_mot_non_joint_aligned_libero_A.sh",
        env_overrides={
            "OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG": "true",
            "TRANSFORMER_SUBDIR": "/tmp/open_wam_transformer",
        },
    )

    assert argv[:2] == [
        "--config-name",
        "configs/experiments/deprecated/mot_libero_latent_local_full_segment_non_joint_aligned.yaml",
    ]
    assert "--transformer-subdir" in argv
    assert argv[argv.index("--transformer-subdir") + 1] == "/tmp/open_wam_transformer"
    for value in FIXED_128_VALUES:
        assert value not in argv


def test_deprecated_full_segment_name_delegates_to_renamed_nonjoint_launcher_when_allowed() -> None:
    argv = _launcher_train_argv(
        "scripts/run_mot_full_segment_nonjoint_libero.sh",
        env_overrides={"OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG": "true"},
    )

    assert argv[:4] == [
        "--config-name",
        "mot_libero_latent_local_video_then_action_heng_compatible",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value in argv


def test_libero_posttrain_launcher_can_print_exact_train_argv_without_running() -> None:
    env = os.environ.copy()
    env.update(
        {
            "OPEN_WAM_PRINT_TRAIN_ARGV": "1",
            "CONFIG_NAME": "parallel_stream_libero_lingbot_m1_joint_heng_compatible",
            "NGPU": "4",
        }
    )
    result = subprocess.run(
        [
            str(REPO_ROOT / "scripts/run_parallel_stream_posttrain_libero.sh"),
            "--num-steps",
            "4000",
            "--set",
            "training.learning_rate=2e-5",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    argv = json.loads(result.stdout)

    assert argv[:4] == [
        "--config-name",
        "parallel_stream_libero_lingbot_m1_joint_heng_compatible",
        "--devices",
        "4",
    ]
    assert "data.sample_construction.segment_frames=128" in argv
    assert argv[-4:] == ["--num-steps", "4000", "--set", "training.learning_rate=2e-5"]


def test_libero_posttrain_launcher_dry_run_uses_python3_without_venv_path() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "OPEN_WAM_PRINT_TRAIN_ARGV": "1",
        "CONFIG_NAME": "parallel_stream_libero_lingbot_m1_joint_heng_compatible",
        "NGPU": "4",
    }
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_parallel_stream_posttrain_libero.sh"),
            "--num-steps",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    argv = json.loads(result.stdout)

    assert argv[:4] == [
        "--config-name",
        "parallel_stream_libero_lingbot_m1_joint_heng_compatible",
        "--devices",
        "4",
    ]
    assert "data.sample_construction.segment_frames=128" in argv
