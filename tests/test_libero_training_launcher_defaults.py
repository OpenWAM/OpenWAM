from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess

import pytest
import yaml

from open_wam.configs.enums import (
    GeneralistTrainingParadigm,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    MoTGeneralistTrainingMode,
    SampleOrderMode,
)
from open_wam.configs import load_experiment_config
from open_wam.utils.config_overrides import apply_config_overrides, parse_override_assignments


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts/libero_fixed128_rollout_context_defaults.sh"
POSTTRAIN_LAUNCHERS = (
    "scripts/run_causal_video_prediction_posttrain_libero.sh",
    "scripts/run_mot_nonjoint_posttrain_libero.sh",
    "scripts/run_mot_posttrain_libero.sh",
    "scripts/run_parallel_stream_posttrain_libero.sh",
    "scripts/run_post_decoded_posttrain_libero.sh",
    "scripts/run_post_latent_posttrain_libero.sh",
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
    "policy_variant.proprio_context_mode=per_chunk_additive",
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


def _launcher_train_argv(
    relative_path: str,
    *script_args: str,
    env_overrides: dict[str, str] | None = None,
) -> list[str]:
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
        ["bash", str(REPO_ROOT / relative_path), *script_args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_posttrain_launchers_delegate_to_shared_process_owner() -> None:
    expected_projects = {
        "scripts/run_causal_video_prediction_posttrain_libero.sh": "openwam-causal-video-libero",
        "scripts/run_parallel_stream_posttrain_libero.sh": "lingbot-va-posttrain-libero",
        "scripts/run_mot_posttrain_libero.sh": "openwam-method5-libero",
        "scripts/run_post_decoded_posttrain_libero.sh": (
            "openwam-method4-post-decoded-libero-video-conditioned"
        ),
        "scripts/run_post_latent_posttrain_libero.sh": (
            "openwam-method4-post-latent-libero-video-conditioned"
        ),
        "scripts/run_mot_nonjoint_posttrain_libero.sh": "openwam-libero-policy-train",
    }
    helper_source = HELPER_PATH.read_text(encoding="utf-8")

    assert set(expected_projects) == set(POSTTRAIN_LAUNCHERS)
    assert helper_source.count("open_wam_launch_training()") == 1
    assert helper_source.count("python -m torch.distributed.run") == 1
    assert helper_source.count("python -m open_wam.training.train") == 1
    assert "open_wam_append_fixed128_rollout_context_args" in helper_source
    for relative_path, project in expected_projects.items():
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "libero_fixed128_rollout_context_defaults.sh" in source
        assert source.count("open_wam_launch_training") == 1
        assert "torch.distributed.run" not in source
        assert "OPEN_WAM_TRAIN_ARGS" not in source
        assert project in source


def test_shared_posttrain_process_owner_preserves_distributed_command_and_env(
    tmp_path: Path,
) -> None:
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
printf 'ARG=%s\\n' "$@"
printf 'TOKENIZERS_PARALLELISM=%s\\n' "${TOKENIZERS_PARALLELISM:-}"
printf 'PYTORCH_CUDA_ALLOC_CONF=%s\\n' "${PYTORCH_CUDA_ALLOC_CONF:-}"
printf 'WANDB_MODE=%s\\n' "${WANDB_MODE:-}"
printf 'WANDB_PROJECT=%s\\n' "${WANDB_PROJECT:-}"
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    for variable in (
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_CUDA_ALLOC_CONF",
        "WANDB_MODE",
        "WANDB_PROJECT",
    ):
        env.pop(variable, None)
    env.update(
        {
            "PATH": f"{tmp_path}:{env['PATH']}",
            "NGPU": "4",
            "LOG_RANK": "2",
            "MASTER_PORT": "29677",
            "OPEN_WAM_ENABLE_FIXED128_ROLLOUT_CONTEXT": "0",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_mot_posttrain_libero.sh"),
            "--num-steps",
            "17",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    lines = result.stdout.splitlines()
    args = [line.removeprefix("ARG=") for line in lines if line.startswith("ARG=")]
    assert args == [
        "run",
        "python",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=4",
        "--local-ranks-filter=2",
        "--master_port",
        "29677",
        "--tee",
        "3",
        "-m",
        "open_wam.training.train",
        "--config-name",
        "mot_libero_latent_local_joint_heng_compatible",
        "--devices",
        "4",
        "--num-steps",
        "17",
    ]
    assert "TOKENIZERS_PARALLELISM=false" in lines
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in lines
    assert "WANDB_MODE=online" in lines
    assert "WANDB_PROJECT=openwam-method5-libero" in lines


def _launcher_realtime_argv(
    relative_path: str,
    *script_args: str,
    env_overrides: dict[str, str] | None = None,
) -> list[str]:
    env = os.environ.copy()
    env["OPEN_WAM_PRINT_REALTIME_ARGV"] = "1"
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / relative_path), *script_args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def _launcher_realtime_result(
    relative_path: str,
    *script_args: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["OPEN_WAM_PRINT_REALTIME_ARGV"] = "1"
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", str(REPO_ROOT / relative_path), *script_args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_mot_visualization_module():
    from open_wam.evals import libero_mot_rollout

    return libero_mot_rollout


def _set_override_tokens(argv: list[str]) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--set":
            tokens.append(argv[index + 1])
            index += 2
        else:
            index += 1
    return tokens


def _arg_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _config_path_from_train_argv(argv: list[str]) -> Path:
    config_name = argv[argv.index("--config-name") + 1]
    config_path = Path(config_name)
    if not config_path.is_absolute():
        if config_path.suffix:
            config_path = REPO_ROOT / config_path
        else:
            config_path = REPO_ROOT / "configs/experiments" / f"{config_name}.yaml"
    return config_path


def _config_path_from_realtime_argv(argv: list[str]) -> Path:
    config_path = Path(argv[argv.index("--cfg") + 1])
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return config_path


def _resolved_config_from_train_argv(argv: list[str]):
    config = load_experiment_config(_config_path_from_train_argv(argv))
    return apply_config_overrides(
        config,
        parse_override_assignments(_set_override_tokens(argv)),
    )


def _resolved_config_from_realtime_argv(argv: list[str]):
    config = load_experiment_config(_config_path_from_realtime_argv(argv))
    return apply_config_overrides(
        config,
        parse_override_assignments(_set_override_tokens(argv)),
    )


def _assert_gjd_ablation_config(config, *, method: str, ablation: str) -> None:
    if method == "m1":
        probs = config.policy_variant.joint_denoise_training_mode_probs
        joint = JointDenoiseTrainingMode.JOINT
        fdm = JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
        idm = JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION
    elif method == "m5":
        probs = config.policy_variant.mot_generalist_training_mode_probs
        joint = MoTGeneralistTrainingMode.JOINT
        fdm = MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO
        idm = MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION
    else:
        raise AssertionError(f"Unexpected method {method!r}")

    if ablation == "pure_joint":
        assert probs[joint] == 1.0
        assert probs[fdm] == 0.0
        assert probs[idm] == 0.0
        assert config.policy_variant.generalist_training_paradigm == GeneralistTrainingParadigm.DEMO_ONLY
        assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
        assert config.data.generalist_dynamics_mixture.train_latent_root is None
        assert config.data.generalist_dynamics_mixture.val_latent_root is None
    else:
        assert probs[joint] == 0.6
        assert probs[fdm] == 0.2
        assert probs[idm] == 0.2
        assert config.policy_variant.generalist_training_paradigm == GeneralistTrainingParadigm.MIXED_DYNAMICS
        assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
        assert config.data.generalist_dynamics_mixture.train_latent_root is not None
        assert config.data.generalist_dynamics_mixture.val_latent_root is not None
    assert config.policy_variant.generalist_mode_text_token is (ablation == "mode_token")


def _gjd_raw_config(*, method: str) -> dict:
    if method == "m1":
        config_path = (
            REPO_ROOT
            / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
        )
    elif method == "m5":
        config_path = (
            REPO_ROOT / "configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml"
        )
    else:
        raise AssertionError(f"Unexpected method {method!r}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _assert_gjd_fullseg_w64_raw_config(raw: dict) -> None:
    sample = raw["data"]["sample_construction"]

    assert raw["data"]["replay_status_policy"] == "include_all"
    assert raw["data"]["val_replay_status_policy"] is None
    assert raw["data"]["require_replay_status"] is False
    assert raw["data"]["val_require_replay_status"] is False
    assert sample["mode"] == "uniform_segment"
    assert sample["sample_order_mode"] == "replacement"
    assert sample["chunk_size"] == 4
    assert sample["window_size"] == 64
    assert sample["randomize_geometry"] is True
    assert sample["segment_min_frames"] == 1000
    assert sample["segment_max_frames"] == 1000
    assert sample["segment_length_stride"] == 1
    assert sample["segment_locality_block_size"] == 1
    assert sample["randomize_segment_length"] is False
    assert sample["randomize_segment_start"] is False
    assert sample["require_full_segment"] is True
    assert sample["task_start_power"] == 0.0
    assert sample["demo_count_power"] == 0.0
    assert sample["trajectory_start_power"] == 0.0
    assert sample["sample_weight_mode"] == "uniform"
    assert "segment_frames" not in sample
    assert "target_alignment" not in sample
    assert "condition_source_frame_offset" not in sample
    assert "start_padding_frames" not in sample
    assert raw["training"]["window_size"] == 64
    assert raw["training"]["sample_loss_weight_mode"] == "none"
    assert raw["training"]["num_steps"] == 20000
    assert raw["policy_variant"]["joint_timestep_coupling"] == "independent"
    assert raw["policy_variant"]["generalist_training_paradigm"] == "mixed_dynamics"
    assert raw["policy_variant"]["generalist_mode_text_token"] is False
    assert raw["trainer"]["checkpoint_mode"] == "full_training_state"
    assert raw["trainer"]["save_interval"] == 100
    assert raw["trainer"]["max_checkpoints_to_keep"] == 3
    mixture = raw["data"]["generalist_dynamics_mixture"]
    assert mixture["train_latent_root"] == "${paths.datasets.libero_gjd_counterfactual_train_latent_root}"
    assert mixture["val_latent_root"] == "${paths.datasets.libero_gjd_counterfactual_val_latent_root}"
    assert mixture["conditional_history_frames"] is None


def _deprecated_launcher_result(
    relative_path: str,
    *,
    allow_deprecated: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG", None)
    if allow_deprecated:
        env["OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG"] = "true"
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
    ):
        args = _default_args_for(config_name)

        for value in FIXED_128_VALUES:
            assert value in args


def test_m1_gjd_config_deprecates_fixed128_for_fullseg_w64() -> None:
    argv = _launcher_train_argv("scripts/run_gjd_libero.sh", "train", "--method=m1", "--ablation=vanilla")

    assert argv[:4] == [
        "--config-name",
        "parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value not in argv

    raw = _gjd_raw_config(method="m1")
    _assert_gjd_fullseg_w64_raw_config(raw)
    assert raw["policy_variant"]["proprio_context_mode"] == "per_chunk_additive"
    assert "parallel_sequence_contract" not in raw["policy_variant"]
    assert raw["policy_variant"]["attn_window"] == 30
    assert raw["policy_variant"]["preserve_video_pretrain_history"] is True


def test_gjd_launcher_help_labels_m5_standard_and_m1_known_issue() -> None:
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/run_gjd_libero.sh"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "M5 is the maintained standard GJD contract" in result.stdout
    assert "Known M1 GJD issue" in result.stdout
    assert "context_condition_latent_source=single_frame_condition_latent" in result.stdout
    assert "M1's legacy-prefix path supports only pure joint" in result.stdout
    assert "policy_variant.context_condition_latent_source=video_latents" in result.stdout
    assert "data.sample_construction.condition_source_frame_offset=0" in result.stdout
    assert "rollout uses run_libero_realtime_sandbox.py" in result.stdout


def test_m1_gjd_launcher_prints_known_issue_notice() -> None:
    env = os.environ.copy()
    env.update({"OPEN_WAM_PRINT_TRAIN_ARGV": "1", "NGPU": "1"})
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_gjd_libero.sh"),
            "train",
            "--method=m1",
            "--ablation=vanilla",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    argv = json.loads(result.stdout)
    assert argv[:2] == [
        "--config-name",
        "parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible",
    ]
    assert "known M1 GJD issue" in result.stderr
    assert "M5 is the maintained standard GJD contract" in result.stderr
    assert "conditional FDM/IDM needs full clean modality slots" in result.stderr
    assert "loss_frame_start=0" in result.stderr
    assert "context_condition_latent_source=video_latents" in result.stderr
    assert "condition_source_frame_offset=0" in result.stderr


def test_m5_gjd_config_deprecates_fixed128_for_legacy_prefix_fullseg_w64() -> None:
    argv = _launcher_train_argv("scripts/run_m5_gjd_posttrain_libero.sh")

    assert argv[:4] == [
        "--config-name",
        "mot_libero_latent_local_generalist_joint_denoising_heng_compatible",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value not in argv

    raw = _gjd_raw_config(method="m5")
    _assert_gjd_fullseg_w64_raw_config(raw)
    assert raw["policy_variant"]["parallel_sequence_contract"] == "legacy_prefix_single_frame_perchunk_proprio"
    assert "proprio_context_mode" not in raw["policy_variant"]
    assert raw["policy_variant"]["noisy_video_condition_prob"] == 0.5


def test_m5_gjd_pure_joint_launcher_keeps_fullseg_w64_sampler() -> None:
    argv = _launcher_train_argv(
        "scripts/run_m5_gjd_posttrain_libero.sh",
        env_overrides={"M5_GJD_VARIANT": "pure_joint"},
    )

    for value in FIXED_128_VALUES:
        assert value not in argv
    _assert_gjd_ablation_config(_resolved_config_from_train_argv(argv), method="m5", ablation="pure_joint")


def test_m5_gjd_mode_token_launcher_keeps_fullseg_w64_sampler() -> None:
    argv = _launcher_train_argv(
        "scripts/run_m5_gjd_posttrain_libero.sh",
        env_overrides={"M5_GJD_VARIANT": "mode_token"},
    )

    for value in FIXED_128_VALUES:
        assert value not in argv
    assert "policy_variant.generalist_mode_text_token=true" in argv
    _assert_gjd_ablation_config(_resolved_config_from_train_argv(argv), method="m5", ablation="mode_token")


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
    assert _default_args_for("parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible") == []
    assert _default_args_for("mot_libero_latent_local_joint") == []
    assert _default_args_for("mot_libero_latent_local_generalist_joint_denoising_heng_compatible") == []
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


def test_mot_gjd_posttrain_launcher_exposes_named_ablation_overrides() -> None:
    vanilla = _launcher_train_argv("scripts/run_mot_gjd_posttrain_libero.sh")
    assert vanilla[:4] == [
        "--config-name",
        "mot_libero_latent_local_generalist_joint_denoising_heng_compatible",
        "--devices",
        "1",
    ]
    assert "policy_variant.generalist_mode_text_token=false" in vanilla
    for value in FIXED_128_VALUES:
        assert value not in vanilla

    pure_joint = _launcher_train_argv(
        "scripts/run_mot_gjd_posttrain_libero.sh",
        env_overrides={"M5_GJD_ABLATION": "pure_joint"},
    )
    assert any(token.startswith("policy_variant.mot_generalist_training_mode_probs=") for token in pure_joint)
    pure_joint_config = _resolved_config_from_train_argv(pure_joint)
    _assert_gjd_ablation_config(pure_joint_config, method="m5", ablation="pure_joint")

    mode_token = _launcher_train_argv(
        "scripts/run_mot_gjd_posttrain_libero.sh",
        env_overrides={"M5_GJD_ABLATION": "mode_token"},
    )
    assert any(token.startswith("policy_variant.mot_generalist_training_mode_probs=") for token in mode_token)
    assert "policy_variant.generalist_mode_text_token=true" in mode_token
    _assert_gjd_ablation_config(_resolved_config_from_train_argv(mode_token), method="m5", ablation="mode_token")


def test_unified_gjd_train_launcher_covers_m1_and_m5_ablation_surfaces() -> None:
    expected_configs = {
        "m1": "parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible",
        "m5": "mot_libero_latent_local_generalist_joint_denoising_heng_compatible",
    }
    expected_prob_prefixes = {
        "m1": "policy_variant.joint_denoise_training_mode_probs=",
        "m5": "policy_variant.mot_generalist_training_mode_probs=",
    }
    for method in ("m1", "m5"):
        for ablation in ("vanilla", "pure_joint", "mode_token"):
            argv = _launcher_train_argv(
                "scripts/run_gjd_libero.sh",
                "train",
                f"--method={method}",
                f"--ablation={ablation}",
            )
            assert argv[:4] == [
                "--config-name",
                expected_configs[method],
                "--devices",
                "1",
            ]
            assert any(token.startswith(expected_prob_prefixes[method]) for token in argv)
            assert f"policy_variant.generalist_mode_text_token={str(ablation == 'mode_token').lower()}" in argv
            _assert_gjd_ablation_config(_resolved_config_from_train_argv(argv), method=method, ablation=ablation)
            for value in FIXED_128_VALUES:
                assert value not in argv


def test_unified_gjd_train_launcher_assigns_safe_default_run_identity() -> None:
    vanilla = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--method=m5",
        "--ablation=vanilla",
    )
    pure_joint = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--method=m5",
        "--ablation=pure_joint",
    )

    vanilla_save_root = vanilla[vanilla.index("--save-root") + 1]
    pure_joint_save_root = pure_joint[pure_joint.index("--save-root") + 1]
    assert vanilla_save_root.startswith("runs/gjd_libero_m5_vanilla_")
    assert pure_joint_save_root.startswith("runs/gjd_libero_m5_pure_joint_")
    assert vanilla_save_root != pure_joint_save_root


def test_unified_gjd_train_launcher_assigns_comparison_wandb_project() -> None:
    default_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--method=m1",
        "--ablation=vanilla",
    )
    assert default_argv[default_argv.index("--wandb-project") + 1] == "openwam-gjd-libero"

    explicit_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--method=m1",
        "--ablation=vanilla",
        "--wandb-project",
        "custom-project",
    )
    assert explicit_argv.count("--wandb-project") == 1
    assert explicit_argv[explicit_argv.index("--wandb-project") + 1] == "custom-project"

    env_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--method=m5",
        "--ablation=mode_token",
        env_overrides={"WANDB_PROJECT": "env-project"},
    )
    assert "--wandb-project" not in env_argv


def test_unified_gjd_train_launcher_preserves_explicit_output_identity() -> None:
    save_root_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--method=m1",
        "--ablation=vanilla",
        "--save-root",
        "/tmp/openwam-gjd-explicit",
    )
    assert save_root_argv.count("--save-root") == 1
    assert save_root_argv[save_root_argv.index("--save-root") + 1] == "/tmp/openwam-gjd-explicit"

    run_name_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--method=m1",
        "--ablation=vanilla",
        "--run-name",
        "explicit-gjd-run",
    )
    assert "--save-root" not in run_name_argv
    assert run_name_argv[run_name_argv.index("--run-name") + 1] == "explicit-gjd-run"

    default_root_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--method=m1",
        "--ablation=vanilla",
        "--set",
        "trainer.default_root_dir=/tmp/openwam-gjd-runs",
    )
    assert "--save-root" not in default_root_argv
    assert default_root_argv[default_root_argv.index("--run-name") + 1].startswith("gjd_libero_m1_vanilla_")
    assert "trainer.default_root_dir=/tmp/openwam-gjd-runs" in default_root_argv

    checkpoint_dir_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--method=m5",
        "--ablation=mode_token",
        "--checkpoint-dir",
        "/tmp/openwam-gjd-checkpoints",
    )
    assert "--save-root" not in checkpoint_dir_argv
    assert checkpoint_dir_argv[checkpoint_dir_argv.index("--run-name") + 1].startswith("gjd_libero_m5_mode_token_")
    assert checkpoint_dir_argv[checkpoint_dir_argv.index("--checkpoint-dir") + 1] == "/tmp/openwam-gjd-checkpoints"


def test_unified_gjd_configs_default_to_step3500_video_only_initialization() -> None:
    m1_config = _resolved_config_from_train_argv(
        _launcher_train_argv(
            "scripts/run_gjd_libero.sh",
            "train",
            "--method=m1",
            "--ablation=vanilla",
        )
    )
    m5_config = _resolved_config_from_train_argv(
        _launcher_train_argv(
            "scripts/run_gjd_libero.sh",
            "train",
            "--method=m5",
            "--ablation=vanilla",
        )
    )

    assert "checkpoint_step_3500/transformer" in str(m1_config.backbone.transformer_subdir)
    assert "checkpoint_step_3500/transformer" in str(m5_config.backbone.transformer_subdir)
    assert str(m1_config.backbone.exported_runtime_action_init_mode) == "random"
    assert str(m5_config.backbone.exported_runtime_action_init_mode) == "random"
    assert m1_config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT
    assert m5_config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT


def test_mot_gjd_realtime_launcher_exposes_named_ablation_overrides() -> None:
    argv = _launcher_realtime_argv(
        "scripts/run_libero_mot_gjd_realtime_sandbox.sh",
        env_overrides={"M5_GJD_ABLATION": "mode_token"},
    )

    assert argv[:3] == [
        str(REPO_ROOT / "scripts/run_libero_mot_visualization.py"),
        "--cfg",
        "configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml",
    ]
    assert _arg_value(argv, "--frontend-encode-mode") == "lingbot_streaming_vae"
    assert _arg_value(argv, "--mot-inference-window-size") == "30"
    assert _arg_value(argv, "--startup-model-obs-frames") == "1"
    assert _arg_value(argv, "--startup-env-init-steps") == "5"
    assert _arg_value(argv, "--max-timestep") == "1500"
    assert _arg_value(argv, "--max-chunks") == "100"
    assert "--allow-deprecated-libero-config" not in argv
    assert "policy_variant.generalist_mode_text_token=true" in argv
    for value in FIXED_128_VALUES:
        assert value not in argv


def test_unified_gjd_realtime_launcher_assigns_safe_default_artifact_identity() -> None:
    suffixes: set[str] = set()
    for method in ("m1", "m5"):
        for ablation in ("vanilla", "pure_joint", "mode_token"):
            argv = _launcher_realtime_argv(
                "scripts/run_gjd_libero.sh",
                "rollout",
                f"--method={method}",
                f"--ablation={ablation}",
            )
            suffix = argv[argv.index("--suffix") + 1]
            assert suffix == f"gjd_libero_{method}_{ablation}"
            suffixes.add(suffix)

    assert len(suffixes) == 6

    explicit_suffix = _launcher_realtime_argv(
        "scripts/run_gjd_libero.sh",
        "rollout",
        "--method=m5",
        "--ablation=vanilla",
        "--suffix",
        "manual_suffix",
    )
    assert explicit_suffix.count("--suffix") == 1
    assert explicit_suffix[explicit_suffix.index("--suffix") + 1] == "manual_suffix"


def test_unified_gjd_realtime_launcher_covers_m1_and_m5_ablation_surfaces() -> None:
    expected_cfgs = {
        "m1": "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml",
        "m5": "configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml",
    }
    expected_prob_prefixes = {
        "m1": "policy_variant.joint_denoise_training_mode_probs=",
        "m5": "policy_variant.mot_generalist_training_mode_probs=",
    }
    for method in ("m1", "m5"):
        for ablation in ("vanilla", "pure_joint", "mode_token"):
            argv = _launcher_realtime_argv(
                "scripts/run_gjd_libero.sh",
                "rollout",
                f"--method={method}",
                f"--ablation={ablation}",
            )
            expected_script = (
                str(REPO_ROOT / "scripts/run_libero_realtime_sandbox.py")
                if method == "m1"
                else str(REPO_ROOT / "scripts/run_libero_mot_visualization.py")
            )
            assert argv[:3] == [
                expected_script,
                "--cfg",
                expected_cfgs[method],
            ]
            if method == "m5":
                assert _arg_value(argv, "--frontend-encode-mode") == "lingbot_streaming_vae"
                assert _arg_value(argv, "--mot-inference-window-size") == "30"
                assert _arg_value(argv, "--startup-model-obs-frames") == "1"
                assert _arg_value(argv, "--startup-env-init-steps") == "5"
                assert _arg_value(argv, "--max-timestep") == "1500"
                assert _arg_value(argv, "--max-chunks") == "100"
                assert "--allow-deprecated-libero-config" not in argv
            assert any(token.startswith(expected_prob_prefixes[method]) for token in argv)
            assert f"policy_variant.generalist_mode_text_token={str(ablation == 'mode_token').lower()}" in argv
            _assert_gjd_ablation_config(_resolved_config_from_realtime_argv(argv), method=method, ablation=ablation)
            for value in FIXED_128_VALUES:
                assert value not in argv


def test_m5_gjd_realtime_launcher_rejects_deprecated_frontend_encode_mode() -> None:
    result = _launcher_realtime_result(
        "scripts/run_gjd_libero.sh",
        "rollout",
        "--method=m5",
        "--ablation=pure_joint",
        "--frontend-encode-mode",
        "rolling_offline",
        "--mot-inference-window-size=64",
        "--max-chunks",
        "7",
    )

    assert result.returncode == 2
    assert "requires --frontend-encode-mode lingbot_streaming_vae" in result.stderr
    assert "rolling_offline" in result.stderr


def test_m5_gjd_realtime_launcher_rejects_deprecated_frontend_encode_mode_env() -> None:
    result = _launcher_realtime_result(
        "scripts/run_gjd_libero.sh",
        "rollout",
        "--method=m5",
        "--ablation=vanilla",
        env_overrides={"GJD_M5_FRONTEND_ENCODE_MODE": "rolling_offline"},
    )

    assert result.returncode == 2
    assert "requires --frontend-encode-mode lingbot_streaming_vae" in result.stderr


def test_mot_visualization_deprecates_non_streaming_frontend_encode_modes() -> None:
    mot_viz = _load_mot_visualization_module()

    mot_viz._require_current_frontend_encode_mode(
        "lingbot_streaming_vae",
        allow_deprecated=False,
        source="test",
    )
    with pytest.raises(ValueError, match="rolling_offline.*deprecated"):
        mot_viz._require_current_frontend_encode_mode(
            "rolling_offline",
            allow_deprecated=False,
            source="test",
        )
    mot_viz._require_current_frontend_encode_mode(
        "rolling_offline",
        allow_deprecated=True,
        source="test",
    )


def test_m5_gjd_sampled_eval_wrapper_fails_closed() -> None:
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/run_m5_gjd_libero_eval.sh")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "deprecated and now fails closed" in result.stderr
    assert "run_gjd_libero.sh rollout --method m5" in result.stderr

    help_result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/run_m5_gjd_libero_eval.sh"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "Do not use" in help_result.stdout
    assert "run_libero_sampled_eval.py for GJD" in help_result.stdout


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
        assert "Removed LIBERO M1/M5 config" in result.stderr


@pytest.mark.parametrize("allow_deprecated", (False, True))
def test_removed_mot_wrapper_scripts_always_fail_closed(allow_deprecated: bool) -> None:
    for relative_path in TOP_LEVEL_DEPRECATED_MOT_WRAPPER_STUBS:
        result = _deprecated_launcher_result(
            relative_path,
            allow_deprecated=allow_deprecated,
        )

        assert result.returncode == 2
        assert "Removed LIBERO launcher" in result.stderr
        assert "run_mot_nonjoint_posttrain_libero.sh" in result.stderr
        assert "there is no runtime opt-in" in result.stderr


def test_removed_mot_configs_reject_explicit_opt_in() -> None:
    env = os.environ.copy()
    env.update(
        {
            "OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG": "true",
            "OPEN_WAM_PRINT_TRAIN_ARGV": "1",
            "CONFIG_NAME": "mot_libero_latent_local_joint",
            "NGPU": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/run_mot_posttrain_libero.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Removed LIBERO M1/M5 config" in result.stderr
    assert "Git history retains the historical YAML" in result.stderr


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
