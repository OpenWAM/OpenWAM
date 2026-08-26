from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from open_wam.configs import load_experiment_config
from open_wam.configs.enums import (
    ContextConditionLatentSource,
    DynamicsObjective,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    ProprioContextMode,
    SampleOrderMode,
    SampleTargetAlignment,
    SampleWeightMode,
    VideoActionSequenceContract,
    WindowSamplingMode,
)
from open_wam.utils.config_overrides import (
    apply_config_overrides,
    parse_override_assignments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_HELPER_PATH = REPO_ROOT / "scripts/training_launcher_common.sh"
LIBERO_COMPATIBILITY_PATH = REPO_ROOT / "scripts/libero_legacy_compatibility.sh"
POSTTRAIN_LAUNCHERS = (
    "scripts/run_causal_video_prediction_posttrain_libero.sh",
    "scripts/run_dual_expert_posttrain_libero.sh",
    "scripts/run_parallel_stream_posttrain_libero.sh",
)
TOP_LEVEL_DEPRECATED_DUAL_EXPERT_WRAPPER_STUBS = (
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
STANDARD_POLICY_PROGRAMS = tuple(
    (f"{architecture}_libero_{program}", program)
    for architecture in ("dual_expert", "parallel_stream")
    for program in (
        "video_then_action",
        "action_then_video",
        "joint",
        "decoupled_same_step",
        "video_noisy_to_action",
        "action_noisy_to_video",
    )
)
SEQUENCE_CONTRACT_POLICY_KEYS = {
    "proprio_context_mode",
    "context_condition_latent_source",
    "history_stream_visibility",
    "use_condition_latents",
    "require_condition_latents",
}
SEQUENCE_CONTRACT_SAMPLE_KEYS = {
    "condition_source_frame_offset",
    "start_padding_frames",
    "target_alignment",
    "rollout_context_policy",
}


def _normalized_legacy_config_name(config_name: str) -> str:
    command = f"""
set -euo pipefail
source {str(LIBERO_COMPATIBILITY_PATH)!r}
open_wam_normalize_config_name {config_name!r}
"""
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _reject_config_override_result(*args: str) -> subprocess.CompletedProcess[str]:
    command = f"""
set -euo pipefail
source {str(TRAINING_HELPER_PATH)!r}
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


def _launcher_train_result(
    relative_path: str,
    *script_args: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"OPEN_WAM_PRINT_TRAIN_ARGV": "1", "NGPU": "1"})
    return subprocess.run(
        ["bash", str(REPO_ROOT / relative_path), *script_args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_posttrain_launchers_delegate_to_shared_process_owner() -> None:
    expected_projects = {
        "scripts/run_causal_video_prediction_posttrain_libero.sh": "openwam-causal-video-libero",
        "scripts/run_parallel_stream_posttrain_libero.sh": "lingbot-va-posttrain-libero",
        "scripts/run_dual_expert_posttrain_libero.sh": "openwam-dual-expert-libero",
    }
    helper_source = TRAINING_HELPER_PATH.read_text(encoding="utf-8")

    assert set(expected_projects) == set(POSTTRAIN_LAUNCHERS)
    assert helper_source.count("open_wam_launch_training()") == 1
    assert helper_source.count("python -m torch.distributed.run") == 1
    assert helper_source.count("python -m open_wam.cli.train") == 1
    assert "libero" not in helper_source.lower()
    assert "fixed128" not in helper_source.lower()
    assert "sample_construction" not in helper_source
    for relative_path, project in expected_projects.items():
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "training_launcher_common.sh" in source
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
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_dual_expert_posttrain_libero.sh"),
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
        "open_wam.cli.train",
        "--config-name",
        "dual_expert_libero_joint",
        "--devices",
        "4",
        "--num-steps",
        "17",
    ]
    assert "TOKENIZERS_PARALLELISM=false" in lines
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in lines
    assert "WANDB_MODE=online" in lines
    assert "WANDB_PROJECT=openwam-dual-expert-libero" in lines


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


def _load_dual_expert_visualization_module():
    from open_wam.evals import libero_dual_expert_rollout

    return libero_dual_expert_rollout


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


def _assert_gjd_ablation_config(
    config,
    *,
    architecture: str,
    ablation: str,
    real_demo_weight: float = 1.0,
    counterfactual_weight: float = 1.0,
) -> None:
    if architecture not in {"parallel_stream", "dual_expert"}:
        raise AssertionError(f"Unexpected architecture {architecture!r}")
    joint = DynamicsObjective.JOINT
    fdm = DynamicsObjective.ACTION_CONDITIONED_VIDEO
    idm = DynamicsObjective.VIDEO_CONDITIONED_ACTION
    mixture = config.data.dynamics_routing
    probs = mixture.mode_probabilities()
    if not mixture.active_routes:
        probs[joint] = 1.0
    route_weights = {
        (route.source.value, route.mode): route.weight
        for route in mixture.active_routes
    }

    if ablation == "pure_joint":
        assert probs[joint] == 1.0
        assert probs[fdm] == 0.0
        assert probs[idm] == 0.0
        assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
        assert config.data.dynamics_routing.train_latent_root is None
        assert config.data.dynamics_routing.val_latent_root is None
        assert config.validation.auxiliary_tasks == ()
    elif ablation == "pure_fdm":
        assert probs[joint] == 0.0
        assert probs[fdm] == 1.0
        assert probs[idm] == 0.0
        assert route_weights == {
            ("real_demo", fdm): real_demo_weight,
            ("counterfactual_dynamics", fdm): counterfactual_weight,
        }
        assert [task.mode_override for task in config.validation.auxiliary_tasks] == [
            fdm
        ]
    elif ablation == "pure_idm":
        assert probs[joint] == 0.0
        assert probs[fdm] == 0.0
        assert probs[idm] == 1.0
        assert route_weights == {
            ("real_demo", idm): real_demo_weight,
            ("counterfactual_dynamics", idm): counterfactual_weight,
        }
        assert [task.mode_override for task in config.validation.auxiliary_tasks] == [
            idm
        ]
    else:
        assert probs[joint] == 0.6
        assert probs[fdm] == 0.2
        assert probs[idm] == 0.2
        assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
        assert config.data.dynamics_routing.train_latent_root is not None
        assert config.data.dynamics_routing.val_latent_root is not None
    assert config.policy_variant.generalist_mode_text_token is (ablation == "mode_token")


def _gjd_raw_config(*, architecture: str) -> dict:
    if architecture == "parallel_stream":
        config_path = (
            REPO_ROOT
            / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
        )
    elif architecture == "dual_expert":
        config_path = (
            REPO_ROOT / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
        )
    else:
        raise AssertionError(f"Unexpected architecture {architecture!r}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _assert_high_success_planning_config(
    config,
    *,
    expected_num_steps: int = 10000,
) -> None:
    sample = config.data.sample_construction
    policy = config.policy_variant

    assert config.data.replay_status_policy.value == "include_all"
    assert config.data.val_replay_status_policy is None
    assert config.data.require_replay_status is False
    assert config.data.val_require_replay_status is False
    assert config.data.train_batch_size == 1
    assert sample.mode == WindowSamplingMode.UNIFORM_SEGMENT
    assert sample.sample_order_mode == SampleOrderMode.REPLACEMENT
    assert sample.chunk_size == 4
    assert sample.window_size == 64
    assert sample.randomize_geometry is True
    assert sample.segment_min_frames == 1000
    assert sample.segment_max_frames == 1000
    assert sample.segment_length_stride == 1
    assert sample.segment_locality_block_size == 1
    assert sample.randomize_segment_length is False
    assert sample.randomize_segment_start is False
    assert sample.require_full_segment is True
    assert sample.task_start_power == 0.0
    assert sample.demo_count_power == 0.0
    assert sample.trajectory_start_power == 0.0
    assert sample.sample_weight_mode == SampleWeightMode.UNIFORM
    assert policy.sequence_contract == (
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert policy.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert policy.context_condition_latent_source == (
        ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert policy.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert policy.use_condition_latents is True
    assert policy.require_condition_latents is True
    assert policy.noisy_video_condition_prob == pytest.approx(0.5)
    assert sample.condition_source_frame_offset == -1
    assert sample.start_padding_frames == 0
    assert sample.target_alignment == SampleTargetAlignment.LEGACY
    assert config.training.chunk_size == 4
    assert config.training.window_size == 64
    assert config.training.sample_loss_weight_mode.value == "none"
    assert config.training.gradient_accumulation_steps == 10
    assert config.training.num_steps == expected_num_steps
    assert config.trainer.checkpoint_mode.value == "full_training_state"


@pytest.mark.parametrize(
    ("config_name", "program"),
    STANDARD_POLICY_PROGRAMS,
)
def test_policy_program_configs_own_architecture_neutral_high_success_recipe(
    config_name: str,
    program: str,
) -> None:
    config_path = REPO_ROOT / "configs/experiments" / f"{config_name}.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    sample = raw["data"]["sample_construction"]
    policy = raw["policy_variant"]

    assert raw["data"]["replay_status_policy"] == "include_all"
    assert raw["data"]["val_replay_status_policy"] is None
    assert raw["data"]["require_replay_status"] is False
    assert raw["data"]["val_require_replay_status"] is False
    assert raw["data"]["train_batch_size"] == 1
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
    assert SEQUENCE_CONTRACT_SAMPLE_KEYS.isdisjoint(sample)
    assert policy["program"] == program
    assert policy["sequence_contract"] == (
        "legacy_prefix_single_frame_perchunk_proprio"
    )
    assert SEQUENCE_CONTRACT_POLICY_KEYS.isdisjoint(policy)
    assert policy["noisy_video_condition_prob"] == pytest.approx(0.5)
    assert raw["training"]["window_size"] == 64
    assert raw["training"]["sample_loss_weight_mode"] == "none"
    assert raw["training"]["gradient_accumulation_steps"] == 10
    assert raw["training"]["num_steps"] == 10000

    _assert_high_success_planning_config(load_experiment_config(config_path))


def _assert_gjd_fullseg_w64_raw_config(raw: dict) -> None:
    sample = raw["data"]["sample_construction"]

    assert raw["data"]["replay_status_policy"] == "include_all"
    assert raw["data"]["val_replay_status_policy"] is None
    assert raw["data"]["require_replay_status"] is False
    assert raw["data"]["val_require_replay_status"] is False
    assert raw["data"]["train_batch_size"] == 1
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
    assert raw["training"]["gradient_accumulation_steps"] == 10
    assert raw["training"]["num_steps"] == 20000
    assert raw["policy_variant"]["joint_timestep_coupling"] == "independent"
    assert "dynamics_routing_requirement" not in raw["policy_variant"]
    assert raw["policy_variant"]["generalist_mode_text_token"] is False
    assert raw["policy_variant"]["sequence_contract"] == (
        "legacy_prefix_single_frame_perchunk_proprio"
    )
    assert raw["policy_variant"]["noisy_video_condition_prob"] == pytest.approx(0.5)
    assert SEQUENCE_CONTRACT_POLICY_KEYS.isdisjoint(raw["policy_variant"])
    assert raw["trainer"]["checkpoint_mode"] == "full_training_state"
    assert raw["trainer"]["save_interval"] == 100
    assert raw["trainer"]["max_checkpoints_to_keep"] == 3
    mixture = raw["data"]["dynamics_routing"]
    assert mixture["train_latent_root"] == "${paths.datasets.libero_gjd_counterfactual_train_latent_root}"
    assert mixture["val_latent_root"] == "${paths.datasets.libero_gjd_counterfactual_val_latent_root}"
    assert mixture["routes"] == [
        {"source": "real_demo", "mode": "joint", "weight": 0.6},
        {"source": "real_demo", "mode": "action_conditioned_video", "weight": 0.1},
        {"source": "real_demo", "mode": "video_conditioned_action", "weight": 0.1},
        {"source": "counterfactual_dynamics", "mode": "action_conditioned_video", "weight": 0.1},
        {"source": "counterfactual_dynamics", "mode": "video_conditioned_action", "weight": 0.1},
    ]


def _planning_recipe_snapshot(config) -> tuple:
    sample = config.data.sample_construction
    policy = config.policy_variant
    return (
        config.data.replay_status_policy,
        config.data.val_replay_status_policy,
        config.data.require_replay_status,
        config.data.val_require_replay_status,
        config.data.train_batch_size,
        sample.mode,
        sample.sample_order_mode,
        sample.chunk_size,
        sample.window_size,
        sample.randomize_geometry,
        sample.segment_min_frames,
        sample.segment_max_frames,
        sample.segment_length_stride,
        sample.segment_locality_block_size,
        sample.randomize_segment_length,
        sample.randomize_segment_start,
        sample.require_full_segment,
        sample.task_start_power,
        sample.demo_count_power,
        sample.trajectory_start_power,
        sample.sample_weight_mode,
        sample.condition_source_frame_offset,
        sample.start_padding_frames,
        sample.target_alignment,
        policy.sequence_contract,
        policy.proprio_context_mode,
        policy.context_condition_latent_source,
        policy.history_stream_visibility,
        policy.use_condition_latents,
        policy.require_condition_latents,
        policy.noisy_video_condition_prob,
        config.training.chunk_size,
        config.training.window_size,
        config.training.sample_loss_weight_mode,
        config.training.gradient_accumulation_steps,
    )


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
def test_gjd_real_joint_uses_canonical_joint_planning_recipe(architecture: str) -> None:
    joint = load_experiment_config(
        REPO_ROOT / "configs/experiments" / f"{architecture}_libero_joint.yaml"
    )
    gjd = load_experiment_config(
        REPO_ROOT
        / "configs/experiments"
        / f"{architecture}_libero_generalist_joint_denoising.yaml"
    )

    # Denoising-mode probabilities, timestep coupling, and total step budget
    # remain GJD method knobs; the shipped planning recipe and sequence semantics do not.
    assert _planning_recipe_snapshot(gjd) == _planning_recipe_snapshot(joint)


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


def test_sampling_geometry_is_owned_by_configs_not_launcher_name_dispatch() -> None:
    helper_source = TRAINING_HELPER_PATH.read_text(encoding="utf-8")

    for value in FIXED_128_VALUES:
        assert value not in helper_source


def test_parallel_stream_gjd_uses_shared_planning_recipe() -> None:
    argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=parallel_stream",
        "--ablation=vanilla",
    )

    assert argv[:4] == [
        "--config-name",
        "parallel_stream_libero_generalist_joint_denoising",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value not in argv

    raw = _gjd_raw_config(architecture="parallel_stream")
    _assert_gjd_fullseg_w64_raw_config(raw)
    assert "proprio_context_mode" not in raw["policy_variant"]
    assert raw["policy_variant"]["attn_window"] == 30
    assert "preserve_video_pretrain_history" not in raw["policy_variant"]


def test_gjd_launcher_help_describes_shared_planning_and_conditional_contracts() -> None:
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/run_gjd_libero.sh"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "architecture choices under one GJD" in result.stdout
    assert "GJD real-joint samples use the same legacy-prefix" in result.stdout
    assert "context_condition_latent_source=single_frame_condition_latent" in result.stdout
    assert "Dynamics-routed FDM/IDM samples remain target-only" in result.stdout
    assert "Both architectures consume the in-sequence t0" in result.stdout
    assert "external planning-prefix path for these rows" in result.stdout
    assert "parallel_stream rollout uses run_libero_realtime_sandbox.py" in result.stdout


def test_parallel_stream_gjd_launcher_prints_shared_contract_notice() -> None:
    env = os.environ.copy()
    env.update({"OPEN_WAM_PRINT_TRAIN_ARGV": "1", "NGPU": "1"})
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_gjd_libero.sh"),
            "train",
            "--architecture=parallel_stream",
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
        "parallel_stream_libero_generalist_joint_denoising",
    ]
    assert "shared GJD defaults" in result.stderr
    assert "real_joint uses the configured full-segment W64 recipe" in result.stderr
    assert "Conditional FDM/IDM uses target-only t0 + future layout" in result.stderr
    assert "exposes only t0 as clean history" in result.stderr
    assert "Both architectures bypass external planning-prefix handling" in result.stderr
    assert "known parallel_stream GJD issue" not in result.stderr


def test_dual_expert_gjd_config_deprecates_fixed128_for_legacy_prefix_fullseg_w64() -> None:
    argv = _launcher_train_argv("scripts/run_m5_gjd_posttrain_libero.sh")

    assert argv[:4] == [
        "--config-name",
        "dual_expert_libero_generalist_joint_denoising",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value not in argv

    raw = _gjd_raw_config(architecture="dual_expert")
    _assert_gjd_fullseg_w64_raw_config(raw)
    assert raw["policy_variant"]["sequence_contract"] == "legacy_prefix_single_frame_perchunk_proprio"
    assert "proprio_context_mode" not in raw["policy_variant"]
    assert raw["policy_variant"]["noisy_video_condition_prob"] == 0.5


def test_m5_gjd_pure_joint_launcher_keeps_fullseg_w64_sampler() -> None:
    argv = _launcher_train_argv(
        "scripts/run_m5_gjd_posttrain_libero.sh",
        env_overrides={"M5_GJD_VARIANT": "pure_joint"},
    )

    for value in FIXED_128_VALUES:
        assert value not in argv
    _assert_gjd_ablation_config(
        _resolved_config_from_train_argv(argv),
        architecture="dual_expert",
        ablation="pure_joint",
    )


def test_m5_gjd_mode_token_launcher_keeps_fullseg_w64_sampler() -> None:
    argv = _launcher_train_argv(
        "scripts/run_m5_gjd_posttrain_libero.sh",
        env_overrides={"M5_GJD_VARIANT": "mode_token"},
    )

    for value in FIXED_128_VALUES:
        assert value not in argv
    assert "policy_variant.generalist_mode_text_token=true" in argv
    _assert_gjd_ablation_config(
        _resolved_config_from_train_argv(argv),
        architecture="dual_expert",
        ablation="mode_token",
    )


def test_legacy_config_name_normalization_is_isolated_from_training_launcher() -> None:
    assert _normalized_legacy_config_name(
        "parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
    ) == "parallel_stream_libero_video_then_action"


def test_policy_program_aliases_only_normalize_identity() -> None:
    aliases = {
        "mot_libero_latent_local_video_then_action_heng_compatible.yaml": (
            "dual_expert_libero_video_then_action"
        ),
        "parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml": (
            "parallel_stream_libero_video_then_action"
        ),
        "/tmp/configs/experiments/parallel_stream_libero_joint.yml": (
            "parallel_stream_libero_joint"
        ),
    }
    for config_name, expected in aliases.items():
        assert _normalized_legacy_config_name(config_name) == expected


def test_launchers_reject_late_cli_config_overrides() -> None:
    for args in (
        ("--config-name", "dual_expert_libero_latent_local_joint"),
        ("--config-name=dual_expert_libero_latent_local_joint",),
        ("--cfg", "configs/experiments/deprecated/dual_expert_libero_latent_local_joint.yaml"),
        ("--config=configs/experiments/deprecated/dual_expert_libero_latent_local_joint.yaml",),
    ):
        result = _reject_config_override_result(*args)

        assert result.returncode == 2
        assert "set CONFIG_NAME=... instead" in result.stderr

    assert _reject_config_override_result("--set", "training.num_steps=1").returncode == 0


def test_dual_expert_posttrain_launcher_uses_canonical_high_success_joint_config() -> None:
    argv = _launcher_train_argv("scripts/run_dual_expert_posttrain_libero.sh")

    assert argv[:4] == [
        "--config-name",
        "dual_expert_libero_joint",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value not in argv
    _assert_high_success_planning_config(_resolved_config_from_train_argv(argv))


def test_parallel_stream_posttrain_launcher_uses_canonical_high_success_joint_config() -> None:
    argv = _launcher_train_argv("scripts/run_parallel_stream_posttrain_libero.sh")

    assert argv[:4] == [
        "--config-name",
        "parallel_stream_libero_joint",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value not in argv
    _assert_high_success_planning_config(_resolved_config_from_train_argv(argv))


def test_legacy_nonjoint_launcher_preserves_video_then_action_default() -> None:
    argv = _launcher_train_argv("scripts/run_mot_nonjoint_posttrain_libero.sh")

    assert argv[:4] == [
        "--config-name",
        "dual_expert_libero_video_then_action",
        "--devices",
        "1",
    ]
    for value in FIXED_128_VALUES:
        assert value not in argv
    _assert_high_success_planning_config(_resolved_config_from_train_argv(argv))


def test_legacy_m5_gjd_launcher_delegates_named_ablation_overrides() -> None:
    vanilla = _launcher_train_argv("scripts/run_mot_gjd_posttrain_libero.sh")
    assert vanilla[:4] == [
        "--config-name",
        "dual_expert_libero_generalist_joint_denoising",
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
    assert "data.dynamics_routing.routes=[]" in pure_joint
    pure_joint_config = _resolved_config_from_train_argv(pure_joint)
    _assert_gjd_ablation_config(
        pure_joint_config,
        architecture="dual_expert",
        ablation="pure_joint",
    )

    mode_token = _launcher_train_argv(
        "scripts/run_mot_gjd_posttrain_libero.sh",
        env_overrides={"M5_GJD_ABLATION": "mode_token"},
    )
    assert not any(
        token.startswith("data.dynamics_routing.routes=") for token in mode_token
    )
    assert "policy_variant.generalist_mode_text_token=true" in mode_token
    _assert_gjd_ablation_config(
        _resolved_config_from_train_argv(mode_token),
        architecture="dual_expert",
        ablation="mode_token",
    )


def test_unified_gjd_train_launcher_covers_architecture_and_ablation_surfaces() -> None:
    expected_configs = {
        "parallel_stream": "parallel_stream_libero_generalist_joint_denoising",
        "dual_expert": "dual_expert_libero_generalist_joint_denoising",
    }
    for architecture in ("parallel_stream", "dual_expert"):
        for ablation in ("vanilla", "pure_joint", "mode_token"):
            argv = _launcher_train_argv(
                "scripts/run_gjd_libero.sh",
                "train",
                f"--architecture={architecture}",
                f"--ablation={ablation}",
            )
            assert argv[:4] == [
                "--config-name",
                expected_configs[architecture],
                "--devices",
                "1",
            ]
            assert any(
                token.startswith("data.dynamics_routing.routes=")
                for token in argv
            ) is (ablation == "pure_joint")
            assert f"policy_variant.generalist_mode_text_token={str(ablation == 'mode_token').lower()}" in argv
            _assert_gjd_ablation_config(
                _resolved_config_from_train_argv(argv),
                architecture=architecture,
                ablation=ablation,
            )
            for value in FIXED_128_VALUES:
                assert value not in argv


@pytest.mark.parametrize("architecture", ["parallel_stream", "dual_expert"])
@pytest.mark.parametrize("ablation", ["pure_fdm", "pure_idm"])
def test_unified_gjd_train_launcher_supports_pure_conditional_source_ratios(
    architecture: str,
    ablation: str,
) -> None:
    argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        f"--architecture={architecture}",
        f"--ablation={ablation}",
        "--real-demo-weight=3",
        "--counterfactual-weight=1",
    )

    config = _resolved_config_from_train_argv(argv)
    _assert_gjd_ablation_config(
        config,
        architecture=architecture,
        ablation=ablation,
        real_demo_weight=3.0,
        counterfactual_weight=1.0,
    )
    assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
    assert config.data.sample_construction.window_size == 64
    assert config.training.window_size == 64


@pytest.mark.parametrize("architecture", ("parallel_stream", "dual_expert"))
def test_unified_gjd_launcher_rejects_pure_conditional_rollout(
    architecture: str,
) -> None:
    result = _launcher_train_result(
        "scripts/run_gjd_libero.sh",
        "rollout",
        f"--architecture={architecture}",
        "--ablation=pure_fdm",
    )

    assert result.returncode == 2
    assert "offline conditional mode" in result.stderr


def test_unified_gjd_launcher_rejects_source_ratio_for_nonconditional_ablation() -> None:
    result = _launcher_train_result(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=dual_expert",
        "--ablation=vanilla",
        "--real-demo-weight=3",
    )

    assert result.returncode == 2
    assert "apply only to pure_fdm or pure_idm" in result.stderr


def test_unified_gjd_train_launcher_keeps_historical_method_aliases() -> None:
    for method, architecture in (("m1", "parallel_stream"), ("m5", "dual_expert")):
        legacy_argv = _launcher_train_argv(
            "scripts/run_gjd_libero.sh",
            "train",
            f"--method={method}",
            "--ablation=mode_token",
        )
        canonical_argv = _launcher_train_argv(
            "scripts/run_gjd_libero.sh",
            "train",
            f"--architecture={architecture}",
            "--ablation=mode_token",
        )
        # Default run IDs contain timestamps and PIDs; everything semantic must match.
        legacy_argv[legacy_argv.index("--save-root") + 1] = "<default-save-root>"
        canonical_argv[canonical_argv.index("--save-root") + 1] = "<default-save-root>"
        assert legacy_argv == canonical_argv


def test_unified_gjd_train_launcher_assigns_safe_default_run_identity() -> None:
    vanilla = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=dual_expert",
        "--ablation=vanilla",
    )
    pure_joint = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=dual_expert",
        "--ablation=pure_joint",
    )

    vanilla_save_root = vanilla[vanilla.index("--save-root") + 1]
    pure_joint_save_root = pure_joint[pure_joint.index("--save-root") + 1]
    assert vanilla_save_root.startswith("runs/gjd_libero_dual_expert_vanilla_")
    assert pure_joint_save_root.startswith("runs/gjd_libero_dual_expert_pure_joint_")
    assert vanilla_save_root != pure_joint_save_root


def test_unified_gjd_train_launcher_assigns_comparison_wandb_project() -> None:
    default_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=parallel_stream",
        "--ablation=vanilla",
    )
    assert default_argv[default_argv.index("--wandb-project") + 1] == "openwam-gjd-libero"

    explicit_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=parallel_stream",
        "--ablation=vanilla",
        "--wandb-project",
        "custom-project",
    )
    assert explicit_argv.count("--wandb-project") == 1
    assert explicit_argv[explicit_argv.index("--wandb-project") + 1] == "custom-project"

    env_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=dual_expert",
        "--ablation=mode_token",
        env_overrides={"WANDB_PROJECT": "env-project"},
    )
    assert "--wandb-project" not in env_argv


def test_unified_gjd_train_launcher_preserves_explicit_output_identity() -> None:
    save_root_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=parallel_stream",
        "--ablation=vanilla",
        "--save-root",
        "/tmp/openwam-gjd-explicit",
    )
    assert save_root_argv.count("--save-root") == 1
    assert save_root_argv[save_root_argv.index("--save-root") + 1] == "/tmp/openwam-gjd-explicit"

    run_name_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=parallel_stream",
        "--ablation=vanilla",
        "--run-name",
        "explicit-gjd-run",
    )
    assert "--save-root" not in run_name_argv
    assert run_name_argv[run_name_argv.index("--run-name") + 1] == "explicit-gjd-run"

    default_root_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=parallel_stream",
        "--ablation=vanilla",
        "--set",
        "trainer.default_root_dir=/tmp/openwam-gjd-runs",
    )
    assert "--save-root" not in default_root_argv
    assert default_root_argv[default_root_argv.index("--run-name") + 1].startswith(
        "gjd_libero_parallel_stream_vanilla_"
    )
    assert "trainer.default_root_dir=/tmp/openwam-gjd-runs" in default_root_argv

    checkpoint_dir_argv = _launcher_train_argv(
        "scripts/run_gjd_libero.sh",
        "train",
        "--architecture=dual_expert",
        "--ablation=mode_token",
        "--checkpoint-dir",
        "/tmp/openwam-gjd-checkpoints",
    )
    assert "--save-root" not in checkpoint_dir_argv
    assert checkpoint_dir_argv[checkpoint_dir_argv.index("--run-name") + 1].startswith(
        "gjd_libero_dual_expert_mode_token_"
    )
    assert checkpoint_dir_argv[checkpoint_dir_argv.index("--checkpoint-dir") + 1] == "/tmp/openwam-gjd-checkpoints"


def test_unified_gjd_configs_default_to_step3500_video_only_initialization() -> None:
    parallel_stream_config = _resolved_config_from_train_argv(
        _launcher_train_argv(
            "scripts/run_gjd_libero.sh",
            "train",
            "--architecture=parallel_stream",
            "--ablation=vanilla",
        )
    )
    dual_expert_config = _resolved_config_from_train_argv(
        _launcher_train_argv(
            "scripts/run_gjd_libero.sh",
            "train",
            "--architecture=dual_expert",
            "--ablation=vanilla",
        )
    )

    assert "checkpoint_step_3500/transformer" in str(
        parallel_stream_config.backbone.runtime_backbone_artifact_path
    )
    assert "checkpoint_step_3500/transformer" in str(
        dual_expert_config.backbone.runtime_backbone_artifact_path
    )
    assert str(parallel_stream_config.backbone.exported_runtime_action_init_mode) == "random"
    assert str(dual_expert_config.backbone.exported_runtime_action_init_mode) == "random"
    assert (
        parallel_stream_config.policy_variant.joint_timestep_coupling
        == JointTimestepCoupling.INDEPENDENT
    )
    assert (
        dual_expert_config.policy_variant.joint_timestep_coupling
        == JointTimestepCoupling.INDEPENDENT
    )


def test_legacy_m5_gjd_realtime_launcher_delegates_named_ablation_overrides() -> None:
    argv = _launcher_realtime_argv(
        "scripts/run_libero_mot_gjd_realtime_sandbox.sh",
        env_overrides={"M5_GJD_ABLATION": "mode_token"},
    )

    assert argv[:3] == [
        str(REPO_ROOT / "scripts/run_libero_dual_expert_visualization.py"),
        "--cfg",
        "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml",
    ]
    assert _arg_value(argv, "--frontend-encode-mode") == "lingbot_streaming_vae"
    assert _arg_value(argv, "--dual-expert-inference-window-size") == "30"
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
    for architecture in ("parallel_stream", "dual_expert"):
        for ablation in ("vanilla", "pure_joint", "mode_token"):
            argv = _launcher_realtime_argv(
                "scripts/run_gjd_libero.sh",
                "rollout",
                f"--architecture={architecture}",
                f"--ablation={ablation}",
            )
            suffix = argv[argv.index("--suffix") + 1]
            assert suffix == f"gjd_libero_{architecture}_{ablation}"
            suffixes.add(suffix)

    assert len(suffixes) == 6

    explicit_suffix = _launcher_realtime_argv(
        "scripts/run_gjd_libero.sh",
        "rollout",
        "--architecture=dual_expert",
        "--ablation=vanilla",
        "--suffix",
        "manual_suffix",
    )
    assert explicit_suffix.count("--suffix") == 1
    assert explicit_suffix[explicit_suffix.index("--suffix") + 1] == "manual_suffix"


def test_unified_gjd_realtime_launcher_covers_architecture_and_ablation_surfaces() -> None:
    expected_cfgs = {
        "parallel_stream": "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml",
        "dual_expert": "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml",
    }
    for architecture in ("parallel_stream", "dual_expert"):
        for ablation in ("vanilla", "pure_joint", "mode_token"):
            argv = _launcher_realtime_argv(
                "scripts/run_gjd_libero.sh",
                "rollout",
                f"--architecture={architecture}",
                f"--ablation={ablation}",
            )
            expected_script = (
                str(REPO_ROOT / "scripts/run_libero_realtime_sandbox.py")
                if architecture == "parallel_stream"
                else str(REPO_ROOT / "scripts/run_libero_dual_expert_visualization.py")
            )
            assert argv[:3] == [
                expected_script,
                "--cfg",
                expected_cfgs[architecture],
            ]
            if architecture == "dual_expert":
                assert _arg_value(argv, "--frontend-encode-mode") == "lingbot_streaming_vae"
                assert _arg_value(argv, "--dual-expert-inference-window-size") == "30"
                assert _arg_value(argv, "--startup-model-obs-frames") == "1"
                assert _arg_value(argv, "--startup-env-init-steps") == "5"
                assert _arg_value(argv, "--max-timestep") == "1500"
                assert _arg_value(argv, "--max-chunks") == "100"
                assert "--allow-deprecated-libero-config" not in argv
            assert any(
                token.startswith("data.dynamics_routing.routes=")
                for token in argv
            ) is (ablation == "pure_joint")
            assert f"policy_variant.generalist_mode_text_token={str(ablation == 'mode_token').lower()}" in argv
            _assert_gjd_ablation_config(
                _resolved_config_from_realtime_argv(argv),
                architecture=architecture,
                ablation=ablation,
            )
            for value in FIXED_128_VALUES:
                assert value not in argv


def test_dual_expert_gjd_realtime_launcher_rejects_deprecated_frontend_encode_mode() -> None:
    result = _launcher_realtime_result(
        "scripts/run_gjd_libero.sh",
        "rollout",
        "--architecture=dual_expert",
        "--ablation=pure_joint",
        "--frontend-encode-mode",
        "rolling_offline",
        "--dual-expert-inference-window-size=64",
        "--max-chunks",
        "7",
    )

    assert result.returncode == 2
    assert "requires --frontend-encode-mode lingbot_streaming_vae" in result.stderr
    assert "rolling_offline" in result.stderr


def test_dual_expert_gjd_realtime_launcher_rejects_deprecated_frontend_encode_mode_env() -> None:
    result = _launcher_realtime_result(
        "scripts/run_gjd_libero.sh",
        "rollout",
        "--architecture=dual_expert",
        "--ablation=vanilla",
        env_overrides={"GJD_M5_FRONTEND_ENCODE_MODE": "rolling_offline"},
    )

    assert result.returncode == 2
    assert "requires --frontend-encode-mode lingbot_streaming_vae" in result.stderr


def test_dual_expert_visualization_deprecates_non_streaming_frontend_encode_modes() -> None:
    from open_wam.evals import libero_dual_expert_runtime

    libero_dual_expert_runtime._require_current_frontend_encode_mode(
        "lingbot_streaming_vae",
        allow_deprecated=False,
        source="test",
    )
    with pytest.raises(ValueError, match="rolling_offline.*deprecated"):
        libero_dual_expert_runtime._require_current_frontend_encode_mode(
            "rolling_offline",
            allow_deprecated=False,
            source="test",
        )
    libero_dual_expert_runtime._require_current_frontend_encode_mode(
        "rolling_offline",
        allow_deprecated=True,
        source="test",
    )


def test_dual_expert_launchers_reject_legacy_configs_by_default() -> None:
    for relative_path, config_name in (
        ("scripts/run_dual_expert_posttrain_libero.sh", "dual_expert_libero_latent_local_joint"),
        (
            "scripts/run_mot_nonjoint_posttrain_libero.sh",
            "dual_expert_libero_latent_local_full_segment_non_joint_aligned",
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
        assert "Removed LIBERO policy config" in result.stderr


@pytest.mark.parametrize("allow_deprecated", (False, True))
def test_removed_dual_expert_wrapper_scripts_always_fail_closed(allow_deprecated: bool) -> None:
    for relative_path in TOP_LEVEL_DEPRECATED_DUAL_EXPERT_WRAPPER_STUBS:
        result = _deprecated_launcher_result(
            relative_path,
            allow_deprecated=allow_deprecated,
        )

        assert result.returncode == 2
        assert "Removed LIBERO launcher" in result.stderr
        assert "run_dual_expert_posttrain_libero.sh" in result.stderr
        assert "there is no runtime opt-in" in result.stderr


def test_removed_dual_expert_configs_reject_explicit_opt_in() -> None:
    env = os.environ.copy()
    env.update(
        {
            "OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG": "true",
            "OPEN_WAM_PRINT_TRAIN_ARGV": "1",
            "CONFIG_NAME": "dual_expert_libero_latent_local_joint",
            "NGPU": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/run_dual_expert_posttrain_libero.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Removed LIBERO policy config" in result.stderr
    assert "Git history retains the historical YAML" in result.stderr


def test_libero_posttrain_launcher_can_print_exact_train_argv_without_running() -> None:
    env = os.environ.copy()
    env.update(
        {
            "OPEN_WAM_PRINT_TRAIN_ARGV": "1",
            "CONFIG_NAME": "parallel_stream_libero_joint",
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
        "parallel_stream_libero_joint",
        "--devices",
        "4",
    ]
    for value in FIXED_128_VALUES:
        assert value not in argv
    _assert_high_success_planning_config(_resolved_config_from_train_argv(argv))
    assert argv[-4:] == ["--num-steps", "4000", "--set", "training.learning_rate=2e-5"]


def test_libero_posttrain_launcher_dry_run_uses_python3_without_venv_path() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "OPEN_WAM_PRINT_TRAIN_ARGV": "1",
        "CONFIG_NAME": "parallel_stream_libero_joint",
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
        "parallel_stream_libero_joint",
        "--devices",
        "4",
    ]
    for value in FIXED_128_VALUES:
        assert value not in argv
    _assert_high_success_planning_config(_resolved_config_from_train_argv(argv))
