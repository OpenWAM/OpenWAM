from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_wam.configs import (
    CurrentBlockCoupling,
    GeneralistDenoisingMode,
    VideoActionProgram,
    load_experiment_config,
    resolve_video_action_program_semantics,
    validate_config_file,
)
from open_wam.utils.config_overrides import apply_config_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]

COUPLING_PROGRAMS = (
    VideoActionProgram.VIDEO_THEN_ACTION,
    VideoActionProgram.ACTION_THEN_VIDEO,
    VideoActionProgram.JOINT,
    VideoActionProgram.DECOUPLED_SAME_STEP,
    VideoActionProgram.VIDEO_NOISY_TO_ACTION,
    VideoActionProgram.ACTION_NOISY_TO_VIDEO,
)


@pytest.mark.parametrize("program", COUPLING_PROGRAMS)
def test_video_action_program_resolves_same_named_coupling(
    program: VideoActionProgram,
) -> None:
    resolved_program, coupling = resolve_video_action_program_semantics(
        program=program,
        current_block_coupling=None,
    )

    assert resolved_program == program
    assert coupling == CurrentBlockCoupling(program.value)


def test_generalist_program_resolves_joint_coupling() -> None:
    program, coupling = resolve_video_action_program_semantics(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        current_block_coupling=None,
    )

    assert program == VideoActionProgram.GENERALIST_JOINT_DENOISING
    assert coupling == CurrentBlockCoupling.JOINT


@pytest.mark.parametrize(
    "program",
    (VideoActionProgram.FORWARD_DYNAMICS, VideoActionProgram.INVERSE_DYNAMICS),
)
def test_standalone_conditional_program_resolves_joint_coupling(
    program: VideoActionProgram,
) -> None:
    resolved_program, coupling = resolve_video_action_program_semantics(
        program=program,
        current_block_coupling=None,
    )

    assert resolved_program is program
    assert coupling == CurrentBlockCoupling.JOINT


def test_standalone_conditional_program_override_rederives_mode_and_coupling() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )

    updated = apply_config_overrides(
        config,
        {"policy_variant.program": VideoActionProgram.INVERSE_DYNAMICS.value},
    )

    assert updated.policy_variant.program == VideoActionProgram.INVERSE_DYNAMICS
    assert updated.policy_variant.current_block_coupling == CurrentBlockCoupling.JOINT
    assert updated.policy_variant.generalist_denoising_mode_probs == {
        mode: float(mode == GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION)
        for mode in GeneralistDenoisingMode
    }


@pytest.mark.parametrize(
    "program",
    (
        VideoActionProgram.JOINT,
        VideoActionProgram.GENERALIST_JOINT_DENOISING,
        VideoActionProgram.VIDEO_THEN_ACTION,
    ),
)
def test_leaving_standalone_conditional_program_does_not_retain_derived_mode(
    program: VideoActionProgram,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )

    with pytest.raises(
        ValueError,
        match="dynamics_routed.*generalist_denoising_mode_probs",
    ):
        apply_config_overrides(
            config,
            {"policy_variant.program": program.value},
        )


def test_legacy_coupling_derives_public_program() -> None:
    program, coupling = resolve_video_action_program_semantics(
        program=None,
        current_block_coupling="action_then_video",
    )

    assert program == VideoActionProgram.ACTION_THEN_VIDEO
    assert coupling == CurrentBlockCoupling.ACTION_THEN_VIDEO


def test_conflicting_program_and_coupling_fail_at_config_boundary() -> None:
    with pytest.raises(ValueError, match="program.*conflicts"):
        resolve_video_action_program_semantics(
            program="video_then_action",
            current_block_coupling="joint",
        )


@pytest.mark.parametrize("architecture", ("parallel_stream", "dual_expert"))
@pytest.mark.parametrize(
    "program",
    (
        VideoActionProgram.VIDEO_THEN_ACTION,
        VideoActionProgram.ACTION_THEN_VIDEO,
        VideoActionProgram.DECOUPLED_SAME_STEP,
        VideoActionProgram.VIDEO_NOISY_TO_ACTION,
        VideoActionProgram.ACTION_NOISY_TO_VIDEO,
    ),
)
def test_program_only_override_replaces_previously_derived_coupling(
    architecture: str,
    program: VideoActionProgram,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs" / "experiments" / f"{architecture}_libero_joint.yaml"
    )

    updated = apply_config_overrides(
        config,
        {"policy_variant.program": program.value},
    )

    assert updated.policy_variant.program == program
    assert updated.policy_variant.current_block_coupling == CurrentBlockCoupling(
        program.value
    )


@pytest.mark.parametrize("architecture", ("parallel_stream", "dual_expert"))
def test_legacy_coupling_only_override_replaces_previously_derived_program(
    architecture: str,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs" / "experiments" / f"{architecture}_libero_joint.yaml"
    )

    updated = apply_config_overrides(
        config,
        {"policy_variant.current_block_coupling": "action_then_video"},
    )

    assert updated.policy_variant.program == VideoActionProgram.ACTION_THEN_VIDEO
    assert (
        updated.policy_variant.current_block_coupling
        == CurrentBlockCoupling.ACTION_THEN_VIDEO
    )


@pytest.mark.parametrize("architecture", ("parallel_stream", "dual_expert"))
def test_conflicting_program_and_coupling_overrides_still_fail(
    architecture: str,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs" / "experiments" / f"{architecture}_libero_joint.yaml"
    )

    with pytest.raises(ValueError, match="program.*conflicts"):
        apply_config_overrides(
            config,
            {
                "policy_variant.program": "video_then_action",
                "policy_variant.current_block_coupling": "joint",
            },
        )


def test_static_validation_rejects_unknown_video_action_program(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["program"] = "typo_program"
    config_path = tmp_path / "invalid_program.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "policy_variant.program" and "typo_program" in issue.message
        for issue in report.errors
    )


def test_static_validation_rejects_standalone_conditional_parallel_stream(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["program"] = VideoActionProgram.FORWARD_DYNAMICS.value
    raw["policy_variant"]["generalist_training_paradigm"] = "dynamics_routed"
    raw["data"]["train_batch_size"] = 1
    raw["data"]["val_batch_size"] = 1
    config_path = tmp_path / "parallel_stream_standalone_fdm.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "policy_variant.program" and "dual_expert" in issue.message
        for issue in report.errors
    )


def test_static_validation_requires_matching_conditional_dynamics_source(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    mixture = raw["data"]["generalist_dynamics_mixture"]
    mixture["real_action_conditioned_video_weight"] = 0.0
    mixture["counterfactual_action_conditioned_video_weight"] = 0.0
    config_path = tmp_path / "conditional_fdm_without_fdm_source.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "data.generalist_dynamics_mixture"
        and "positive real or counterfactual" in issue.message
        for issue in report.errors
    )


@pytest.mark.parametrize("architecture", ("parallel_stream", "dual_expert"))
def test_static_validation_requires_dynamics_routing_for_conditional_gjd(
    tmp_path: Path,
    architecture: str,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments"
        / f"{architecture}_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["generalist_training_paradigm"] = "demo_only"
    raw["policy_variant"]["generalist_denoising_mode_probs"] = {
        "joint": 0.0,
        "action_conditioned_video": 1.0,
        "video_conditioned_action": 0.0,
    }
    config_path = tmp_path / f"{architecture}_conditional_gjd_demo_only.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "policy_variant.generalist_training_paradigm"
        and "target-only t0" in issue.message
        for issue in report.errors
    )


@pytest.mark.parametrize("architecture", ("parallel_stream", "dual_expert"))
@pytest.mark.parametrize("program", COUPLING_PROGRAMS)
def test_canonical_libero_configs_select_program_not_low_level_coupling(
    architecture: str,
    program: VideoActionProgram,
) -> None:
    config_path = (
        REPO_ROOT
        / "configs"
        / "experiments"
        / f"{architecture}_libero_{program.value}.yaml"
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = load_experiment_config(config_path)

    assert raw["policy_variant"]["program"] == program.value
    assert "current_block_coupling" not in raw["policy_variant"]
    assert config.policy_variant.program == program
    assert config.policy_variant.current_block_coupling == CurrentBlockCoupling(
        program.value
    )


@pytest.mark.parametrize("architecture", ("parallel_stream", "dual_expert"))
def test_canonical_generalist_configs_select_generalist_program(
    architecture: str,
) -> None:
    config_path = (
        REPO_ROOT
        / "configs"
        / "experiments"
        / f"{architecture}_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = load_experiment_config(config_path)

    assert raw["policy_variant"]["program"] == "generalist_joint_denoising"
    assert "current_block_coupling" not in raw["policy_variant"]
    assert (
        config.policy_variant.program
        == VideoActionProgram.GENERALIST_JOINT_DENOISING
    )
    assert config.policy_variant.current_block_coupling == CurrentBlockCoupling.JOINT
