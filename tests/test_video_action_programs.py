from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_wam.configs import (
    CurrentBlockCoupling,
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
