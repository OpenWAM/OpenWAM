from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_wam.configs import (
    CurrentBlockCoupling,
    DynamicsObjective,
    JointTimestepCoupling,
    ParallelStreamVariantProfile,
    VideoActionProgram,
    current_block_coupling_for_program,
    load_experiment_config,
    serialize_experiment_config,
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
SHARED_VIDEO_ACTION_SEMANTIC_FIELDS = (
    "program",
    "current_block_coupling",
    "joint_timestep_coupling",
    "generalist_mode_text_token",
    "proprio_context_mode",
    "history_stream_visibility",
    "context_condition_latent_source",
    "use_condition_latents",
    "require_condition_latents",
    "sequence_contract",
    "noisy_video_condition_prob",
)


@pytest.mark.parametrize(
    "config_suffix",
    tuple(program.value for program in COUPLING_PROGRAMS)
    + (VideoActionProgram.GENERALIST_JOINT_DENOISING.value,),
)
def test_dual_and_parallel_profiles_share_program_semantics(
    config_suffix: str,
) -> None:
    """Changing architecture must not silently change public program semantics."""

    configs = [
        load_experiment_config(
            REPO_ROOT
            / "configs"
            / "experiments"
            / f"{architecture}_libero_{config_suffix}.yaml"
        )
        for architecture in ("dual_expert", "parallel_stream")
    ]

    for field_name in SHARED_VIDEO_ACTION_SEMANTIC_FIELDS:
        assert getattr(configs[0].policy_variant, field_name) == getattr(
            configs[1].policy_variant,
            field_name,
        ), field_name


@pytest.mark.parametrize(
    ("config_name", "program"),
    (
        ("dual_expert_libero_joint", VideoActionProgram.JOINT),
        (
            "dual_expert_libero_video_noisy_to_action",
            VideoActionProgram.VIDEO_NOISY_TO_ACTION,
        ),
        (
            "dual_expert_libero_action_noisy_to_video",
            VideoActionProgram.ACTION_NOISY_TO_VIDEO,
        ),
        ("parallel_stream_libero_joint", VideoActionProgram.JOINT),
        (
            "parallel_stream_libero_video_noisy_to_action",
            VideoActionProgram.VIDEO_NOISY_TO_ACTION,
        ),
        (
            "parallel_stream_libero_action_noisy_to_video",
            VideoActionProgram.ACTION_NOISY_TO_VIDEO,
        ),
        ("parallel_stream_libero_action_conditioned_smoke", VideoActionProgram.JOINT),
    ),
)
def test_joint_like_configs_use_independent_timestep_clocks(
    config_name: str,
    program: VideoActionProgram,
) -> None:
    config_path = REPO_ROOT / "configs" / "experiments" / f"{config_name}.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert raw["policy_variant"]["joint_timestep_coupling"] == "independent"

    config = load_experiment_config(config_path)
    assert config.policy_variant.program is program
    assert (
        config.policy_variant.joint_timestep_coupling
        is JointTimestepCoupling.INDEPENDENT
    )


def test_every_video_action_program_has_explicit_current_block_coupling() -> None:
    expected = {
        VideoActionProgram.VIDEO_THEN_ACTION: CurrentBlockCoupling.VIDEO_THEN_ACTION,
        VideoActionProgram.ACTION_THEN_VIDEO: CurrentBlockCoupling.ACTION_THEN_VIDEO,
        VideoActionProgram.JOINT: CurrentBlockCoupling.JOINT,
        VideoActionProgram.DECOUPLED_SAME_STEP: CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        VideoActionProgram.VIDEO_NOISY_TO_ACTION: CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        VideoActionProgram.ACTION_NOISY_TO_VIDEO: CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
        VideoActionProgram.GENERALIST_JOINT_DENOISING: CurrentBlockCoupling.JOINT,
        VideoActionProgram.FORWARD_DYNAMICS: CurrentBlockCoupling.JOINT,
        VideoActionProgram.INVERSE_DYNAMICS: CurrentBlockCoupling.JOINT,
    }

    assert set(expected) == set(VideoActionProgram)
    assert {
        program: current_block_coupling_for_program(program)
        for program in VideoActionProgram
    } == expected


@pytest.mark.parametrize(
    ("config_name", "architecture_name"),
    (
        ("dual_expert_libero_generalist_joint_denoising", "dual_expert"),
        ("parallel_stream_libero_generalist_joint_denoising", "parallel_stream"),
    ),
)
@pytest.mark.parametrize(
    ("program", "objective"),
    (
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
    ),
)
def test_each_architecture_gjd_profile_can_select_fixed_dynamics(
    config_name: str,
    architecture_name: str,
    program: VideoActionProgram,
    objective: DynamicsObjective,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments" / f"{config_name}.yaml"
    )
    routes = [
        {"source": "real_demo", "mode": objective.value, "weight": 3.0},
        {
            "source": "counterfactual_dynamics",
            "mode": objective.value,
            "weight": 1.0,
        },
    ]

    updated = apply_config_overrides(
        config,
        {
            "policy_variant.program": program.value,
            "data.dynamics_routing.routes": routes,
            "validation.auxiliary_tasks": [
                {
                    "name": "conditional_dynamics_val",
                    "dataset_split": "val",
                    "source": "dataset",
                    "max_batches": 16,
                    "report_prefix": "val_conditional_dynamics",
                }
            ],
        },
    )

    assert updated.policy_variant.name.value == architecture_name
    assert updated.policy_variant.program is program
    assert updated.policy_variant.current_block_coupling is CurrentBlockCoupling.JOINT
    assert updated.data.dynamics_routing.mode_probabilities() == {
        mode: float(mode is objective) for mode in DynamicsObjective
    }


@pytest.mark.parametrize(
    ("program", "objective"),
    (
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
    ),
)
def test_fixed_dynamics_overrides_preserve_cross_architecture_semantics(
    program: VideoActionProgram,
    objective: DynamicsObjective,
) -> None:
    routes = [
        {"source": "real_demo", "mode": objective.value, "weight": 3.0},
        {
            "source": "counterfactual_dynamics",
            "mode": objective.value,
            "weight": 1.0,
        },
    ]
    configs = [
        apply_config_overrides(
            load_experiment_config(
                REPO_ROOT
                / "configs"
                / "experiments"
                / f"{architecture}_libero_generalist_joint_denoising.yaml"
            ),
            {
                "policy_variant.program": program.value,
                "data.dynamics_routing.routes": routes,
                "validation.auxiliary_tasks": [
                    {
                        "name": "conditional_dynamics_val",
                        "dataset_split": "val",
                        "source": "dataset",
                        "max_batches": 16,
                        "report_prefix": "val_conditional_dynamics",
                    }
                ],
            },
        )
        for architecture in ("dual_expert", "parallel_stream")
    ]

    for field_name in SHARED_VIDEO_ACTION_SEMANTIC_FIELDS:
        assert getattr(configs[0].policy_variant, field_name) == getattr(
            configs[1].policy_variant,
            field_name,
        ), field_name
    assert (
        configs[0].data.dynamics_routing.mode_probabilities()
        == configs[1].data.dynamics_routing.mode_probabilities()
    )


def test_standalone_conditional_program_and_routes_override_together() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )

    with pytest.raises(ValueError, match="accepts only.*video_conditioned_action"):
        apply_config_overrides(
            config,
            {"policy_variant.program": VideoActionProgram.INVERSE_DYNAMICS.value},
        )

    updated = apply_config_overrides(
        config,
        {
            "policy_variant.program": VideoActionProgram.INVERSE_DYNAMICS.value,
            "data.dynamics_routing.routes": [
                {
                    "source": "real_demo",
                    "mode": "video_conditioned_action",
                    "weight": 3.0,
                },
                {
                    "source": "counterfactual_dynamics",
                    "mode": "video_conditioned_action",
                    "weight": 1.0,
                },
            ],
        },
    )

    assert updated.policy_variant.program == VideoActionProgram.INVERSE_DYNAMICS
    assert updated.policy_variant.current_block_coupling == CurrentBlockCoupling.JOINT
    assert updated.data.dynamics_routing.mode_probabilities() == {
        mode: float(mode == DynamicsObjective.VIDEO_CONDITIONED_ACTION)
        for mode in DynamicsObjective
    }


@pytest.mark.parametrize(
    "program",
    (VideoActionProgram.FORWARD_DYNAMICS, VideoActionProgram.INVERSE_DYNAMICS),
)
def test_standalone_conditional_program_rejects_joint_clock_coupling(
    program: VideoActionProgram,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    overrides: dict[str, object] = {
        "policy_variant.program": program.value,
        "policy_variant.joint_timestep_coupling": "match_sigma",
    }
    if program is VideoActionProgram.INVERSE_DYNAMICS:
        overrides["data.dynamics_routing.routes"] = [
            {
                "source": "real_demo",
                "mode": "video_conditioned_action",
                "weight": 1.0,
            }
        ]

    with pytest.raises(ValueError, match="does not define.*noise clock.*requires"):
        apply_config_overrides(config, overrides)


@pytest.mark.parametrize(
    "program",
    (
        VideoActionProgram.JOINT,
        VideoActionProgram.VIDEO_THEN_ACTION,
    ),
)
def test_dynamics_routes_reject_non_dynamics_programs(
    program: VideoActionProgram,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )

    with pytest.raises(
        ValueError,
        match="Active .*dynamics_routing.routes.*require",
    ):
        apply_config_overrides(
            config,
            {"policy_variant.program": program.value},
        )


def test_gjd_accepts_a_single_conditional_mode_route_set() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )

    updated = apply_config_overrides(
        config,
        {
            "policy_variant.program": (
                VideoActionProgram.GENERALIST_JOINT_DENOISING.value
            )
        },
    )

    assert (
        updated.policy_variant.program is VideoActionProgram.GENERALIST_JOINT_DENOISING
    )
    assert updated.data.dynamics_routing.mode_probabilities() == {
        mode: float(mode is DynamicsObjective.ACTION_CONDITIONED_VIDEO)
        for mode in DynamicsObjective
    }


@pytest.mark.parametrize(
    ("override", "match"),
    (
        (
            {"policy_variant.program": VideoActionProgram.JOINT.value},
            "Active `data.dynamics_routing.routes` require",
        ),
        (
            {
                "policy_variant.variant_profile": (
                    ParallelStreamVariantProfile.STANDARD.value
                )
            },
            "cannot be set for Parallel Stream",
        ),
    ),
)
def test_parallel_stream_generalist_program_and_profile_are_atomic(
    override: dict[str, str],
    match: str,
) -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )

    with pytest.raises(ValueError, match=match):
        apply_config_overrides(config, override)


@pytest.mark.parametrize(
    ("field", "value", "issue_path", "message"),
    (
        (
            "program",
            VideoActionProgram.JOINT.value,
            "data.dynamics_routing.routes",
            "Active routes require",
        ),
        (
            "variant_profile",
            ParallelStreamVariantProfile.STANDARD.value,
            "policy_variant.variant_profile",
            "derives this value",
        ),
    ),
)
def test_static_parallel_stream_generalist_program_and_profile_are_atomic(
    tmp_path: Path,
    field: str,
    value: str,
    issue_path: str,
    message: str,
) -> None:
    source = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["policy_variant"][field] = value
    config_path = tmp_path / f"mismatched_{field}.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert any(
        issue.path == issue_path and message in issue.message for issue in report.errors
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
    assert (
        updated.policy_variant.joint_timestep_coupling
        == JointTimestepCoupling.INDEPENDENT
    )


@pytest.mark.parametrize("architecture", ("parallel_stream", "dual_expert"))
def test_static_validation_rejects_inapplicable_staged_timestep_coupling(
    tmp_path: Path,
    architecture: str,
) -> None:
    source = (
        REPO_ROOT
        / "configs"
        / "experiments"
        / f"{architecture}_libero_video_then_action.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["policy_variant"]["joint_timestep_coupling"] = "match_sigma"
    config_path = tmp_path / "staged_with_joint_clock.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "policy_variant.joint_timestep_coupling"
        and "requires `independent`" in issue.message
        for issue in report.errors
    )


def test_static_validation_rejects_inapplicable_strict_dynamics_coupling(
    tmp_path: Path,
) -> None:
    source = (
        REPO_ROOT
        / "configs"
        / "experiments"
        / "dual_expert_libero_conditional_dynamics.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["policy_variant"]["joint_timestep_coupling"] = "match_sigma"
    config_path = tmp_path / "strict_fdm_with_joint_clock.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "policy_variant.joint_timestep_coupling"
        and "requires `independent`" in issue.message
        for issue in report.errors
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runtime_mode", "lingbot_exact_action_conditioned"),
        ("current_block_coupling", "action_then_video"),
        ("variant_profile", "standard"),
        ("video_condition_on_action", False),
    ),
)
def test_parallel_stream_rejects_derived_control_override(
    field: str,
    value: object,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    )

    with pytest.raises(ValueError, match=r"select `policy_variant.program`"):
        apply_config_overrides(
            config,
            {f"policy_variant.{field}": value},
        )


def test_parallel_stream_serializes_only_the_authoritative_program() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    )

    policy = serialize_experiment_config(config)["policy_variant"]

    assert policy["program"] == VideoActionProgram.JOINT.value
    assert {
        "runtime_mode",
        "current_block_coupling",
        "variant_profile",
        "video_condition_on_action",
    }.isdisjoint(policy)


def test_parallel_stream_conflicting_program_and_coupling_overrides_fail() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    )

    with pytest.raises(ValueError, match=r"select `policy_variant.program`"):
        apply_config_overrides(
            config,
            {
                "policy_variant.program": "video_then_action",
                "policy_variant.current_block_coupling": "joint",
            },
        )


@pytest.mark.parametrize(
    "field",
    ("runtime_mode", "current_block_coupling", "video_can_attend_action"),
)
def test_dual_expert_rejects_removed_execution_controls(field: str) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    )

    with pytest.raises(ValueError, match=r"select `policy_variant.program`"):
        apply_config_overrides(config, {f"policy_variant.{field}": "joint"})


@pytest.mark.parametrize(
    "field",
    ("runtime_mode", "current_block_coupling", "video_can_attend_action"),
)
def test_static_validation_rejects_removed_dual_expert_controls(
    tmp_path: Path,
    field: str,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"][field] = "joint"
    config_path = tmp_path / f"dual_expert_with_{field}.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(issue.path == f"policy_variant.{field}" for issue in report.errors)


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


def test_static_validation_accepts_standalone_conditional_parallel_stream(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["program"] = VideoActionProgram.FORWARD_DYNAMICS.value
    raw["data"]["train_batch_size"] = 1
    raw["data"]["val_batch_size"] = 1
    raw["data"]["dynamics_routing"] = {
        "routes": [
            {
                "source": "real_demo",
                "mode": "action_conditioned_video",
                "weight": 1.0,
            }
        ]
    }
    config_path = tmp_path / "parallel_stream_standalone_fdm.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert report.ok
    assert not any(issue.path == "policy_variant.program" for issue in report.errors)


def test_static_validation_requires_matching_conditional_dynamics_source(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    mixture = raw["data"]["dynamics_routing"]
    mixture["routes"] = []
    config_path = tmp_path / "conditional_fdm_without_fdm_source.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "data.dynamics_routing.routes"
        and "positive route" in issue.message
        for issue in report.errors
    )


def test_strict_dynamics_rejects_mismatched_auxiliary_validation_mode() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )

    with pytest.raises(ValueError, match="idm_val.*fixes mode"):
        apply_config_overrides(
            config,
            {
                "validation.auxiliary_tasks": [
                    {
                        "name": "idm_val",
                        "mode_override": "video_conditioned_action",
                        "max_batches": 1,
                    }
                ]
            },
        )


def test_static_validation_rejects_mismatched_strict_dynamics_probe(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["validation"]["auxiliary_tasks"][0]["mode_override"] = (
        "video_conditioned_action"
    )
    config_path = tmp_path / "mismatched_strict_probe.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "validation.auxiliary_tasks[0].mode_override"
        and "fixes auxiliary validation mode" in issue.message
        for issue in report.errors
    )


def test_static_validation_requires_routes_for_strict_conditional_program(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["dynamics_routing"]["routes"] = []
    config_path = tmp_path / "strict_fdm_without_routes.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "data.dynamics_routing.routes"
        and "requires at least one positive route" in issue.message
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
        config.policy_variant.program == VideoActionProgram.GENERALIST_JOINT_DENOISING
    )
    assert config.policy_variant.current_block_coupling == CurrentBlockCoupling.JOINT
