from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from open_wam.configs import load_experiment_config
from open_wam.utils.libero_paradigm import (
    collect_current_libero_policy_paradigm_issues,
    removed_libero_policy_config_reason,
    require_current_libero_policy_paradigm,
    require_current_libero_script,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_libero_compatibility_guard_does_not_select_a_proprio_recipe() -> None:
    config_path = (
        REPO_ROOT / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    )
    config = load_experiment_config(config_path)

    issues = collect_current_libero_policy_paradigm_issues(config, config_path=config_path)

    assert issues == []


def test_libero_compatibility_guard_accepts_maintained_gjd_configs() -> None:
    for config_path in (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml",
        REPO_ROOT / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml",
    ):
        config = load_experiment_config(config_path)

        assert collect_current_libero_policy_paradigm_issues(config, config_path=config_path) == []


def test_libero_compatibility_guard_accepts_architecture_neutral_policy_programs() -> None:
    for architecture in ("dual_expert", "parallel_stream"):
        for program in (
            "video_then_action",
            "action_then_video",
            "joint",
            "decoupled_same_step",
            "video_noisy_to_action",
            "action_noisy_to_video",
        ):
            config_path = REPO_ROOT / "configs/experiments" / f"{architecture}_libero_{program}.yaml"
            config = load_experiment_config(config_path)

            assert collect_current_libero_policy_paradigm_issues(config, config_path=config_path) == []


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
def test_libero_compatibility_guard_allows_alternative_sampling_recipes(architecture: str) -> None:
    config_path = REPO_ROOT / "configs/experiments" / f"{architecture}_libero_joint.yaml"
    config = load_experiment_config(config_path)
    config = replace(
        config,
        data=replace(
            config.data,
            sample_construction=replace(
                config.data.sample_construction,
                mode="hierarchical_fixed_segment",
                segment_frames=128,
                segment_min_frames=None,
                segment_max_frames=None,
                window_size=30,
                randomize_geometry=False,
                randomize_segment_length=False,
                randomize_segment_start=False,
                require_full_segment=False,
                sample_order_mode="epoch_order",
                target_alignment="next_after_context",
                rollout_context_policy="one_frame",
                start_padding_frames=0,
            ),
        ),
    )

    assert collect_current_libero_policy_paradigm_issues(config, config_path=config_path) == []


def test_libero_compatibility_guard_does_not_own_gjd_sampling_geometry() -> None:
    config_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    config = load_experiment_config(config_path)
    config = replace(
        config,
        data=replace(
            config.data,
            sample_construction=replace(
                config.data.sample_construction,
                mode="hierarchical_fixed_segment",
                segment_frames=128,
                segment_min_frames=None,
                segment_max_frames=None,
                window_size=30,
                randomize_geometry=False,
                randomize_segment_length=False,
                randomize_segment_start=False,
                require_full_segment=False,
                sample_order_mode="epoch_order",
                target_alignment="next_after_context",
                rollout_context_policy="one_frame",
                start_padding_frames=0,
            ),
        ),
    )

    assert collect_current_libero_policy_paradigm_issues(config, config_path=config_path) == []


def test_libero_paradigm_guard_prefers_resolved_config_over_legacy_wrapper_path() -> None:
    config_path = (
        REPO_ROOT / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    )
    legacy_wrapper_path = REPO_ROOT / "configs/experiments/deprecated/parallel_stream_libero_lingbot_exact_local.yaml"
    config = load_experiment_config(config_path)
    assert collect_current_libero_policy_paradigm_issues(config, config_path=legacy_wrapper_path) == []


def test_libero_paradigm_guard_rejects_known_legacy_m5_config() -> None:
    config_path = REPO_ROOT / "configs/experiments/deprecated/mot_libero_latent_local_joint.yaml"
    current_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    config = load_experiment_config(current_path)
    config = replace(config, name="mot_libero_latent_local_joint")

    with pytest.raises(ValueError, match="refuses retired LIBERO policy config"):
        require_current_libero_policy_paradigm(
            config,
            config_path=config_path,
            source="test",
        )

    assert removed_libero_policy_config_reason(config_path) == (
        "retired dual-expert joint config; use dual_expert_libero_joint"
    )


def test_libero_paradigm_guard_ignores_non_libero_smoke_config() -> None:
    config_path = REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    config = load_experiment_config(config_path)

    assert collect_current_libero_policy_paradigm_issues(config, config_path=config_path) == []


@pytest.mark.parametrize(
    "script_name",
    (
        "scripts/run_libero_exact_realtime_sandbox.py",
        "scripts/run_libero_exact_visualization.py",
        "scripts/run_mot_non_joint_aligned_libero_A.sh",
        "scripts/run_mot_non_joint_action_only_libero_B.sh",
        "scripts/run_mot_full_segment_nonjoint_libero.sh",
    ),
)
def test_libero_script_guard_never_allows_removed_entrypoints(script_name: str) -> None:
    with pytest.raises(ValueError, match="was removed from the maintained Open-WAM runtime"):
        require_current_libero_script(script_name)


def test_libero_script_guard_allows_current_entrypoints_and_explicit_opt_in() -> None:
    require_current_libero_script("scripts/run_libero_realtime_sandbox.py")
    require_current_libero_script("scripts/run_libero_dual_expert_visualization.py")
