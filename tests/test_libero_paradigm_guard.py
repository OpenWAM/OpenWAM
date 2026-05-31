from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from open_wam.configs import ProprioContextMode
from open_wam.utils.config_loader import load_experiment_config
from open_wam.utils.libero_paradigm import (
    collect_current_libero_policy_paradigm_issues,
    deprecated_libero_script_replacement,
    deprecated_libero_policy_config_reason,
    require_current_libero_policy_paradigm,
    require_current_libero_script,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_libero_paradigm_guard_flags_no_proprio_strict_m1_config() -> None:
    config_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
    config = load_experiment_config(config_path)

    issues = collect_current_libero_policy_paradigm_issues(config, config_path=config_path)

    assert issues == [
        "policy_variant.proprio_context_mode='none', "
        "expected 'per_chunk_additive'"
    ]


def test_libero_paradigm_guard_rejects_deprecated_text_token_proprio() -> None:
    config_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
    config = load_experiment_config(config_path)
    config = replace(
        config,
        policy_variant=replace(
            config.policy_variant,
            proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
        ),
    )

    assert collect_current_libero_policy_paradigm_issues(config, config_path=config_path) == [
        "policy_variant.proprio_context_mode='text_context_token', expected 'per_chunk_additive'"
    ]


def test_libero_paradigm_guard_accepts_strict_m1_config_with_per_chunk_proprio() -> None:
    config_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
    config = load_experiment_config(config_path)
    config = replace(
        config,
        policy_variant=replace(
            config.policy_variant,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
    )

    assert collect_current_libero_policy_paradigm_issues(config, config_path=config_path) == []


def test_libero_paradigm_guard_accepts_fullseg_w64_gjd_configs() -> None:
    for config_path in (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml",
        REPO_ROOT / "configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml",
    ):
        config = load_experiment_config(config_path)

        assert collect_current_libero_policy_paradigm_issues(config, config_path=config_path) == []


def test_libero_paradigm_guard_flags_fixed128_gjd_config() -> None:
    config_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
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

    issues = collect_current_libero_policy_paradigm_issues(config, config_path=config_path)

    assert "data.sample_construction.mode='hierarchical_fixed_segment', expected 'uniform_segment'" in issues
    assert "data.sample_construction.window_size=30, expected 64" in issues


def test_libero_paradigm_guard_prefers_resolved_config_over_legacy_wrapper_path() -> None:
    config_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
    legacy_wrapper_path = REPO_ROOT / "configs/experiments/deprecated/parallel_stream_libero_lingbot_exact_local.yaml"
    config = load_experiment_config(config_path)
    config = replace(
        config,
        policy_variant=replace(
            config.policy_variant,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
    )

    assert collect_current_libero_policy_paradigm_issues(config, config_path=legacy_wrapper_path) == []


def test_libero_paradigm_guard_rejects_known_legacy_m5_config() -> None:
    config_path = REPO_ROOT / "configs/experiments/deprecated/mot_libero_latent_local_joint.yaml"
    config = load_experiment_config(config_path)

    with pytest.raises(ValueError, match="Refuses deprecated LIBERO M1/M5 config|refuses deprecated LIBERO M1/M5 config"):
        require_current_libero_policy_paradigm(
            config,
            config_path=config_path,
            source="test",
        )

    assert deprecated_libero_policy_config_reason(config_path) == (
        "legacy M5 joint config without strict one-frame fixed-128 rollout parity"
    )


def test_libero_paradigm_guard_ignores_non_libero_smoke_config() -> None:
    config_path = REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    config = load_experiment_config(config_path)

    assert collect_current_libero_policy_paradigm_issues(config, config_path=config_path) == []


def test_libero_script_guard_rejects_legacy_visualization_entrypoints() -> None:
    with pytest.raises(ValueError, match="run_libero_realtime_sandbox.py"):
        require_current_libero_script("scripts/deprecated/run_libero_exact_visualization.py")

    assert deprecated_libero_script_replacement("scripts/deprecated/run_libero_mot_visualization.py") == (
        "scripts/run_libero_realtime_sandbox.py or scripts/run_libero_sampled_eval.py"
    )


def test_libero_script_guard_allows_current_entrypoints_and_explicit_opt_in() -> None:
    require_current_libero_script("scripts/run_libero_realtime_sandbox.py")
    require_current_libero_script("scripts/deprecated/run_libero_exact_visualization.py", allow_deprecated=True)
