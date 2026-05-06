from __future__ import annotations

from pathlib import Path

import pytest

from open_wam.configs.static_schema import validate_config_file, validate_config_files


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_static_validator_accepts_public_tiny_configs() -> None:
    reports = validate_config_files(
        (
            REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml",
            REPO_ROOT / "configs/evals/public_tiny_synthetic_contract.yaml",
        )
    )

    assert all(report.ok for report in reports)


@pytest.mark.unit
def test_static_validator_catches_enum_typos(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
name: bad
data:
  dataset_name: bad
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 4
    action_horizon: 2
    state_dim: 3
    state_horizon: 1
backbone:
  implementation: not_a_backbone
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
  action_dim: 4
  action_horizon: 2
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid BackboneImplementation" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_warns_for_legacy_action_head() -> None:
    report = validate_config_file(REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml")

    assert report.ok
    assert any(issue.path == "action_head" for issue in report.warnings)


@pytest.mark.unit
def test_static_validator_checks_joint_denoise_mode_probabilities(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_joint_probs.yaml"
    config_path.write_text(
        """
name: bad_joint_probs
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  joint_denoise_training_mode_probs:
    joint: 0.0
    typo_mode: 1.0
    action_conditioned_video: -0.2
    video_conditioned_action: .nan
action_decoder:
  name: lingbot_parallel_decoder
  action_dim: 30
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid JointDenoiseTrainingMode" in issue.message for issue in report.errors)
    assert any(issue.path.endswith("action_conditioned_video") for issue in report.errors)
    assert any(issue.path.endswith("video_conditioned_action") and "finite" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_rejects_boolean_joint_denoise_probability(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_joint_bool_prob.yaml"
    config_path.write_text(
        """
name: bad_joint_bool_prob
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  joint_denoise_training_mode_probs:
    joint: true
    action_conditioned_video: 0.0
    video_conditioned_action: 0.0
action_decoder:
  name: lingbot_parallel_decoder
  action_dim: 30
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path.endswith("joint") and "numeric" in issue.message for issue in report.errors)


@pytest.mark.parametrize("raw_probability", [".nan", ".inf"])
def test_static_validator_rejects_non_finite_joint_denoise_probability(
    tmp_path: Path,
    raw_probability: str,
) -> None:
    config_path = tmp_path / "bad_joint_non_finite_prob.yaml"
    config_path.write_text(
        f"""
name: bad_joint_non_finite_prob
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  joint_denoise_training_mode_probs:
    joint: 0.6
    action_conditioned_video: {raw_probability}
    video_conditioned_action: 0.2
action_decoder:
  name: lingbot_parallel_decoder
  action_dim: 30
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(
        issue.path.endswith("action_conditioned_video") and "finite" in issue.message
        for issue in report.errors
    )
