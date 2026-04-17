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
