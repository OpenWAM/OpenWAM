from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_wam.configs import (
    EVALUATION_CONFIG_ALIASES,
    EXPERIMENT_CONFIG_ALIASES,
    DeprecatedConfigNameWarning,
    load_experiment_config,
    resolve_config_path_alias,
    validate_config_file,
)
from open_wam.evals.evaluation_contracts import resolve_evaluation_request
from open_wam.training import TrainCliOverrides, resolve_experiment_config_path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "configs" / "experiments"
EVALUATION_ROOT = REPO_ROOT / "configs" / "evals"


def test_canonical_config_alias_targets_exist_without_duplicate_legacy_yaml() -> None:
    assert len(EXPERIMENT_CONFIG_ALIASES) == 18
    assert len(EVALUATION_CONFIG_ALIASES) == 3
    assert len(set(EXPERIMENT_CONFIG_ALIASES.values())) == len(
        EXPERIMENT_CONFIG_ALIASES
    )
    assert len(set(EVALUATION_CONFIG_ALIASES.values())) == len(
        EVALUATION_CONFIG_ALIASES
    )

    for old_stem, canonical_stem in EXPERIMENT_CONFIG_ALIASES.items():
        assert not (EXPERIMENT_ROOT / f"{old_stem}.yaml").exists()
        canonical_path = EXPERIMENT_ROOT / f"{canonical_stem}.yaml"
        assert canonical_path.is_file()
        assert (
            yaml.safe_load(canonical_path.read_text(encoding="utf-8"))["name"]
            == canonical_stem
        )
        assert "heng_compatible" not in canonical_stem

    for old_stem, canonical_stem in EVALUATION_CONFIG_ALIASES.items():
        assert not (EVALUATION_ROOT / f"{old_stem}.yaml").exists()
        canonical_path = EVALUATION_ROOT / f"{canonical_stem}.yaml"
        assert canonical_path.is_file()
        assert (
            yaml.safe_load(canonical_path.read_text(encoding="utf-8"))["name"]
            == canonical_stem
        )
        assert "heng_eval" not in canonical_stem


def test_retired_experiment_name_resolves_to_canonical_owner() -> None:
    old_path = EXPERIMENT_ROOT / "mot_libero_latent_local_joint_heng_compatible.yaml"

    with pytest.warns(DeprecatedConfigNameWarning, match="mot_libero_joint"):
        resolved = resolve_config_path_alias(old_path)
    with pytest.warns(DeprecatedConfigNameWarning, match="mot_libero_joint"):
        config = load_experiment_config(old_path)

    assert resolved == EXPERIMENT_ROOT / "mot_libero_joint.yaml"
    assert config.name == "mot_libero_joint"


def test_retired_bare_config_name_resolves_through_training_cli() -> None:
    overrides = TrainCliOverrides(
        config_name="mot_libero_latent_local_video_then_action_heng_compatible"
    )

    with pytest.warns(
        DeprecatedConfigNameWarning, match="mot_libero_video_then_action"
    ):
        resolved = resolve_experiment_config_path(overrides)

    assert resolved == EXPERIMENT_ROOT / "mot_libero_video_then_action.yaml"


def test_retired_eval_name_resolves_end_to_end() -> None:
    old_path = EVALUATION_ROOT / "parallel_stream_libero_lingbot_exact_heng_eval.yaml"

    with pytest.warns(
        DeprecatedConfigNameWarning, match="parallel_stream_libero_lingbot_exact_eval"
    ):
        request = resolve_evaluation_request(old_path)

    assert (
        request.experiment_config_path
        == EXPERIMENT_ROOT / "parallel_stream_libero_lingbot_exact.yaml"
    )


def test_static_validation_accepts_retired_name_through_alias() -> None:
    old_path = (
        EXPERIMENT_ROOT / "parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
    )

    with pytest.warns(
        DeprecatedConfigNameWarning, match="parallel_stream_libero_lingbot_exact"
    ):
        report = validate_config_file(old_path, repo_root=REPO_ROOT)

    assert report.ok
    assert (
        report.source_path
        == EXPERIMENT_ROOT / "parallel_stream_libero_lingbot_exact.yaml"
    )


def test_existing_historical_copy_wins_over_alias(tmp_path: Path) -> None:
    copied_path = tmp_path / "mot_libero_latent_local_joint_heng_compatible.yaml"
    copied_path.write_text("name: historical_copy\n", encoding="utf-8")

    assert resolve_config_path_alias(copied_path) == copied_path
