from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from open_wam.configs import GenericDataConfig, load_experiment_config
from open_wam.data import (
    DATASET_ADAPTERS,
    DatasetArtifactKind,
    DatasetArtifactPreflightError,
    DatasetArtifactRequirement,
    register_dataset_adapter,
)
from open_wam.data.lerobot_v2_latent_artifacts import (
    resolve_local_lerobot_latent_artifacts,
)
from open_wam.training import TrainingRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_adapter_artifact_preflight_is_extension_owned(tmp_path: Path) -> None:
    dataset_type = "test_artifact_preflight_adapter"
    required_file = tmp_path / "manifest.json"

    def build_raw(config):
        raise AssertionError("builder must not run during artifact preflight")

    def resolve_artifacts(config):
        assert config.dataset_type == dataset_type
        return (
            DatasetArtifactRequirement(
                name="manifest",
                path=required_file,
                kind=DatasetArtifactKind.FILE,
                required=True,
                config_path="data.adapter_options.manifest",
                purpose="the extension indexes episodes from this manifest",
            ),
        )

    register_dataset_adapter(
        dataset_type,
        raw_builder=build_raw,
        artifact_resolver=resolve_artifacts,
    )
    config = GenericDataConfig(dataset_name="test", dataset_type=dataset_type)

    with pytest.raises(DatasetArtifactPreflightError, match=r"manifest\.json"):
        DATASET_ADAPTERS.preflight_artifacts(config)

    required_file.write_text("{}", encoding="utf-8")
    statuses = DATASET_ADAPTERS.preflight_artifacts(config)
    assert len(statuses) == 1
    assert statuses[0].available is True


def test_optional_adapter_artifact_does_not_fail(tmp_path: Path) -> None:
    dataset_type = "test_optional_artifact_adapter"

    def build_raw(config):
        raise AssertionError("builder must not run during artifact preflight")

    register_dataset_adapter(
        dataset_type,
        raw_builder=build_raw,
        artifact_resolver=lambda config: (
            DatasetArtifactRequirement(
                name="optional labels",
                path=tmp_path / "missing.jsonl",
                kind=DatasetArtifactKind.FILE,
                required=False,
                config_path="data.adapter_options.labels",
                purpose="optional filtering",
            ),
        ),
    )

    statuses = DATASET_ADAPTERS.preflight_artifacts(
        GenericDataConfig(dataset_name="test", dataset_type=dataset_type)
    )
    assert statuses[0].available is False
    assert statuses[0].requirement.required is False


def test_blank_artifact_path_is_unconfigured_not_current_directory() -> None:
    requirement = DatasetArtifactRequirement(
        name="dataset root",
        path=" ",
        kind=DatasetArtifactKind.DIRECTORY,
        required=True,
        config_path="data.local_root",
        purpose="the adapter needs an explicit root",
    )

    assert requirement.resolved_path is None


def test_adapter_discovery_failure_uses_typed_preflight_error(
    tmp_path: Path,
) -> None:
    dataset_type = "test_artifact_discovery_failure"

    def fail_discovery(config):
        raise FileNotFoundError(tmp_path / "meta/info.json")

    register_dataset_adapter(
        dataset_type,
        artifact_resolver=fail_discovery,
    )

    with pytest.raises(
        DatasetArtifactPreflightError,
        match="artifact discovery failed",
    ):
        DATASET_ADAPTERS.preflight_artifacts(
            GenericDataConfig(dataset_name="test", dataset_type=dataset_type)
        )


def test_local_latent_replay_defaults_are_resolved_per_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "member_a"
    second = tmp_path / "member_b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(
        "open_wam.data.lerobot_v2_latent_artifacts.discover_local_lerobot_repo_bundles",
        lambda root: (
            SimpleNamespace(root=first),
            SimpleNamespace(root=second),
        ),
    )
    config = GenericDataConfig(
        dataset_name="test",
        dataset_type="lerobot_v2_latent_local",
        local_root=str(tmp_path),
        replay_status_path=None,
        require_replay_status=True,
    )

    requirements = resolve_local_lerobot_latent_artifacts(config)
    replay_paths = {
        requirement.resolved_path
        for requirement in requirements
        if requirement.name == "training replay-status metadata"
    }

    assert replay_paths == {
        first / "meta/replay_status.jsonl",
        second / "meta/replay_status.jsonl",
    }


def test_local_latent_preflight_rejects_existing_non_repository_root(
    tmp_path: Path,
) -> None:
    config = GenericDataConfig(
        dataset_name="test",
        dataset_type="lerobot_v2_latent_local",
        local_root=str(tmp_path),
        replay_status_path=None,
        require_replay_status=False,
    )

    with pytest.raises(DatasetArtifactPreflightError, match="No local LeRobot repo"):
        DATASET_ADAPTERS.preflight_artifacts(config)


def test_shared_validation_policy_can_require_replay_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "open_wam.data.lerobot_v2_latent_artifacts.discover_local_lerobot_repo_bundles",
        lambda root: (SimpleNamespace(root=tmp_path),),
    )
    config = GenericDataConfig(
        dataset_name="test",
        dataset_type="lerobot_v2_latent_local",
        local_root=str(tmp_path),
        replay_status_policy="include_all",
        val_replay_status_policy="failure_only",
        require_replay_status=False,
        val_require_replay_status=True,
    )

    requirements = resolve_local_lerobot_latent_artifacts(config)
    replay_requirement = next(
        requirement
        for requirement in requirements
        if requirement.name == "shared replay-status metadata"
    )

    assert replay_requirement.required is True
    assert replay_requirement.resolved_path == tmp_path / "meta/replay_status.jsonl"


def test_training_runtime_runs_artifact_preflight_before_model_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    observed: list[str] = []

    def fail_preflight(data_config):
        observed.append(data_config.dataset_type)
        raise DatasetArtifactPreflightError("missing test artifact")

    monkeypatch.setattr(
        "open_wam.training.runtime.preflight_dataset_artifacts",
        fail_preflight,
    )
    monkeypatch.setattr(
        "open_wam.training.runtime.build_variant_pipeline_from_config",
        lambda config: pytest.fail("model construction ran before artifact preflight"),
    )

    with pytest.raises(DatasetArtifactPreflightError, match="missing test artifact"):
        TrainingRuntime.from_config(config)

    assert observed == ["synthetic_multiview"]
