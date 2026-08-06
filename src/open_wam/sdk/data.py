"""Stable dataset sample and adapter-registration contracts."""

from open_wam.data.artifacts import (
    DatasetArtifactKind,
    DatasetArtifactPreflightError,
    DatasetArtifactRequirement,
    DatasetArtifactStatus,
    check_dataset_artifacts,
    require_dataset_artifacts,
)
from open_wam.data.contracts import WAMSample
from open_wam.data.latent_contracts import LatentWAMSample
from open_wam.data.registries import (
    DatasetAdapterSpec,
    DatasetArtifactResolver,
    DatasetPairBuilder,
    LatentDatasetPairBuilder,
    preflight_dataset_artifacts,
    register_dataset_adapter,
    registered_dataset_adapters,
)

__all__ = [
    "DatasetAdapterSpec",
    "DatasetArtifactKind",
    "DatasetArtifactPreflightError",
    "DatasetArtifactRequirement",
    "DatasetArtifactResolver",
    "DatasetArtifactStatus",
    "DatasetPairBuilder",
    "LatentDatasetPairBuilder",
    "LatentWAMSample",
    "WAMSample",
    "check_dataset_artifacts",
    "preflight_dataset_artifacts",
    "register_dataset_adapter",
    "registered_dataset_adapters",
    "require_dataset_artifacts",
]
