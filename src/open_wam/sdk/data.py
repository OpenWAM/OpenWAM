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
from open_wam.data.encoded_dynamics_dataset import (
    ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1,
    EncodedDynamicsArtifact,
    load_encoded_dynamics_artifact,
    migrate_encoded_dynamics_artifact,
    preflight_encoded_dynamics_artifact,
)
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
    "ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1",
    "DatasetAdapterSpec",
    "DatasetArtifactKind",
    "DatasetArtifactPreflightError",
    "DatasetArtifactRequirement",
    "DatasetArtifactResolver",
    "DatasetArtifactStatus",
    "DatasetPairBuilder",
    "EncodedDynamicsArtifact",
    "LatentDatasetPairBuilder",
    "LatentWAMSample",
    "WAMSample",
    "check_dataset_artifacts",
    "load_encoded_dynamics_artifact",
    "migrate_encoded_dynamics_artifact",
    "preflight_dataset_artifacts",
    "preflight_encoded_dynamics_artifact",
    "register_dataset_adapter",
    "registered_dataset_adapters",
    "require_dataset_artifacts",
]
