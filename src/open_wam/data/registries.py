from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch.utils.data import Dataset

from open_wam.configs import DataConfig
from open_wam.registry import Registry

from .artifacts import (
    DatasetArtifactPreflightError,
    DatasetArtifactRequirement,
    DatasetArtifactStatus,
    require_dataset_artifacts,
)
from .contracts import WAMSample
from .latent_contracts import LatentWAMSample

RawDatasetPair = tuple[Dataset[WAMSample], Dataset[WAMSample]]
LatentDatasetPair = tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]
DatasetPairBuilder = Callable[[DataConfig], RawDatasetPair]
LatentDatasetPairBuilder = Callable[[DataConfig], LatentDatasetPair]
DatasetArtifactResolver = Callable[
    [DataConfig], Sequence[DatasetArtifactRequirement]
]


@dataclass(frozen=True)
class DatasetAdapterSpec:
    """Builders exposed by one dataset type.

    An adapter may support raw RGB samples, pre-encoded latent samples, or
    both, and may declare adapter-owned artifact preflight. Keeping these
    capabilities under one key makes them visible to callers and prevents raw,
    latent, and startup contracts from drifting.
    """

    dataset_type: str
    raw_builder: DatasetPairBuilder | None = None
    latent_builder: LatentDatasetPairBuilder | None = None
    artifact_resolver: DatasetArtifactResolver | None = None
    description: str | None = None


class DatasetAdapterRegistry(Registry[str, DatasetAdapterSpec]):
    """Registry for dataset adapters selected by ``DataConfig.dataset_type``."""

    def register_adapter(
        self,
        dataset_type: str,
        *,
        raw_builder: DatasetPairBuilder | None = None,
        latent_builder: LatentDatasetPairBuilder | None = None,
        artifact_resolver: DatasetArtifactResolver | None = None,
        description: str | None = None,
        replace: bool = False,
    ) -> None:
        normalized_type = dataset_type.strip()
        if not normalized_type:
            raise ValueError("Dataset adapter type must be a non-empty string.")
        if raw_builder is None and latent_builder is None and artifact_resolver is None:
            raise ValueError(
                f"Dataset adapter {normalized_type!r} must provide a raw builder, latent builder, "
                "or artifact resolver."
            )
        if raw_builder is not None and not callable(raw_builder):
            raise TypeError("Dataset adapter raw_builder must be callable.")
        if latent_builder is not None and not callable(latent_builder):
            raise TypeError("Dataset adapter latent_builder must be callable.")
        if artifact_resolver is not None and not callable(artifact_resolver):
            raise TypeError("Dataset adapter artifact_resolver must be callable.")

        current = self.get(normalized_type)
        if current is not None and not replace:
            collisions = []
            if raw_builder is not None and current.raw_builder is not None:
                collisions.append("raw")
            if latent_builder is not None and current.latent_builder is not None:
                collisions.append("latent")
            if artifact_resolver is not None and current.artifact_resolver is not None:
                collisions.append("artifact resolver")
            if collisions:
                joined = " and ".join(collisions)
                raise ValueError(
                    f"Dataset adapter {normalized_type!r} already has a {joined} builder. "
                    "Pass replace=True only for an intentional override."
                )

        existing_raw_builder = None if current is None else current.raw_builder
        existing_latent_builder = None if current is None else current.latent_builder
        existing_artifact_resolver = None if current is None else current.artifact_resolver
        existing_description = None if current is None else current.description
        spec = DatasetAdapterSpec(
            dataset_type=normalized_type,
            raw_builder=raw_builder if raw_builder is not None else existing_raw_builder,
            latent_builder=(
                latent_builder if latent_builder is not None else existing_latent_builder
            ),
            artifact_resolver=(
                artifact_resolver
                if artifact_resolver is not None
                else existing_artifact_resolver
            ),
            description=description if description is not None else existing_description,
        )
        super().register(
            normalized_type,
            spec,
            description=spec.description,
            replace=current is not None,
        )

    def require_raw_builder(self, dataset_type: str) -> DatasetPairBuilder:
        spec = self.get(dataset_type)
        if spec is not None and spec.raw_builder is not None:
            return spec.raw_builder
        supported = ", ".join(
            entry.key for entry in self.entries() if entry.value.raw_builder is not None
        )
        raise ValueError(
            f"Unsupported raw dataset_type {dataset_type!r}. "
            f"Registered raw dataset types: {supported}"
        )

    def require_latent_builder(self, dataset_type: str) -> LatentDatasetPairBuilder:
        spec = self.get(dataset_type)
        if spec is not None and spec.latent_builder is not None:
            return spec.latent_builder
        supported = ", ".join(
            entry.key for entry in self.entries() if entry.value.latent_builder is not None
        )
        raise ValueError(
            f"Unsupported latent dataset_type {dataset_type!r}. "
            f"Registered latent dataset types: {supported}"
        )

    def preflight_artifacts(
        self,
        data_config: DataConfig,
    ) -> tuple[DatasetArtifactStatus, ...]:
        spec = self.get(data_config.dataset_type)
        if spec is None or spec.artifact_resolver is None:
            return ()
        try:
            requirements = tuple(spec.artifact_resolver(data_config))
        except DatasetArtifactPreflightError:
            raise
        except (OSError, ValueError) as exc:
            raise DatasetArtifactPreflightError(
                "Dataset artifact discovery failed for "
                f"dataset_type={data_config.dataset_type!r}: {exc}"
            ) from exc
        invalid = [
            requirement
            for requirement in requirements
            if not isinstance(requirement, DatasetArtifactRequirement)
        ]
        if invalid:
            raise TypeError(
                f"Artifact resolver for dataset_type={data_config.dataset_type!r} "
                "must return DatasetArtifactRequirement values."
            )
        return require_dataset_artifacts(
            requirements,
            dataset_type=data_config.dataset_type,
        )


DATASET_ADAPTERS = DatasetAdapterRegistry("dataset adapter")


def register_dataset_adapter(
    dataset_type: str,
    *,
    raw_builder: DatasetPairBuilder | None = None,
    latent_builder: LatentDatasetPairBuilder | None = None,
    artifact_resolver: DatasetArtifactResolver | None = None,
    description: str | None = None,
    replace: bool = False,
) -> None:
    """Register raw and/or latent builders for one dataset type."""

    DATASET_ADAPTERS.register_adapter(
        dataset_type,
        raw_builder=raw_builder,
        latent_builder=latent_builder,
        artifact_resolver=artifact_resolver,
        description=description,
        replace=replace,
    )


def register_dataset_builder(
    dataset_type: str,
    builder: DatasetPairBuilder,
    *,
    artifact_resolver: DatasetArtifactResolver | None = None,
    description: str | None = None,
    replace: bool = False,
) -> None:
    """Compatibility helper for registering a raw RGB dataset builder."""

    register_dataset_adapter(
        dataset_type,
        raw_builder=builder,
        artifact_resolver=artifact_resolver,
        description=description,
        replace=replace,
    )


def register_latent_dataset_builder(
    dataset_type: str,
    builder: LatentDatasetPairBuilder,
    *,
    artifact_resolver: DatasetArtifactResolver | None = None,
    description: str | None = None,
    replace: bool = False,
) -> None:
    """Compatibility helper for registering a pre-encoded latent builder."""

    register_dataset_adapter(
        dataset_type,
        latent_builder=builder,
        artifact_resolver=artifact_resolver,
        description=description,
        replace=replace,
    )


def preflight_dataset_artifacts(
    data_config: DataConfig,
) -> tuple[DatasetArtifactStatus, ...]:
    """Run the selected adapter's declared filesystem checks."""

    return DATASET_ADAPTERS.preflight_artifacts(data_config)


def registered_dataset_adapters() -> tuple[str, ...]:
    """Return dataset adapter identifiers registered in this process."""

    return DATASET_ADAPTERS.keys()


__all__ = [
    "DATASET_ADAPTERS",
    "DatasetAdapterRegistry",
    "DatasetAdapterSpec",
    "DatasetArtifactResolver",
    "DatasetPairBuilder",
    "LatentDatasetPair",
    "LatentDatasetPairBuilder",
    "RawDatasetPair",
    "preflight_dataset_artifacts",
    "register_dataset_adapter",
    "register_dataset_builder",
    "register_latent_dataset_builder",
    "registered_dataset_adapters",
]
