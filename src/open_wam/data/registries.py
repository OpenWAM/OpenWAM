from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch.utils.data import Dataset

from open_wam.configs import DataConfig
from open_wam.registry import Registry

from .contracts import WAMSample
from .latent_contracts import LatentWAMSample

RawDatasetPair = tuple[Dataset[WAMSample], Dataset[WAMSample]]
LatentDatasetPair = tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]
DatasetPairBuilder = Callable[[DataConfig], RawDatasetPair]
LatentDatasetPairBuilder = Callable[[DataConfig], LatentDatasetPair]


@dataclass(frozen=True)
class DatasetAdapterSpec:
    """Builders exposed by one dataset type.

    An adapter may support raw RGB samples, pre-encoded latent samples, or
    both. Keeping both builders under one key makes the capability visible to
    callers and prevents raw and latent registries from drifting.
    """

    dataset_type: str
    raw_builder: DatasetPairBuilder | None = None
    latent_builder: LatentDatasetPairBuilder | None = None
    description: str | None = None


class DatasetAdapterRegistry(Registry[str, DatasetAdapterSpec]):
    """Registry for dataset adapters selected by ``DataConfig.dataset_type``."""

    def register_adapter(
        self,
        dataset_type: str,
        *,
        raw_builder: DatasetPairBuilder | None = None,
        latent_builder: LatentDatasetPairBuilder | None = None,
        description: str | None = None,
        replace: bool = False,
    ) -> None:
        normalized_type = dataset_type.strip()
        if not normalized_type:
            raise ValueError("Dataset adapter type must be a non-empty string.")
        if raw_builder is None and latent_builder is None:
            raise ValueError(
                f"Dataset adapter {normalized_type!r} must provide a raw or latent builder."
            )
        if raw_builder is not None and not callable(raw_builder):
            raise TypeError("Dataset adapter raw_builder must be callable.")
        if latent_builder is not None and not callable(latent_builder):
            raise TypeError("Dataset adapter latent_builder must be callable.")

        current = self.get(normalized_type)
        if current is not None and not replace:
            collisions = []
            if raw_builder is not None and current.raw_builder is not None:
                collisions.append("raw")
            if latent_builder is not None and current.latent_builder is not None:
                collisions.append("latent")
            if collisions:
                joined = " and ".join(collisions)
                raise ValueError(
                    f"Dataset adapter {normalized_type!r} already has a {joined} builder. "
                    "Pass replace=True only for an intentional override."
                )

        existing_raw_builder = None if current is None else current.raw_builder
        existing_latent_builder = None if current is None else current.latent_builder
        existing_description = None if current is None else current.description
        spec = DatasetAdapterSpec(
            dataset_type=normalized_type,
            raw_builder=raw_builder if raw_builder is not None else existing_raw_builder,
            latent_builder=(
                latent_builder if latent_builder is not None else existing_latent_builder
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


DATASET_ADAPTERS = DatasetAdapterRegistry("dataset adapter")


def register_dataset_adapter(
    dataset_type: str,
    *,
    raw_builder: DatasetPairBuilder | None = None,
    latent_builder: LatentDatasetPairBuilder | None = None,
    description: str | None = None,
    replace: bool = False,
) -> None:
    """Register raw and/or latent builders for one dataset type."""

    DATASET_ADAPTERS.register_adapter(
        dataset_type,
        raw_builder=raw_builder,
        latent_builder=latent_builder,
        description=description,
        replace=replace,
    )


def register_dataset_builder(
    dataset_type: str,
    builder: DatasetPairBuilder,
    *,
    description: str | None = None,
    replace: bool = False,
) -> None:
    """Compatibility helper for registering a raw RGB dataset builder."""

    register_dataset_adapter(
        dataset_type,
        raw_builder=builder,
        description=description,
        replace=replace,
    )


def register_latent_dataset_builder(
    dataset_type: str,
    builder: LatentDatasetPairBuilder,
    *,
    description: str | None = None,
    replace: bool = False,
) -> None:
    """Compatibility helper for registering a pre-encoded latent builder."""

    register_dataset_adapter(
        dataset_type,
        latent_builder=builder,
        description=description,
        replace=replace,
    )


__all__ = [
    "DATASET_ADAPTERS",
    "DatasetAdapterRegistry",
    "DatasetAdapterSpec",
    "DatasetPairBuilder",
    "LatentDatasetPair",
    "LatentDatasetPairBuilder",
    "RawDatasetPair",
    "register_dataset_adapter",
    "register_dataset_builder",
    "register_latent_dataset_builder",
]
