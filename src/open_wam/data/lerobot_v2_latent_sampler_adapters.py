"""Distributed sampler adapters for local LeRobot latent policies."""

from __future__ import annotations

from collections.abc import Sequence, Sized
from typing import Protocol

from open_wam.configs import DataConfig

from .distributed_sampling import (
    EpochOffsetDistributedSampler,
    EpochOrderDistributedSampler,
    EpochOrderSource,
    WeightedReplacementDistributedSampler,
)


__all__ = [
    "HierarchicalFixedSegmentTrainSampler",
    "LocalLatentEpochOrderSampler",
    "LocalLatentWeightedTrainSampler",
]


class _WeightedLocalLatentSource(Protocol):
    """Dataset fields needed by replacement sampling."""

    data_config: DataConfig
    sample_weights: Sequence[float]

    def __len__(self) -> int: ...


class LocalLatentWeightedTrainSampler(WeightedReplacementDistributedSampler):
    """Replacement train sampler for weighted local latent examples."""

    def __init__(
        self,
        dataset: _WeightedLocalLatentSource,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            weights=dataset.sample_weights,
            base_seed=int(dataset.data_config.split_seed),
            world_size=world_size,
            rank=rank,
            empty_dataset_message=(
                "Weighted local latent sampling requires a non-empty dataset."
            ),
        )


class LocalLatentEpochOrderSampler(EpochOrderDistributedSampler):
    """Sampler backed by a dataset-provided epoch order."""

    def __init__(
        self,
        dataset: EpochOrderSource,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message=(
                "Epoch-order local latent sampling requires a non-empty dataset."
            ),
            empty_order_message=(
                "Epoch-order local latent sampler received an empty order."
            ),
        )


class HierarchicalFixedSegmentTrainSampler(EpochOffsetDistributedSampler):
    """Deterministic sampler for hierarchical fixed-segment draw keys."""

    def __init__(
        self,
        dataset: Sized,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message=(
                "Hierarchical fixed-segment sampling requires a non-empty dataset."
            ),
        )
