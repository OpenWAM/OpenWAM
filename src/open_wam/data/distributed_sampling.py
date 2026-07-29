"""Generic deterministic samplers for sharded dataset index streams."""

from __future__ import annotations

from collections.abc import Iterator, Sequence, Sized
import math
from typing import Protocol

import torch
from torch.utils.data import Sampler


__all__ = [
    "DistributedIndexSampler",
    "EpochOffsetDistributedSampler",
    "EpochOrderDistributedSampler",
    "EpochOrderSource",
    "PaddedEpochOffsetDistributedSampler",
    "UnpaddedEpochOrderDistributedSampler",
    "WeightedReplacementDistributedSampler",
]


class EpochOrderSource(Protocol):
    """Dataset contract for constructing one deterministic global epoch order."""

    def __len__(self) -> int: ...

    def build_epoch_index_order(self, *, epoch: int) -> list[int]: ...


class DistributedIndexSampler(Sampler[int]):
    """Common rank sharding and epoch state for deterministic index samplers."""

    def __init__(
        self,
        dataset: Sized,
        *,
        world_size: int = 1,
        rank: int = 0,
        empty_dataset_message: str | None = "Distributed sampling requires a non-empty dataset.",
    ) -> None:
        if len(dataset) <= 0 and empty_dataset_message is not None:
            raise ValueError(empty_dataset_message)
        if world_size <= 0:
            raise ValueError(f"`world_size` must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"`rank` must be in [0, world_size), got rank={rank}, world_size={world_size}.")
        self.dataset = dataset
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.epoch = 0
        self._num_samples = int(math.ceil(len(dataset) / float(self.world_size)))
        self._total_size = self._num_samples * self.world_size

    def __len__(self) -> int:
        return self._num_samples

    @property
    def num_samples(self) -> int:
        """Number of indices emitted by this rank."""

        return self._num_samples

    @property
    def total_size(self) -> int:
        """Padded global index count shared by all ranks."""

        return self._total_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_shard(self, indices: Sequence[int]) -> Iterator[int]:
        return iter(int(index) for index in indices[self.rank : self._total_size : self.world_size])


class WeightedReplacementDistributedSampler(DistributedIndexSampler):
    """Deterministically sample weighted global indices, then shard by rank."""

    def __init__(
        self,
        dataset: Sized,
        *,
        weights: Sequence[float],
        base_seed: int,
        world_size: int = 1,
        rank: int = 0,
        empty_dataset_message: str = "Weighted replacement sampling requires a non-empty dataset.",
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message=empty_dataset_message,
        )
        if len(weights) != len(dataset):
            raise ValueError(
                "Weighted replacement sampling requires one weight per dataset item, "
                f"got weights={len(weights)}, dataset={len(dataset)}."
            )
        self.weights = tuple(float(weight) for weight in weights)
        self.base_seed = int(base_seed)

    def __iter__(self) -> Iterator[int]:
        weights = torch.tensor(self.weights, dtype=torch.double)
        if float(weights.sum().item()) <= 0:
            weights = torch.ones(len(self.dataset), dtype=torch.double)
        generator = torch.Generator()
        seed = (self.base_seed + self.epoch * 1_000_003) & 0x7FFF_FFFF_FFFF_FFFF
        generator.manual_seed(seed)
        sampled = torch.multinomial(
            weights,
            num_samples=self._total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return self._rank_shard(sampled)


class EpochOrderDistributedSampler(DistributedIndexSampler):
    """Shard a deterministic global order supplied by the dataset."""

    dataset: EpochOrderSource

    def __init__(
        self,
        dataset: EpochOrderSource,
        *,
        world_size: int = 1,
        rank: int = 0,
        empty_dataset_message: str | None = "Epoch-order sampling requires a non-empty dataset.",
        empty_order_message: str | None = "Epoch-order sampler received an empty order.",
        cache_order: bool = False,
        geometry_from_order: bool = False,
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message=empty_dataset_message,
        )
        self.empty_order_message = empty_order_message
        self.cache_order = bool(cache_order)
        self.geometry_from_order = bool(geometry_from_order)
        if self.geometry_from_order and not self.cache_order:
            raise ValueError("Order-derived sampler geometry requires `cache_order=True`.")
        self._epoch_order: tuple[int, ...] | None = None
        if self.cache_order:
            self._refresh_epoch_order()

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._epoch_order = None
        if self.cache_order:
            self._refresh_epoch_order()

    def _build_padded_epoch_order(self) -> tuple[int, ...]:
        order = self.dataset.build_epoch_index_order(epoch=self.epoch)
        if not order:
            if self.geometry_from_order:
                self._num_samples = 0
                self._total_size = 0
            if self.empty_order_message is not None:
                raise ValueError(self.empty_order_message)
            return ()
        if self.geometry_from_order:
            self._num_samples = int(math.ceil(len(order) / float(self.world_size)))
            self._total_size = self._num_samples * self.world_size
        if len(order) < self._total_size:
            repeats = int(math.ceil(self._total_size / len(order)))
            order = (order * repeats)[: self._total_size]
        else:
            order = order[: self._total_size]
        return tuple(int(index) for index in order)

    def _refresh_epoch_order(self) -> None:
        self._epoch_order = self._build_padded_epoch_order()

    def __iter__(self) -> Iterator[int]:
        order = self._epoch_order
        if order is None:
            order = self._build_padded_epoch_order()
        return self._rank_shard(order)


class UnpaddedEpochOrderDistributedSampler(DistributedIndexSampler):
    """Shard an epoch order without padding ranks to equal lengths."""

    dataset: EpochOrderSource

    def __init__(
        self,
        dataset: EpochOrderSource,
        *,
        world_size: int = 1,
        rank: int = 0,
        empty_dataset_message: str | None = "Unpadded epoch-order sampling requires a non-empty dataset.",
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message=empty_dataset_message,
        )

    def __len__(self) -> int:
        return len(range(self.rank, len(self.dataset), self.world_size))

    @property
    def num_samples(self) -> int:
        return len(self)

    @property
    def total_size(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        order = self.dataset.build_epoch_index_order(epoch=self.epoch)
        return iter(int(index) for index in order[self.rank :: self.world_size])


class EpochOffsetDistributedSampler(DistributedIndexSampler):
    """Emit epoch-offset global draw keys while preserving rank coordination."""

    def _epoch_span(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        epoch_offset = int(self.epoch) * self._epoch_span()
        return iter(
            epoch_offset + global_index
            for global_index in range(self.rank, self._total_size, self.world_size)
        )


class PaddedEpochOffsetDistributedSampler(EpochOffsetDistributedSampler):
    """Use the padded global size as the non-overlapping epoch-key span."""

    def _epoch_span(self) -> int:
        return self._total_size
