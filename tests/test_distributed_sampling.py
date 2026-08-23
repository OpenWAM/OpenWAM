from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from open_wam.data import (
    EpochOffsetDistributedSampler as PublicEpochOffsetDistributedSampler,
    EpochOrderDistributedSampler as PublicEpochOrderDistributedSampler,
    HierarchicalSampleIndex as PublicHierarchicalSampleIndex,
    PaddedEpochOffsetDistributedSampler as PublicPaddedEpochOffsetDistributedSampler,
    UnpaddedEpochOrderDistributedSampler as PublicUnpaddedEpochOrderDistributedSampler,
    WeightedReplacementDistributedSampler as PublicWeightedReplacementDistributedSampler,
    draw_hierarchical_sample_index as public_draw_hierarchical_sample_index,
)
from open_wam.data.distributed_sampling import (
    EpochOffsetDistributedSampler,
    EpochOrderDistributedSampler,
    HierarchicalSampleIndex,
    PaddedEpochOffsetDistributedSampler,
    UnpaddedEpochOrderDistributedSampler,
    WeightedReplacementDistributedSampler,
    draw_hierarchical_sample_index,
    stable_int_seed,
    weighted_choice_index,
)
from open_wam.data.dynamics_routing import DynamicsRoutingDistributedSampler
from open_wam.data.factory import resolve_dataset_loader_spec
from open_wam.data.lerobot_consortium import ConsortiumTrainSampler
from open_wam.data.lerobot_v2_latent import (
    HierarchicalFixedSegmentTaskSpec as LegacyHierarchicalTaskSpec,
    HierarchicalFixedSegmentTrainSampler as LegacyHierarchicalTrainSampler,
    HierarchicalFixedSegmentWindowSpec as LegacyHierarchicalWindowSpec,
    LocalLatentEpochOrderSampler as LegacyLocalLatentEpochOrderSampler,
    LocalLatentWeightedTrainSampler as LegacyLocalLatentWeightedTrainSampler,
)
from open_wam.data.lerobot_v2_latent_sampling import (
    HierarchicalFixedSegmentTaskSpec,
    HierarchicalFixedSegmentTrainSampler,
    HierarchicalFixedSegmentWindowSpec,
    LocalLatentEpochOrderSampler,
    LocalLatentWeightedTrainSampler,
)
from open_wam.data.mixed_video import MixedVideoTrainSampler


class _SizedDataset:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length


class _EpochOrderDataset(_SizedDataset):
    def __init__(self, order: list[int]) -> None:
        super().__init__(len(order))
        self.order = order
        self.requested_epochs: list[int] = []

    def build_epoch_index_order(self, *, epoch: int) -> list[int]:
        self.requested_epochs.append(epoch)
        return list(self.order)


def test_distributed_samplers_are_public_identity_exports() -> None:
    assert PublicEpochOffsetDistributedSampler is EpochOffsetDistributedSampler
    assert PublicEpochOrderDistributedSampler is EpochOrderDistributedSampler
    assert PublicPaddedEpochOffsetDistributedSampler is PaddedEpochOffsetDistributedSampler
    assert PublicUnpaddedEpochOrderDistributedSampler is UnpaddedEpochOrderDistributedSampler
    assert PublicWeightedReplacementDistributedSampler is WeightedReplacementDistributedSampler
    assert PublicHierarchicalSampleIndex is HierarchicalSampleIndex
    assert public_draw_hierarchical_sample_index is draw_hierarchical_sample_index


def test_stable_int_seed_is_repeatable_across_signed_and_large_values() -> None:
    assert stable_int_seed(7, 17, 0) == 2258631717021994766
    assert stable_int_seed(7, 53, 1_000_003) == 8044077716504970128
    assert stable_int_seed(-1) == 8858027199621451829
    assert stable_int_seed(2**80, -(2**70), 17) == 2212125351011225462


def test_weighted_choice_preserves_rng_order_and_zero_mass_fallback() -> None:
    weighted_rng = random.Random(7)
    assert [weighted_choice_index((1.0, 2.0, 3.0), weighted_rng) for _ in range(6)] == [
        1,
        0,
        2,
        0,
        2,
        1,
    ]

    zero_rng = random.Random(7)
    assert [weighted_choice_index((0.0, 0.0, 0.0), zero_rng) for _ in range(6)] == [
        1,
        0,
        1,
        2,
        0,
        0,
    ]


def test_hierarchical_draw_selects_task_window_and_inclusive_start() -> None:
    draws = [
        draw_hierarchical_sample_index(
            seed_values=(123, 17, index),
            task_weights=(2.0, 5.0),
            task_specs=(
                SimpleNamespace(
                    windows=(
                        SimpleNamespace(mass_within_task=1.0, start_min=-3, start_max=5),
                        SimpleNamespace(mass_within_task=3.0, start_min=7, start_max=11),
                    )
                ),
                SimpleNamespace(
                    windows=(
                        SimpleNamespace(mass_within_task=0.0, start_min=100, start_max=100),
                        SimpleNamespace(mass_within_task=8.0, start_min=20, start_max=27),
                    )
                ),
            ),
        )
        for index in (0, 1, 2, 9, 1_000_003)
    ]

    assert draws == [
        HierarchicalSampleIndex(task_index=1, window_index=1, start=24),
        HierarchicalSampleIndex(task_index=0, window_index=0, start=4),
        HierarchicalSampleIndex(task_index=1, window_index=1, start=26),
        HierarchicalSampleIndex(task_index=1, window_index=1, start=26),
        HierarchicalSampleIndex(task_index=1, window_index=1, start=26),
    ]


@pytest.mark.parametrize(
    ("task_weights", "task_specs", "message"),
    (
        ((), (), "at least one task"),
        ((1.0,), (), "one task specification per task weight"),
        ((1.0,), (SimpleNamespace(windows=()),), "at least one window"),
    ),
)
def test_hierarchical_draw_rejects_inconsistent_tables(
    task_weights: tuple[float, ...],
    task_specs: tuple[SimpleNamespace, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        draw_hierarchical_sample_index(
            seed_values=(1, 17, 0),
            task_weights=task_weights,
            task_specs=task_specs,
        )


def test_dataset_samplers_are_thin_generic_contract_adapters() -> None:
    assert issubclass(LocalLatentWeightedTrainSampler, WeightedReplacementDistributedSampler)
    assert issubclass(LocalLatentEpochOrderSampler, EpochOrderDistributedSampler)
    assert issubclass(HierarchicalFixedSegmentTrainSampler, EpochOffsetDistributedSampler)
    assert issubclass(MixedVideoTrainSampler, EpochOrderDistributedSampler)
    assert issubclass(ConsortiumTrainSampler, UnpaddedEpochOrderDistributedSampler)
    assert issubclass(
        DynamicsRoutingDistributedSampler,
        PaddedEpochOffsetDistributedSampler,
    )


def test_loader_spec_uses_dataset_validation_sampler_hook() -> None:
    class _ValidationSamplerDataset(_SizedDataset):
        def build_validation_sampler(self, *, world_size: int, rank: int):
            return PaddedEpochOffsetDistributedSampler(
                self,
                world_size=world_size,
                rank=rank,
            )

    spec = resolve_dataset_loader_spec(
        _ValidationSamplerDataset(5),
        split="val",
        world_size=4,
        rank=2,
    )

    assert spec.shuffle is False
    assert isinstance(spec.sampler, PaddedEpochOffsetDistributedSampler)
    assert list(spec.sampler) == [2, 6]


def test_local_latent_sampling_compatibility_exports_preserve_identity() -> None:
    assert LegacyHierarchicalTaskSpec is HierarchicalFixedSegmentTaskSpec
    assert LegacyHierarchicalTrainSampler is HierarchicalFixedSegmentTrainSampler
    assert LegacyHierarchicalWindowSpec is HierarchicalFixedSegmentWindowSpec
    assert LegacyLocalLatentEpochOrderSampler is LocalLatentEpochOrderSampler
    assert LegacyLocalLatentWeightedTrainSampler is LocalLatentWeightedTrainSampler


def test_weighted_replacement_sampler_matches_global_torch_draw_and_rank_shards() -> None:
    dataset = _SizedDataset(6)
    weights = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    expected_generator = torch.Generator().manual_seed(23 + 2 * 1_000_003)
    expected = torch.multinomial(
        torch.tensor(weights, dtype=torch.double),
        num_samples=6,
        replacement=True,
        generator=expected_generator,
    ).tolist()

    rank_orders: list[list[int]] = []
    for rank in range(2):
        sampler = WeightedReplacementDistributedSampler(
            dataset,
            weights=weights,
            base_seed=23,
            world_size=2,
            rank=rank,
        )
        sampler.set_epoch(2)
        rank_orders.append(list(sampler))

    assert rank_orders[0] == expected[0::2]
    assert rank_orders[1] == expected[1::2]


def test_weighted_replacement_sampler_falls_back_to_uniform_for_zero_mass() -> None:
    dataset = _SizedDataset(4)
    sampler = WeightedReplacementDistributedSampler(
        dataset,
        weights=(0.0, 0.0, 0.0, 0.0),
        base_seed=11,
    )

    order = list(sampler)

    assert len(order) == len(dataset)
    assert all(0 <= index < len(dataset) for index in order)


def test_epoch_order_sampler_pads_before_rank_sharding() -> None:
    dataset = _EpochOrderDataset([4, 3, 2, 1, 0])
    rank_zero = EpochOrderDistributedSampler(dataset, world_size=2, rank=0)
    rank_one = EpochOrderDistributedSampler(dataset, world_size=2, rank=1)
    rank_zero.set_epoch(7)
    rank_one.set_epoch(7)

    assert list(rank_zero) == [4, 2, 0]
    assert list(rank_one) == [3, 1, 4]
    assert dataset.requested_epochs == [7, 7]


def test_epoch_order_sampler_can_cache_and_refresh_global_order() -> None:
    dataset = _EpochOrderDataset([3, 2, 1, 0])
    sampler = EpochOrderDistributedSampler(dataset, cache_order=True)

    assert dataset.requested_epochs == [0]
    assert list(sampler) == [3, 2, 1, 0]
    assert list(sampler) == [3, 2, 1, 0]
    assert dataset.requested_epochs == [0]

    sampler.set_epoch(4)

    assert dataset.requested_epochs == [0, 4]
    assert list(sampler) == [3, 2, 1, 0]
    assert dataset.requested_epochs == [0, 4]


def test_epoch_order_sampler_can_derive_rank_geometry_from_weighted_order() -> None:
    dataset = _EpochOrderDataset([6, 5, 4, 3, 2, 1, 0])
    dataset.length = 3
    sampler = EpochOrderDistributedSampler(
        dataset,
        world_size=2,
        rank=1,
        cache_order=True,
        geometry_from_order=True,
    )

    assert len(sampler) == 4
    assert sampler.total_size == 8
    assert list(sampler) == [5, 3, 1, 6]


def test_unpadded_epoch_order_sampler_preserves_uneven_rank_lengths() -> None:
    dataset = _EpochOrderDataset([4, 3, 2, 1, 0])
    rank_zero = UnpaddedEpochOrderDistributedSampler(dataset, world_size=2, rank=0)
    rank_one = UnpaddedEpochOrderDistributedSampler(dataset, world_size=2, rank=1)

    assert len(rank_zero) == 3
    assert len(rank_one) == 2
    assert rank_zero.num_samples == 3
    assert rank_one.num_samples == 2
    assert rank_zero.total_size == rank_one.total_size == 5
    assert list(rank_zero) == [4, 2, 0]
    assert list(rank_one) == [3, 1]


def test_epoch_offset_sampler_coordinates_nondivisible_rank_draw_keys() -> None:
    dataset = _SizedDataset(5)
    samplers = [
        EpochOffsetDistributedSampler(dataset, world_size=4, rank=rank)
        for rank in range(4)
    ]
    for sampler in samplers:
        sampler.set_epoch(1)

    assert [list(sampler) for sampler in samplers] == [
        [5, 9],
        [6, 10],
        [7, 11],
        [8, 12],
    ]


def test_padded_epoch_offset_sampler_uses_nonoverlapping_padded_epochs() -> None:
    dataset = _SizedDataset(5)
    samplers = [
        PaddedEpochOffsetDistributedSampler(dataset, world_size=4, rank=rank)
        for rank in range(4)
    ]
    for sampler in samplers:
        sampler.set_epoch(1)

    assert [list(sampler) for sampler in samplers] == [
        [8, 12],
        [9, 13],
        [10, 14],
        [11, 15],
    ]


@pytest.mark.parametrize(
    ("world_size", "rank", "message"),
    (
        (0, 0, "world_size"),
        (2, -1, "rank"),
        (2, 2, "rank"),
    ),
)
def test_distributed_sampler_rejects_invalid_rank_geometry(
    world_size: int,
    rank: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EpochOffsetDistributedSampler(
            _SizedDataset(1),
            world_size=world_size,
            rank=rank,
        )
