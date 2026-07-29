from __future__ import annotations

import pytest
import torch

from open_wam.data import (
    EpochOffsetDistributedSampler as PublicEpochOffsetDistributedSampler,
    EpochOrderDistributedSampler as PublicEpochOrderDistributedSampler,
    WeightedReplacementDistributedSampler as PublicWeightedReplacementDistributedSampler,
)
from open_wam.data.distributed_sampling import (
    EpochOffsetDistributedSampler,
    EpochOrderDistributedSampler,
    WeightedReplacementDistributedSampler,
)


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
    assert PublicWeightedReplacementDistributedSampler is WeightedReplacementDistributedSampler


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
