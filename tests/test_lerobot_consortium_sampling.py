from __future__ import annotations

import pickle
import random

import pytest

from open_wam.configs import ConsortiumRandomMode, ConsortiumWeightMode
from open_wam.data import ConsortiumEpochOrderPlan as PublicEpochOrderPlan
from open_wam.data.lerobot_consortium_sampling import ConsortiumEpochOrderPlan

MEMBER_INDICES = {
    "alpha": (0, 1, 2, 3, 4),
    "beta": (5, 6),
    "gamma": (7, 8, 9, 10),
}
MEMBER_WEIGHTS = {"alpha": 1.0, "beta": 3.0, "gamma": 0.5}


@pytest.mark.parametrize(
    ("weight_mode", "random_mode", "expected_epoch_0", "expected_epoch_1"),
    (
        (
            ConsortiumWeightMode.MANUAL_OVERRIDE,
            ConsortiumRandomMode.NONE,
            (5, 0, 6, 5, 7, 6, 1, 5, 6, 2, 5),
            (5, 0, 6, 5, 7, 6, 1, 5, 6, 2, 5),
        ),
        (
            ConsortiumWeightMode.MANUAL_OVERRIDE,
            ConsortiumRandomMode.WITHIN_DATASET,
            (5, 4, 6, 5, 8, 6, 3, 5, 6, 1, 5),
            (5, 4, 6, 5, 9, 6, 0, 5, 6, 1, 5),
        ),
        (
            ConsortiumWeightMode.MANUAL_OVERRIDE,
            ConsortiumRandomMode.TRAJECTORY_GLOBAL,
            (5, 1, 5, 6, 3, 6, 2, 6, 5, 8, 5),
            (6, 3, 2, 7, 5, 5, 6, 0, 6, 5, 6),
        ),
        (
            ConsortiumWeightMode.PROPORTIONAL_THEN_MANUAL_SCALE,
            ConsortiumRandomMode.NONE,
            (5, 0, 7, 6, 1, 5, 2, 6, 8, 3, 5),
            (5, 0, 7, 6, 1, 5, 2, 6, 8, 3, 5),
        ),
        (
            ConsortiumWeightMode.PROPORTIONAL_THEN_MANUAL_SCALE,
            ConsortiumRandomMode.WITHIN_DATASET,
            (5, 4, 8, 6, 3, 5, 1, 6, 10, 0, 5),
            (5, 4, 9, 6, 0, 5, 1, 6, 7, 3, 5),
        ),
        (
            ConsortiumWeightMode.PROPORTIONAL_THEN_MANUAL_SCALE,
            ConsortiumRandomMode.TRAJECTORY_GLOBAL,
            (6, 1, 6, 5, 3, 5, 2, 5, 8, 10, 4),
            (4, 3, 2, 8, 6, 6, 5, 0, 5, 6, 7),
        ),
        (
            ConsortiumWeightMode.PROPORTIONAL_TO_SIZE,
            ConsortiumRandomMode.NONE,
            (0, 7, 5, 1, 8, 2, 9, 3, 6, 10, 4),
            (0, 7, 5, 1, 8, 2, 9, 3, 6, 10, 4),
        ),
        (
            ConsortiumWeightMode.PROPORTIONAL_TO_SIZE,
            ConsortiumRandomMode.WITHIN_DATASET,
            (4, 8, 5, 3, 10, 1, 9, 0, 6, 7, 2),
            (4, 9, 5, 0, 7, 1, 8, 3, 6, 10, 2),
        ),
        (
            ConsortiumWeightMode.PROPORTIONAL_TO_SIZE,
            ConsortiumRandomMode.TRAJECTORY_GLOBAL,
            (5, 1, 8, 10, 3, 6, 2, 0, 9, 7, 4),
            (4, 3, 2, 10, 8, 1, 6, 0, 7, 5, 9),
        ),
    ),
)
def test_consortium_epoch_order_plan_preserves_frozen_schedules(
    weight_mode: ConsortiumWeightMode,
    random_mode: ConsortiumRandomMode,
    expected_epoch_0: tuple[int, ...],
    expected_epoch_1: tuple[int, ...],
) -> None:
    plan = ConsortiumEpochOrderPlan.from_member_indices(
        member_indices=MEMBER_INDICES,
        member_weights=MEMBER_WEIGHTS,
        random_mode=random_mode,
        weight_mode=weight_mode,
        sampling_seed=1729,
    )

    random.seed(314159)
    assert plan.build_epoch_index_order(epoch=0) == expected_epoch_0
    assert plan.build_epoch_index_order(epoch=1) == expected_epoch_1
    assert (random.random(), random.random()) == (
        0.19236379321481523,
        0.2868424512347926,
    )
    assert pickle.loads(pickle.dumps(plan)) == plan


def test_consortium_epoch_order_plan_is_public() -> None:
    assert PublicEpochOrderPlan is ConsortiumEpochOrderPlan
