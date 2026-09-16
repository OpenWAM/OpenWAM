from __future__ import annotations

import random
import numpy as np
import torch
from open_wam.utils import seeding


def test_rng_snapshot_replays_python_numpy_and_torch_draws() -> None:
    outer_snapshot = seeding.snapshot_rng_state()
    try:
        random.seed(41)
        np.random.seed(42)
        torch.manual_seed(43)
        snapshot = seeding.snapshot_rng_state()
        expected = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )

        seeding.restore_rng_state(snapshot)
        actual = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )

        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert torch.equal(actual[2], expected[2])
    finally:
        seeding.restore_rng_state(outer_snapshot)


def test_preserve_rng_state_isolates_auxiliary_draws() -> None:
    outer_snapshot = seeding.snapshot_rng_state()
    try:
        random.seed(51)
        np.random.seed(52)
        torch.manual_seed(53)
        expected_snapshot = seeding.snapshot_rng_state()

        with seeding.preserve_rng_state():
            random.random()
            np.random.rand()
            torch.rand(3)

        expected = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )
        seeding.restore_rng_state(expected_snapshot)
        actual = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )

        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert torch.equal(actual[2], expected[2])
    finally:
        seeding.restore_rng_state(outer_snapshot)
