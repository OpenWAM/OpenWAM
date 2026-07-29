from __future__ import annotations

import pytest
import torch

from open_wam.data import pack_temporal_sequence as public_pack_temporal_sequence
from open_wam.data.sequence_packing import pack_temporal_sequence


def test_public_sequence_packer_is_the_canonical_function() -> None:
    assert public_pack_temporal_sequence is pack_temporal_sequence


@pytest.mark.parametrize("left_pad", [False, True])
def test_pack_temporal_sequence_preserves_values_and_builds_mask(left_pad: bool) -> None:
    sequence = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    output, mask = pack_temporal_sequence(
        sequence=sequence,
        target_dim=3,
        target_length=4,
        left_pad=left_pad,
        sequence_name="state",
    )

    start = 2 if left_pad else 0
    expected = torch.zeros(4, 3)
    expected[start : start + 2, :2] = sequence.detach().float()
    expected_mask = torch.zeros_like(expected)
    expected_mask[start : start + 2, :2] = 1.0
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert torch.equal(output, expected)
    assert torch.equal(mask, expected_mask)

    weights = torch.arange(output.numel(), dtype=output.dtype).reshape_as(output)
    (output * weights).sum().backward()
    assert torch.equal(sequence.grad, weights[start : start + 2, :2].double())


def test_pack_temporal_sequence_truncates_only_when_requested() -> None:
    sequence = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    output, mask = pack_temporal_sequence(
        sequence=sequence,
        target_dim=2,
        target_length=4,
        left_pad=True,
        truncate_to_target_length=True,
    )

    assert torch.equal(output, sequence[:4])
    assert torch.equal(mask, torch.ones_like(output))
    legacy_output, legacy_mask = pack_temporal_sequence(
        sequence=sequence,
        target_dim=2,
        target_length=4,
        left_pad=True,
    )
    assert torch.equal(legacy_output, sequence[-4:])
    assert torch.equal(legacy_mask, torch.ones_like(legacy_output))


def test_pack_temporal_sequence_rejects_invalid_rank_and_width() -> None:
    with pytest.raises(
        ValueError,
        match=r"Expected action tensor with shape \[T, D\], got \(2,\)\.",
    ):
        pack_temporal_sequence(
            sequence=torch.zeros(2),
            target_dim=2,
            target_length=2,
            sequence_name="action",
        )

    with pytest.raises(
        ValueError,
        match="Raw action dim 3 exceeds configured target dim 2",
    ):
        pack_temporal_sequence(
            sequence=torch.zeros(2, 3),
            target_dim=2,
            target_length=2,
            sequence_name="action",
        )
