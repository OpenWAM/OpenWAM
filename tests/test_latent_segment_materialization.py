from __future__ import annotations

import re

import pytest
import torch

from open_wam.configs import LatentTemporalLayout
from open_wam.data import (
    LatentSegmentMaterializationPlan as PublicLatentSegmentMaterializationPlan,
    plan_latent_segment_materialization as public_plan_latent_segment_materialization,
    slice_latent_segment_with_zero_order_hold as public_slice_latent_segment_with_zero_order_hold,
)
from open_wam.data.latent_segment_materialization import (
    LatentSegmentMaterializationPlan,
    plan_latent_segment_materialization,
    slice_latent_segment_with_zero_order_hold,
)


def test_latent_segment_materialization_contract_is_public() -> None:
    assert PublicLatentSegmentMaterializationPlan is LatentSegmentMaterializationPlan
    assert public_plan_latent_segment_materialization is plan_latent_segment_materialization
    assert (
        public_slice_latent_segment_with_zero_order_hold
        is slice_latent_segment_with_zero_order_hold
    )


def test_plan_regular_segment_tracks_tail_hold_and_raw_alignment() -> None:
    plan = plan_latent_segment_materialization(
        source_latent_frames=5,
        raw_frame_ids=list(range(17)),
        latent_start=3,
        segment_length=4,
        latent_temporal_layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
    )

    assert plan == LatentSegmentMaterializationPlan(
        tensor_latent_start=3,
        tensor_segment_length=4,
        valid_latent_frames=2,
        padded_latent_frames=2,
        pre_start_frames=0,
        loss_frame_start=0,
        loss_frame_end=2,
        sample_start_frame=9,
        sample_end_frame=17,
        anchor_frame_index=16,
        observed_frame_ids=(12, 16, 16, 16),
        boundary_metadata={
            "logical_frame_start": 3,
            "logical_frame_end": 7,
            "effective_frame_start": 3,
            "effective_frame_end": 7,
            "effective_segment_frames": 4,
            "head_padded_frame_count": 0,
            "tail_padded_frame_count": 2,
            "startup_context_frames": 0,
            "compact_boundary_padding": False,
        },
        chunk_size_for_boundary=None,
    )


def test_plan_regular_segment_tracks_startup_hold_and_loss_boundary() -> None:
    plan = plan_latent_segment_materialization(
        source_latent_frames=5,
        raw_frame_ids=list(range(17)),
        latent_start=-2,
        segment_length=4,
        latent_temporal_layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
        start_padding_frames=2,
    )

    assert plan.tensor_latent_start == -2
    assert plan.tensor_segment_length == 4
    assert plan.valid_latent_frames == 4
    assert plan.padded_latent_frames == 0
    assert plan.pre_start_frames == 3
    assert plan.loss_frame_start == 3
    assert plan.loss_frame_end == 4
    assert plan.sample_start_frame == 0
    assert plan.sample_end_frame == 5
    assert plan.observed_frame_ids == (0, 0, 0, 4)
    assert plan.anchor_frame_index == 4


def test_plan_rollout_parity_segment_keeps_context_and_supervision_separate() -> None:
    plan = plan_latent_segment_materialization(
        source_latent_frames=12,
        raw_frame_ids=list(range(45)),
        latent_start=5,
        segment_length=5,
        latent_temporal_layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
        compact_boundary_padding=True,
        compact_boundary_chunk_size=3,
        compact_boundary_context_prefix_frames=1,
        rollout_parity_target_alignment=True,
    )

    assert plan.tensor_latent_start == 4
    assert plan.tensor_segment_length == 6
    assert plan.loss_frame_start == 1
    assert plan.loss_frame_end == 6
    assert plan.pre_start_frames == 0
    assert plan.chunk_size_for_boundary == 3
    assert plan.boundary_metadata["target_frame_start"] == 5
    assert plan.boundary_metadata["context_prefix_frames_in_sample"] == 1
    assert plan.boundary_metadata["rollout_parity_target_alignment"] is True


@pytest.mark.parametrize(
    ("latent_start", "segment_length", "expected", "expected_gradient"),
    [
        (-2, 5, [1.0, 1.0, 1.0, 2.0, 3.0], [3.0, 1.0, 1.0]),
        (1, 4, [2.0, 3.0, 3.0, 3.0], [0.0, 1.0, 3.0]),
        (0, 3, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
    ],
)
def test_slice_latent_segment_zero_order_hold_preserves_values_and_gradients(
    latent_start: int,
    segment_length: int,
    expected: list[float],
    expected_gradient: list[float],
) -> None:
    source = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1).requires_grad_()
    result = slice_latent_segment_with_zero_order_hold(
        video_latents=source,
        latent_start=latent_start,
        segment_length=segment_length,
    )

    assert result.is_contiguous()
    assert result.flatten().tolist() == expected
    result.sum().backward()
    assert source.grad is not None
    assert source.grad.flatten().tolist() == expected_gradient


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {
                "source_latent_frames": 0,
                "raw_frame_ids": [0],
                "latent_start": 0,
                "segment_length": 1,
            },
            ValueError,
            "Uniform segment sampling requires at least one source latent frame.",
        ),
        (
            {
                "source_latent_frames": 3,
                "raw_frame_ids": [0, 1, 2],
                "latent_start": -1,
                "segment_length": 1,
            },
            IndexError,
            "latent_start=-1 is outside source_latent_frames=3.",
        ),
        (
            {
                "source_latent_frames": 3,
                "raw_frame_ids": [],
                "latent_start": 0,
                "segment_length": 1,
            },
            ValueError,
            "Uniform segment sampling requires non-empty frame ids.",
        ),
    ],
)
def test_plan_latent_segment_materialization_preserves_errors(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=f"^{re.escape(message)}$"):
        plan_latent_segment_materialization(
            **kwargs,
            latent_temporal_layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
        )
