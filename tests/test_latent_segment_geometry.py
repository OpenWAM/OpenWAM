from __future__ import annotations

import pytest

from open_wam.data import (
    compact_boundary_start_range,
    resolve_compact_boundary_segment,
    resolve_rollout_parity_boundary_segment,
    rollout_parity_start_range,
)


def test_compact_boundary_preserves_conditioning_chunk_and_supervision() -> None:
    assert compact_boundary_start_range(
        source_latent_frames=10,
        segment_length=6,
        start_padding_frames=0,
        chunk_size=2,
    ) == (0, 7, 8)
    assert resolve_compact_boundary_segment(
        source_latent_frames=10,
        latent_start=0,
        segment_length=6,
        start_padding_frames=0,
        chunk_size=2,
    ) == {
        "logical_frame_start": 0,
        "logical_frame_end": 6,
        "target_frame_start": 0,
        "target_frame_end": 6,
        "effective_start": 0,
        "effective_end": 6,
        "effective_frame_start": 0,
        "effective_frame_end": 6,
        "effective_segment_frames": 6,
        "supervised_start": 0,
        "supervised_end": 6,
        "loss_frame_start": 2,
        "loss_frame_end": 6,
        "head_padded_frame_count": 0,
        "tail_padded_frame_count": 0,
        "startup_context_frames": 0,
        "context_prefix_frames_requested": 0,
        "context_prefix_frames_in_sample": 0,
        "context_prefix_real_frames": 0,
        "context_prefix_truncated_frames": 0,
        "chunk_size_for_boundary": 2,
        "compact_boundary_padding": True,
    }


def test_compact_boundary_aligns_prefixed_supervision_to_chunk_size() -> None:
    boundary = resolve_compact_boundary_segment(
        source_latent_frames=10,
        latent_start=3,
        segment_length=5,
        start_padding_frames=0,
        chunk_size=2,
        context_prefix_frames=1,
    )

    assert boundary["effective_frame_start"] == 2
    assert boundary["context_prefix_frames_in_sample"] == 1
    assert boundary["supervised_start"] == 1
    assert boundary["loss_frame_start"] == 2
    assert boundary["loss_frame_end"] == 6


def test_rollout_parity_keeps_one_real_context_frame_outside_loss() -> None:
    assert rollout_parity_start_range(source_latent_frames=10) == (1, 9, 9)
    assert resolve_rollout_parity_boundary_segment(
        source_latent_frames=10,
        latent_start=3,
        target_frame_count=4,
        context_frames=1,
        chunk_size=2,
    ) == {
        "logical_frame_start": 2,
        "logical_frame_end": 7,
        "target_frame_start": 3,
        "target_frame_end": 7,
        "effective_start": 2,
        "effective_end": 7,
        "effective_frame_start": 2,
        "effective_frame_end": 7,
        "effective_segment_frames": 5,
        "supervised_start": 1,
        "supervised_end": 5,
        "loss_frame_start": 1,
        "loss_frame_end": 5,
        "head_padded_frame_count": 0,
        "tail_padded_frame_count": 0,
        "startup_context_frames": 0,
        "context_prefix_frames_requested": 1,
        "context_prefix_frames_in_sample": 1,
        "context_prefix_real_frames": 1,
        "context_prefix_truncated_frames": 0,
        "chunk_size_for_boundary": 2,
        "compact_boundary_padding": True,
        "rollout_parity_target_alignment": True,
    }


def test_segment_geometry_rejects_sources_without_supervised_targets() -> None:
    assert compact_boundary_start_range(
        source_latent_frames=1,
        segment_length=4,
        start_padding_frames=0,
        chunk_size=1,
    ) == (0, -1, 0)
    assert rollout_parity_start_range(source_latent_frames=1) == (0, -1, 0)

    with pytest.raises(IndexError, match="not eligible"):
        resolve_rollout_parity_boundary_segment(
            source_latent_frames=1,
            latent_start=0,
            target_frame_count=4,
            context_frames=1,
            chunk_size=1,
        )
