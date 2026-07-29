from __future__ import annotations

import math
from typing import NotRequired, TypedDict


class LatentSegmentBoundary(TypedDict):
    """Materialized and supervised bounds in latent-frame coordinates.

    Logical bounds may extend beyond the source trajectory. Effective bounds
    identify materialized frames, while supervision and loss bounds are local
    offsets into that effective segment.
    """

    logical_frame_start: int
    logical_frame_end: int
    target_frame_start: int
    target_frame_end: int
    effective_start: int
    effective_end: int
    effective_frame_start: int
    effective_frame_end: int
    effective_segment_frames: int
    supervised_start: int
    supervised_end: int
    loss_frame_start: int
    loss_frame_end: int
    head_padded_frame_count: int
    tail_padded_frame_count: int
    startup_context_frames: int
    context_prefix_frames_requested: int
    context_prefix_frames_in_sample: int
    context_prefix_real_frames: int
    context_prefix_truncated_frames: int
    chunk_size_for_boundary: int
    compact_boundary_padding: bool
    rollout_parity_target_alignment: NotRequired[bool]


def compact_boundary_start_range(
    *,
    source_latent_frames: int,
    segment_length: int,
    start_padding_frames: int,
    chunk_size: int,
    context_prefix_frames: int = 0,
) -> tuple[int, int, int]:
    """Return the contiguous eligible target-start range."""

    source_latent_frames = int(source_latent_frames)
    segment_length = int(segment_length)
    start_padding_frames = max(0, int(start_padding_frames))
    chunk_size = max(1, int(chunk_size))
    context_prefix_frames = max(0, int(context_prefix_frames))
    if source_latent_frames + start_padding_frames <= max(chunk_size, start_padding_frames):
        return (0, -1, 0)
    if context_prefix_frames > 0:
        candidate_start_min = -start_padding_frames
    elif start_padding_frames > 0:
        candidate_start_min = max(chunk_size, start_padding_frames) - segment_length - start_padding_frames + 1
    else:
        candidate_start_min = 0
    eligible_starts: list[int] = []
    for latent_start in range(int(candidate_start_min), source_latent_frames):
        boundary = _compact_boundary_metadata_unchecked(
            source_latent_frames=source_latent_frames,
            latent_start=latent_start,
            segment_length=segment_length,
            start_padding_frames=start_padding_frames,
            chunk_size=chunk_size,
            context_prefix_frames=context_prefix_frames,
        )
        if (
            int(boundary["effective_segment_frames"]) > chunk_size
            and int(boundary["supervised_end"]) > int(boundary["loss_frame_start"])
        ):
            eligible_starts.append(int(latent_start))
    if not eligible_starts:
        return (0, -1, 0)
    start_min = min(eligible_starts)
    start_max = max(eligible_starts)
    eligible_start_count = len(eligible_starts)
    if eligible_start_count != start_max - start_min + 1:
        raise ValueError(
            "Compact boundary sampler expected contiguous eligible starts, got "
            f"start_min={start_min}, start_max={start_max}, eligible_count={eligible_start_count}."
        )
    return int(start_min), int(start_max), int(eligible_start_count)


def _compact_boundary_metadata_unchecked(
    *,
    source_latent_frames: int,
    latent_start: int,
    segment_length: int,
    start_padding_frames: int,
    chunk_size: int,
    context_prefix_frames: int = 0,
) -> LatentSegmentBoundary:
    source_latent_frames = int(source_latent_frames)
    latent_start = int(latent_start)
    segment_length = int(segment_length)
    start_padding_frames = max(0, int(start_padding_frames))
    chunk_size = max(1, int(chunk_size))
    context_prefix_frames = max(0, int(context_prefix_frames))

    target_start = int(latent_start)
    target_end = int(target_start + segment_length)
    logical_start = int(target_start - context_prefix_frames)
    logical_end = int(target_end)
    target_material_start = max(target_start, -start_padding_frames)
    if context_prefix_frames > 0:
        # Rollout-history prefix may only draw real pre-target frames.
        # Virtual startup frames are materialized only when they are part of
        # the sampled target segment itself, not to satisfy context.
        real_prefix_start = max(0, target_start - context_prefix_frames)
        effective_start = min(target_material_start, real_prefix_start)
    else:
        effective_start = target_material_start
    effective_end = min(logical_end, source_latent_frames)
    effective_segment_frames = effective_end - effective_start
    supervised_real_start = max(0, target_start)
    supervised_real_end = min(source_latent_frames, target_end)
    supervised_start = max(0, supervised_real_start - effective_start)
    supervised_end = max(supervised_start, supervised_real_end - effective_start)
    if context_prefix_frames > 0:
        aligned_supervised_start = int(math.ceil(float(supervised_start) / float(chunk_size))) * chunk_size
    else:
        aligned_supervised_start = int(supervised_start)
    loss_frame_start = max(chunk_size, aligned_supervised_start)
    real_context_start = max(0, effective_start)
    real_context_end = min(max(0, target_start), source_latent_frames, effective_end)
    real_context_frames = max(0, real_context_end - real_context_start)
    prefix_frames_in_sample = real_context_frames
    return {
        "logical_frame_start": int(logical_start),
        "logical_frame_end": int(logical_end),
        "target_frame_start": int(target_start),
        "target_frame_end": int(target_end),
        "effective_start": int(effective_start),
        "effective_end": int(effective_end),
        "effective_frame_start": int(effective_start),
        "effective_frame_end": int(effective_end),
        "effective_segment_frames": int(effective_segment_frames),
        "supervised_start": int(supervised_start),
        "supervised_end": int(supervised_end),
        "loss_frame_start": int(loss_frame_start),
        "loss_frame_end": int(supervised_end),
        "head_padded_frame_count": max(0, int(effective_start - logical_start)),
        "tail_padded_frame_count": max(0, int(logical_end - effective_end)),
        "startup_context_frames": max(0, min(0, effective_end) - effective_start),
        "context_prefix_frames_requested": int(context_prefix_frames),
        "context_prefix_frames_in_sample": int(prefix_frames_in_sample),
        "context_prefix_real_frames": int(real_context_frames),
        "context_prefix_truncated_frames": max(0, int(context_prefix_frames - prefix_frames_in_sample)),
        "chunk_size_for_boundary": int(chunk_size),
        "compact_boundary_padding": True,
    }


def resolve_compact_boundary_segment(
    *,
    source_latent_frames: int,
    latent_start: int,
    segment_length: int,
    start_padding_frames: int,
    chunk_size: int,
    context_prefix_frames: int = 0,
) -> LatentSegmentBoundary:
    """Resolve one eligible compact segment or reject its target start."""

    source_latent_frames = int(source_latent_frames)
    latent_start = int(latent_start)
    segment_length = int(segment_length)
    start_padding_frames = max(0, int(start_padding_frames))
    chunk_size = max(1, int(chunk_size))
    context_prefix_frames = max(0, int(context_prefix_frames))
    start_min, start_max, eligible_start_count = compact_boundary_start_range(
        source_latent_frames=source_latent_frames,
        segment_length=segment_length,
        start_padding_frames=start_padding_frames,
        chunk_size=chunk_size,
        context_prefix_frames=context_prefix_frames,
    )
    if eligible_start_count <= 0 or latent_start < start_min or latent_start > start_max:
        raise IndexError(
            "Compact boundary segment start is not eligible: "
            f"latent_start={latent_start}, start_min={start_min}, start_max={start_max}, "
            f"source_latent_frames={source_latent_frames}, segment_length={segment_length}, "
            f"start_padding_frames={start_padding_frames}, chunk_size={chunk_size}, "
            f"context_prefix_frames={context_prefix_frames}."
        )

    boundary = _compact_boundary_metadata_unchecked(
        source_latent_frames=source_latent_frames,
        latent_start=latent_start,
        segment_length=segment_length,
        start_padding_frames=start_padding_frames,
        chunk_size=chunk_size,
        context_prefix_frames=context_prefix_frames,
    )
    if int(boundary["effective_segment_frames"]) <= chunk_size or int(boundary["supervised_end"]) <= int(
        boundary["loss_frame_start"]
    ):
        raise IndexError(
            "Compact boundary segment has no supervised frame after the conditioning chunk: "
            f"latent_start={latent_start}, effective_segment_frames={boundary['effective_segment_frames']}, "
            f"supervised_start={boundary['supervised_start']}, supervised_end={boundary['supervised_end']}, "
            f"loss_frame_start={boundary['loss_frame_start']}, chunk_size={chunk_size}, "
            f"context_prefix_frames={context_prefix_frames}."
        )
    return boundary


def rollout_parity_start_range(*, source_latent_frames: int) -> tuple[int, int, int]:
    """Return eligible first targets for strict one-observation rollout parity."""

    source_latent_frames = int(source_latent_frames)
    if source_latent_frames <= 1:
        return (0, -1, 0)
    return (1, source_latent_frames - 1, source_latent_frames - 1)


def _rollout_parity_metadata_unchecked(
    *,
    source_latent_frames: int,
    latent_start: int,
    target_frame_count: int,
    context_frames: int,
    chunk_size: int,
) -> LatentSegmentBoundary:
    """Build strict rollout-parity sample bounds.

    `latent_start` is the first supervised/generated target frame. Context is
    materialized from real frames immediately before it and is never
    supervised. Missing future tail frames remain logical metadata only.
    """

    source_latent_frames = int(source_latent_frames)
    latent_start = int(latent_start)
    target_frame_count = int(target_frame_count)
    context_frames = max(1, int(context_frames))
    chunk_size = max(1, int(chunk_size))

    target_start = int(latent_start)
    target_end = int(target_start + target_frame_count)
    logical_start = int(target_start - context_frames)
    logical_end = int(target_end)
    effective_start = max(0, target_start - context_frames)
    effective_end = min(source_latent_frames, target_end)
    effective_segment_frames = effective_end - effective_start
    context_frames_in_sample = max(0, target_start - effective_start)
    supervised_start = context_frames_in_sample
    supervised_end = max(supervised_start, effective_end - effective_start)
    return {
        "logical_frame_start": int(logical_start),
        "logical_frame_end": int(logical_end),
        "target_frame_start": int(target_start),
        "target_frame_end": int(target_end),
        "effective_start": int(effective_start),
        "effective_end": int(effective_end),
        "effective_frame_start": int(effective_start),
        "effective_frame_end": int(effective_end),
        "effective_segment_frames": int(effective_segment_frames),
        "supervised_start": int(supervised_start),
        "supervised_end": int(supervised_end),
        "loss_frame_start": int(supervised_start),
        "loss_frame_end": int(supervised_end),
        "head_padded_frame_count": 0,
        "tail_padded_frame_count": max(0, int(logical_end - effective_end)),
        "startup_context_frames": 0,
        "context_prefix_frames_requested": int(context_frames),
        "context_prefix_frames_in_sample": int(context_frames_in_sample),
        "context_prefix_real_frames": int(context_frames_in_sample),
        "context_prefix_truncated_frames": max(0, int(context_frames - context_frames_in_sample)),
        "chunk_size_for_boundary": int(chunk_size),
        "compact_boundary_padding": True,
        "rollout_parity_target_alignment": True,
    }


def resolve_rollout_parity_boundary_segment(
    *,
    source_latent_frames: int,
    latent_start: int,
    target_frame_count: int,
    context_frames: int,
    chunk_size: int,
) -> LatentSegmentBoundary:
    """Resolve one strict rollout-parity target segment."""

    source_latent_frames = int(source_latent_frames)
    latent_start = int(latent_start)
    target_frame_count = int(target_frame_count)
    context_frames = max(1, int(context_frames))
    chunk_size = max(1, int(chunk_size))
    start_min, start_max, eligible_start_count = rollout_parity_start_range(
        source_latent_frames=source_latent_frames,
    )
    if eligible_start_count <= 0 or latent_start < start_min or latent_start > start_max:
        raise IndexError(
            "Rollout-parity segment start is not eligible: "
            f"latent_start={latent_start}, start_min={start_min}, start_max={start_max}, "
            f"source_latent_frames={source_latent_frames}, target_frame_count={target_frame_count}."
        )

    boundary = _rollout_parity_metadata_unchecked(
        source_latent_frames=source_latent_frames,
        latent_start=latent_start,
        target_frame_count=target_frame_count,
        context_frames=context_frames,
        chunk_size=chunk_size,
    )
    if int(boundary["context_prefix_frames_in_sample"]) <= 0:
        raise IndexError(
            "Rollout-parity fixed segment requires at least one real context frame before supervision."
        )
    if int(boundary["supervised_end"]) <= int(boundary["loss_frame_start"]):
        raise IndexError(
            "Rollout-parity fixed segment has no supervised real target frame: "
            f"latent_start={latent_start}, effective_segment_frames={boundary['effective_segment_frames']}, "
            f"loss_frame_start={boundary['loss_frame_start']}, loss_frame_end={boundary['loss_frame_end']}."
        )
    return boundary


__all__ = [
    "LatentSegmentBoundary",
    "compact_boundary_start_range",
    "resolve_compact_boundary_segment",
    "resolve_rollout_parity_boundary_segment",
    "rollout_parity_start_range",
]
