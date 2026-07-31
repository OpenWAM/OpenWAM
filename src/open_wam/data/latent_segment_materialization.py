from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import LatentTemporalLayout

from .latent_segment_geometry import (
    resolve_compact_boundary_segment,
    resolve_rollout_parity_boundary_segment,
)
from .latent_temporal import (
    observed_frame_ids_for_latent_segment,
    raw_span_for_latent_range,
)


@dataclass(frozen=True)
class LatentSegmentMaterializationPlan:
    """Resolved tensor, raw-frame, padding, and supervision coordinates."""

    tensor_latent_start: int
    tensor_segment_length: int
    valid_latent_frames: int
    padded_latent_frames: int
    pre_start_frames: int
    loss_frame_start: int
    loss_frame_end: int
    sample_start_frame: int
    sample_end_frame: int
    anchor_frame_index: int
    observed_frame_ids: tuple[int, ...]
    boundary_metadata: dict[str, int | bool]
    chunk_size_for_boundary: int | None


def plan_latent_segment_materialization(
    *,
    source_latent_frames: int,
    raw_frame_ids: list[int],
    latent_start: int,
    segment_length: int,
    latent_temporal_layout: LatentTemporalLayout | str,
    start_padding_frames: int = 0,
    compact_boundary_padding: bool = False,
    compact_boundary_chunk_size: int | None = None,
    compact_boundary_context_prefix_frames: int = 0,
    rollout_parity_target_alignment: bool = False,
) -> LatentSegmentMaterializationPlan:
    """Resolve one segment before tensors and row supervision are assembled."""

    source_latent_frames = int(source_latent_frames)
    if source_latent_frames <= 0:
        raise ValueError("Uniform segment sampling requires at least one source latent frame.")
    start_padding_frames = max(0, int(start_padding_frames))

    chunk_size_for_boundary: int | None = None
    if compact_boundary_padding:
        chunk_size_for_boundary = max(
            1,
            int(
                compact_boundary_chunk_size
                if compact_boundary_chunk_size is not None
                else 1
            ),
        )
        if rollout_parity_target_alignment:
            boundary = resolve_rollout_parity_boundary_segment(
                source_latent_frames=source_latent_frames,
                latent_start=latent_start,
                target_frame_count=segment_length,
                context_frames=compact_boundary_context_prefix_frames,
                chunk_size=chunk_size_for_boundary,
            )
        else:
            boundary = resolve_compact_boundary_segment(
                source_latent_frames=source_latent_frames,
                latent_start=latent_start,
                segment_length=segment_length,
                start_padding_frames=start_padding_frames,
                chunk_size=chunk_size_for_boundary,
                context_prefix_frames=compact_boundary_context_prefix_frames,
            )
        tensor_latent_start = int(boundary["effective_start"])
        tensor_segment_length = int(boundary["effective_segment_frames"])
        loss_frame_start = int(boundary["loss_frame_start"])
        loss_frame_end = int(boundary["supervised_end"])
        pre_start_frames = int(boundary["startup_context_frames"])
        valid_latent_frames = tensor_segment_length
        padded_latent_frames = int(boundary["head_padded_frame_count"]) + int(
            boundary["tail_padded_frame_count"]
        )
    else:
        min_latent_start = -start_padding_frames
        if latent_start < min_latent_start or latent_start >= source_latent_frames:
            raise IndexError(
                f"latent_start={latent_start} is outside source_latent_frames={source_latent_frames}."
            )
        tensor_latent_start = int(latent_start)
        tensor_segment_length = int(segment_length)
        valid_latent_frames = max(
            0,
            min(segment_length, source_latent_frames - latent_start),
        )
        padded_latent_frames = max(0, segment_length - valid_latent_frames)
        pre_start_frames = 0
        if start_padding_frames > 0 and latent_start <= 0:
            pre_start_frames = max(0, min(segment_length, 1 - latent_start))
        loss_frame_start = min(pre_start_frames, valid_latent_frames)
        loss_frame_end = valid_latent_frames
        boundary = {
            "logical_frame_start": int(latent_start),
            "logical_frame_end": int(latent_start + segment_length),
            "effective_frame_start": int(tensor_latent_start),
            "effective_frame_end": int(tensor_latent_start + tensor_segment_length),
            "effective_segment_frames": int(tensor_segment_length),
            "head_padded_frame_count": 0,
            "tail_padded_frame_count": int(padded_latent_frames),
            "startup_context_frames": int(pre_start_frames),
            "compact_boundary_padding": False,
        }

    if tensor_segment_length <= 0:
        raise IndexError(
            f"latent_start={latent_start} is outside source_latent_frames={source_latent_frames}."
        )
    if not raw_frame_ids:
        raise ValueError("Uniform segment sampling requires non-empty frame ids.")

    observed_frame_ids = observed_frame_ids_for_latent_segment(
        raw_frame_ids=raw_frame_ids,
        source_latent_frames=source_latent_frames,
        latent_start=tensor_latent_start,
        segment_length=tensor_segment_length,
        layout=latent_temporal_layout,
    )
    source_latent_start = max(0, tensor_latent_start)
    valid_latent_end = min(
        source_latent_frames,
        max(0, tensor_latent_start + tensor_segment_length),
    )
    _, _, sample_start_frame, sample_end_frame = raw_span_for_latent_range(
        raw_frame_ids=raw_frame_ids,
        source_latent_frames=source_latent_frames,
        latent_start=source_latent_start,
        latent_end=valid_latent_end,
        layout=latent_temporal_layout,
    )
    return LatentSegmentMaterializationPlan(
        tensor_latent_start=tensor_latent_start,
        tensor_segment_length=tensor_segment_length,
        valid_latent_frames=valid_latent_frames,
        padded_latent_frames=padded_latent_frames,
        pre_start_frames=pre_start_frames,
        loss_frame_start=loss_frame_start,
        loss_frame_end=loss_frame_end,
        sample_start_frame=sample_start_frame,
        sample_end_frame=sample_end_frame,
        anchor_frame_index=observed_frame_ids[-1],
        observed_frame_ids=tuple(observed_frame_ids),
        boundary_metadata=dict(boundary),
        chunk_size_for_boundary=chunk_size_for_boundary,
    )


def slice_latent_segment_with_zero_order_hold(
    *,
    video_latents: torch.Tensor,
    latent_start: int,
    segment_length: int,
) -> torch.Tensor:
    """Slice a latent segment and hold the first/last source frame at its edges."""

    if latent_start < 0:
        source_indices = torch.arange(
            latent_start,
            latent_start + segment_length,
            dtype=torch.long,
            device=video_latents.device,
        ).clamp_(0, video_latents.shape[1] - 1)
        return video_latents.index_select(
            dim=1,
            index=source_indices,
        ).contiguous()
    latent_end = latent_start + segment_length
    valid_slice = video_latents[
        :,
        latent_start : min(latent_end, video_latents.shape[1]),
    ].contiguous()
    if valid_slice.shape[1] == segment_length:
        return valid_slice
    output = torch.zeros(
        video_latents.shape[0],
        segment_length,
        video_latents.shape[2],
        video_latents.shape[3],
        dtype=video_latents.dtype,
        device=video_latents.device,
    )
    if valid_slice.shape[1] > 0:
        output[:, : valid_slice.shape[1]] = valid_slice
        output[:, valid_slice.shape[1] :] = valid_slice[:, -1:].expand(
            -1,
            segment_length - valid_slice.shape[1],
            -1,
            -1,
        )
    return output.contiguous()
