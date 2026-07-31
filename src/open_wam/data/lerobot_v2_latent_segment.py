"""Typed segment assembly for local LeRobot latent datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from open_wam.configs import DataConfig, LatentTemporalLayout, SampleStateAnchorMode

from .latent_segment_materialization import (
    plan_latent_segment_materialization,
    slice_latent_segment_with_zero_order_hold,
)
from .lerobot_v2_latent_storage import LocalEpisodeWindow
from .lerobot_v2_latent_supervision import LocalLatentSupervisionAssembler


__all__ = [
    "LocalLatentSegment",
    "LocalLatentSegmentAssembler",
]


@dataclass(frozen=True)
class LocalLatentSegment:
    """Materialized latent, supervision, and coordinate contract for one segment."""

    video_latents: torch.Tensor
    condition_latents: torch.Tensor | None
    actions: torch.Tensor
    action_mask: torch.Tensor
    action_target_metadata: dict[str, Any]
    state: torch.Tensor
    state_mask: torch.Tensor
    proprio_context_state: torch.Tensor
    proprio_context_state_mask: torch.Tensor
    proprio_context_frames: torch.Tensor
    proprio_context_frames_mask: torch.Tensor
    sample_start_frame: int
    sample_end_frame: int
    anchor_frame_index: int
    state_anchor_frame: int
    proprio_context_frame_index: int
    proprio_context_local_frame: int
    observed_frame_ids: list[int]
    latent_temporal_layout: LatentTemporalLayout | str
    action_start_index: int
    action_end_index: int
    valid_latent_frames: int
    padded_latent_frames: int
    loss_frame_start: int
    loss_frame_end: int
    start_padding_frames: int
    pre_start_frames: int
    boundary_metadata: dict[str, Any]


class LocalLatentSegmentAssembler:
    """Assemble one selected local latent segment without owning selection or I/O."""

    def __init__(
        self,
        data_config: DataConfig,
        *,
        supervision_assembler: LocalLatentSupervisionAssembler | None = None,
    ) -> None:
        self.data_config = data_config
        self.supervision_assembler = (
            supervision_assembler
            if supervision_assembler is not None
            else LocalLatentSupervisionAssembler(data_config)
        )

    def build(
        self,
        *,
        video_latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
        rows: list[dict[str, Any]],
        raw_frame_ids: Sequence[int],
        window: LocalEpisodeWindow,
        latent_start: int,
        segment_length: int,
        start_padding_frames: int,
        compact_boundary_padding: bool = False,
        compact_boundary_chunk_size: int | None = None,
        compact_boundary_context_prefix_frames: int = 0,
        rollout_parity_target_alignment: bool = False,
    ) -> LocalLatentSegment:
        """Materialize selected latents and aligned action/state supervision."""

        materialization = plan_latent_segment_materialization(
            source_latent_frames=int(video_latents.shape[1]),
            raw_frame_ids=raw_frame_ids,
            latent_start=latent_start,
            segment_length=segment_length,
            latent_temporal_layout=self.data_config.latent_temporal_layout,
            start_padding_frames=start_padding_frames,
            compact_boundary_padding=compact_boundary_padding,
            compact_boundary_chunk_size=(
                compact_boundary_chunk_size
                if compact_boundary_chunk_size is not None
                else self.data_config.sample_construction.chunk_size
            ),
            compact_boundary_context_prefix_frames=(
                compact_boundary_context_prefix_frames
            ),
            rollout_parity_target_alignment=rollout_parity_target_alignment,
        )
        observed_frame_ids = list(materialization.observed_frame_ids)
        sampled_window = LocalEpisodeWindow(
            repo_root=window.repo_root,
            episode_index=window.episode_index,
            start_frame=materialization.sample_start_frame,
            end_frame=min(materialization.sample_end_frame, len(rows)),
        )
        actions, action_mask, action_target_metadata = (
            self.supervision_assembler.build_lingbot_window_action_targets(
                rows=rows,
                window=sampled_window,
                observed_frame_ids=observed_frame_ids,
                latent_num_frames=materialization.tensor_segment_length,
                leading_zero_action_frames=(
                    materialization.pre_start_frames
                    if materialization.pre_start_frames > 0
                    else 1
                ),
                leading_zero_action_mask=(
                    0.0
                    if materialization.pre_start_frames > 0
                    or rollout_parity_target_alignment
                    else 1.0
                ),
            )
        )
        proprio_context_local_frame = max(
            0,
            min(
                len(observed_frame_ids) - 1,
                int(materialization.loss_frame_start) - 1,
            ),
        )
        proprio_context_frame_index = observed_frame_ids[
            proprio_context_local_frame
        ]
        state_anchor_frame = self.resolve_state_anchor_frame(
            observed_frame_ids=observed_frame_ids,
            sample_start_frame=materialization.sample_start_frame,
            anchor_frame_index=materialization.anchor_frame_index,
            proprio_context_frame_index=proprio_context_frame_index,
        )
        state, state_mask = (
            self.supervision_assembler.extract_state_history_at_frame(
                rows=rows,
                anchor_frame_index=state_anchor_frame,
            )
        )
        proprio_context_state, proprio_context_state_mask = (
            self.supervision_assembler.extract_proprio_context_state_sequence(
                rows=rows,
                observed_frame_ids=observed_frame_ids,
                chunk_size=(
                    int(materialization.chunk_size_for_boundary)
                    if compact_boundary_padding
                    and materialization.chunk_size_for_boundary is not None
                    else max(
                        1,
                        int(self.data_config.sample_construction.chunk_size),
                    )
                ),
                loss_frame_start=materialization.loss_frame_start,
            )
        )
        proprio_context_frames, proprio_context_frames_mask = (
            self.supervision_assembler.extract_proprio_context_frames(
                rows=rows,
                observed_frame_ids=observed_frame_ids,
            )
        )
        materialized_video_latents = slice_latent_segment_with_zero_order_hold(
            video_latents=video_latents,
            latent_start=materialization.tensor_latent_start,
            segment_length=materialization.tensor_segment_length,
        )
        materialized_condition_latents = (
            slice_latent_segment_with_zero_order_hold(
                video_latents=condition_latents,
                latent_start=materialization.tensor_latent_start,
                segment_length=materialization.tensor_segment_length,
            )
            if condition_latents is not None
            else None
        )
        return LocalLatentSegment(
            video_latents=materialized_video_latents,
            condition_latents=materialized_condition_latents,
            actions=actions,
            action_mask=action_mask,
            action_target_metadata=action_target_metadata,
            state=state,
            state_mask=state_mask,
            proprio_context_state=proprio_context_state,
            proprio_context_state_mask=proprio_context_state_mask,
            proprio_context_frames=proprio_context_frames,
            proprio_context_frames_mask=proprio_context_frames_mask,
            sample_start_frame=materialization.sample_start_frame,
            sample_end_frame=materialization.sample_end_frame,
            anchor_frame_index=materialization.anchor_frame_index,
            state_anchor_frame=state_anchor_frame,
            proprio_context_frame_index=proprio_context_frame_index,
            proprio_context_local_frame=proprio_context_local_frame,
            observed_frame_ids=observed_frame_ids,
            latent_temporal_layout=self.data_config.latent_temporal_layout,
            action_start_index=materialization.sample_start_frame,
            action_end_index=materialization.sample_start_frame
            + int(actions.shape[0]),
            valid_latent_frames=materialization.valid_latent_frames,
            padded_latent_frames=materialization.padded_latent_frames,
            loss_frame_start=materialization.loss_frame_start,
            loss_frame_end=materialization.loss_frame_end,
            start_padding_frames=start_padding_frames,
            pre_start_frames=materialization.pre_start_frames,
            boundary_metadata=materialization.boundary_metadata,
        )

    def resolve_state_anchor_frame(
        self,
        *,
        observed_frame_ids: Sequence[int],
        sample_start_frame: int,
        anchor_frame_index: int,
        proprio_context_frame_index: int | None = None,
    ) -> int:
        """Resolve which raw frame anchors the sample-level state history."""

        mode = self.data_config.sample_construction.state_anchor_mode
        if mode == SampleStateAnchorMode.PROPRIO_CONTEXT_FRAME:
            if proprio_context_frame_index is None:
                return int(anchor_frame_index)
            return int(proprio_context_frame_index)
        if mode == SampleStateAnchorMode.SAMPLE_START_FRAME:
            return int(sample_start_frame)
        if mode == SampleStateAnchorMode.FIRST_OBSERVED_FRAME:
            if not observed_frame_ids:
                raise ValueError(
                    "state_anchor_mode=first_observed_frame requires "
                    "non-empty observed_frame_ids."
                )
            return int(observed_frame_ids[0])
        if mode == SampleStateAnchorMode.ANCHOR_FRAME:
            return int(anchor_frame_index)
        raise ValueError(f"Unsupported sample state_anchor_mode {mode!r}.")
