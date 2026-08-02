"""Uniform-segment dataset for local LeRobot latent repositories."""

from __future__ import annotations

import math
from typing import Any

from torch.utils.data import Sampler

from open_wam.configs import DataConfig, SampleOrderMode, WindowSamplingMode

from .latent_contracts import LatentWAMSample
from .lerobot_v2_latent_base_dataset import LocalLeRobotLatentWindowDataset
from .lerobot_v2_latent_sampler_adapters import (
    LocalLatentEpochOrderSampler,
    LocalLatentWeightedTrainSampler,
)
from .lerobot_v2_latent_storage import LocalEpisodeWindow
from .lerobot_v2_latent_uniform_policy import (
    LocalLatentUniformSegmentSamplingPlan as _LocalLatentUniformSegmentSamplingPlan,
)

__all__ = ["UniformSegmentLocalLeRobotLatentDataset"]


class UniformSegmentLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Uniform latent-start segment sampler over all eligible trajectories."""

    def __init__(
        self, data_config: DataConfig, windows: list[LocalEpisodeWindow]
    ) -> None:
        super().__init__(data_config, windows)
        plan = _LocalLatentUniformSegmentSamplingPlan.from_windows(
            data_config=data_config,
            windows=self.windows,
            window_task_texts=self._window_task_texts,
            task_demo_counts=self._task_demo_counts,
            dataset_mean_task_demo_count=self.dataset_mean_task_demo_count,
        )
        self._uniform_segment_sampling_plan = plan
        self._segment_length_candidates = plan.segment_length_candidates
        self._virtual_index = plan.virtual_index
        self._virtual_indices_by_window = plan.materialize_virtual_indices_by_window()
        self._task_virtual_start_counts = plan.materialize_task_virtual_start_counts()
        self.dataset_mean_task_virtual_start_count = (
            plan.dataset_mean_task_virtual_start_count
        )
        self.dataset_mean_valid_action_steps = plan.dataset_mean_valid_action_steps
        self.sample_weights = plan.sample_weights

    def __len__(self) -> int:
        return len(self._virtual_index)

    def build_train_sampler(
        self, *, world_size: int = 1, rank: int = 0
    ) -> Sampler[int]:
        if (
            self.data_config.sample_construction.sample_order_mode
            == SampleOrderMode.REPLACEMENT
        ):
            return LocalLatentWeightedTrainSampler(
                self, world_size=world_size, rank=rank
            )
        return LocalLatentEpochOrderSampler(self, world_size=world_size, rank=rank)

    def build_epoch_index_order(self, *, epoch: int) -> list[int]:
        return self._uniform_segment_sampling_plan.build_epoch_index_order(epoch=epoch)

    def _sample_weight_metadata(self, index: int) -> dict[str, Any]:
        return self._uniform_segment_sampling_plan.sample_weight_metadata(index)

    def __getitem__(self, index: int) -> LatentWAMSample:
        window_index, virtual_latent_start = self._virtual_index[index]
        window = self.windows[window_index]
        source = self._load_sample_source(
            window,
            include_condition_latents=True,
        )
        rows = source.rows
        full_video_latents = source.video_latents
        start_padding_frames = (
            self._uniform_segment_sampling_plan.resolve_start_padding_frames(
                self.data_config,
                window,
            )
        )
        segment_length, latent_start = (
            self._uniform_segment_sampling_plan.sample_segment_geometry(
                index=index,
                source_latent_frames=int(full_video_latents.shape[1]),
                virtual_latent_start=virtual_latent_start,
                start_padding_frames=start_padding_frames,
            )
        )
        sampled_chunk_size, sampled_window_size = (
            self._uniform_segment_sampling_plan.sample_attention_geometry(
                segment_length=segment_length
            )
        )
        segment = self._segment_assembler.build(
            video_latents=full_video_latents,
            condition_latents=source.condition_latents,
            rows=rows,
            raw_frame_ids=source.raw_frame_ids,
            window=window,
            latent_start=latent_start,
            segment_length=segment_length,
            start_padding_frames=start_padding_frames,
        )

        conditioning = source.conditioning_for_frame(
            segment.sample_start_frame,
            empty_text_embedding=self.empty_text_embedding,
        )

        return LatentWAMSample(
            video_latents=segment.video_latents,
            actions=segment.actions,
            action_mask=segment.action_mask,
            state=segment.state,
            state_mask=segment.state_mask,
            task_text=conditioning.task_text,
            text_context=conditioning.text_context,
            negative_text_context=conditioning.negative_text_context,
            condition_latents=segment.condition_latents,
            proprio_context_state=segment.proprio_context_state,
            proprio_context_state_mask=segment.proprio_context_state_mask,
            proprio_context_frames=segment.proprio_context_frames,
            proprio_context_frames_mask=segment.proprio_context_frames_mask,
            metadata={
                "repo_root": str(window.repo_root),
                "dataset_id": str(window.repo_root),
                "episode_index": window.episode_index,
                "segment_start_frame": window.start_frame,
                "segment_end_frame": window.end_frame,
                "sample_start_frame": segment.sample_start_frame,
                "sample_end_frame": segment.sample_end_frame,
                "observation_start": segment.sample_start_frame,
                "observation_frame_indices": segment.observed_frame_ids,
                "window_sampling_mode": WindowSamplingMode.UNIFORM_SEGMENT,
                "window_start_frame": segment.sample_start_frame,
                "window_end_frame": segment.sample_end_frame,
                "anchor_frame_index": segment.anchor_frame_index,
                "state_anchor_frame": segment.state_anchor_frame,
                "proprio_context_frame_index": segment.proprio_context_frame_index,
                "proprio_context_local_frame": segment.proprio_context_local_frame,
                "proprio_context_chunk_count": int(
                    segment.proprio_context_state.shape[0]
                ),
                "proprio_context_frame_count": int(
                    segment.proprio_context_frames.shape[0]
                ),
                "observed_frame_ids": segment.observed_frame_ids,
                "latent_temporal_layout": segment.latent_temporal_layout,
                "task_index": conditioning.task_index,
                "latent_layout": source.latent_layout_metadata,
                "condition_latent_layout": source.condition_layout_metadata,
                "has_condition_latents": segment.condition_latents is not None,
                "state_source_key": self.data_config.action_target.pose_source_key,
                "action_representation": self.data_config.action_target.representation,
                "virtual_sample_index": index,
                "trajectory_window_index": window_index,
                "virtual_latent_start": virtual_latent_start,
                "subwindow_latent_start": latent_start,
                "subwindow_latent_end": latent_start + segment_length,
                "segment_length_frames": segment_length,
                "segment_valid_latent_frames": segment.valid_latent_frames,
                "segment_padded_latent_frames": segment.padded_latent_frames,
                "tail_padding_mode": "none"
                if segment.padded_latent_frames == 0
                else "zero_hold",
                "subwindow_action_start": segment.action_start_index,
                "subwindow_action_end": segment.action_end_index,
                **self._uniform_segment_attention_metadata(
                    latent_start=latent_start,
                    segment_length=segment_length,
                    valid_latent_frames=segment.valid_latent_frames,
                    loss_frame_start=segment.loss_frame_start,
                    loss_frame_end=segment.loss_frame_end,
                    sample_start_frame=segment.sample_start_frame,
                    start_padding_frames=segment.start_padding_frames,
                    pre_start_frames=segment.pre_start_frames,
                    sampled_chunk_size=sampled_chunk_size,
                    sampled_window_size=sampled_window_size,
                ),
                **segment.action_target_metadata,
                **self._action_loss_metadata(segment.action_mask),
                **self._sample_weight_metadata(index),
            },
        )

    def _uniform_segment_attention_metadata(
        self,
        *,
        latent_start: int,
        segment_length: int,
        valid_latent_frames: int,
        loss_frame_start: int,
        loss_frame_end: int,
        sample_start_frame: int,
        start_padding_frames: int,
        pre_start_frames: int,
        emit_explicit_loss_ranges: bool = False,
        context_prefix_enabled: bool = False,
        sampled_chunk_size: int | None = None,
        sampled_window_size: int | None = None,
    ) -> dict[str, Any]:
        sample_cfg = self.data_config.sample_construction
        metadata: dict[str, Any] = {
            "latent_loss_frame_start": int(loss_frame_start),
            "latent_loss_frame_end": int(loss_frame_end),
            # Runtime grid ids use latent-frame positions. `sample_start_frame`
            # remains the raw dataset/action-row frame index.
            "latent_frame_start": int(latent_start),
            "frame_shift": int(latent_start),
            "start_padding_frames": int(start_padding_frames),
            "segment_pre_start_frames": int(pre_start_frames),
            "start_padding_mode": "repeat_first_latent"
            if int(pre_start_frames) > 0
            else "none",
        }
        if int(pre_start_frames) > 0 or bool(emit_explicit_loss_ranges):
            metadata.update(
                {
                    "loss_frame_start": int(loss_frame_start),
                    "loss_frame_end": int(loss_frame_end),
                    "action_loss_frame_start": int(loss_frame_start),
                    "action_loss_frame_end": int(loss_frame_end),
                }
            )
        chunk_size = max(
            1,
            int(
                sampled_chunk_size
                if sampled_chunk_size is not None
                else sample_cfg.chunk_size
            ),
        )
        window_size = max(
            1,
            int(
                sampled_window_size
                if sampled_window_size is not None
                else sample_cfg.window_size
            ),
        )
        if (
            chunk_size > 1
            or window_size > 1
            or sampled_chunk_size is not None
            or sampled_window_size is not None
        ):
            metadata["sampled_chunk_size"] = chunk_size
            metadata["sampled_window_size"] = window_size
            if emit_explicit_loss_ranges and bool(context_prefix_enabled):
                metadata["history_frames"] = max(
                    1, min(int(loss_frame_start), max(1, int(segment_length) - 1))
                )
            else:
                history_frames = int(math.ceil(window_size / 2.0)) * chunk_size  # noqa: RUF046
                metadata["history_frames"] = max(
                    1, min(history_frames, max(1, int(segment_length) - chunk_size))
                )
        return metadata
