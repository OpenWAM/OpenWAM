"""Hierarchical fixed-segment dataset for local LeRobot latent repositories."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from torch.utils.data import Sampler

from open_wam.configs import (
    DataConfig,
    PaddedTargetPolicy,
    SampleTargetAlignment,
    TailPaddingPolicy,
    WindowSamplingMode,
)

from .latent_contracts import LatentWAMSample
from .latent_hierarchical_sampling import LocalLatentHierarchicalSegmentPlan
from .lerobot_v2_latent_base_dataset import LocalLeRobotLatentWindowDataset
from .lerobot_v2_latent_sampler_adapters import HierarchicalFixedSegmentTrainSampler
from .lerobot_v2_latent_storage import LocalEpisodeWindow
from .lerobot_v2_latent_uniform_dataset import (
    UniformSegmentLocalLeRobotLatentDataset,
)
from .lerobot_v2_latent_uniform_policy import (
    LocalLatentUniformSegmentSamplingPlan as _LocalLatentUniformSegmentSamplingPlan,
)

__all__ = ["HierarchicalFixedSegmentLocalLeRobotLatentDataset"]


class HierarchicalFixedSegmentLocalLeRobotLatentDataset(
    UniformSegmentLocalLeRobotLatentDataset
):
    """Shared fixed-length hierarchical task/trajectory/start sampler."""

    def __init__(
        self, data_config: DataConfig, windows: list[LocalEpisodeWindow]
    ) -> None:
        LocalLeRobotLatentWindowDataset.__init__(self, data_config, windows)
        sample_cfg = self.data_config.sample_construction
        if sample_cfg.tail_padding_policy != TailPaddingPolicy.ZERO_ORDER_HOLD:
            raise ValueError(
                "Hierarchical fixed-segment sampling currently requires zero-order-hold tail padding."
            )
        if sample_cfg.padded_target_policy != PaddedTargetPolicy.MASK_LOSS:
            raise ValueError(
                "Hierarchical fixed-segment sampling currently requires masked padded targets."
            )
        if sample_cfg.segment_frames is None:
            raise ValueError(
                "Hierarchical fixed-segment sampling requires `sample_construction.segment_frames`."
            )
        if max(int(data_config.train_batch_size), int(data_config.val_batch_size)) > 1:
            raise ValueError(
                "Hierarchical fixed-segment compact boundary sampling currently requires "
                "`data.train_batch_size <= 1` and `data.val_batch_size <= 1` because the latent collate "
                "path stacks compact variable-length tensors directly."
            )
        hierarchical_plan = LocalLatentHierarchicalSegmentPlan.from_windows(
            data_config=data_config,
            windows=self.windows,
            window_task_texts=self._window_task_texts,
            task_demo_counts=self._task_demo_counts,
        )
        sampling_plan = hierarchical_plan.sampling_plan
        self.segment_frames = hierarchical_plan.segment_frames
        self._window_start_ranges_by_chunk = (
            hierarchical_plan.window_start_ranges_by_chunk
        )
        self._task_specs = sampling_plan.task_specs
        self._task_weights = sampling_plan.task_weights
        self._task_mass_total = sampling_plan.task_mass_total
        self._task_specs_by_text = sampling_plan.task_specs_by_text
        self._epoch_sample_count = sampling_plan.epoch_sample_count
        self._hierarchical_sampling_plan = sampling_plan
        self._hierarchical_segment_plan = hierarchical_plan

    def __len__(self) -> int:
        return self._epoch_sample_count

    def build_train_sampler(
        self, *, world_size: int = 1, rank: int = 0
    ) -> Sampler[int]:
        return HierarchicalFixedSegmentTrainSampler(
            self, world_size=world_size, rank=rank
        )

    def resolve_hierarchical_sample_key(self, index: int) -> dict[str, Any]:
        """Resolve one sampler/dataloader index without loading tensors."""

        return self._hierarchical_segment_plan.resolve_sample_key(index).as_metadata()

    def iter_hierarchical_eligible_start_keys(self) -> Iterator[tuple[int, int, int]]:
        """Yield every concrete trajectory/start/chunk key that must be reachable."""

        return self._hierarchical_segment_plan.iter_eligible_start_keys()

    def __getitem__(self, index: int) -> LatentWAMSample:
        task_spec, window_spec, latent_start, sampled_chunk_size = (
            self._hierarchical_segment_plan.draw(index)
        )
        window_index = int(window_spec.window_index)
        window = self.windows[window_index]
        source = self._load_sample_source(
            window,
            include_condition_latents=True,
        )
        rows = source.rows
        full_video_latents = source.video_latents
        start_padding_frames = (
            _LocalLatentUniformSegmentSamplingPlan.resolve_start_padding_frames(
                self.data_config,
                window,
            )
        )
        segment = self._segment_assembler.build(
            video_latents=full_video_latents,
            condition_latents=source.condition_latents,
            rows=rows,
            raw_frame_ids=source.raw_frame_ids,
            window=window,
            latent_start=latent_start,
            segment_length=self.segment_frames,
            start_padding_frames=start_padding_frames,
            compact_boundary_padding=True,
            compact_boundary_chunk_size=sampled_chunk_size,
            compact_boundary_context_prefix_frames=(
                self._hierarchical_segment_plan.context_prefix_frames(
                    sampled_chunk_size
                )
            ),
            rollout_parity_target_alignment=(
                self.data_config.sample_construction.target_alignment
                == SampleTargetAlignment.NEXT_AFTER_CONTEXT
            ),
        )

        conditioning = source.conditioning_for_frame(
            segment.sample_start_frame,
            empty_text_embedding=self.empty_text_embedding,
        )

        boundary_metadata = dict(segment.boundary_metadata)
        effective_latent_start = int(
            boundary_metadata.get("effective_frame_start", latent_start)
        )
        tail_padded_frame_count = int(
            boundary_metadata.get(
                "tail_padded_frame_count", segment.padded_latent_frames
            )
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
                "window_sampling_mode": WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
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
                "virtual_sample_index": int(index),
                "trajectory_window_index": window_index,
                "virtual_latent_start": latent_start,
                "subwindow_latent_start": latent_start,
                "subwindow_latent_end": latent_start + self.segment_frames,
                "segment_length_frames": self.segment_frames,
                "segment_valid_latent_frames": segment.valid_latent_frames,
                "segment_padded_latent_frames": segment.padded_latent_frames,
                "tail_padding_mode": "none"
                if tail_padded_frame_count == 0
                else "zero_order_hold",
                "subwindow_action_start": segment.action_start_index,
                "subwindow_action_end": segment.action_end_index,
                **self._uniform_segment_attention_metadata(
                    latent_start=effective_latent_start,
                    segment_length=int(
                        boundary_metadata.get(
                            "effective_segment_frames", self.segment_frames
                        )
                    ),
                    valid_latent_frames=segment.valid_latent_frames,
                    loss_frame_start=segment.loss_frame_start,
                    loss_frame_end=segment.loss_frame_end,
                    sample_start_frame=segment.sample_start_frame,
                    start_padding_frames=segment.start_padding_frames,
                    pre_start_frames=segment.pre_start_frames,
                    emit_explicit_loss_ranges=True,
                    context_prefix_enabled=int(
                        boundary_metadata.get("context_prefix_frames_requested", 0)
                    )
                    > 0,
                    sampled_chunk_size=sampled_chunk_size,
                    sampled_window_size=max(
                        1, int(self.data_config.sample_construction.window_size)
                    ),
                ),
                **boundary_metadata,
                **segment.action_target_metadata,
                **self._action_loss_metadata(
                    segment.action_mask,
                    loss_frame_start=segment.loss_frame_start,
                    loss_frame_end=segment.loss_frame_end,
                    latent_num_frames=int(
                        boundary_metadata.get(
                            "effective_segment_frames", self.segment_frames
                        )
                    ),
                ),
                **self._hierarchical_segment_plan.sample_metadata(
                    index=index,
                    task_spec=task_spec,
                    window_spec=window_spec,
                ),
            },
        )
