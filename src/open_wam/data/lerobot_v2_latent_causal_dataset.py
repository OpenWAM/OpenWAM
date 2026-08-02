"""Causal prefix/suffix dataset for local LeRobot latent repositories."""

from __future__ import annotations

import torch

from open_wam.configs import DataConfig, WindowSamplingMode

from .latent_causal_sampling import LatentCausalPrefixSuffixWindowPlanner
from .latent_contracts import LatentWAMSample
from .lerobot_v2_latent_base_dataset import LocalLeRobotLatentWindowDataset
from .lerobot_v2_latent_storage import LocalEpisodeWindow

__all__ = ["CausalPrefixSuffixLocalLeRobotLatentDataset"]


class CausalPrefixSuffixLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Bucketed causal prefix/suffix video-only samples over local latent exports."""

    def __init__(
        self,
        data_config: DataConfig,
        windows: list[LocalEpisodeWindow],
    ) -> None:
        super().__init__(data_config, windows)
        self._causal_sampling_planner = (
            LatentCausalPrefixSuffixWindowPlanner.from_data_config(data_config)
        )

    def __getitem__(self, index: int) -> LatentWAMSample:
        window = self.windows[index]
        source = self._load_sample_source(
            window,
            include_condition_latents=False,
        )
        rows = source.rows
        full_video_latents = source.video_latents
        plan = self._causal_sampling_planner.plan(
            raw_frame_ids=source.raw_frame_ids,
            source_latent_frames=int(full_video_latents.shape[1]),
            row_count=len(rows),
            sample_index=index,
        )
        if plan is None:
            buckets = self._causal_sampling_planner.buckets
            raise ValueError(
                "No valid causal prefix/suffix sample could be drawn from the local latent segment. "
                f"episode_index={window.episode_index}, latent_frames={full_video_latents.shape[1]}, "
                f"configured_buckets={[(bucket.observed_frames, bucket.future_frames) for bucket in buckets]}."
            )

        padded_latents = torch.zeros(
            full_video_latents.shape[0],
            plan.padded_video_frames,
            full_video_latents.shape[2],
            full_video_latents.shape[3],
            dtype=full_video_latents.dtype,
        )
        padded_latents[:, : plan.valid_video_frames] = full_video_latents[
            :, plan.latent_start : plan.latent_end
        ]
        video_latents = padded_latents.contiguous()
        actions = torch.zeros(
            self.data_config.action_schema.action_horizon,
            self.data_config.action_schema.action_dim,
            dtype=torch.float32,
        )
        action_mask = torch.zeros_like(actions)
        state = torch.zeros(
            self.data_config.action_schema.state_horizon,
            self.data_config.action_schema.state_dim,
            dtype=torch.float32,
        )
        state_mask = torch.zeros_like(state)
        observed_frame_ids = list(plan.observed_frame_ids)

        conditioning = source.conditioning_for_frame(
            plan.sample_start_frame,
            empty_text_embedding=self.empty_text_embedding,
        )

        return LatentWAMSample(
            video_latents=video_latents,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            task_text=conditioning.task_text,
            text_context=conditioning.text_context,
            negative_text_context=conditioning.negative_text_context,
            metadata={
                "repo_root": str(window.repo_root),
                "dataset_id": str(window.repo_root),
                "episode_index": window.episode_index,
                "segment_start_frame": window.start_frame,
                "segment_end_frame": window.end_frame,
                "sample_start_frame": plan.sample_start_frame,
                "sample_end_frame": plan.sample_end_frame,
                "observation_start": plan.sample_start_frame,
                "observation_frame_indices": observed_frame_ids,
                "window_sampling_mode": WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
                "window_start_frame": plan.sample_start_frame,
                "window_end_frame": plan.sample_end_frame,
                "anchor_frame_index": plan.sample_start_frame,
                "observed_frame_ids": observed_frame_ids,
                "latent_temporal_layout": plan.latent_temporal_layout,
                "task_index": conditioning.task_index,
                "latent_layout": source.latent_layout_metadata,
                "action_representation": self.data_config.action_target.representation,
                "subwindow_latent_start": plan.latent_start,
                "subwindow_latent_end": plan.latent_end,
                "observed_prefix_frames": plan.observed_prefix_frames,
                "future_suffix_frames": plan.future_suffix_frames,
                "valid_video_frames": plan.valid_video_frames,
                "padded_video_frames": int(video_latents.shape[1]),
                **self._action_loss_metadata(action_mask),
                **self._sample_weight_metadata(index),
            },
        )
