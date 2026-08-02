"""Base and full-segment datasets for local LeRobot latent repositories."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from open_wam.configs import (
    DataConfig,
    LatentWindowProfile,
    SampleOrderMode,
    SampleWeightMode,
    WindowSamplingMode,
)

from .latent_contracts import LatentWAMSample
from .latent_temporal import (
    observed_frame_ids_for_latent_segment,
    raw_span_for_latent_range,
)
from .lerobot_v2_latent_sampler_adapters import LocalLatentWeightedTrainSampler
from .lerobot_v2_latent_segment import LocalLatentSegmentAssembler
from .lerobot_v2_latent_source import (
    LocalLatentSampleSource,
    LocalLatentSampleSourceLoader,
)
from .lerobot_v2_latent_storage import (
    LocalEpisodeWindow,
    LocalLatentRepository,
    load_empty_text_embedding,
)
from .lerobot_v2_latent_supervision import LocalLatentSupervisionAssembler
from .lerobot_v2_latent_weighting import LocalLatentWindowWeightPlan

__all__ = [
    "FullSegmentLocalLeRobotLatentDataset",
    "LocalLeRobotLatentWindowDataset",
]


class LocalLeRobotLatentWindowDataset(Dataset[LatentWAMSample]):
    """Latent-first local-repo dataset for LingBot-style post-training exports."""

    def __init__(
        self, data_config: DataConfig, windows: list[LocalEpisodeWindow]
    ) -> None:
        if data_config.local_root is None:
            raise ValueError(
                "Local latent datasets require `data.local_root` in the experiment config."
            )
        self.data_config = data_config
        self.windows = list(windows)
        self.empty_text_embedding = load_empty_text_embedding(data_config)
        repository = LocalLatentRepository(data_config)
        self._repo_bundles = repository.repo_bundles
        self._episode_cache = repository.episode_cache
        self._latent_view_cache = repository.latent_view_cache
        self._latent_repository = repository
        self._sample_source_loader = LocalLatentSampleSourceLoader(repository)
        self._supervision_assembler = LocalLatentSupervisionAssembler(data_config)
        self._segment_assembler = LocalLatentSegmentAssembler(
            data_config,
            supervision_assembler=self._supervision_assembler,
        )

        if not self.windows:
            raise ValueError(
                "No valid latent windows were constructed. "
                f"Check local_root={data_config.local_root!r} and latent_camera_names={data_config.latent_camera_names!r}."
            )
        weight_plan = LocalLatentWindowWeightPlan.from_windows(
            data_config=data_config,
            windows=self.windows,
            repo_bundles=self._repo_bundles,
        )
        self._window_weight_plan = weight_plan
        self._window_valid_action_steps = weight_plan.window_valid_action_steps
        self.dataset_mean_valid_action_steps = (
            weight_plan.dataset_mean_valid_action_steps
        )
        self._window_task_texts = weight_plan.window_task_texts
        self._task_demo_counts = weight_plan.task_demo_counts
        self.dataset_mean_task_demo_count = weight_plan.dataset_mean_task_demo_count
        self.sample_weights = weight_plan.sample_weights

    def __len__(self) -> int:
        return len(self.windows)

    def build_train_sampler(
        self, *, world_size: int = 1, rank: int = 0
    ) -> Sampler[int] | None:
        sample_cfg = self.data_config.sample_construction
        if (
            sample_cfg.sample_weight_mode == SampleWeightMode.UNIFORM
            and sample_cfg.sample_order_mode == SampleOrderMode.EPOCH_ORDER
        ):
            return None
        return LocalLatentWeightedTrainSampler(self, world_size=world_size, rank=rank)

    def _sample_weight_metadata(self, index: int) -> dict[str, Any]:
        return self._window_weight_plan.sample_weight_metadata(index)

    def task_text_for_window_index(self, index: int) -> str:
        """Return the resolved task label for one physical window."""

        return self._window_weight_plan.task_text_for_window_index(index)

    def _load_sample_source(
        self,
        window: LocalEpisodeWindow,
        *,
        include_condition_latents: bool,
    ) -> LocalLatentSampleSource:
        return self._sample_source_loader.load(
            window,
            include_condition_latents=include_condition_latents,
        )

    def _action_loss_metadata(
        self,
        action_mask: torch.Tensor | None,
        *,
        loss_frame_start: int | None = None,
        loss_frame_end: int | None = None,
        latent_num_frames: int | None = None,
    ) -> dict[str, Any]:
        if action_mask is None:
            valid_steps = int(self.data_config.action_schema.action_horizon)
            valid_values = valid_steps * int(self.data_config.action_schema.action_dim)
        else:
            effective_mask = action_mask.float()
            if (
                loss_frame_start is not None
                and loss_frame_end is not None
                and latent_num_frames is not None
                and int(latent_num_frames) > 0
                and effective_mask.shape[0] % int(latent_num_frames) == 0
            ):
                action_per_frame = effective_mask.shape[0] // int(latent_num_frames)
                frame_mask = torch.zeros_like(effective_mask)
                frame_start = max(0, int(loss_frame_start)) * action_per_frame
                frame_end = (
                    min(int(latent_num_frames), int(loss_frame_end)) * action_per_frame
                )
                if frame_end > frame_start:
                    frame_mask[frame_start:frame_end] = 1.0
                effective_mask = effective_mask * frame_mask
            reduced = effective_mask.sum(dim=-1)
            valid_steps = int((reduced > 0).sum().item())
            valid_values = int(effective_mask.sum().item())
        return {
            "valid_action_steps": valid_steps,
            "valid_action_values": valid_values,
            "dataset_mean_valid_action_steps": self.dataset_mean_valid_action_steps,
        }

    def __getitem__(self, index: int) -> LatentWAMSample:
        window = self.windows[index]
        source = self._load_sample_source(
            window,
            include_condition_latents=False,
        )
        rows = source.rows
        video_latents = source.video_latents
        raw_frame_ids = source.raw_frame_ids
        observed_frame_ids = observed_frame_ids_for_latent_segment(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=int(video_latents.shape[1]),
            latent_start=0,
            segment_length=int(video_latents.shape[1]),
            layout=self.data_config.latent_temporal_layout,
        )
        _, _, observation_start, observation_end = raw_span_for_latent_range(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=int(video_latents.shape[1]),
            latent_start=0,
            latent_end=int(video_latents.shape[1]),
            layout=self.data_config.latent_temporal_layout,
        )
        anchor_frame_index = observed_frame_ids[-1]
        sampled_window = LocalEpisodeWindow(
            repo_root=window.repo_root,
            episode_index=window.episode_index,
            start_frame=observation_start,
            end_frame=min(observation_end, len(rows)),
        )

        actions, action_mask, action_target_metadata = (
            self._build_full_segment_action_targets(
                rows=rows,
                window=sampled_window,
                observed_frame_ids=observed_frame_ids,
                latent_num_frames=int(video_latents.shape[1]),
            )
        )
        state, state_mask = self._supervision_assembler.extract_state_history_at_frame(
            rows=rows,
            anchor_frame_index=anchor_frame_index,
        )
        proprio_context_state, proprio_context_state_mask = (
            self._supervision_assembler.extract_proprio_context_state_sequence(
                rows=rows,
                observed_frame_ids=observed_frame_ids,
                chunk_size=1,
                loss_frame_start=0,
            )
        )

        conditioning = source.conditioning_for_frame(
            anchor_frame_index,
            empty_text_embedding=self.empty_text_embedding,
        )

        return LatentWAMSample(
            video_latents=video_latents,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            proprio_context_state=proprio_context_state,
            proprio_context_state_mask=proprio_context_state_mask,
            task_text=conditioning.task_text,
            text_context=conditioning.text_context,
            negative_text_context=conditioning.negative_text_context,
            metadata={
                "repo_root": str(window.repo_root),
                "dataset_id": str(window.repo_root),
                "episode_index": window.episode_index,
                "segment_start_frame": window.start_frame,
                "segment_end_frame": window.end_frame,
                "sample_start_frame": observation_start,
                "sample_end_frame": observation_end,
                "observation_start": observation_start,
                "observation_frame_indices": observed_frame_ids,
                "window_sampling_mode": WindowSamplingMode.FULL_SEGMENT,
                "window_start_frame": observation_start,
                "window_end_frame": observation_end,
                "anchor_frame_index": anchor_frame_index,
                "observed_frame_ids": observed_frame_ids,
                "latent_temporal_layout": self.data_config.latent_temporal_layout,
                "task_index": conditioning.task_index,
                "latent_layout": source.latent_layout_metadata,
                "state_source_key": self.data_config.action_target.pose_source_key,
                "action_representation": self.data_config.action_target.representation,
                "proprio_context_chunk_count": int(proprio_context_state.shape[0]),
                **action_target_metadata,
                **self._action_loss_metadata(action_mask),
                **self._sample_weight_metadata(index),
            },
        )

    def _build_full_segment_action_targets(
        self,
        *,
        rows: list[dict[str, Any]],
        window: LocalEpisodeWindow,
        observed_frame_ids: list[int],
        latent_num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if (
            self.data_config.latent_window_profile
            == LatentWindowProfile.EXACT_CHUNKED_WINDOW
        ):
            return self._supervision_assembler.build_lingbot_window_action_targets(
                rows=rows,
                window=window,
                observed_frame_ids=observed_frame_ids,
                latent_num_frames=latent_num_frames,
            )
        if (
            self.data_config.latent_window_profile
            == LatentWindowProfile.STANDARD_POLICY_WINDOW
        ):
            return (
                self._supervision_assembler.build_standard_policy_window_action_targets(
                    rows=rows,
                    observation_start=int(observed_frame_ids[0]),
                )
            )
        raise ValueError(
            f"Unsupported latent_window_profile: {self.data_config.latent_window_profile!r}"
        )


class FullSegmentLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Current LingBot-style long-window latent dataset view."""

    def __init__(
        self, data_config: DataConfig, windows: list[LocalEpisodeWindow]
    ) -> None:
        super().__init__(data_config, windows)
        self.sample_index = tuple(self.windows)
