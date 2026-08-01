from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from open_wam.configs import (
    DataConfig,
    DataSplit,
    LatentWindowProfile,
    PaddedTargetPolicy,
    SampleOrderMode,
    SampleWeightMode,
    SampleTargetAlignment,
    TailPaddingPolicy,
    WindowSamplingMode,
)

from .latent_causal_sampling import LatentCausalPrefixSuffixWindowPlanner
from .latent_contracts import LatentWAMSample
from .latent_hierarchical_sampling import LocalLatentHierarchicalSegmentPlan
from .latent_temporal import (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET
    as CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET,
    latent_anchor_positions,
    observed_frame_ids_for_latent_segment,
    raw_span_for_latent_range,
)
from .lerobot_v2 import LeRobotV2Metadata
# Keep storage symbols importable from this historical module while ownership
# lives in the repository adapter.
from .lerobot_v2_latent_storage import (
    LocalEpisodeWindow,
    LocalLatentRepository,
    LocalRepoBundle as LocalRepoBundle,
    assemble_canonical_latents,
    condition_latent_offset_mismatches,
    discover_local_lerobot_repo_bundles,
    latent_filename as latent_filename,
    load_empty_text_embedding,
    load_lerobot_v2_local_metadata as load_lerobot_v2_local_metadata,
    read_json_local as read_json_local,
    read_jsonl_local as read_jsonl_local,
    reshape_latent_payload as reshape_latent_payload,
    resolve_latent_root as resolve_latent_root,
    scan_local_latent_windows,
    split_local_episode_indices as split_local_episode_indices,
)
from .lerobot_v2_latent_sampling import (
    HierarchicalFixedSegmentSamplingPlan,
    HierarchicalFixedSegmentTaskSpec as HierarchicalFixedSegmentTaskSpec,
    HierarchicalFixedSegmentTrainSampler
    as HierarchicalFixedSegmentTrainSampler,
    HierarchicalFixedSegmentWindowSpec as HierarchicalFixedSegmentWindowSpec,
    LocalLatentEpochOrderSampler as LocalLatentEpochOrderSampler,
    LocalLatentUniformSegmentSamplingPlan as _LocalLatentUniformSegmentSamplingPlan,
    LocalLatentWindowWeightPlan,
    LocalLatentWeightedTrainSampler as LocalLatentWeightedTrainSampler,
    build_hierarchical_fixed_segment_task_specs,
)
from .lerobot_v2_latent_segment import LocalLatentSegmentAssembler
from .lerobot_v2_latent_split import LocalLatentTrainValWindowPlanner
from .lerobot_v2_latent_supervision import LocalLatentSupervisionAssembler

_COMPATIBILITY_EXPORTS = (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET,
    HierarchicalFixedSegmentSamplingPlan,
    HierarchicalFixedSegmentTaskSpec,
    HierarchicalFixedSegmentTrainSampler,
    HierarchicalFixedSegmentWindowSpec,
    LocalLatentEpochOrderSampler,
    LocalLatentWeightedTrainSampler,
    build_hierarchical_fixed_segment_task_specs,
    discover_local_lerobot_repo_bundles,
    latent_anchor_positions,
    LocalRepoBundle,
    latent_filename,
    load_lerobot_v2_local_metadata,
    read_json_local,
    read_jsonl_local,
    reshape_latent_payload,
    resolve_latent_root,
    scan_local_latent_windows,
    split_local_episode_indices,
)


class LocalLeRobotLatentWindowDataset(Dataset[LatentWAMSample]):
    """Latent-first local-repo dataset for LingBot-style post-training exports."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
        if data_config.local_root is None:
            raise ValueError("Local latent datasets require `data.local_root` in the experiment config.")
        self.data_config = data_config
        self.windows = list(windows)
        self.empty_text_embedding = self._load_empty_text_embedding()
        repository = LocalLatentRepository(data_config)
        self._repo_bundles = repository.repo_bundles
        self._episode_cache = repository.episode_cache
        self._latent_view_cache = repository.latent_view_cache
        self._latent_repository = repository
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
        self.dataset_mean_task_demo_count = (
            weight_plan.dataset_mean_task_demo_count
        )
        self.sample_weights = weight_plan.sample_weights

    def __len__(self) -> int:
        return len(self.windows)

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int] | None:
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
                frame_end = min(int(latent_num_frames), int(loss_frame_end)) * action_per_frame
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
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        latent_payloads = self._load_window_latents(window, repo_bundle.metadata)
        video_latents, latent_layout_metadata = self._assemble_canonical_latents(latent_payloads)
        assert video_latents is not None

        primary_payload = latent_payloads[self.data_config.latent_camera_names[0]]
        raw_frame_ids = [int(value) for value in list(primary_payload.get("frame_ids", []))]
        if not raw_frame_ids:
            raw_frame_ids = list(window.observation_frame_indices)
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

        actions, action_mask, action_target_metadata = self._build_full_segment_action_targets(
            rows=rows,
            window=sampled_window,
            observed_frame_ids=observed_frame_ids,
            latent_num_frames=int(video_latents.shape[1]),
        )
        state, state_mask = self._extract_state_history_at_frame(
            rows=rows,
            anchor_frame_index=anchor_frame_index,
        )
        proprio_context_state, proprio_context_state_mask = self._extract_proprio_context_state_sequence(
            rows=rows,
            observed_frame_ids=observed_frame_ids,
            chunk_size=1,
            loss_frame_start=0,
        )

        text_context = primary_payload.get("text_emb")
        if isinstance(text_context, torch.Tensor):
            text_context = text_context.to(dtype=torch.float32)
        else:
            text_context = None
        negative_text_context = self.empty_text_embedding.clone() if self.empty_text_embedding is not None else None

        episode_record = repo_bundle.episodes_by_index.get(window.episode_index)
        task_index = int(rows[min(anchor_frame_index, len(rows) - 1)].get("task_index", 0)) if rows else 0
        task_text = repo_bundle.metadata.tasks_by_index.get(task_index)
        if task_text is None and episode_record is not None and episode_record.tasks:
            task_text = episode_record.tasks[0]

        return LatentWAMSample(
            video_latents=video_latents,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            proprio_context_state=proprio_context_state,
            proprio_context_state_mask=proprio_context_state_mask,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
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
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
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
        if self.data_config.latent_window_profile == LatentWindowProfile.EXACT_CHUNKED_WINDOW:
            return self._build_lingbot_window_action_targets(
                rows=rows,
                window=window,
                observed_frame_ids=observed_frame_ids,
                latent_num_frames=latent_num_frames,
            )
        if self.data_config.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW:
            return self._build_standard_policy_window_action_targets(
                rows=rows,
                observation_start=int(observed_frame_ids[0]),
            )
        raise ValueError(f"Unsupported latent_window_profile: {self.data_config.latent_window_profile!r}")

    def _build_standard_policy_window_action_targets(
        self,
        *,
        rows: list[dict[str, Any]],
        observation_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self._supervision_assembler.build_standard_policy_window_action_targets(
            rows=rows,
            observation_start=observation_start,
        )

    def _load_empty_text_embedding(self) -> torch.Tensor | None:
        return load_empty_text_embedding(self.data_config)

    def _load_window_latents(
        self,
        window: LocalEpisodeWindow,
        metadata: LeRobotV2Metadata,
    ) -> dict[str, dict[str, Any]]:
        return self._latent_repository.load_window_latents(window, metadata)

    def _assemble_canonical_latents(
        self,
        latent_payloads: dict[str, dict[str, Any]],
        *,
        payload_key: str = "latent",
        require_payload_key: bool = True,
    ) -> tuple[torch.Tensor | None, dict[str, dict[str, int]]]:
        return assemble_canonical_latents(
            self.data_config,
            latent_payloads,
            payload_key=payload_key,
            require_payload_key=require_payload_key,
        )

    def _condition_latent_offset_mismatches(
        self,
        latent_payloads: dict[str, dict[str, Any]],
        *,
        expected_offset: int,
    ) -> list[str]:
        return condition_latent_offset_mismatches(
            self.data_config,
            latent_payloads,
            expected_offset=expected_offset,
        )

    def _load_canonical_window_latents(
        self,
        window: LocalEpisodeWindow,
        metadata: LeRobotV2Metadata,
    ) -> tuple[
        torch.Tensor,
        dict[str, dict[str, int]],
        dict[str, Any],
        torch.Tensor | None,
        dict[str, dict[str, int]],
    ]:
        return self._latent_repository.load_canonical_window_latents(
            window,
            metadata,
        )

    def _build_lingbot_window_action_targets(
        self,
        *,
        rows: list[dict[str, Any]],
        window: LocalEpisodeWindow,
        observed_frame_ids: list[int],
        latent_num_frames: int,
        leading_zero_action_frames: int = 1,
        leading_zero_action_mask: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self._supervision_assembler.build_lingbot_window_action_targets(
            rows=rows,
            window=window,
            observed_frame_ids=observed_frame_ids,
            latent_num_frames=latent_num_frames,
            leading_zero_action_frames=leading_zero_action_frames,
            leading_zero_action_mask=leading_zero_action_mask,
        )

    def _extract_state_history_at_frame(
        self,
        *,
        rows: list[dict[str, Any]],
        anchor_frame_index: int,
        state_horizon: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._supervision_assembler.extract_state_history_at_frame(
            rows=rows,
            anchor_frame_index=anchor_frame_index,
            state_horizon=state_horizon,
        )

    def _extract_proprio_context_state_sequence(
        self,
        *,
        rows: list[dict[str, Any]],
        observed_frame_ids: list[int],
        chunk_size: int,
        loss_frame_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._supervision_assembler.extract_proprio_context_state_sequence(
            rows=rows,
            observed_frame_ids=observed_frame_ids,
            chunk_size=chunk_size,
            loss_frame_start=loss_frame_start,
        )

    def _load_episode_rows(
        self,
        repo_root: Path,
        episode_index: int,
        metadata: LeRobotV2Metadata,
    ) -> list[dict[str, Any]]:
        return self._latent_repository.load_episode_rows(
            repo_root,
            episode_index,
            metadata,
        )

class UniformSegmentLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Uniform latent-start segment sampler over all eligible trajectories."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
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
        self._virtual_indices_by_window = (
            plan.materialize_virtual_indices_by_window()
        )
        self._task_virtual_start_counts = (
            plan.materialize_task_virtual_start_counts()
        )
        self.dataset_mean_task_virtual_start_count = (
            plan.dataset_mean_task_virtual_start_count
        )
        self.dataset_mean_valid_action_steps = (
            plan.dataset_mean_valid_action_steps
        )
        self.sample_weights = plan.sample_weights

    def __len__(self) -> int:
        return len(self._virtual_index)

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int]:
        if self.data_config.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT:
            return LocalLatentWeightedTrainSampler(self, world_size=world_size, rank=rank)
        return LocalLatentEpochOrderSampler(self, world_size=world_size, rank=rank)

    def build_epoch_index_order(self, *, epoch: int) -> list[int]:
        return self._uniform_segment_sampling_plan.build_epoch_index_order(
            epoch=epoch
        )

    def _sample_weight_metadata(self, index: int) -> dict[str, Any]:
        return self._uniform_segment_sampling_plan.sample_weight_metadata(index)

    def __getitem__(self, index: int) -> LatentWAMSample:
        window_index, virtual_latent_start = self._virtual_index[index]
        window = self.windows[window_index]
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        (
            full_video_latents,
            latent_layout_metadata,
            primary_payload,
            full_condition_latents,
            condition_layout_metadata,
        ) = self._load_canonical_window_latents(
            window,
            repo_bundle.metadata,
        )
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
        raw_frame_ids = [
            int(value)
            for value in list(primary_payload.get("frame_ids", []))
        ]
        if not raw_frame_ids:
            raw_frame_ids = list(window.observation_frame_indices)
        segment = self._segment_assembler.build(
            video_latents=full_video_latents,
            condition_latents=full_condition_latents,
            rows=rows,
            raw_frame_ids=raw_frame_ids,
            window=window,
            latent_start=latent_start,
            segment_length=segment_length,
            start_padding_frames=start_padding_frames,
        )

        task_index = int(rows[min(segment.sample_start_frame, len(rows) - 1)].get("task_index", 0)) if rows else 0
        episode_record = repo_bundle.episodes_by_index.get(window.episode_index)
        task_text = repo_bundle.metadata.tasks_by_index.get(task_index)
        if task_text is None and episode_record is not None and episode_record.tasks:
            task_text = episode_record.tasks[0]

        text_context = primary_payload.get("text_emb")
        if isinstance(text_context, torch.Tensor):
            text_context = text_context.to(dtype=torch.float32)
        else:
            text_context = None
        negative_text_context = self.empty_text_embedding.clone() if self.empty_text_embedding is not None else None

        return LatentWAMSample(
            video_latents=segment.video_latents,
            actions=segment.actions,
            action_mask=segment.action_mask,
            state=segment.state,
            state_mask=segment.state_mask,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
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
                "proprio_context_chunk_count": int(segment.proprio_context_state.shape[0]),
                "proprio_context_frame_count": int(segment.proprio_context_frames.shape[0]),
                "observed_frame_ids": segment.observed_frame_ids,
                "latent_temporal_layout": segment.latent_temporal_layout,
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "condition_latent_layout": condition_layout_metadata,
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
                "tail_padding_mode": "none" if segment.padded_latent_frames == 0 else "zero_hold",
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
            "start_padding_mode": "repeat_first_latent" if int(pre_start_frames) > 0 else "none",
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
        chunk_size = max(1, int(sampled_chunk_size if sampled_chunk_size is not None else sample_cfg.chunk_size))
        window_size = max(1, int(sampled_window_size if sampled_window_size is not None else sample_cfg.window_size))
        if chunk_size > 1 or window_size > 1 or sampled_chunk_size is not None or sampled_window_size is not None:
            metadata["sampled_chunk_size"] = chunk_size
            metadata["sampled_window_size"] = window_size
            if emit_explicit_loss_ranges and bool(context_prefix_enabled):
                metadata["history_frames"] = max(1, min(int(loss_frame_start), max(1, int(segment_length) - 1)))
            else:
                history_frames = int(math.ceil(window_size / 2.0)) * chunk_size
                metadata["history_frames"] = max(1, min(history_frames, max(1, int(segment_length) - chunk_size)))
        return metadata

class HierarchicalFixedSegmentLocalLeRobotLatentDataset(UniformSegmentLocalLeRobotLatentDataset):
    """Shared fixed-length hierarchical task/trajectory/start sampler."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
        LocalLeRobotLatentWindowDataset.__init__(self, data_config, windows)
        sample_cfg = self.data_config.sample_construction
        if sample_cfg.tail_padding_policy != TailPaddingPolicy.ZERO_ORDER_HOLD:
            raise ValueError("Hierarchical fixed-segment sampling currently requires zero-order-hold tail padding.")
        if sample_cfg.padded_target_policy != PaddedTargetPolicy.MASK_LOSS:
            raise ValueError("Hierarchical fixed-segment sampling currently requires masked padded targets.")
        if sample_cfg.segment_frames is None:
            raise ValueError("Hierarchical fixed-segment sampling requires `sample_construction.segment_frames`.")
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

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int]:
        return HierarchicalFixedSegmentTrainSampler(self, world_size=world_size, rank=rank)

    def resolve_hierarchical_sample_key(self, index: int) -> dict[str, Any]:
        """Resolve one sampler/dataloader index without loading tensors."""

        return self._hierarchical_segment_plan.resolve_sample_key(
            index
        ).as_metadata()

    def iter_hierarchical_eligible_start_keys(self) -> Iterator[tuple[int, int, int]]:
        """Yield every concrete trajectory/start/chunk key that must be reachable."""

        return self._hierarchical_segment_plan.iter_eligible_start_keys()

    def __getitem__(self, index: int) -> LatentWAMSample:
        task_spec, window_spec, latent_start, sampled_chunk_size = (
            self._hierarchical_segment_plan.draw(index)
        )
        window_index = int(window_spec.window_index)
        window = self.windows[window_index]
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        (
            full_video_latents,
            latent_layout_metadata,
            primary_payload,
            full_condition_latents,
            condition_layout_metadata,
        ) = self._load_canonical_window_latents(
            window,
            repo_bundle.metadata,
        )
        raw_frame_ids = [
            int(value)
            for value in list(primary_payload.get("frame_ids", []))
        ]
        if not raw_frame_ids:
            raw_frame_ids = list(window.observation_frame_indices)
        start_padding_frames = (
            _LocalLatentUniformSegmentSamplingPlan.resolve_start_padding_frames(
                self.data_config,
                window,
            )
        )
        segment = self._segment_assembler.build(
            video_latents=full_video_latents,
            condition_latents=full_condition_latents,
            rows=rows,
            raw_frame_ids=raw_frame_ids,
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
                self.data_config.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
            ),
        )

        task_index = int(rows[min(segment.sample_start_frame, len(rows) - 1)].get("task_index", 0)) if rows else 0
        episode_record = repo_bundle.episodes_by_index.get(window.episode_index)
        task_text = repo_bundle.metadata.tasks_by_index.get(task_index)
        if task_text is None and episode_record is not None and episode_record.tasks:
            task_text = episode_record.tasks[0]

        text_context = primary_payload.get("text_emb")
        if isinstance(text_context, torch.Tensor):
            text_context = text_context.to(dtype=torch.float32)
        else:
            text_context = None
        negative_text_context = self.empty_text_embedding.clone() if self.empty_text_embedding is not None else None

        boundary_metadata = dict(segment.boundary_metadata)
        effective_latent_start = int(boundary_metadata.get("effective_frame_start", latent_start))
        tail_padded_frame_count = int(boundary_metadata.get("tail_padded_frame_count", segment.padded_latent_frames))

        return LatentWAMSample(
            video_latents=segment.video_latents,
            actions=segment.actions,
            action_mask=segment.action_mask,
            state=segment.state,
            state_mask=segment.state_mask,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
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
                "proprio_context_chunk_count": int(segment.proprio_context_state.shape[0]),
                "proprio_context_frame_count": int(segment.proprio_context_frames.shape[0]),
                "observed_frame_ids": segment.observed_frame_ids,
                "latent_temporal_layout": segment.latent_temporal_layout,
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "condition_latent_layout": condition_layout_metadata,
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
                "tail_padding_mode": "none" if tail_padded_frame_count == 0 else "zero_order_hold",
                "subwindow_action_start": segment.action_start_index,
                "subwindow_action_end": segment.action_end_index,
                **self._uniform_segment_attention_metadata(
                    latent_start=effective_latent_start,
                    segment_length=int(boundary_metadata.get("effective_segment_frames", self.segment_frames)),
                    valid_latent_frames=segment.valid_latent_frames,
                    loss_frame_start=segment.loss_frame_start,
                    loss_frame_end=segment.loss_frame_end,
                    sample_start_frame=segment.sample_start_frame,
                    start_padding_frames=segment.start_padding_frames,
                    pre_start_frames=segment.pre_start_frames,
                    emit_explicit_loss_ranges=True,
                    context_prefix_enabled=int(boundary_metadata.get("context_prefix_frames_requested", 0)) > 0,
                    sampled_chunk_size=sampled_chunk_size,
                    sampled_window_size=max(1, int(self.data_config.sample_construction.window_size)),
                ),
                **boundary_metadata,
                **segment.action_target_metadata,
                **self._action_loss_metadata(
                    segment.action_mask,
                    loss_frame_start=segment.loss_frame_start,
                    loss_frame_end=segment.loss_frame_end,
                    latent_num_frames=int(boundary_metadata.get("effective_segment_frames", self.segment_frames)),
                ),
                **self._hierarchical_segment_plan.sample_metadata(
                    index=index,
                    task_spec=task_spec,
                    window_spec=window_spec,
                ),
            },
        )


class FullSegmentLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Current LingBot-style long-window latent dataset view."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
        super().__init__(data_config, windows)
        self.sample_index = tuple(self.windows)


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
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        latent_payloads = self._load_window_latents(window, repo_bundle.metadata)
        full_video_latents, latent_layout_metadata = self._assemble_canonical_latents(latent_payloads)
        primary_payload = latent_payloads[self.data_config.latent_camera_names[0]]
        raw_frame_ids = [
            int(value)
            for value in list(primary_payload.get("frame_ids", []))
        ]
        if not raw_frame_ids:
            raw_frame_ids = list(window.observation_frame_indices)
        plan = self._causal_sampling_planner.plan(
            raw_frame_ids=raw_frame_ids,
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

        task_index = int(rows[min(plan.sample_start_frame, len(rows) - 1)].get("task_index", 0)) if rows else 0
        episode_record = repo_bundle.episodes_by_index.get(window.episode_index)
        task_text = repo_bundle.metadata.tasks_by_index.get(task_index)
        if task_text is None and episode_record is not None and episode_record.tasks:
            task_text = episode_record.tasks[0]

        text_context = primary_payload.get("text_emb")
        if isinstance(text_context, torch.Tensor):
            text_context = text_context.to(dtype=torch.float32)
        else:
            text_context = None
        negative_text_context = self.empty_text_embedding.clone() if self.empty_text_embedding is not None else None

        return LatentWAMSample(
            video_latents=video_latents,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
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
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
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


def build_local_lerobot_latent_train_val_datasets(
    data_config: DataConfig,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    window_plan = LocalLatentTrainValWindowPlanner(data_config).plan()
    train_windows = list(window_plan.train_windows)
    val_windows = list(window_plan.val_windows)

    dataset_cls: type[Dataset[LatentWAMSample]]
    if data_config.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT:
        dataset_cls = FullSegmentLocalLeRobotLatentDataset
    elif data_config.sample_construction.mode == WindowSamplingMode.UNIFORM_SEGMENT:
        segment_min_frames = int(data_config.sample_construction.segment_min_frames or data_config.num_frames)
        segment_max_frames = int(data_config.sample_construction.segment_max_frames or segment_min_frames)
        if (
            segment_min_frames != segment_max_frames
            and (data_config.train_batch_size != 1 or data_config.val_batch_size != 1)
        ):
            raise ValueError(
                "Uniform segment sampling with variable segment lengths requires train/val batch size 1 because "
                "latent/action tensor lengths vary across examples."
            )
        dataset_cls = UniformSegmentLocalLeRobotLatentDataset
    elif data_config.sample_construction.mode == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT:
        dataset_cls = HierarchicalFixedSegmentLocalLeRobotLatentDataset
    elif data_config.sample_construction.mode == WindowSamplingMode.CAUSAL_PREFIX_SUFFIX:
        dataset_cls = CausalPrefixSuffixLocalLeRobotLatentDataset
    else:
        raise ValueError(
            f"Unsupported sample_construction.mode for local latent datasets: "
            f"{data_config.sample_construction.mode!r}"
        )

    val_data_config = replace(data_config, split=DataSplit.VAL)
    return (
        dataset_cls(data_config=data_config, windows=train_windows),
        dataset_cls(data_config=val_data_config, windows=val_windows),
    )
