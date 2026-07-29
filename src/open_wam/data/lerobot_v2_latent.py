from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import partial
import math
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from open_wam.configs import (
    ActionTargetRepresentation,
    DataConfig,
    DataSplit,
    GripperRepresentation,
    LatentTemporalLayout,
    LatentWindowProfile,
    PaddedTargetPolicy,
    ReplayStatusPolicy,
    RolloutContextPolicy,
    SampleOrderMode,
    SampleWeightMode,
    SampleStateAnchorMode,
    SampleTargetAlignment,
    SegmentContextPolicy,
    TailPaddingPolicy,
    WindowSamplingMode,
)

from .action_transforms import (
    build_absolute_joint_position_targets,
    expected_joint_position_target_dim,
    normalize_action_targets,
)
from .distributed_sampling import (
    EpochOffsetDistributedSampler,
    EpochOrderDistributedSampler,
    WeightedReplacementDistributedSampler,
)
from .latent_contracts import LatentWAMSample
from .latent_segment_geometry import (
    compact_boundary_start_range,
    resolve_compact_boundary_segment,
    resolve_rollout_parity_boundary_segment,
    rollout_parity_start_range,
)
from .latent_temporal import (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET,
    latent_anchor_positions,
    latent_raw_boundaries,
    observed_frame_ids_for_latent_segment,
    raw_span_for_latent_range,
)
from .lerobot_v2 import LeRobotV2Metadata
# Keep storage symbols importable from this historical module while ownership
# lives in the repository adapter.
from .lerobot_v2_latent_storage import (
    LocalEpisodeWindow,
    LocalRepoBundle as LocalRepoBundle,
    discover_local_lerobot_repo_bundles,
    latent_filename,
    load_lerobot_v2_local_metadata as load_lerobot_v2_local_metadata,
    read_json_local as read_json_local,
    read_jsonl_local as read_jsonl_local,
    reshape_latent_payload,
    resolve_latent_root,
    scan_local_latent_windows,
    split_local_episode_indices as split_local_episode_indices,
)
from .replay_status import load_replay_status_records, split_episode_indices_by_replay_status
from .row_action_targets import build_row_action_targets, resolve_row_key
from .sequence_packing import pack_temporal_sequence


_TRUNCATING_SEQUENCE_PACKER = partial(
    pack_temporal_sequence,
    truncate_to_target_length=True,
)


@dataclass(frozen=True)
class HierarchicalFixedSegmentWindowSpec:
    """One eligible trajectory/chunk geometry for hierarchical fixed-segment sampling."""

    window_index: int
    task_text: str
    sampled_chunk_size: int
    start_min: int
    start_max: int
    eligible_start_count: int
    mass_within_task: float


@dataclass(frozen=True)
class HierarchicalFixedSegmentTaskSpec:
    """Task-level sampling mass and trajectory candidates."""

    task_text: str
    eligible_start_count: int
    demo_count: int
    task_mass: float
    windows: tuple[HierarchicalFixedSegmentWindowSpec, ...]
    window_mass_total: float


class LocalLeRobotLatentWindowDataset(Dataset[LatentWAMSample]):
    """Latent-first local-repo dataset for LingBot-style post-training exports."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
        if data_config.local_root is None:
            raise ValueError("Local latent datasets require `data.local_root` in the experiment config.")
        self.data_config = data_config
        self.windows = list(windows)
        self.empty_text_embedding = self._load_empty_text_embedding()
        repo_roots = [data_config.local_root]
        if data_config.val_local_root and data_config.val_local_root not in repo_roots:
            repo_roots.append(data_config.val_local_root)
        self._repo_bundles = {
            str(bundle.root): bundle
            for repo_root in repo_roots
            for bundle in discover_local_lerobot_repo_bundles(repo_root)
        }
        self._episode_cache: OrderedDict[tuple[str, int], list[dict[str, Any]]] = OrderedDict()
        self._latent_view_cache: OrderedDict[
            tuple[str, int, int, int],
            tuple[
                torch.Tensor,
                dict[str, dict[str, int]],
                dict[str, Any],
                torch.Tensor | None,
                dict[str, dict[str, int]],
            ],
        ] = OrderedDict()

        if not self.windows:
            raise ValueError(
                "No valid latent windows were constructed. "
                f"Check local_root={data_config.local_root!r} and latent_camera_names={data_config.latent_camera_names!r}."
            )
        self._window_valid_action_steps = tuple(self._estimate_window_valid_action_steps(window) for window in self.windows)
        self.dataset_mean_valid_action_steps = self._estimate_mean_valid_action_steps()
        self._window_task_texts = tuple(self._window_task_text(window) for window in self.windows)
        self._task_demo_counts = self._estimate_task_demo_counts()
        self.dataset_mean_task_demo_count = self._estimate_mean_task_demo_count()
        self.sample_weights = self._build_sample_weights()

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

    def _estimate_mean_valid_action_steps(self) -> float:
        positive_estimates = [value for value in self._window_valid_action_steps if value > 0]
        if not positive_estimates:
            return float(max(1, self.data_config.action_schema.action_horizon))
        return float(sum(positive_estimates) / len(positive_estimates))

    def _estimate_window_valid_action_steps(self, window: LocalEpisodeWindow) -> int:
        if (
            self.data_config.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
            and self.data_config.latent_window_profile == LatentWindowProfile.EXACT_CHUNKED_WINDOW
        ):
            observed_frame_ids = window.observation_frame_indices
            frame_stride = 1
            if len(observed_frame_ids) > 1:
                frame_stride = max(1, int(observed_frame_ids[1] - observed_frame_ids[0]))
            prefix_actions = int(self.data_config.action_schema.action_horizon // max(1, self.data_config.num_frames))
            window_span = max(0, window.end_frame - window.start_frame)
            raw_action_steps = max(len(observed_frame_ids), window_span)
            return max(0, int(prefix_actions + raw_action_steps))
        if self.data_config.sample_construction.mode == WindowSamplingMode.CAUSAL_PREFIX_SUFFIX:
            return 0
        return max(0, int(self.data_config.action_schema.action_horizon))

    def _window_task_text(self, window: LocalEpisodeWindow) -> str:
        repo_bundle = self._repo_bundles.get(str(window.repo_root))
        if repo_bundle is not None:
            episode_record = repo_bundle.episodes_by_index.get(window.episode_index)
            if episode_record is not None and episode_record.tasks:
                return str(episode_record.tasks[0])
        return f"{window.repo_root}:episode:{window.episode_index}"

    def _estimate_task_demo_counts(self) -> Counter[str]:
        demo_keys_by_task: dict[str, set[tuple[str, int]]] = {}
        for window, task_text in zip(self.windows, self._window_task_texts, strict=True):
            demo_keys_by_task.setdefault(task_text, set()).add((str(window.repo_root), int(window.episode_index)))
        return Counter({task_text: len(demo_keys) for task_text, demo_keys in demo_keys_by_task.items()})

    def _estimate_mean_task_demo_count(self) -> float:
        if not self._task_demo_counts:
            return 1.0
        return float(sum(self._task_demo_counts.values()) / len(self._task_demo_counts))

    def _build_sample_weights(self) -> tuple[float, ...]:
        mode = self.data_config.sample_construction.sample_weight_mode
        if mode == SampleWeightMode.UNIFORM:
            return tuple(1.0 for _ in self.windows)

        reference_steps = max(1.0, float(self.dataset_mean_valid_action_steps))
        reference_task_count = max(1.0, float(self.dataset_mean_task_demo_count))
        weights: list[float] = []
        for index, valid_steps in enumerate(self._window_valid_action_steps):
            weight = 1.0
            if mode in {
                SampleWeightMode.VALID_ACTION_STEPS,
                SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT,
            }:
                weight *= max(1.0, float(valid_steps)) / reference_steps
            if mode in {
                SampleWeightMode.INVERSE_TASK_DEMO_COUNT,
                SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT,
            }:
                task_count = max(1, self._task_demo_counts[self._window_task_texts[index]])
                weight *= reference_task_count / float(task_count)
            if self.data_config.sample_construction.sample_weight_min is not None:
                weight = max(float(self.data_config.sample_construction.sample_weight_min), weight)
            if self.data_config.sample_construction.sample_weight_max is not None:
                weight = min(float(self.data_config.sample_construction.sample_weight_max), weight)
            weights.append(float(weight))

        if not any(weight > 0 for weight in weights):
            return tuple(1.0 for _ in self.windows)
        return tuple(weights)

    def _sample_weight_metadata(self, index: int) -> dict[str, Any]:
        task_text = self._window_task_texts[index]
        return {
            "train_sample_weight": self.sample_weights[index],
            "train_sample_weight_mode": self.data_config.sample_construction.sample_weight_mode,
            "eligible_task_demo_count": self._task_demo_counts[task_text],
            "dataset_mean_eligible_task_demo_count": self.dataset_mean_task_demo_count,
        }

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
        action_horizon = int(self.data_config.action_schema.action_horizon)
        action_rows = rows[observation_start : observation_start + action_horizon]
        target_state_rows = rows[observation_start : observation_start + action_horizon]
        actions, action_mask, metadata = self._build_action_targets(
            action_rows=action_rows,
            target_state_rows=target_state_rows,
        )
        metadata = dict(metadata)
        metadata["latent_window_profile"] = self.data_config.latent_window_profile
        return actions, action_mask, metadata

    def _load_empty_text_embedding(self) -> torch.Tensor | None:
        configured_path = self.data_config.empty_text_embedding_path
        if configured_path is not None:
            configured_candidate = Path(configured_path)
            if not configured_candidate.exists():
                raise FileNotFoundError(
                    "Configured `data.empty_text_embedding_path` does not exist: "
                    f"{configured_candidate}"
                )
            candidate_path = configured_candidate
        else:
            candidate_path = Path(self.data_config.local_root) / "empty_emb.pt"
            if not candidate_path.exists():
                return None
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, torch.Tensor):
            raise TypeError(
                "Expected `empty_text_embedding_path` to point at a tensor checkpoint, "
                f"got {type(payload)!r} from {candidate_path}"
            )
        if payload.ndim == 3 and payload.shape[0] == 1:
            payload = payload.squeeze(0)
        return payload.to(dtype=torch.float32).contiguous()

    def _load_window_latents(
        self,
        window: LocalEpisodeWindow,
        metadata: LeRobotV2Metadata,
    ) -> dict[str, dict[str, Any]]:
        latent_root = resolve_latent_root(window.repo_root, self.data_config)
        chunk_dir = latent_root / f"chunk-{window.episode_index // metadata.chunk_size:03d}"
        payloads: dict[str, dict[str, Any]] = {}
        for camera_name in self.data_config.latent_camera_names:
            latent_path = chunk_dir / camera_name / latent_filename(
                episode_index=window.episode_index,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
            )
            payload = torch.load(latent_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected latent payload mapping at {latent_path}, got {type(payload).__name__}.")
            payloads[camera_name] = payload
        return payloads

    def _assemble_canonical_latents(
        self,
        latent_payloads: dict[str, dict[str, Any]],
        *,
        payload_key: str = "latent",
        require_payload_key: bool = True,
    ) -> tuple[torch.Tensor | None, dict[str, dict[str, int]]]:
        canonical_latents = None
        metadata: dict[str, dict[str, int]] = {}
        for view_layout, camera_name in zip(self.data_config.view_layout, self.data_config.latent_camera_names, strict=True):
            payload = latent_payloads[camera_name]
            if payload_key not in payload:
                if require_payload_key:
                    raise KeyError(f"Expected key {payload_key!r} in latent payload for camera {camera_name!r}.")
                return None, {}
            view_latents = reshape_latent_payload(payload, payload_key=payload_key)
            latent_height = int(view_latents.shape[1])
            latent_width = int(view_latents.shape[2])
            stride_h = max(1, view_layout.height // latent_height)
            stride_w = max(1, view_layout.width // latent_width)
            top = view_layout.top // stride_h
            left = view_layout.left // stride_w
            full_height = self.data_config.canonical_height // stride_h
            full_width = self.data_config.canonical_width // stride_w

            if canonical_latents is None:
                frames = int(view_latents.shape[0])
                channels = int(view_latents.shape[-1])
                canonical_latents = torch.zeros(
                    frames,
                    full_height,
                    full_width,
                    channels,
                    dtype=view_latents.dtype,
                )
            canonical_latents[:, top : top + latent_height, left : left + latent_width, :] = view_latents
            metadata[camera_name] = {
                "latent_height": latent_height,
                "latent_width": latent_width,
                "top": top,
                "left": left,
            }

        if canonical_latents is None:
            raise ValueError("Expected at least one latent camera payload.")
        return canonical_latents.permute(3, 0, 1, 2).contiguous().to(dtype=torch.float32), metadata

    def _condition_latent_offset_mismatches(
        self,
        latent_payloads: dict[str, dict[str, Any]],
        *,
        expected_offset: int,
    ) -> list[str]:
        mismatches: list[str] = []
        for camera_name in self.data_config.latent_camera_names:
            payload = latent_payloads[camera_name]
            if "condition_latent" not in payload:
                continue
            payload_offset = payload.get("condition_source_frame_offset")
            if payload_offset is None:
                if int(expected_offset) == 0:
                    # Legacy optional condition-latent payloads predate explicit source-frame
                    # metadata. They are valid for the unshifted default path, but shifted
                    # single-frame condition latents must be regenerated with metadata.
                    continue
                mismatches.append(f"{camera_name}: missing condition_source_frame_offset")
                continue
            if int(payload_offset) != int(expected_offset):
                mismatches.append(f"{camera_name}: payload={int(payload_offset)} expected={int(expected_offset)}")
                continue
            payload_policy = payload.get("condition_source_frame_policy")
            if payload_policy is None:
                if int(expected_offset) == 0:
                    continue
                mismatches.append(f"{camera_name}: missing condition_source_frame_policy")
                continue
            if payload_policy != CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET:
                mismatches.append(
                    f"{camera_name}: condition_source_frame_policy={payload_policy!r} "
                    f"expected {CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET!r}"
                )
                continue
        return mismatches

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
        cache_key = (str(window.repo_root), window.episode_index, window.start_frame, window.end_frame)
        if cache_key in self._latent_view_cache:
            self._latent_view_cache.move_to_end(cache_key)
            return self._latent_view_cache[cache_key]

        latent_payloads = self._load_window_latents(window, metadata)
        video_latents, latent_layout_metadata = self._assemble_canonical_latents(latent_payloads)
        assert video_latents is not None
        condition_latents, condition_layout_metadata = self._assemble_canonical_latents(
            latent_payloads,
            payload_key="condition_latent",
            require_payload_key=False,
        )
        if condition_latents is not None:
            expected_offset = int(self.data_config.sample_construction.condition_source_frame_offset)
            mismatches = self._condition_latent_offset_mismatches(
                latent_payloads,
                expected_offset=expected_offset,
            )
            if mismatches:
                if expected_offset == 0:
                    condition_latents = None
                    condition_layout_metadata = {}
                else:
                    preview = "; ".join(mismatches[:4])
                    raise ValueError(
                        "Latent payload condition_source_frame_offset/policy does not match "
                        f"`sample_construction.condition_source_frame_offset={expected_offset}`. "
                        "Re-run scripts/augment_lerobot_latents_with_single_frame_condition.py "
                        f"with --source-frame-offset {expected_offset} --overwrite. "
                        f"Mismatches: {preview}"
                    )
        primary_payload = dict(latent_payloads[self.data_config.latent_camera_names[0]])
        payload = (video_latents, latent_layout_metadata, primary_payload, condition_latents, condition_layout_metadata)
        self._latent_view_cache[cache_key] = payload
        while len(self._latent_view_cache) > max(1, int(self.data_config.episode_cache_size)):
            self._latent_view_cache.popitem(last=False)
        return payload

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
        action_target = self.data_config.action_target
        if action_target.representation not in {
            ActionTargetRepresentation.RAW,
            ActionTargetRepresentation.ABSOLUTE_JOINT_POSITION,
        }:
            raise ValueError(
                "Long-window local latent datasets currently support only "
                "`action_target.representation=raw` or `absolute_joint_position` for LingBot-compatible exact "
                "training."
            )
        if latent_num_frames <= 0:
            raise ValueError("Expected at least one latent frame in the local latent window.")
        if not observed_frame_ids:
            raise ValueError("Expected non-empty frame_ids metadata for the local latent window.")

        frame_stride = 1
        if len(observed_frame_ids) > 1:
            frame_stride = max(1, int(observed_frame_ids[1] - observed_frame_ids[0]))
        prefix_actions = int(self.data_config.action_schema.action_horizon // max(1, self.data_config.num_frames))
        required_action_num = latent_num_frames * prefix_actions
        leading_zero_action_frames = max(0, int(leading_zero_action_frames))
        leading_action_steps = leading_zero_action_frames * prefix_actions

        action_start_offset = max(0, int(observed_frame_ids[0] - window.start_frame))
        raw_window_rows = rows[window.start_frame : window.end_frame]
        aligned_rows = raw_window_rows[action_start_offset:]
        if action_target.representation == ActionTargetRepresentation.RAW:
            source_actions = torch.stack(
                [
                    torch.tensor(row[resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                    for row in aligned_rows
                ],
                dim=0,
            )
            source_actions = normalize_action_targets(
                source_actions,
                normalization=action_target.normalization,
            )
            source_mask = torch.ones_like(source_actions, dtype=torch.float32)
            action_dim = source_actions.shape[-1]
            target_family_metadata: dict[str, Any] = {
                "action_target_normalization_mode": str(action_target.normalization.mode),
            }
        else:
            joint_position_source = torch.stack(
                [
                    torch.tensor(row[resolve_row_key(row, action_target.joint_position_source_key)], dtype=torch.float32)
                    for row in aligned_rows
                ],
                dim=0,
            )
            raw_action_sequence = torch.stack(
                [
                    torch.tensor(row[resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                    for row in aligned_rows
                ],
                dim=0,
            )
            gripper_position_sequence = None
            if (
                action_target.include_gripper
                and action_target.gripper_representation != GripperRepresentation.ACTION_COMMAND
            ):
                gripper_position_sequence = torch.stack(
                    [
                        torch.tensor(
                            row[resolve_row_key(row, action_target.gripper_position_source_key)],
                            dtype=torch.float32,
                        )
                        for row in aligned_rows
                    ],
                    dim=0,
                )
            source_actions, source_mask, target_family_metadata = build_absolute_joint_position_targets(
                joint_position_source,
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
                gripper_position_sequence=gripper_position_sequence,
                raw_action_sequence=raw_action_sequence,
                gripper_action_index=action_target.gripper_action_index,
                normalization=action_target.joint_position_normalization,
            )
            action_dim = source_actions.shape[-1]
            expected_dim = expected_joint_position_target_dim(
                joint_dim=joint_position_source.shape[-1],
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
            )
            if action_dim != expected_dim:
                raise ValueError(
                    "Derived absolute-joint target dim mismatch: "
                    f"derived={action_dim}, expected={expected_dim}."
                )
        if action_dim != self.data_config.action_schema.action_dim:
            raise ValueError(
                "Configured action_dim does not match local latent supervision: "
                f"configured={self.data_config.action_schema.action_dim}, source={action_dim}."
            )

        leading_fill = torch.zeros(leading_action_steps, action_dim, dtype=torch.float32)
        leading_mask = torch.ones_like(leading_fill, dtype=torch.float32)
        if action_target.representation == ActionTargetRepresentation.ABSOLUTE_JOINT_POSITION and leading_action_steps > 0:
            leading_fill = source_actions[0:1].expand(leading_action_steps, -1).contiguous()
            leading_mask = source_mask[0:1].expand(leading_action_steps, -1).contiguous()

        padded_actions = torch.cat(
            [
                leading_fill,
                source_actions,
            ],
            dim=0,
        )
        padded_mask = torch.cat([leading_mask, source_mask], dim=0)
        if padded_actions.shape[0] < required_action_num:
            padded_actions = torch.cat(
                [
                    padded_actions,
                    torch.zeros(required_action_num - padded_actions.shape[0], action_dim, dtype=torch.float32),
                ],
                dim=0,
            )
            padded_mask = torch.cat(
                [
                    padded_mask,
                    torch.zeros(required_action_num - padded_mask.shape[0], action_dim, dtype=torch.float32),
                ],
                dim=0,
            )
        actions = padded_actions[:required_action_num].contiguous()

        action_mask = padded_mask[:required_action_num].contiguous()
        if leading_action_steps > 0 and float(leading_zero_action_mask) <= 0.0:
            action_mask[:leading_action_steps] = 0.0
        if source_actions.shape[0] + leading_action_steps < required_action_num:
            action_mask[source_actions.shape[0] + leading_action_steps :] = 0.0
        return actions, action_mask, {
            "lingbot_window_action_alignment": {
                "latent_num_frames": latent_num_frames,
                "raw_frame_count": len(observed_frame_ids),
                "frame_stride": frame_stride,
                "prefix_actions": prefix_actions,
                "required_action_num": required_action_num,
                "action_start_offset": action_start_offset,
                "leading_zero_action_frames": leading_zero_action_frames,
                "leading_zero_action_steps": leading_action_steps,
                "leading_zero_action_mask": float(leading_zero_action_mask),
            },
            **target_family_metadata,
        }

    def _build_action_targets(
        self,
        *,
        action_rows: list[dict[str, Any]],
        target_state_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return build_row_action_targets(
            data_config=self.data_config,
            action_rows=action_rows,
            target_state_rows=target_state_rows,
            extract_sequence=self._extract_sequence,
            pack_sequence=_TRUNCATING_SEQUENCE_PACKER,
            reference_source_subject="Local latent LeRobot datasets",
        )

    def _extract_sequence(
        self,
        *,
        rows: list[dict[str, Any]],
        key: str,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows:
            return (
                torch.zeros(target_length, target_dim, dtype=torch.float32),
                torch.zeros(target_length, target_dim, dtype=torch.float32),
            )
        sequence = torch.stack(
            [torch.tensor(row[resolve_row_key(row, key)], dtype=torch.float32) for row in rows],
            dim=0,
        )
        return pack_temporal_sequence(
            sequence=sequence,
            target_dim=target_dim,
            target_length=target_length,
            left_pad=left_pad,
            sequence_name=key,
            truncate_to_target_length=True,
        )

    def _extract_state_history_at_frame(
        self,
        *,
        rows: list[dict[str, Any]],
        anchor_frame_index: int,
        state_horizon: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_horizon = int(
            self.data_config.action_schema.state_horizon if state_horizon is None else state_horizon
        )
        anchor = max(0, min(int(anchor_frame_index), len(rows) - 1)) if rows else 0
        state_start = max(0, anchor - resolved_horizon + 1)
        return self._extract_sequence(
            rows=rows[state_start : anchor + 1],
            key=self.data_config.action_target.pose_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=resolved_horizon,
            left_pad=True,
        )

    def _extract_state_at_frame(
        self,
        *,
        rows: list[dict[str, Any]],
        frame_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state, state_mask = self._extract_state_history_at_frame(
            rows=rows,
            anchor_frame_index=frame_index,
            state_horizon=1,
        )
        return state[0], state_mask[0]

    def _extract_proprio_context_state_sequence(
        self,
        *,
        rows: list[dict[str, Any]],
        observed_frame_ids: list[int],
        chunk_size: int,
        loss_frame_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not observed_frame_ids:
            raise ValueError("Per-chunk proprio context requires non-empty observed_frame_ids.")
        resolved_chunk_size = max(1, int(chunk_size))
        chunk_count = int(math.ceil(len(observed_frame_ids) / float(resolved_chunk_size)))
        states: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for chunk_index in range(chunk_count):
            local_context_index = max(
                0,
                min(
                    len(observed_frame_ids) - 1,
                    int(loss_frame_start) + chunk_index * resolved_chunk_size - 1,
                ),
            )
            frame_index = int(observed_frame_ids[local_context_index])
            state, state_mask = self._extract_state_at_frame(rows=rows, frame_index=frame_index)
            states.append(state)
            masks.append(state_mask)
        return torch.stack(states, dim=0), torch.stack(masks, dim=0)

    def _extract_proprio_context_frames(
        self,
        *,
        rows: list[dict[str, Any]],
        observed_frame_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_dim = int(self.data_config.action_schema.state_dim)
        if state_dim <= 0:
            raise ValueError("Per-frame proprio context requires positive data.action_schema.state_dim.")
        if not observed_frame_ids:
            return (
                torch.zeros(0, state_dim, dtype=torch.float32),
                torch.zeros(0, state_dim, dtype=torch.float32),
            )
        if not rows:
            return (
                torch.zeros(len(observed_frame_ids), state_dim, dtype=torch.float32),
                torch.zeros(len(observed_frame_ids), state_dim, dtype=torch.float32),
            )
        states: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for frame_index in observed_frame_ids:
            state, state_mask = self._extract_state_history_at_frame(
                rows=rows,
                anchor_frame_index=int(frame_index),
                state_horizon=1,
            )
            states.append(state[-1])
            masks.append(state_mask[-1])
        return torch.stack(states, dim=0), torch.stack(masks, dim=0)

    def _load_episode_rows(
        self,
        repo_root: Path,
        episode_index: int,
        metadata: LeRobotV2Metadata,
    ) -> list[dict[str, Any]]:
        cache_key = (str(repo_root), episode_index)
        if cache_key in self._episode_cache:
            self._episode_cache.move_to_end(cache_key)
            return self._episode_cache[cache_key]

        path = repo_root / metadata.data_path_template.format(
            episode_chunk=episode_index // metadata.chunk_size,
            episode_index=episode_index,
        )
        rows = pq.read_table(path).to_pylist()
        self._episode_cache[cache_key] = rows
        while len(self._episode_cache) > self.data_config.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return rows

    @staticmethod
    def _build_raw_bucket_boundaries(
        *,
        raw_frame_count: int,
        latent_num_frames: int,
        latent_temporal_layout: LatentTemporalLayout | str = LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
    ) -> list[int]:
        return latent_raw_boundaries(
            raw_frame_count=raw_frame_count,
            latent_num_frames=latent_num_frames,
            layout=latent_temporal_layout,
        )


class LocalLatentWeightedTrainSampler(WeightedReplacementDistributedSampler):
    """Replacement train sampler for weighted local latent examples."""

    def __init__(self, dataset: LocalLeRobotLatentWindowDataset, *, world_size: int = 1, rank: int = 0) -> None:
        super().__init__(
            dataset,
            weights=dataset.sample_weights,
            base_seed=int(dataset.data_config.split_seed),
            world_size=world_size,
            rank=rank,
            empty_dataset_message="Weighted local latent sampling requires a non-empty dataset.",
        )


class LocalLatentEpochOrderSampler(EpochOrderDistributedSampler):
    """Sampler backed by a dataset-provided epoch order."""

    def __init__(self, dataset: "UniformSegmentLocalLeRobotLatentDataset", *, world_size: int = 1, rank: int = 0) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message="Epoch-order local latent sampling requires a non-empty dataset.",
            empty_order_message="Epoch-order local latent sampler received an empty order.",
        )


class HierarchicalFixedSegmentTrainSampler(EpochOffsetDistributedSampler):
    """Deterministic step-wise sampler for hierarchical fixed-segment draw keys."""

    def __init__(
        self,
        dataset: "HierarchicalFixedSegmentLocalLeRobotLatentDataset",
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message="Hierarchical fixed-segment sampling requires a non-empty dataset.",
        )


def _stable_int_seed(*values: int) -> int:
    """Build a stable 63-bit seed without relying on Python's randomized hash."""

    seed = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        mixed = (int(value) + 0x9E3779B97F4A7C15) & mask
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
        seed ^= mixed ^ (mixed >> 31)
        seed &= mask
    return seed & 0x7FFF_FFFF_FFFF_FFFF


def _weighted_choice_index(weights: tuple[float, ...], rng: random.Random) -> int:
    total = float(sum(weights))
    if total <= 0.0:
        return int(rng.randrange(len(weights)))
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += float(weight)
        if threshold <= cumulative:
            return index
    return len(weights) - 1


class UniformSegmentLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Uniform latent-start segment sampler over all eligible trajectories."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
        super().__init__(data_config, windows)
        self._segment_length_candidates = self._resolve_segment_length_candidates()
        self._virtual_index = self._build_virtual_index()
        if not self._virtual_index:
            raise ValueError("Uniform segment sampling requires at least one latent start.")
        self._virtual_indices_by_window = self._build_virtual_indices_by_window()
        self._task_virtual_start_counts = self._estimate_task_virtual_start_counts()
        self.dataset_mean_task_virtual_start_count = self._estimate_mean_task_virtual_start_count()
        self.dataset_mean_valid_action_steps = self._estimate_virtual_mean_valid_action_steps()
        self.sample_weights = self._build_virtual_sample_weights()

    def __len__(self) -> int:
        return len(self._virtual_index)

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int]:
        if self.data_config.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT:
            return LocalLatentWeightedTrainSampler(self, world_size=world_size, rank=rank)
        return LocalLatentEpochOrderSampler(self, world_size=world_size, rank=rank)

    def build_epoch_index_order(self, *, epoch: int) -> list[int]:
        rng = random.Random(self.data_config.split_seed + epoch * 1_000_003)
        if self.data_config.sample_construction.sample_weight_mode == SampleWeightMode.UNIFORM:
            per_window = {
                window_index: list(indices)
                for window_index, indices in self._virtual_indices_by_window.items()
            }
            for indices in per_window.values():
                rng.shuffle(indices)
        else:
            weights = torch.tensor(self.sample_weights, dtype=torch.double)
            if float(weights.sum().item()) <= 0:
                weights = torch.ones(len(self), dtype=torch.double)
            generator = torch.Generator()
            generator.manual_seed((self.data_config.split_seed + epoch * 1_000_003) & 0x7FFF_FFFF_FFFF_FFFF)
            sampled = torch.multinomial(weights, num_samples=len(self), replacement=True, generator=generator).tolist()
            per_window: dict[int, list[int]] = {}
            for virtual_index in sampled:
                window_index, _ = self._virtual_index[int(virtual_index)]
                per_window.setdefault(window_index, []).append(int(virtual_index))
            for indices in per_window.values():
                rng.shuffle(indices)

        window_order = list(per_window)
        rng.shuffle(window_order)
        block_size = max(1, int(self.data_config.sample_construction.segment_locality_block_size))
        ordered: list[int] = []
        active = list(window_order)
        while active:
            next_active: list[int] = []
            for window_index in active:
                indices = per_window[window_index]
                take = indices[:block_size]
                del indices[:block_size]
                ordered.extend(take)
                if indices:
                    next_active.append(window_index)
            active = next_active
        return ordered

    def _resolve_segment_length_candidates(self) -> tuple[int, ...]:
        sample_cfg = self.data_config.sample_construction
        min_frames = int(sample_cfg.segment_min_frames or self.data_config.num_frames)
        max_frames = int(sample_cfg.segment_max_frames or min_frames)
        stride = max(1, int(sample_cfg.segment_length_stride))
        if min_frames > max_frames:
            raise ValueError(
                "Uniform segment sampling requires segment_min_frames <= segment_max_frames, "
                f"got min={min_frames}, max={max_frames}."
            )
        candidates = list(range(min_frames, max_frames + 1, stride))
        if candidates[-1] != max_frames:
            candidates.append(max_frames)
        return tuple(candidates)

    def _build_virtual_index(self) -> tuple[tuple[int, int], ...]:
        virtual_index: list[tuple[int, int]] = []
        min_segment_length = min(self._segment_length_candidates)
        for window_index, window in enumerate(self.windows):
            source_latent_frames = max(1, int(window.latent_num_frames))
            start_padding_frames = self._window_start_padding_frames(window)
            min_latent_start = -start_padding_frames
            logical_source_frames = source_latent_frames + start_padding_frames
            for latent_start in range(min_latent_start, source_latent_frames):
                if self.data_config.sample_construction.require_full_segment:
                    if logical_source_frames < min_segment_length and latent_start > min_latent_start:
                        continue
                    max_length_from_start = source_latent_frames - latent_start
                    if logical_source_frames >= min_segment_length and max_length_from_start < min_segment_length:
                        continue
                virtual_index.append((window_index, latent_start))
        return tuple(virtual_index)

    def _window_start_padding_frames(self, window: LocalEpisodeWindow) -> int:
        padding_frames = max(0, int(self.data_config.sample_construction.start_padding_frames))
        if padding_frames <= 0:
            return 0
        return padding_frames if int(window.observation_start) == 0 else 0

    def _build_virtual_indices_by_window(self) -> dict[int, list[int]]:
        by_window: dict[int, list[int]] = {}
        for virtual_index, (window_index, _) in enumerate(self._virtual_index):
            by_window.setdefault(window_index, []).append(virtual_index)
        return by_window

    def _estimate_task_virtual_start_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for window_index, _ in self._virtual_index:
            task_text = self._window_task_texts[window_index]
            counts[task_text] = counts.get(task_text, 0) + 1
        return counts

    def _estimate_mean_task_virtual_start_count(self) -> float:
        positive = [count for count in self._task_virtual_start_counts.values() if count > 0]
        if not positive:
            return 1.0
        return float(sum(positive) / len(positive))

    def _estimate_virtual_mean_valid_action_steps(self) -> float:
        estimates = [
            self._estimate_virtual_valid_action_steps(virtual_index)
            for virtual_index in range(len(self._virtual_index))
        ]
        positive = [value for value in estimates if value > 0]
        if not positive:
            return float(max(1, self.data_config.action_schema.action_horizon))
        return float(sum(positive) / len(positive))

    def _estimate_virtual_valid_action_steps(self, virtual_index: int) -> float:
        window_index, latent_start = self._virtual_index[virtual_index]
        window = self.windows[window_index]
        source_latent_frames = int(window.latent_num_frames)
        start_padding_frames = self._window_start_padding_frames(window)
        estimates = [
            self._estimate_segment_valid_action_steps(
                window=window,
                latent_start=latent_start,
                segment_length=segment_length,
            )
            for segment_length in self._eligible_segment_lengths(
                source_latent_frames=source_latent_frames,
                start_padding_frames=start_padding_frames,
            )
        ]
        return float(sum(estimates) / len(estimates))

    def _estimate_segment_valid_action_steps(
        self,
        *,
        window: LocalEpisodeWindow,
        latent_start: int,
        segment_length: int,
    ) -> int:
        raw_frame_ids = list(window.observation_frame_indices)
        source_latent_frames = len(raw_frame_ids)
        if not raw_frame_ids or source_latent_frames <= 0:
            return 0
        observed_frame_ids = self._segment_observed_frame_ids(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=source_latent_frames,
            latent_start=latent_start,
            segment_length=segment_length,
            latent_temporal_layout=self.data_config.latent_temporal_layout,
        )
        frame_stride = 1
        if len(observed_frame_ids) > 1:
            frame_stride = max(1, int(observed_frame_ids[1] - observed_frame_ids[0]))
        prefix_actions = int(self.data_config.action_schema.action_horizon // max(1, self.data_config.num_frames))
        source_latent_start = max(0, latent_start)
        valid_latent_end = min(source_latent_frames, max(0, latent_start + segment_length))
        _, _, sample_start_frame, sample_end_frame = raw_span_for_latent_range(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=source_latent_frames,
            latent_start=source_latent_start,
            latent_end=valid_latent_end,
            layout=self.data_config.latent_temporal_layout,
        )
        raw_action_steps = max(0, sample_end_frame - sample_start_frame)
        required_action_steps = max(1, segment_length * prefix_actions)
        leading_valid_action_steps = prefix_actions
        if self._window_start_padding_frames(window) > 0 and latent_start <= 0:
            leading_valid_action_steps = 0
        return min(required_action_steps, leading_valid_action_steps + raw_action_steps)

    def _build_virtual_sample_weights(self) -> tuple[float, ...]:
        mode = self.data_config.sample_construction.sample_weight_mode
        if mode == SampleWeightMode.UNIFORM:
            return tuple(1.0 for _ in self._virtual_index)
        reference_steps = max(1.0, float(self.dataset_mean_valid_action_steps))
        reference_task_count = max(1.0, float(self.dataset_mean_task_demo_count))
        weights: list[float] = []
        for virtual_index, (window_index, _) in enumerate(self._virtual_index):
            weight = 1.0
            if mode in {
                SampleWeightMode.VALID_ACTION_STEPS,
                SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT,
            }:
                weight *= max(1.0, self._estimate_virtual_valid_action_steps(virtual_index)) / reference_steps
            if mode in {
                SampleWeightMode.INVERSE_TASK_DEMO_COUNT,
                SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT,
            }:
                task_text = self._window_task_texts[window_index]
                task_count = max(1, self._task_demo_counts[task_text])
                weight *= reference_task_count / float(task_count)
            if mode == SampleWeightMode.TASK_VIRTUAL_START_COUNT_POWER:
                task_text = self._window_task_texts[window_index]
                task_start_count = max(1.0, float(self._task_virtual_start_counts[task_text]))
                reference_start_count = max(1.0, float(self.dataset_mean_task_virtual_start_count))
                power = float(self.data_config.sample_construction.sample_weight_length_power)
                weight *= (task_start_count / reference_start_count) ** (power - 1.0)
            if self.data_config.sample_construction.sample_weight_min is not None:
                weight = max(float(self.data_config.sample_construction.sample_weight_min), weight)
            if self.data_config.sample_construction.sample_weight_max is not None:
                weight = min(float(self.data_config.sample_construction.sample_weight_max), weight)
            weights.append(float(weight))
        if not any(weight > 0 for weight in weights):
            return tuple(1.0 for _ in self._virtual_index)
        return tuple(weights)

    def _sample_weight_metadata(self, index: int) -> dict[str, Any]:
        window_index, _ = self._virtual_index[index]
        task_text = self._window_task_texts[window_index]
        return {
            "train_sample_weight": self.sample_weights[index],
            "train_sample_weight_mode": self.data_config.sample_construction.sample_weight_mode,
            "eligible_task_demo_count": self._task_demo_counts[task_text],
            "dataset_mean_eligible_task_demo_count": self.dataset_mean_task_demo_count,
            "eligible_task_virtual_start_count": self._task_virtual_start_counts[task_text],
            "dataset_mean_eligible_task_virtual_start_count": self.dataset_mean_task_virtual_start_count,
            "sample_weight_length_power": self.data_config.sample_construction.sample_weight_length_power,
        }

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
        segment_length, latent_start = self._sample_segment_geometry(
            index=index,
            source_latent_frames=int(full_video_latents.shape[1]),
            virtual_latent_start=virtual_latent_start,
            start_padding_frames=self._window_start_padding_frames(window),
        )
        sampled_chunk_size, sampled_window_size = self._sample_uniform_segment_attention_geometry(
            segment_length=segment_length
        )
        subwindow = self._build_uniform_segment(
            video_latents=full_video_latents,
            condition_latents=full_condition_latents,
            rows=rows,
            primary_payload=primary_payload,
            window=window,
            latent_start=latent_start,
            segment_length=segment_length,
        )

        task_index = int(rows[min(subwindow["sample_start_frame"], len(rows) - 1)].get("task_index", 0)) if rows else 0
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
            video_latents=subwindow["video_latents"],
            actions=subwindow["actions"],
            action_mask=subwindow["action_mask"],
            state=subwindow["state"],
            state_mask=subwindow["state_mask"],
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            condition_latents=subwindow["condition_latents"],
            proprio_context_state=subwindow["proprio_context_state"],
            proprio_context_state_mask=subwindow["proprio_context_state_mask"],
            proprio_context_frames=subwindow["proprio_context_frames"],
            proprio_context_frames_mask=subwindow["proprio_context_frames_mask"],
            metadata={
                "repo_root": str(window.repo_root),
                "dataset_id": str(window.repo_root),
                "episode_index": window.episode_index,
                "segment_start_frame": window.start_frame,
                "segment_end_frame": window.end_frame,
                "sample_start_frame": subwindow["sample_start_frame"],
                "sample_end_frame": subwindow["sample_end_frame"],
                "observation_start": subwindow["sample_start_frame"],
                "observation_frame_indices": subwindow["observed_frame_ids"],
                "window_sampling_mode": WindowSamplingMode.UNIFORM_SEGMENT,
                "window_start_frame": subwindow["sample_start_frame"],
                "window_end_frame": subwindow["sample_end_frame"],
                "anchor_frame_index": subwindow["anchor_frame_index"],
                "state_anchor_frame": subwindow["state_anchor_frame"],
                "proprio_context_frame_index": subwindow["proprio_context_frame_index"],
                "proprio_context_local_frame": subwindow["proprio_context_local_frame"],
                "proprio_context_chunk_count": int(subwindow["proprio_context_state"].shape[0]),
                "proprio_context_frame_count": int(subwindow["proprio_context_frames"].shape[0]),
                "observed_frame_ids": subwindow["observed_frame_ids"],
                "latent_temporal_layout": subwindow["latent_temporal_layout"],
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "condition_latent_layout": condition_layout_metadata,
                "has_condition_latents": subwindow["condition_latents"] is not None,
                "state_source_key": self.data_config.action_target.pose_source_key,
                "action_representation": self.data_config.action_target.representation,
                "virtual_sample_index": index,
                "trajectory_window_index": window_index,
                "virtual_latent_start": virtual_latent_start,
                "subwindow_latent_start": latent_start,
                "subwindow_latent_end": latent_start + segment_length,
                "segment_length_frames": segment_length,
                "segment_valid_latent_frames": subwindow["valid_latent_frames"],
                "segment_padded_latent_frames": subwindow["padded_latent_frames"],
                "tail_padding_mode": "none" if subwindow["padded_latent_frames"] == 0 else "zero_hold",
                "subwindow_action_start": subwindow["action_start_index"],
                "subwindow_action_end": subwindow["action_end_index"],
                **self._uniform_segment_attention_metadata(
                    latent_start=latent_start,
                    segment_length=segment_length,
                    valid_latent_frames=subwindow["valid_latent_frames"],
                    loss_frame_start=subwindow["loss_frame_start"],
                    loss_frame_end=subwindow["loss_frame_end"],
                    sample_start_frame=subwindow["sample_start_frame"],
                    start_padding_frames=subwindow["start_padding_frames"],
                    pre_start_frames=subwindow["pre_start_frames"],
                    sampled_chunk_size=sampled_chunk_size,
                    sampled_window_size=sampled_window_size,
                ),
                **subwindow["action_target_metadata"],
                **self._action_loss_metadata(subwindow["action_mask"]),
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

    def _sample_uniform_segment_attention_geometry(self, *, segment_length: int) -> tuple[int, int]:
        sample_cfg = self.data_config.sample_construction
        max_chunk_size = max(1, min(int(sample_cfg.chunk_size), int(segment_length)))
        if bool(sample_cfg.randomize_geometry) and max_chunk_size > 1:
            sampled_chunk_size = int(random.randint(1, max_chunk_size))
        else:
            sampled_chunk_size = max_chunk_size

        max_window_size = max(1, int(sample_cfg.window_size))
        if bool(sample_cfg.randomize_geometry) and max_window_size >= 4:
            sampled_window_size = int(random.randint(4, max_window_size))
        else:
            sampled_window_size = max_window_size

        return sampled_chunk_size, sampled_window_size

    def _sample_segment_geometry(
        self,
        *,
        index: int,
        source_latent_frames: int,
        virtual_latent_start: int,
        start_padding_frames: int = 0,
    ) -> tuple[int, int]:
        start_padding_frames = max(0, int(start_padding_frames))
        candidates = self._eligible_segment_lengths(
            source_latent_frames=source_latent_frames,
            start_padding_frames=start_padding_frames,
        )
        if self.data_config.sample_construction.randomize_segment_length:
            # Truly random per __getitem__ call: use the global random module
            # which is auto-seeded per process / per worker. Same index across
            # different calls/epochs draws different lengths.
            segment_length = int(random.choice(candidates))
        else:
            split_salt = 17 if self.data_config.split == DataSplit.TRAIN else 53
            seed = (
                int(self.data_config.split_seed)
                + split_salt
                + 1_000_003 * int(index + 1)
            ) & 0x7FFF_FFFF_FFFF_FFFF
            rng = random.Random(seed)
            segment_length = int(candidates[rng.randrange(len(candidates))])

        if self.data_config.sample_construction.randomize_segment_start:
            # With randomize_segment_start=True, virtual_latent_start is only
            # a sampling-frequency slot: longer trajectories still contribute
            # more virtual indices, while the actual segment start is drawn
            # fresh for this __getitem__ call. By default, draw from the full
            # padded logical timeline so startup and tail padding are both
            # represented. require_full_segment keeps the old full-window bound.
            min_start = -start_padding_frames
            if self.data_config.sample_construction.require_full_segment:
                max_start = max(min_start, int(source_latent_frames) - int(segment_length))
            else:
                max_start = max(min_start, int(source_latent_frames) - 1)
            latent_start = int(random.randint(min_start, max_start))
        else:
            latent_start = int(virtual_latent_start)
            if self.data_config.sample_construction.require_full_segment:
                min_start = -start_padding_frames
                max_start = max(min_start, int(source_latent_frames) - int(segment_length))
                latent_start = min(max(latent_start, min_start), max_start)
        return int(segment_length), int(latent_start)

    def _eligible_segment_lengths(
        self,
        *,
        source_latent_frames: int,
        start_padding_frames: int = 0,
    ) -> tuple[int, ...]:
        if not self.data_config.sample_construction.require_full_segment:
            return self._segment_length_candidates
        logical_source_frames = int(source_latent_frames) + max(0, int(start_padding_frames))
        candidates = tuple(length for length in self._segment_length_candidates if length <= logical_source_frames)
        if not candidates and logical_source_frames > 0:
            return (logical_source_frames,)
        if not candidates:
            raise ValueError(
                "Uniform segment sampling with require_full_segment=True found no eligible segment length for "
                f"source_latent_frames={source_latent_frames}; start_padding_frames={start_padding_frames}; "
                f"minimum candidate={min(self._segment_length_candidates)}."
            )
        return candidates

    def _build_uniform_segment(
        self,
        *,
        video_latents: torch.Tensor,
        condition_latents: torch.Tensor | None = None,
        rows: list[dict[str, Any]],
        primary_payload: dict[str, Any],
        window: LocalEpisodeWindow,
        latent_start: int,
        segment_length: int,
        compact_boundary_padding: bool = False,
        compact_boundary_chunk_size: int | None = None,
        compact_boundary_context_prefix_frames: int = 0,
        rollout_parity_target_alignment: bool = False,
    ) -> dict[str, Any]:
        source_latent_frames = int(video_latents.shape[1])
        if source_latent_frames <= 0:
            raise ValueError("Uniform segment sampling requires at least one source latent frame.")
        start_padding_frames = self._window_start_padding_frames(window)
        if compact_boundary_padding:
            chunk_size_for_boundary = max(
                1,
                int(
                    compact_boundary_chunk_size
                    if compact_boundary_chunk_size is not None
                    else self.data_config.sample_construction.chunk_size
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
            padded_latent_frames = int(boundary["head_padded_frame_count"]) + int(boundary["tail_padded_frame_count"])
        else:
            min_latent_start = -start_padding_frames
            if latent_start < min_latent_start or latent_start >= source_latent_frames:
                raise IndexError(f"latent_start={latent_start} is outside source_latent_frames={source_latent_frames}.")
            tensor_latent_start = int(latent_start)
            tensor_segment_length = int(segment_length)
            valid_latent_frames = max(0, min(segment_length, source_latent_frames - latent_start))
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
            raise IndexError(f"latent_start={latent_start} is outside source_latent_frames={source_latent_frames}.")
        raw_frame_ids = [int(value) for value in list(primary_payload.get("frame_ids", []))]
        if not raw_frame_ids:
            raw_frame_ids = list(window.observation_frame_indices)
        if not raw_frame_ids:
            raise ValueError("Uniform segment sampling requires non-empty frame ids.")

        observed_frame_ids = self._segment_observed_frame_ids(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=source_latent_frames,
            latent_start=tensor_latent_start,
            segment_length=tensor_segment_length,
            latent_temporal_layout=self.data_config.latent_temporal_layout,
        )
        source_latent_start = max(0, tensor_latent_start)
        valid_latent_end = min(source_latent_frames, max(0, tensor_latent_start + tensor_segment_length))
        _, _, sample_start_frame, sample_end_frame = raw_span_for_latent_range(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=source_latent_frames,
            latent_start=source_latent_start,
            latent_end=valid_latent_end,
            layout=self.data_config.latent_temporal_layout,
        )
        anchor_frame_index = observed_frame_ids[-1]

        sampled_window = LocalEpisodeWindow(
            repo_root=window.repo_root,
            episode_index=window.episode_index,
            start_frame=sample_start_frame,
            end_frame=min(sample_end_frame, len(rows)),
        )
        actions, action_mask, action_target_metadata = self._build_lingbot_window_action_targets(
            rows=rows,
            window=sampled_window,
            observed_frame_ids=observed_frame_ids,
            latent_num_frames=tensor_segment_length,
            leading_zero_action_frames=pre_start_frames if pre_start_frames > 0 else 1,
            leading_zero_action_mask=(
                0.0 if pre_start_frames > 0 or rollout_parity_target_alignment else 1.0
            ),
        )
        proprio_context_local_frame = max(0, min(len(observed_frame_ids) - 1, int(loss_frame_start) - 1))
        proprio_context_frame_index = observed_frame_ids[proprio_context_local_frame]
        state_anchor_frame = self._resolve_sample_state_anchor_frame(
            observed_frame_ids=observed_frame_ids,
            sample_start_frame=sample_start_frame,
            anchor_frame_index=anchor_frame_index,
            proprio_context_frame_index=proprio_context_frame_index,
        )
        state, state_mask = self._extract_state_history_at_frame(
            rows=rows,
            anchor_frame_index=state_anchor_frame,
        )
        proprio_context_state, proprio_context_state_mask = self._extract_proprio_context_state_sequence(
            rows=rows,
            observed_frame_ids=observed_frame_ids,
            chunk_size=(
                chunk_size_for_boundary
                if compact_boundary_padding
                else max(1, int(self.data_config.sample_construction.chunk_size))
            ),
            loss_frame_start=loss_frame_start,
        )
        proprio_context_frames, proprio_context_frames_mask = self._extract_proprio_context_frames(
            rows=rows,
            observed_frame_ids=observed_frame_ids,
        )
        return {
            "video_latents": self._slice_video_latents_with_zero_hold(
                video_latents=video_latents,
                latent_start=tensor_latent_start,
                segment_length=tensor_segment_length,
            ),
            "condition_latents": (
                self._slice_video_latents_with_zero_hold(
                    video_latents=condition_latents,
                    latent_start=tensor_latent_start,
                    segment_length=tensor_segment_length,
                )
                if condition_latents is not None
                else None
            ),
            "actions": actions,
            "action_mask": action_mask,
            "action_target_metadata": action_target_metadata,
            "state": state,
            "state_mask": state_mask,
            "proprio_context_state": proprio_context_state,
            "proprio_context_state_mask": proprio_context_state_mask,
            "proprio_context_frames": proprio_context_frames,
            "proprio_context_frames_mask": proprio_context_frames_mask,
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "anchor_frame_index": anchor_frame_index,
            "state_anchor_frame": state_anchor_frame,
            "proprio_context_frame_index": proprio_context_frame_index,
            "proprio_context_local_frame": proprio_context_local_frame,
            "observed_frame_ids": observed_frame_ids,
            "latent_temporal_layout": self.data_config.latent_temporal_layout,
            "action_start_index": sample_start_frame,
            "action_end_index": sample_start_frame + int(actions.shape[0]),
            "valid_latent_frames": valid_latent_frames,
            "padded_latent_frames": padded_latent_frames,
            "loss_frame_start": loss_frame_start,
            "loss_frame_end": loss_frame_end,
            "start_padding_frames": start_padding_frames,
            "pre_start_frames": pre_start_frames,
            "boundary_metadata": boundary,
        }

    @staticmethod
    def _segment_observed_frame_ids(
        *,
        raw_frame_ids: list[int],
        source_latent_frames: int,
        latent_start: int,
        segment_length: int,
        latent_temporal_layout: LatentTemporalLayout | str = LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
    ) -> list[int]:
        return observed_frame_ids_for_latent_segment(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=source_latent_frames,
            latent_start=latent_start,
            segment_length=segment_length,
            layout=latent_temporal_layout,
        )

    @staticmethod
    def _slice_video_latents_with_zero_hold(
        *,
        video_latents: torch.Tensor,
        latent_start: int,
        segment_length: int,
    ) -> torch.Tensor:
        if latent_start < 0:
            source_indices = torch.arange(
                latent_start,
                latent_start + segment_length,
                dtype=torch.long,
                device=video_latents.device,
            ).clamp_(0, video_latents.shape[1] - 1)
            return video_latents.index_select(dim=1, index=source_indices).contiguous()
        latent_end = latent_start + segment_length
        valid_slice = video_latents[:, latent_start:min(latent_end, video_latents.shape[1])].contiguous()
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

    def _resolve_sample_state_anchor_frame(
        self,
        *,
        observed_frame_ids: list[int],
        sample_start_frame: int,
        anchor_frame_index: int,
        proprio_context_frame_index: int | None = None,
    ) -> int:
        mode = self.data_config.sample_construction.state_anchor_mode
        if mode == SampleStateAnchorMode.PROPRIO_CONTEXT_FRAME:
            if proprio_context_frame_index is None:
                return int(anchor_frame_index)
            return int(proprio_context_frame_index)
        if mode == SampleStateAnchorMode.SAMPLE_START_FRAME:
            return int(sample_start_frame)
        if mode == SampleStateAnchorMode.FIRST_OBSERVED_FRAME:
            if not observed_frame_ids:
                raise ValueError("state_anchor_mode=first_observed_frame requires non-empty observed_frame_ids.")
            return int(observed_frame_ids[0])
        if mode == SampleStateAnchorMode.ANCHOR_FRAME:
            return int(anchor_frame_index)
        raise ValueError(f"Unsupported sample state_anchor_mode {mode!r}.")


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
        self.segment_frames = int(sample_cfg.segment_frames)
        self._window_start_ranges_by_chunk = self._build_window_start_ranges_by_chunk()
        self._task_specs = self._build_task_specs()
        self._task_weights = tuple(float(task.task_mass) for task in self._task_specs)
        self._task_mass_total = float(sum(self._task_weights))
        self._task_specs_by_text = {task.task_text: task for task in self._task_specs}
        self._epoch_sample_count = sum(
            int(window_spec.eligible_start_count)
            for task_spec in self._task_specs
            for window_spec in task_spec.windows
        )
        if self._epoch_sample_count <= 0:
            raise ValueError("Hierarchical fixed-segment sampling requires at least one eligible start.")

    def __len__(self) -> int:
        return self._epoch_sample_count

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int]:
        return HierarchicalFixedSegmentTrainSampler(self, world_size=world_size, rank=rank)

    def _hierarchical_chunk_size_candidates(self) -> tuple[int, ...]:
        sample_cfg = self.data_config.sample_construction
        max_chunk_size = max(1, int(sample_cfg.chunk_size))
        if sample_cfg.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT:
            return (max_chunk_size,)
        if bool(sample_cfg.randomize_geometry) and max_chunk_size > 1:
            return tuple(range(1, max_chunk_size + 1))
        return (max_chunk_size,)

    def _hierarchical_context_prefix_frames(self, sampled_chunk_size: int) -> int:
        sample_cfg = self.data_config.sample_construction
        if sample_cfg.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT:
            if sample_cfg.rollout_context_frames is not None:
                return max(1, int(sample_cfg.rollout_context_frames))
            if sample_cfg.rollout_context_policy == RolloutContextPolicy.ONE_FRAME:
                return 1
            if sample_cfg.rollout_context_policy == RolloutContextPolicy.ROLLOUT_HISTORY:
                chunk_size = max(1, int(sampled_chunk_size))
                window_size = max(1, int(sample_cfg.window_size))
                history_chunks = max(1, int(math.ceil(window_size / 2.0)))
                return max(1, history_chunks * chunk_size)
            raise ValueError(f"Unsupported rollout_context_policy: {sample_cfg.rollout_context_policy!r}")
        policy = sample_cfg.context_prefix_policy
        if policy == SegmentContextPolicy.NONE:
            return 0
        if policy == SegmentContextPolicy.FIXED:
            return max(0, int(sample_cfg.context_prefix_frames))
        if policy == SegmentContextPolicy.ROLLOUT_HISTORY:
            chunk_size = max(1, int(sampled_chunk_size))
            window_size = max(1, int(sample_cfg.window_size))
            history_chunks = max(1, int(math.ceil(window_size / 2.0)))
            return max(0, min(history_chunks * chunk_size, self.segment_frames - 1))
        raise ValueError(f"Unsupported context_prefix_policy: {policy!r}")

    def _build_window_start_ranges_by_chunk(self) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
        ranges_by_window: list[tuple[tuple[int, int, int, int], ...]] = []
        chunk_size_candidates = self._hierarchical_chunk_size_candidates()
        for window in self.windows:
            source_latent_frames = max(1, int(window.latent_num_frames))
            window_ranges: list[tuple[int, int, int, int]] = []
            for chunk_size in chunk_size_candidates:
                if self.data_config.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT:
                    start_min, start_max, eligible_start_count = rollout_parity_start_range(
                        source_latent_frames=source_latent_frames,
                    )
                else:
                    start_min, start_max, eligible_start_count = compact_boundary_start_range(
                        source_latent_frames=source_latent_frames,
                        segment_length=self.segment_frames,
                        start_padding_frames=self._window_start_padding_frames(window),
                        chunk_size=chunk_size,
                        context_prefix_frames=self._hierarchical_context_prefix_frames(chunk_size),
                    )
                if eligible_start_count > 0:
                    window_ranges.append(
                        (
                            int(chunk_size),
                            int(start_min),
                            int(start_max),
                            int(eligible_start_count),
                        )
                    )
            ranges_by_window.append(tuple(window_ranges))
        return tuple(ranges_by_window)

    def _build_task_specs(self) -> tuple[HierarchicalFixedSegmentTaskSpec, ...]:
        sample_cfg = self.data_config.sample_construction
        window_specs_by_task: dict[str, list[HierarchicalFixedSegmentWindowSpec]] = {}
        eligible_starts_by_task: Counter[str] = Counter()
        for window_index, task_text in enumerate(self._window_task_texts):
            for sampled_chunk_size, start_min, start_max, eligible_start_count in self._window_start_ranges_by_chunk[
                window_index
            ]:
                if eligible_start_count <= 0:
                    continue
                trajectory_mass = float(eligible_start_count) ** float(sample_cfg.trajectory_start_power)
                window_spec = HierarchicalFixedSegmentWindowSpec(
                    window_index=window_index,
                    task_text=task_text,
                    sampled_chunk_size=int(sampled_chunk_size),
                    start_min=int(start_min),
                    start_max=int(start_max),
                    eligible_start_count=int(eligible_start_count),
                    mass_within_task=trajectory_mass,
                )
                window_specs_by_task.setdefault(task_text, []).append(window_spec)
                eligible_starts_by_task[task_text] += int(eligible_start_count)

        task_specs: list[HierarchicalFixedSegmentTaskSpec] = []
        for task_text in sorted(window_specs_by_task):
            eligible_start_count = int(eligible_starts_by_task[task_text])
            demo_count = max(1, int(self._task_demo_counts[task_text]))
            task_mass = (
                float(eligible_start_count) ** float(sample_cfg.task_start_power)
            ) * (float(demo_count) ** float(sample_cfg.demo_count_power))
            if task_mass <= 0.0:
                task_mass = 1.0
            windows = tuple(window_specs_by_task[task_text])
            window_mass_total = float(sum(window.mass_within_task for window in windows))
            if window_mass_total <= 0.0:
                windows = tuple(
                    HierarchicalFixedSegmentWindowSpec(
                        window_index=window.window_index,
                        task_text=window.task_text,
                        sampled_chunk_size=window.sampled_chunk_size,
                        start_min=window.start_min,
                        start_max=window.start_max,
                        eligible_start_count=window.eligible_start_count,
                        mass_within_task=1.0,
                    )
                    for window in windows
                )
                window_mass_total = float(len(windows))
            task_specs.append(
                HierarchicalFixedSegmentTaskSpec(
                    task_text=task_text,
                    eligible_start_count=eligible_start_count,
                    demo_count=demo_count,
                    task_mass=float(task_mass),
                    windows=windows,
                    window_mass_total=window_mass_total,
                )
            )
        if not task_specs:
            raise ValueError("Hierarchical fixed-segment sampling found no eligible task/window starts.")
        return tuple(task_specs)

    def _draw_hierarchical_sample(
        self,
        index: int,
    ) -> tuple[HierarchicalFixedSegmentTaskSpec, HierarchicalFixedSegmentWindowSpec, int, int]:
        split_salt = 17 if self.data_config.split == DataSplit.TRAIN else 53
        rng = random.Random(_stable_int_seed(int(self.data_config.split_seed), split_salt, int(index)))
        task_index = _weighted_choice_index(self._task_weights, rng)
        task_spec = self._task_specs[task_index]
        window_weights = tuple(float(window.mass_within_task) for window in task_spec.windows)
        window_index = _weighted_choice_index(window_weights, rng)
        window_spec = task_spec.windows[window_index]
        latent_start = int(rng.randint(window_spec.start_min, window_spec.start_max))
        return task_spec, window_spec, latent_start, int(window_spec.sampled_chunk_size)

    def resolve_hierarchical_sample_key(self, index: int) -> dict[str, Any]:
        """Resolve one sampler/dataloader index without loading tensors."""

        epoch, epoch_index = divmod(int(index), len(self))
        task_spec, window_spec, latent_start, sampled_chunk_size = self._draw_hierarchical_sample(index)
        window = self.windows[int(window_spec.window_index)]
        context_prefix_frames = self._hierarchical_context_prefix_frames(sampled_chunk_size)
        if self.data_config.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT:
            boundary = resolve_rollout_parity_boundary_segment(
                source_latent_frames=max(1, int(window.latent_num_frames)),
                latent_start=latent_start,
                target_frame_count=self.segment_frames,
                context_frames=context_prefix_frames,
                chunk_size=sampled_chunk_size,
            )
        else:
            boundary = resolve_compact_boundary_segment(
                source_latent_frames=max(1, int(window.latent_num_frames)),
                latent_start=latent_start,
                segment_length=self.segment_frames,
                start_padding_frames=self._window_start_padding_frames(window),
                chunk_size=sampled_chunk_size,
                context_prefix_frames=context_prefix_frames,
            )
        return {
            "epoch": int(epoch),
            "epoch_sample_index": int(epoch_index),
            "task_text": task_spec.task_text,
            "trajectory_window_index": int(window_spec.window_index),
            "latent_start": int(latent_start),
            "start_min": int(window_spec.start_min),
            "start_max": int(window_spec.start_max),
            "window_eligible_start_count": int(window_spec.eligible_start_count),
            "logical_frame_start": int(boundary["logical_frame_start"]),
            "logical_frame_end": int(boundary["logical_frame_end"]),
            "effective_frame_start": int(boundary["effective_frame_start"]),
            "effective_frame_end": int(boundary["effective_frame_end"]),
            "effective_segment_frames": int(boundary["effective_segment_frames"]),
            "supervised_frame_start": int(boundary["supervised_start"]),
            "supervised_frame_end": int(boundary["supervised_end"]),
            "loss_frame_start": int(boundary["loss_frame_start"]),
            "loss_frame_end": int(boundary["loss_frame_end"]),
            "head_padded_frame_count": int(boundary["head_padded_frame_count"]),
            "tail_padded_frame_count": int(boundary["tail_padded_frame_count"]),
            "context_prefix_policy": str(self.data_config.sample_construction.context_prefix_policy),
            "target_alignment": str(self.data_config.sample_construction.target_alignment),
            "rollout_context_policy": str(self.data_config.sample_construction.rollout_context_policy),
            "context_prefix_frames_requested": int(boundary["context_prefix_frames_requested"]),
            "context_prefix_frames_in_sample": int(boundary["context_prefix_frames_in_sample"]),
            "context_prefix_real_frames": int(boundary["context_prefix_real_frames"]),
            "context_prefix_truncated_frames": int(boundary["context_prefix_truncated_frames"]),
            "chunk_size_for_boundary": int(boundary["chunk_size_for_boundary"]),
            "sampled_chunk_size": int(sampled_chunk_size),
            "sampled_window_size": max(1, int(self.data_config.sample_construction.window_size)),
        }

    def iter_hierarchical_eligible_start_keys(self) -> Iterator[tuple[int, int, int]]:
        """Yield every concrete trajectory/start/chunk key that must be reachable."""

        for task_spec in self._task_specs:
            for window_spec in task_spec.windows:
                for latent_start in range(int(window_spec.start_min), int(window_spec.start_max) + 1):
                    yield (
                        int(window_spec.window_index),
                        int(latent_start),
                        int(window_spec.sampled_chunk_size),
                    )

    def _hierarchical_sample_metadata(
        self,
        *,
        index: int,
        task_spec: HierarchicalFixedSegmentTaskSpec,
        window_spec: HierarchicalFixedSegmentWindowSpec,
    ) -> dict[str, Any]:
        sample_cfg = self.data_config.sample_construction
        task_probability = float(task_spec.task_mass) / max(1e-12, self._task_mass_total)
        trajectory_probability = float(window_spec.mass_within_task) / max(1e-12, task_spec.window_mass_total)
        return {
            "hierarchical_global_sample_index": int(index),
            "hierarchical_task_text": task_spec.task_text,
            "hierarchical_task_start_power": float(sample_cfg.task_start_power),
            "hierarchical_demo_count_power": float(sample_cfg.demo_count_power),
            "hierarchical_trajectory_start_power": float(sample_cfg.trajectory_start_power),
            "hierarchical_task_eligible_start_count": int(task_spec.eligible_start_count),
            "hierarchical_task_demo_count": int(task_spec.demo_count),
            "hierarchical_task_mass": float(task_spec.task_mass),
            "hierarchical_task_probability": task_probability,
            "hierarchical_trajectory_eligible_start_count": int(window_spec.eligible_start_count),
            "hierarchical_trajectory_mass": float(window_spec.mass_within_task),
            "hierarchical_trajectory_probability_within_task": trajectory_probability,
            "hierarchical_start_min": int(window_spec.start_min),
            "hierarchical_start_max": int(window_spec.start_max),
            "hierarchical_start_count": int(window_spec.eligible_start_count),
            "hierarchical_task_count": int(len(self._task_specs)),
            "hierarchical_epoch_sample_count": int(self._epoch_sample_count),
            "context_prefix_policy": str(sample_cfg.context_prefix_policy),
            "context_prefix_config_frames": int(sample_cfg.context_prefix_frames),
            "target_alignment": str(sample_cfg.target_alignment),
            "rollout_context_policy": str(sample_cfg.rollout_context_policy),
            "rollout_context_config_frames": (
                None if sample_cfg.rollout_context_frames is None else int(sample_cfg.rollout_context_frames)
            ),
            "tail_padding_policy": str(sample_cfg.tail_padding_policy),
            "padded_target_policy": str(sample_cfg.padded_target_policy),
        }

    def __getitem__(self, index: int) -> LatentWAMSample:
        task_spec, window_spec, latent_start, sampled_chunk_size = self._draw_hierarchical_sample(index)
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
        subwindow = self._build_uniform_segment(
            video_latents=full_video_latents,
            condition_latents=full_condition_latents,
            rows=rows,
            primary_payload=primary_payload,
            window=window,
            latent_start=latent_start,
            segment_length=self.segment_frames,
            compact_boundary_padding=True,
            compact_boundary_chunk_size=sampled_chunk_size,
            compact_boundary_context_prefix_frames=self._hierarchical_context_prefix_frames(sampled_chunk_size),
            rollout_parity_target_alignment=(
                self.data_config.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
            ),
        )

        task_index = int(rows[min(subwindow["sample_start_frame"], len(rows) - 1)].get("task_index", 0)) if rows else 0
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

        boundary_metadata = dict(subwindow["boundary_metadata"])
        effective_latent_start = int(boundary_metadata.get("effective_frame_start", latent_start))
        tail_padded_frame_count = int(boundary_metadata.get("tail_padded_frame_count", subwindow["padded_latent_frames"]))

        return LatentWAMSample(
            video_latents=subwindow["video_latents"],
            actions=subwindow["actions"],
            action_mask=subwindow["action_mask"],
            state=subwindow["state"],
            state_mask=subwindow["state_mask"],
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            condition_latents=subwindow["condition_latents"],
            proprio_context_state=subwindow["proprio_context_state"],
            proprio_context_state_mask=subwindow["proprio_context_state_mask"],
            proprio_context_frames=subwindow["proprio_context_frames"],
            proprio_context_frames_mask=subwindow["proprio_context_frames_mask"],
            metadata={
                "repo_root": str(window.repo_root),
                "dataset_id": str(window.repo_root),
                "episode_index": window.episode_index,
                "segment_start_frame": window.start_frame,
                "segment_end_frame": window.end_frame,
                "sample_start_frame": subwindow["sample_start_frame"],
                "sample_end_frame": subwindow["sample_end_frame"],
                "observation_start": subwindow["sample_start_frame"],
                "observation_frame_indices": subwindow["observed_frame_ids"],
                "window_sampling_mode": WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                "window_start_frame": subwindow["sample_start_frame"],
                "window_end_frame": subwindow["sample_end_frame"],
                "anchor_frame_index": subwindow["anchor_frame_index"],
                "state_anchor_frame": subwindow["state_anchor_frame"],
                "proprio_context_frame_index": subwindow["proprio_context_frame_index"],
                "proprio_context_local_frame": subwindow["proprio_context_local_frame"],
                "proprio_context_chunk_count": int(subwindow["proprio_context_state"].shape[0]),
                "proprio_context_frame_count": int(subwindow["proprio_context_frames"].shape[0]),
                "observed_frame_ids": subwindow["observed_frame_ids"],
                "latent_temporal_layout": subwindow["latent_temporal_layout"],
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "condition_latent_layout": condition_layout_metadata,
                "has_condition_latents": subwindow["condition_latents"] is not None,
                "state_source_key": self.data_config.action_target.pose_source_key,
                "action_representation": self.data_config.action_target.representation,
                "virtual_sample_index": int(index),
                "trajectory_window_index": window_index,
                "virtual_latent_start": latent_start,
                "subwindow_latent_start": latent_start,
                "subwindow_latent_end": latent_start + self.segment_frames,
                "segment_length_frames": self.segment_frames,
                "segment_valid_latent_frames": subwindow["valid_latent_frames"],
                "segment_padded_latent_frames": subwindow["padded_latent_frames"],
                "tail_padding_mode": "none" if tail_padded_frame_count == 0 else "zero_order_hold",
                "subwindow_action_start": subwindow["action_start_index"],
                "subwindow_action_end": subwindow["action_end_index"],
                **self._uniform_segment_attention_metadata(
                    latent_start=effective_latent_start,
                    segment_length=int(boundary_metadata.get("effective_segment_frames", self.segment_frames)),
                    valid_latent_frames=subwindow["valid_latent_frames"],
                    loss_frame_start=subwindow["loss_frame_start"],
                    loss_frame_end=subwindow["loss_frame_end"],
                    sample_start_frame=subwindow["sample_start_frame"],
                    start_padding_frames=subwindow["start_padding_frames"],
                    pre_start_frames=subwindow["pre_start_frames"],
                    emit_explicit_loss_ranges=True,
                    context_prefix_enabled=int(boundary_metadata.get("context_prefix_frames_requested", 0)) > 0,
                    sampled_chunk_size=sampled_chunk_size,
                    sampled_window_size=max(1, int(self.data_config.sample_construction.window_size)),
                ),
                **boundary_metadata,
                **subwindow["action_target_metadata"],
                **self._action_loss_metadata(
                    subwindow["action_mask"],
                    loss_frame_start=subwindow["loss_frame_start"],
                    loss_frame_end=subwindow["loss_frame_end"],
                    latent_num_frames=int(boundary_metadata.get("effective_segment_frames", self.segment_frames)),
                ),
                **self._hierarchical_sample_metadata(
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

    def __getitem__(self, index: int) -> LatentWAMSample:
        window = self.windows[index]
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        latent_payloads = self._load_window_latents(window, repo_bundle.metadata)
        full_video_latents, latent_layout_metadata = self._assemble_canonical_latents(latent_payloads)
        primary_payload = latent_payloads[self.data_config.latent_camera_names[0]]

        subwindow = self._sample_causal_prefix_suffix_subwindow(
            video_latents=full_video_latents,
            rows=rows,
            primary_payload=primary_payload,
            window=window,
            index=index,
        )
        task_index = int(rows[min(subwindow["sample_start_frame"], len(rows) - 1)].get("task_index", 0)) if rows else 0
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
            video_latents=subwindow["video_latents"],
            actions=subwindow["actions"],
            action_mask=subwindow["action_mask"],
            state=subwindow["state"],
            state_mask=subwindow["state_mask"],
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            metadata={
                "repo_root": str(window.repo_root),
                "dataset_id": str(window.repo_root),
                "episode_index": window.episode_index,
                "segment_start_frame": window.start_frame,
                "segment_end_frame": window.end_frame,
                "sample_start_frame": subwindow["sample_start_frame"],
                "sample_end_frame": subwindow["sample_end_frame"],
                "observation_start": subwindow["sample_start_frame"],
                "observation_frame_indices": subwindow["observed_frame_ids"],
                "window_sampling_mode": WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
                "window_start_frame": subwindow["sample_start_frame"],
                "window_end_frame": subwindow["sample_end_frame"],
                "anchor_frame_index": subwindow["sample_start_frame"],
                "observed_frame_ids": subwindow["observed_frame_ids"],
                "latent_temporal_layout": subwindow["latent_temporal_layout"],
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "action_representation": self.data_config.action_target.representation,
                "subwindow_latent_start": subwindow["latent_start_index"],
                "subwindow_latent_end": subwindow["latent_end_index"],
                "observed_prefix_frames": subwindow["observed_prefix_frames"],
                "future_suffix_frames": subwindow["future_suffix_frames"],
                "valid_video_frames": subwindow["valid_video_frames"],
                "padded_video_frames": int(subwindow["video_latents"].shape[1]),
                **self._action_loss_metadata(subwindow["action_mask"]),
                **self._sample_weight_metadata(index),
            },
        )

    def _sample_causal_prefix_suffix_subwindow(
        self,
        *,
        video_latents: torch.Tensor,
        rows: list[dict[str, Any]],
        primary_payload: dict[str, Any],
        window: LocalEpisodeWindow,
        index: int,
    ) -> dict[str, Any]:
        sample_cfg = self.data_config.sample_construction
        padded_num_frames = int(sample_cfg.num_frames)
        buckets = tuple(sample_cfg.causal_prefix_suffix_buckets)
        if not buckets:
            raise ValueError(
                "Causal prefix/suffix sampling requires non-empty `sample_construction.causal_prefix_suffix_buckets`."
            )
        raw_frame_ids = [int(value) for value in list(primary_payload.get("frame_ids", []))]
        if not raw_frame_ids:
            raw_frame_ids = list(window.observation_frame_indices)
        raw_bucket_boundaries = self._build_raw_bucket_boundaries(
            raw_frame_count=len(raw_frame_ids),
            latent_num_frames=int(video_latents.shape[1]),
            latent_temporal_layout=self.data_config.latent_temporal_layout,
        )
        valid_candidates: list[tuple[int, int]] = []
        for bucket_index, bucket in enumerate(buckets):
            total_frames = int(bucket.total_frames)
            if total_frames > int(video_latents.shape[1]):
                continue
            max_latent_start = int(video_latents.shape[1]) - total_frames
            for latent_start in range(max_latent_start + 1):
                latent_end = latent_start + total_frames
                raw_start_position = raw_bucket_boundaries[latent_start]
                raw_end_position = raw_bucket_boundaries[latent_end]
                if raw_start_position >= len(raw_frame_ids) or raw_end_position <= raw_start_position:
                    continue
                sample_end_frame = raw_frame_ids[max(raw_start_position, raw_end_position - 1)] + 1
                if sample_end_frame > len(rows):
                    continue
                valid_candidates.append((latent_start, bucket_index))
        if not valid_candidates:
            raise ValueError(
                "No valid causal prefix/suffix sample could be drawn from the local latent segment. "
                f"episode_index={window.episode_index}, latent_frames={video_latents.shape[1]}, "
                f"configured_buckets={[(bucket.observed_frames, bucket.future_frames) for bucket in buckets]}."
            )

        if self.data_config.split == DataSplit.TRAIN:
            rng = random.Random(random.randrange(1 << 30) + index)
        else:
            rng = random.Random(self.data_config.split_seed + index)
        latent_start, bucket_index = valid_candidates[rng.randrange(len(valid_candidates))]
        bucket = buckets[bucket_index]
        total_frames = int(bucket.total_frames)
        latent_end = latent_start + total_frames
        raw_start_position, raw_end_position, sample_start_frame, sample_end_frame = raw_span_for_latent_range(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=int(video_latents.shape[1]),
            latent_start=latent_start,
            latent_end=latent_end,
            layout=self.data_config.latent_temporal_layout,
        )
        observed_frame_ids = observed_frame_ids_for_latent_segment(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=int(video_latents.shape[1]),
            latent_start=latent_start,
            segment_length=total_frames,
            layout=self.data_config.latent_temporal_layout,
        )
        padded_latents = torch.zeros(
            video_latents.shape[0],
            padded_num_frames,
            video_latents.shape[2],
            video_latents.shape[3],
            dtype=video_latents.dtype,
        )
        padded_latents[:, :total_frames] = video_latents[:, latent_start:latent_end]
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
        return {
            "video_latents": padded_latents.contiguous(),
            "actions": actions,
            "action_mask": action_mask,
            "state": state,
            "state_mask": state_mask,
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "observed_frame_ids": observed_frame_ids,
            "latent_temporal_layout": self.data_config.latent_temporal_layout,
            "latent_start_index": latent_start,
            "latent_end_index": latent_end,
            "observed_prefix_frames": int(bucket.observed_frames),
            "future_suffix_frames": int(bucket.future_frames),
            "valid_video_frames": total_frames,
        }


def build_local_lerobot_latent_train_val_datasets(
    data_config: DataConfig,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    train_windows: list[LocalEpisodeWindow] = []
    val_windows: list[LocalEpisodeWindow] = []
    use_config_replay_status_path = object()

    def _filtered_windows_for_bundles(
        local_root: str,
        *,
        max_episodes: int | None = None,
        configured_replay_status_path: str | None | object = use_config_replay_status_path,
        replay_status_policy: ReplayStatusPolicy | None = None,
        require_replay_status: bool | None = None,
    ) -> list[LocalEpisodeWindow]:
        windows: list[LocalEpisodeWindow] = []
        for bundle in discover_local_lerobot_repo_bundles(local_root):
            repo_windows = scan_local_latent_windows(bundle.root, data_config)
            repo_episodes = [episode.episode_index for episode in bundle.metadata.episodes]
            replay_status_records, replay_status_path = load_replay_status_records(
                bundle.root,
                replay_status_path=(
                    data_config.replay_status_path
                    if configured_replay_status_path is use_config_replay_status_path
                    else configured_replay_status_path
                ),
                require=data_config.require_replay_status
                if require_replay_status is None
                else bool(require_replay_status),
            )
            split = split_episode_indices_by_replay_status(
                repo_episodes,
                replay_status_records=replay_status_records,
                replay_status_path=replay_status_path,
                replay_status_policy=replay_status_policy or data_config.replay_status_policy,
                require_replay_status=(
                    data_config.require_replay_status
                    if require_replay_status is None
                    else bool(require_replay_status)
                ),
                val_replay_status_policy=None,
                val_require_replay_status=None,
                train_fraction=1.0,
                split_seed=data_config.split_seed,
                max_train_episodes=max_episodes,
                max_val_episodes=None,
            )
            episode_set = set(split.train_episodes)
            windows.extend(window for window in repo_windows if window.episode_index in episode_set)
        return windows

    if data_config.val_local_root:
        val_replay_status_path = data_config.val_replay_status_path
        if val_replay_status_path is None and data_config.replay_status_path is not None:
            train_status_path = Path(data_config.replay_status_path).expanduser()
            val_replay_status_path = None if train_status_path.is_absolute() else data_config.replay_status_path
        train_windows = _filtered_windows_for_bundles(
            data_config.local_root or "",
            max_episodes=data_config.max_train_episodes,
        )
        val_windows = _filtered_windows_for_bundles(
            data_config.val_local_root,
            max_episodes=data_config.max_val_episodes,
            configured_replay_status_path=val_replay_status_path,
            replay_status_policy=data_config.val_replay_status_policy or data_config.replay_status_policy,
            require_replay_status=(
                data_config.require_replay_status
                if data_config.val_require_replay_status is None
                else data_config.val_require_replay_status
            ),
        )
    else:
        bundles = discover_local_lerobot_repo_bundles(data_config.local_root or "")
        for bundle in bundles:
            repo_windows = scan_local_latent_windows(bundle.root, data_config)
            repo_episodes = [episode.episode_index for episode in bundle.metadata.episodes]
            replay_status_records, replay_status_path = load_replay_status_records(
                bundle.root,
                replay_status_path=data_config.replay_status_path,
                require=data_config.require_replay_status,
            )
            split = split_episode_indices_by_replay_status(
                repo_episodes,
                replay_status_records=replay_status_records,
                replay_status_path=replay_status_path,
                replay_status_policy=data_config.replay_status_policy,
                require_replay_status=data_config.require_replay_status,
                val_replay_status_policy=data_config.val_replay_status_policy,
                val_require_replay_status=data_config.val_require_replay_status,
                train_fraction=data_config.train_fraction,
                split_seed=data_config.split_seed,
                max_train_episodes=data_config.max_train_episodes,
                max_val_episodes=data_config.max_val_episodes,
            )
            train_episode_set = set(split.train_episodes)
            val_episode_set = set(split.val_episodes)
            repo_train_windows = [window for window in repo_windows if window.episode_index in train_episode_set]
            repo_val_windows = [window for window in repo_windows if window.episode_index in val_episode_set]
            if not split.used_explicit_val_policy and not repo_val_windows and repo_train_windows:
                repo_val_windows = repo_train_windows[:1]
            train_windows.extend(repo_train_windows)
            val_windows.extend(repo_val_windows)

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
