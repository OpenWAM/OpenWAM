from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
import math
import json
import random
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from einops import rearrange
from torch.utils.data import Dataset, Sampler

from open_wam.configs import (
    ActionTargetReferenceSource,
    ActionTargetRepresentation,
    DataConfig,
    DataSplit,
    LatentWindowProfile,
    SampleWeightMode,
    WindowSamplingMode,
)

from .action_transforms import build_relative_pose_targets, expected_pose_target_dim
from .action_mapping import (
    action_mapping_is_active,
    apply_action_mapping,
    resolve_action_source_dim,
)
from .latent_contracts import LatentWAMSample
from .lerobot_v2 import LeRobotEpisodeRecord, LeRobotV2Metadata, _resolve_row_key
from .replay_status import filter_episode_indices_by_replay_status, load_replay_status_records


_LATENT_FILE_PATTERN = re.compile(r"episode_(?P<episode>\d{6})_(?P<start>\d+)_(?P<end>\d+)\.pth$")


@dataclass(frozen=True)
class LocalEpisodeWindow:
    """One latent window over one local episode file."""

    repo_root: Path
    episode_index: int
    start_frame: int
    end_frame: int
    observed_frame_ids: tuple[int, ...] = ()

    @property
    def observation_start(self) -> int:
        if self.observed_frame_ids:
            return int(self.observed_frame_ids[0])
        return int(self.start_frame)

    @property
    def observation_frame_indices(self) -> tuple[int, ...]:
        if self.observed_frame_ids:
            return tuple(int(value) for value in self.observed_frame_ids)
        return tuple(range(int(self.start_frame), int(self.end_frame)))


@dataclass(frozen=True)
class LocalRepoBundle:
    """Metadata and episode lookup for one discovered local repo."""

    root: Path
    metadata: LeRobotV2Metadata
    episodes_by_index: dict[int, LeRobotEpisodeRecord]


class LocalLeRobotLatentWindowDataset(Dataset[LatentWAMSample]):
    """Latent-first local-repo dataset for LingBot-style post-training exports."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
        if data_config.local_root is None:
            raise ValueError("Local latent datasets require `data.local_root` in the experiment config.")
        self.data_config = data_config
        self.windows = list(windows)
        self.empty_text_embedding = self._load_empty_text_embedding()
        self._repo_bundles = {
            str(bundle.root): bundle
            for bundle in discover_local_lerobot_repo_bundles(data_config.local_root)
        }
        self._episode_cache: OrderedDict[tuple[str, int], list[dict[str, Any]]] = OrderedDict()
        self._latent_view_cache: OrderedDict[
            tuple[str, int, int, int],
            tuple[torch.Tensor, dict[str, dict[str, int]], dict[str, Any]],
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
        if self.data_config.sample_construction.sample_weight_mode == SampleWeightMode.UNIFORM:
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
            prefix_actions = frame_stride * int(
                self.data_config.action_schema.action_horizon // max(1, self.data_config.num_frames)
            )
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

    def _action_loss_metadata(self, action_mask: torch.Tensor | None) -> dict[str, Any]:
        if action_mask is None:
            valid_steps = int(self.data_config.action_schema.action_horizon)
            valid_values = valid_steps * int(self.data_config.action_schema.action_dim)
        else:
            reduced = action_mask.float().sum(dim=-1)
            valid_steps = int((reduced > 0).sum().item())
            valid_values = int(action_mask.float().sum().item())
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

        primary_payload = latent_payloads[self.data_config.latent_camera_names[0]]
        observed_frame_ids = [int(value) for value in list(primary_payload.get("frame_ids", []))]
        if not observed_frame_ids:
            observed_frame_ids = list(window.observation_frame_indices)
        observation_start = int(observed_frame_ids[0])
        observation_end = int(observed_frame_ids[-1]) + 1
        anchor_frame_index = observed_frame_ids[-1]

        actions, action_mask, action_target_metadata = self._build_full_segment_action_targets(
            rows=rows,
            window=window,
            observed_frame_ids=observed_frame_ids,
            latent_num_frames=int(video_latents.shape[1]),
        )
        state_start = max(0, anchor_frame_index - self.data_config.action_schema.state_horizon + 1)
        state_rows = rows[state_start : anchor_frame_index + 1]
        state_source_key = self.data_config.action_target.pose_source_key
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=state_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=self.data_config.action_schema.state_horizon,
            left_pad=True,
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
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "state_source_key": state_source_key,
                "action_representation": self.data_config.action_target.representation,
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
    ) -> tuple[torch.Tensor, dict[str, dict[str, int]]]:
        canonical_latents = None
        metadata: dict[str, dict[str, int]] = {}
        for view_layout, camera_name in zip(self.data_config.view_layout, self.data_config.latent_camera_names, strict=True):
            payload = latent_payloads[camera_name]
            view_latents = reshape_latent_payload(payload)
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

    def _load_canonical_window_latents(
        self,
        window: LocalEpisodeWindow,
        metadata: LeRobotV2Metadata,
    ) -> tuple[torch.Tensor, dict[str, dict[str, int]], dict[str, Any]]:
        cache_key = (str(window.repo_root), window.episode_index, window.start_frame, window.end_frame)
        if cache_key in self._latent_view_cache:
            self._latent_view_cache.move_to_end(cache_key)
            return self._latent_view_cache[cache_key]

        latent_payloads = self._load_window_latents(window, metadata)
        video_latents, latent_layout_metadata = self._assemble_canonical_latents(latent_payloads)
        primary_payload = dict(latent_payloads[self.data_config.latent_camera_names[0]])
        payload = (video_latents, latent_layout_metadata, primary_payload)
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_target = self.data_config.action_target
        if action_target.representation != ActionTargetRepresentation.RAW:
            raise ValueError(
                "Long-window local latent datasets currently support only `action_target.representation=raw` "
                "for LingBot-compatible exact training."
            )
        if latent_num_frames <= 0:
            raise ValueError("Expected at least one latent frame in the local latent window.")
        if not observed_frame_ids:
            raise ValueError("Expected non-empty frame_ids metadata for the local latent window.")

        frame_stride = 1
        if len(observed_frame_ids) > 1:
            frame_stride = max(1, int(observed_frame_ids[1] - observed_frame_ids[0]))
        prefix_actions = frame_stride * int(self.data_config.action_schema.action_horizon // max(1, self.data_config.num_frames))
        required_action_num = latent_num_frames * prefix_actions

        action_start_offset = max(0, int(observed_frame_ids[0] - window.start_frame))
        raw_window_rows = rows[window.start_frame : window.end_frame]
        raw_actions = torch.stack(
            [
                torch.tensor(row[_resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                for row in raw_window_rows
            ],
            dim=0,
        )
        raw_actions = raw_actions[action_start_offset:]
        action_dim = raw_actions.shape[-1] if raw_actions.numel() > 0 else self.data_config.action_schema.action_dim
        if raw_actions.shape[-1] != self.data_config.action_schema.action_dim:
            raise ValueError(
                "Configured action_dim does not match raw local latent supervision: "
                f"configured={self.data_config.action_schema.action_dim}, raw={raw_actions.shape[-1]}."
            )

        padded_actions = torch.cat(
            [
                torch.zeros(prefix_actions, action_dim, dtype=torch.float32),
                raw_actions,
            ],
            dim=0,
        )
        if padded_actions.shape[0] < required_action_num:
            padded_actions = torch.cat(
                [
                    padded_actions,
                    torch.zeros(required_action_num - padded_actions.shape[0], action_dim, dtype=torch.float32),
                ],
                dim=0,
            )
        actions = padded_actions[:required_action_num].contiguous()

        action_mask = torch.ones_like(actions, dtype=torch.float32)
        if raw_actions.shape[0] + prefix_actions < required_action_num:
            action_mask[raw_actions.shape[0] + prefix_actions :] = 0.0
        return actions, action_mask, {
            "lingbot_window_action_alignment": {
                "latent_num_frames": latent_num_frames,
                "raw_frame_count": len(observed_frame_ids),
                "frame_stride": frame_stride,
                "prefix_actions": prefix_actions,
                "required_action_num": required_action_num,
                "action_start_offset": action_start_offset,
            }
        }

    def _build_action_targets(
        self,
        *,
        action_rows: list[dict[str, Any]],
        target_state_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_target = self.data_config.action_target
        action_mapping = self.data_config.action_mapping
        target_dim = self.data_config.action_schema.action_dim
        target_length = self.data_config.action_schema.action_horizon

        if action_target.representation == ActionTargetRepresentation.RAW:
            source_dim = resolve_action_source_dim(action_mapping, fallback_dim=target_dim)
            actions, action_mask = self._extract_sequence(
                rows=action_rows,
                key=action_target.source_key,
                target_dim=source_dim,
                target_length=target_length,
            )
            mapped = apply_action_mapping(
                actions,
                action_mask,
                action_mapping,
                target_dim=target_dim,
            )
            return mapped.actions, mapped.action_mask, mapped.metadata

        if action_target.representation == ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
            if action_target.reference_source != ActionTargetReferenceSource.ANCHOR_STATE:
                raise ValueError(
                    "Local latent LeRobot datasets currently support only "
                    f"`reference_source=anchor_state`, got {action_target.reference_source}."
                )
            pose_source = torch.stack(
                [
                    torch.tensor(row[_resolve_row_key(row, action_target.pose_source_key)], dtype=torch.float32)
                    for row in target_state_rows
                ],
                dim=0,
            )
            raw_action_sequence = torch.stack(
                [
                    torch.tensor(row[_resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                    for row in action_rows
                ],
                dim=0,
            )
            relative_targets, relative_mask, metadata = build_relative_pose_targets(
                pose_source,
                state_encoding=action_target.state_encoding,
                rotation_representation=action_target.rotation_representation,
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
                raw_action_sequence=raw_action_sequence,
                gripper_action_index=action_target.gripper_action_index,
            )
            expected_dim = expected_pose_target_dim(
                rotation_representation=action_target.rotation_representation,
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
            )
            target_or_source_dim = resolve_action_source_dim(action_mapping, fallback_dim=target_dim)
            if target_or_source_dim != expected_dim:
                raise ValueError(
                    "Configured action_dim does not match the derived pose-target dimension: "
                    f"configured_dim={target_or_source_dim}, expected={expected_dim} for "
                    f"[rotation_representation={action_target.rotation_representation}, "
                    f"gripper_representation={action_target.gripper_representation}]."
                )
            metadata.update(
                {
                    "reference_source": action_target.reference_source,
                    "pose_source_key": action_target.pose_source_key,
                    "gripper_source_key": action_target.source_key,
                }
            )
            actions, action_mask = self._pack_sequence(
                sequence=relative_targets,
                target_dim=target_or_source_dim,
                target_length=target_length,
            )
            if relative_mask.shape[-1] != relative_targets.shape[-1]:
                raise ValueError("Relative target mask shape must match the relative target tensor shape.")
            action_mask[:, : relative_mask.shape[-1]] = relative_mask
            mapped = apply_action_mapping(
                actions,
                action_mask,
                action_mapping,
                target_dim=target_dim,
            )
            metadata.update(mapped.metadata)
            metadata["action_mapping_applied"] = action_mapping_is_active(action_mapping)
            return mapped.actions, mapped.action_mask, metadata

        raise ValueError(f"Unsupported action target representation: {action_target.representation}")

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
            [torch.tensor(row[_resolve_row_key(row, key)], dtype=torch.float32) for row in rows],
            dim=0,
        )
        return self._pack_sequence(
            sequence=sequence,
            target_dim=target_dim,
            target_length=target_length,
            left_pad=left_pad,
            sequence_name=key,
        )

    def _pack_sequence(
        self,
        *,
        sequence: torch.Tensor,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
        sequence_name: str = "sequence",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 2:
            raise ValueError(
                f"Expected {sequence_name} tensor with shape [T, D], got {tuple(sequence.shape)}."
            )
        raw_dim = sequence.shape[-1]
        if raw_dim > target_dim:
            raise ValueError(f"Raw {sequence_name} dim {raw_dim} exceeds configured target dim {target_dim}.")

        output = torch.zeros(target_length, target_dim, dtype=torch.float32)
        mask = torch.zeros(target_length, target_dim, dtype=torch.float32)
        clipped = sequence[:target_length]
        start_index = target_length - len(clipped) if left_pad else 0
        for index, values in enumerate(clipped):
            output[start_index + index, : raw_dim] = values
            mask[start_index + index, : raw_dim] = 1.0
        return output, mask

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
    def _build_raw_bucket_boundaries(*, raw_frame_count: int, latent_num_frames: int) -> list[int]:
        if raw_frame_count <= 0 or latent_num_frames <= 0:
            raise ValueError(
                "Expected positive raw frame and latent frame counts for aligned subwindow sampling, "
                f"got raw_frame_count={raw_frame_count}, latent_num_frames={latent_num_frames}."
            )
        boundaries = []
        for latent_index in range(latent_num_frames + 1):
            boundaries.append((latent_index * raw_frame_count) // latent_num_frames)
        boundaries[-1] = raw_frame_count
        return boundaries


class LocalLatentWeightedTrainSampler(Sampler[int]):
    """Replacement train sampler for weighted local latent examples."""

    def __init__(self, dataset: LocalLeRobotLatentWindowDataset, *, world_size: int = 1, rank: int = 0) -> None:
        if len(dataset) <= 0:
            raise ValueError("Weighted local latent sampling requires a non-empty dataset.")
        if world_size <= 0:
            raise ValueError(f"`world_size` must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"`rank` must be in [0, world_size), got rank={rank}, world_size={world_size}.")
        self.dataset = dataset
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.epoch = 0
        self._num_samples = int(math.ceil(len(dataset) / float(self.world_size)))
        self._total_size = self._num_samples * self.world_size

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        weights = torch.tensor(self.dataset.sample_weights, dtype=torch.double)
        if float(weights.sum().item()) <= 0:
            weights = torch.ones(len(self.dataset), dtype=torch.double)
        generator = torch.Generator()
        seed = (int(self.dataset.data_config.split_seed) + self.epoch * 1_000_003) & 0x7FFF_FFFF_FFFF_FFFF
        generator.manual_seed(seed)
        sampled = torch.multinomial(
            weights,
            num_samples=self._total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(int(index) for index in sampled[self.rank : self._total_size : self.world_size])


class LocalLatentEpochOrderSampler(Sampler[int]):
    """Sampler backed by a dataset-provided epoch order."""

    def __init__(self, dataset: "UniformSegmentLocalLeRobotLatentDataset", *, world_size: int = 1, rank: int = 0) -> None:
        if len(dataset) <= 0:
            raise ValueError("Epoch-order local latent sampling requires a non-empty dataset.")
        if world_size <= 0:
            raise ValueError(f"`world_size` must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"`rank` must be in [0, world_size), got rank={rank}, world_size={world_size}.")
        self.dataset = dataset
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.epoch = 0
        self._num_samples = int(math.ceil(len(dataset) / float(self.world_size)))
        self._total_size = self._num_samples * self.world_size

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        order = self.dataset.build_epoch_index_order(epoch=self.epoch)
        if not order:
            raise ValueError("Epoch-order local latent sampler received an empty order.")
        if len(order) < self._total_size:
            repeats = int(math.ceil(self._total_size / len(order)))
            order = (order * repeats)[: self._total_size]
        else:
            order = order[: self._total_size]
        return iter(int(index) for index in order[self.rank : self._total_size : self.world_size])


class UniformSegmentLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Uniform latent-start segment sampler over all eligible trajectories."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
        super().__init__(data_config, windows)
        self._segment_length_candidates = self._resolve_segment_length_candidates()
        self._virtual_index = self._build_virtual_index()
        if not self._virtual_index:
            raise ValueError("Uniform segment sampling requires at least one latent start.")
        self._virtual_indices_by_window = self._build_virtual_indices_by_window()
        self.dataset_mean_valid_action_steps = self._estimate_virtual_mean_valid_action_steps()
        self.sample_weights = self._build_virtual_sample_weights()

    def __len__(self) -> int:
        return len(self._virtual_index)

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int]:
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
        for window_index, window in enumerate(self.windows):
            source_latent_frames = max(1, len(window.observation_frame_indices))
            for latent_start in range(source_latent_frames):
                virtual_index.append((window_index, latent_start))
        return tuple(virtual_index)

    def _build_virtual_indices_by_window(self) -> dict[int, list[int]]:
        by_window: dict[int, list[int]] = {}
        for virtual_index, (window_index, _) in enumerate(self._virtual_index):
            by_window.setdefault(window_index, []).append(virtual_index)
        return by_window

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
        estimates = [
            self._estimate_segment_valid_action_steps(
                window=window,
                latent_start=latent_start,
                segment_length=segment_length,
            )
            for segment_length in self._segment_length_candidates
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
        )
        frame_stride = 1
        if len(observed_frame_ids) > 1:
            frame_stride = max(1, int(observed_frame_ids[1] - observed_frame_ids[0]))
        prefix_actions = frame_stride * int(
            self.data_config.action_schema.action_horizon // max(1, self.data_config.num_frames)
        )
        valid_latent_end = min(source_latent_frames, latent_start + segment_length)
        boundaries = self._build_raw_bucket_boundaries(
            raw_frame_count=len(raw_frame_ids),
            latent_num_frames=source_latent_frames,
        )
        raw_start_position = min(boundaries[latent_start], len(raw_frame_ids) - 1)
        raw_end_position = boundaries[valid_latent_end]
        if raw_end_position <= raw_start_position:
            raw_end_position = raw_start_position + 1
        sample_start_frame = raw_frame_ids[raw_start_position]
        sample_end_frame = raw_frame_ids[min(len(raw_frame_ids) - 1, raw_end_position - 1)] + 1
        raw_action_steps = max(0, sample_end_frame - sample_start_frame)
        required_action_steps = max(1, segment_length * prefix_actions)
        return min(required_action_steps, prefix_actions + raw_action_steps)

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
        }

    def __getitem__(self, index: int) -> LatentWAMSample:
        window_index, latent_start = self._virtual_index[index]
        window = self.windows[window_index]
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        full_video_latents, latent_layout_metadata, primary_payload = self._load_canonical_window_latents(
            window,
            repo_bundle.metadata,
        )
        segment_length = self._sample_segment_length(index)
        subwindow = self._build_uniform_segment(
            video_latents=full_video_latents,
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
                "observed_frame_ids": subwindow["observed_frame_ids"],
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "state_source_key": self.data_config.action_target.pose_source_key,
                "action_representation": self.data_config.action_target.representation,
                "virtual_sample_index": index,
                "trajectory_window_index": window_index,
                "subwindow_latent_start": latent_start,
                "subwindow_latent_end": latent_start + segment_length,
                "segment_length_frames": segment_length,
                "segment_valid_latent_frames": subwindow["valid_latent_frames"],
                "segment_padded_latent_frames": subwindow["padded_latent_frames"],
                "tail_padding_mode": "zero_hold",
                "subwindow_action_start": subwindow["action_start_index"],
                "subwindow_action_end": subwindow["action_end_index"],
                **subwindow["action_target_metadata"],
                **self._action_loss_metadata(subwindow["action_mask"]),
                **self._sample_weight_metadata(index),
            },
        )

    def _sample_segment_length(self, index: int) -> int:
        split_salt = 17 if self.data_config.split == DataSplit.TRAIN else 53
        seed = (
            int(self.data_config.split_seed)
            + split_salt
            + 1_000_003 * int(index + 1)
        ) & 0x7FFF_FFFF_FFFF_FFFF
        rng = random.Random(seed)
        return int(self._segment_length_candidates[rng.randrange(len(self._segment_length_candidates))])

    def _build_uniform_segment(
        self,
        *,
        video_latents: torch.Tensor,
        rows: list[dict[str, Any]],
        primary_payload: dict[str, Any],
        window: LocalEpisodeWindow,
        latent_start: int,
        segment_length: int,
    ) -> dict[str, Any]:
        source_latent_frames = int(video_latents.shape[1])
        if source_latent_frames <= 0:
            raise ValueError("Uniform segment sampling requires at least one source latent frame.")
        if latent_start < 0 or latent_start >= source_latent_frames:
            raise IndexError(f"latent_start={latent_start} is outside source_latent_frames={source_latent_frames}.")
        raw_frame_ids = [int(value) for value in list(primary_payload.get("frame_ids", []))]
        if not raw_frame_ids:
            raw_frame_ids = list(window.observation_frame_indices)
        if not raw_frame_ids:
            raise ValueError("Uniform segment sampling requires non-empty frame ids.")

        valid_latent_frames = max(0, min(segment_length, source_latent_frames - latent_start))
        padded_latent_frames = max(0, segment_length - valid_latent_frames)
        observed_frame_ids = self._segment_observed_frame_ids(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=source_latent_frames,
            latent_start=latent_start,
            segment_length=segment_length,
        )
        boundaries = self._build_raw_bucket_boundaries(
            raw_frame_count=len(raw_frame_ids),
            latent_num_frames=source_latent_frames,
        )
        raw_start_position = min(boundaries[latent_start], len(raw_frame_ids) - 1)
        valid_latent_end = min(source_latent_frames, latent_start + segment_length)
        raw_end_position = boundaries[valid_latent_end]
        if raw_end_position <= raw_start_position:
            raw_end_position = raw_start_position + 1
        sample_start_frame = raw_frame_ids[raw_start_position]
        sample_end_frame = raw_frame_ids[min(len(raw_frame_ids) - 1, raw_end_position - 1)] + 1
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
            latent_num_frames=segment_length,
        )
        state_start = max(0, anchor_frame_index - self.data_config.action_schema.state_horizon + 1)
        state_rows = rows[state_start : min(anchor_frame_index + 1, len(rows))]
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=self.data_config.action_target.pose_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=self.data_config.action_schema.state_horizon,
            left_pad=True,
        )
        return {
            "video_latents": self._slice_video_latents_with_zero_hold(
                video_latents=video_latents,
                latent_start=latent_start,
                segment_length=segment_length,
            ),
            "actions": actions,
            "action_mask": action_mask,
            "action_target_metadata": action_target_metadata,
            "state": state,
            "state_mask": state_mask,
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "anchor_frame_index": anchor_frame_index,
            "observed_frame_ids": observed_frame_ids,
            "action_start_index": sample_start_frame,
            "action_end_index": sample_start_frame + int(actions.shape[0]),
            "valid_latent_frames": valid_latent_frames,
            "padded_latent_frames": padded_latent_frames,
        }

    @staticmethod
    def _segment_observed_frame_ids(
        *,
        raw_frame_ids: list[int],
        source_latent_frames: int,
        latent_start: int,
        segment_length: int,
    ) -> list[int]:
        boundaries = LocalLeRobotLatentWindowDataset._build_raw_bucket_boundaries(
            raw_frame_count=len(raw_frame_ids),
            latent_num_frames=source_latent_frames,
        )
        observed_frame_ids: list[int] = []
        for latent_index in range(latent_start, latent_start + segment_length):
            if latent_index >= source_latent_frames:
                observed_frame_ids.append(raw_frame_ids[-1])
                continue
            bucket_start = boundaries[latent_index]
            bucket_end = boundaries[latent_index + 1]
            if bucket_start >= len(raw_frame_ids):
                observed_frame_ids.append(raw_frame_ids[-1])
                continue
            bucket = raw_frame_ids[bucket_start:bucket_end]
            observed_frame_ids.append(bucket[-1] if bucket else raw_frame_ids[bucket_start])
        return observed_frame_ids

    @staticmethod
    def _slice_video_latents_with_zero_hold(
        *,
        video_latents: torch.Tensor,
        latent_start: int,
        segment_length: int,
    ) -> torch.Tensor:
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


class FullSegmentLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Current LingBot-style long-window latent dataset view."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
        super().__init__(data_config, windows)
        self.sample_index = tuple(self.windows)


class RandomSubwindowLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Generic random subwindow view over one exported local latent segment."""

    def __getitem__(self, index: int) -> LatentWAMSample:
        window = self.windows[index]
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        latent_payloads = self._load_window_latents(window, repo_bundle.metadata)
        full_video_latents, latent_layout_metadata = self._assemble_canonical_latents(latent_payloads)
        primary_payload = latent_payloads[self.data_config.latent_camera_names[0]]

        subwindow = self._sample_random_subwindow(
            video_latents=full_video_latents,
            rows=rows,
            primary_payload=primary_payload,
            window=window,
            index=index,
        )
        task_index = int(rows[min(subwindow["anchor_frame_index"], len(rows) - 1)].get("task_index", 0)) if rows else 0
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
                "window_sampling_mode": WindowSamplingMode.RANDOM_SUBWINDOW,
                "window_start_frame": subwindow["sample_start_frame"],
                "window_end_frame": subwindow["sample_end_frame"],
                "anchor_frame_index": subwindow["anchor_frame_index"],
                "observed_frame_ids": subwindow["observed_frame_ids"],
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "state_source_key": self.data_config.action_target.pose_source_key,
                "action_representation": self.data_config.action_target.representation,
                "subwindow_latent_start": subwindow["latent_start_index"],
                "subwindow_latent_end": subwindow["latent_end_index"],
                "subwindow_action_start": subwindow["action_start_index"],
                "subwindow_action_end": subwindow["action_end_index"],
                **subwindow["action_target_metadata"],
                **self._action_loss_metadata(subwindow["action_mask"]),
                **self._sample_weight_metadata(index),
            },
        )

    def _sample_random_subwindow(
        self,
        *,
        video_latents: torch.Tensor,
        rows: list[dict[str, Any]],
        primary_payload: dict[str, Any],
        window: LocalEpisodeWindow,
        index: int,
    ) -> dict[str, Any]:
        sample_cfg = self.data_config.sample_construction
        sample_num_frames = int(sample_cfg.num_frames)
        action_horizon = int(sample_cfg.action_horizon)
        state_horizon = int(sample_cfg.state_horizon)
        if sample_num_frames != int(self.data_config.num_frames):
            raise ValueError(
                "Random local latent subwindow sampling currently expects `sample_construction.num_frames` "
                "to match `data.num_frames` so the shared pipeline contracts stay consistent, "
                f"got sample_num_frames={sample_num_frames}, data.num_frames={self.data_config.num_frames}."
            )
        if action_horizon != int(self.data_config.action_schema.action_horizon):
            raise ValueError(
                "Random local latent subwindow sampling currently expects `sample_construction.action_horizon` "
                "to match `data.action_schema.action_horizon`, "
                f"got sample_action_horizon={action_horizon}, "
                f"schema_action_horizon={self.data_config.action_schema.action_horizon}."
            )
        if state_horizon != int(self.data_config.action_schema.state_horizon):
            raise ValueError(
                "Random local latent subwindow sampling currently expects `sample_construction.state_horizon` "
                "to match `data.action_schema.state_horizon`, "
                f"got sample_state_horizon={state_horizon}, "
                f"schema_state_horizon={self.data_config.action_schema.state_horizon}."
            )
        if video_latents.shape[1] < sample_num_frames:
            raise ValueError(
                "Random subwindow sampling requires at least as many latent frames as the requested sample length, "
                f"got source_latent_frames={video_latents.shape[1]}, requested={sample_num_frames}."
            )

        raw_frame_ids = [int(value) for value in list(primary_payload.get("frame_ids", []))]
        if not raw_frame_ids:
            raw_frame_ids = list(range(window.start_frame, window.end_frame))
        raw_bucket_boundaries = self._build_raw_bucket_boundaries(
            raw_frame_count=len(raw_frame_ids),
            latent_num_frames=int(video_latents.shape[1]),
        )
        max_latent_start = int(video_latents.shape[1]) - sample_num_frames
        valid_latent_starts: list[int] = []
        for latent_start in range(max_latent_start + 1):
            latent_end = latent_start + sample_num_frames
            raw_start_position = raw_bucket_boundaries[latent_start]
            if raw_start_position >= len(raw_frame_ids):
                continue
            sample_start_frame = raw_frame_ids[raw_start_position]
            if sample_start_frame + action_horizon > len(rows):
                continue
            anchor_frame_index = raw_frame_ids[max(raw_start_position, raw_bucket_boundaries[latent_end] - 1)]
            if anchor_frame_index >= len(rows):
                continue
            valid_latent_starts.append(latent_start)

        if not valid_latent_starts:
            raise ValueError(
                "No valid random subwindow could be sampled from the local latent segment. "
                f"episode_index={window.episode_index}, latent_frames={video_latents.shape[1]}, "
                f"requested_num_frames={sample_num_frames}, action_horizon={action_horizon}."
            )

        if self.data_config.split == DataSplit.TRAIN:
            rng = random.Random(random.randrange(1 << 30) + index)
            latent_start = valid_latent_starts[rng.randrange(len(valid_latent_starts))]
        else:
            rng = random.Random(self.data_config.split_seed + index)
            latent_start = valid_latent_starts[rng.randrange(len(valid_latent_starts))]

        latent_end = latent_start + sample_num_frames
        raw_start_position = raw_bucket_boundaries[latent_start]
        raw_end_position = raw_bucket_boundaries[latent_end]
        sample_start_frame = raw_frame_ids[raw_start_position]
        sample_end_frame = raw_frame_ids[max(raw_start_position, raw_end_position - 1)] + 1
        observed_frame_ids = raw_frame_ids[raw_start_position:raw_end_position]
        if not observed_frame_ids:
            observed_frame_ids = [sample_start_frame]
        anchor_frame_index = observed_frame_ids[-1]

        sampled_window = LocalEpisodeWindow(
            repo_root=window.repo_root,
            episode_index=window.episode_index,
            start_frame=sample_start_frame,
            end_frame=sample_end_frame,
        )
        actions, action_mask, action_target_metadata = self._build_lingbot_window_action_targets(
            rows=rows,
            window=sampled_window,
            observed_frame_ids=observed_frame_ids,
            latent_num_frames=sample_num_frames,
        )
        state_start = max(0, anchor_frame_index - state_horizon + 1)
        state_rows = rows[state_start : anchor_frame_index + 1]
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=self.data_config.action_target.pose_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=state_horizon,
            left_pad=True,
        )

        return {
            "video_latents": video_latents[:, latent_start:latent_end].contiguous(),
            "actions": actions,
            "action_mask": action_mask,
            "action_target_metadata": action_target_metadata,
            "state": state,
            "state_mask": state_mask,
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "anchor_frame_index": anchor_frame_index,
            "observed_frame_ids": observed_frame_ids,
            "latent_start_index": latent_start,
            "latent_end_index": latent_end,
            "action_start_index": sample_start_frame,
            "action_end_index": sample_start_frame + action_horizon,
        }


class ContextualSubwindowLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """Method-driven exact-training subwindow with explicit history and current region."""

    def __getitem__(self, index: int) -> LatentWAMSample:
        window = self.windows[index]
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        latent_payloads = self._load_window_latents(window, repo_bundle.metadata)
        full_video_latents, latent_layout_metadata = self._assemble_canonical_latents(latent_payloads)
        primary_payload = latent_payloads[self.data_config.latent_camera_names[0]]

        subwindow = self._sample_contextual_subwindow(
            video_latents=full_video_latents,
            rows=rows,
            primary_payload=primary_payload,
            window=window,
            index=index,
        )
        task_index = int(rows[min(subwindow["anchor_frame_index"], len(rows) - 1)].get("task_index", 0)) if rows else 0
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
                "window_sampling_mode": WindowSamplingMode.CONTEXTUAL_SUBWINDOW,
                "window_start_frame": subwindow["sample_start_frame"],
                "window_end_frame": subwindow["sample_end_frame"],
                "anchor_frame_index": subwindow["anchor_frame_index"],
                "observed_frame_ids": subwindow["observed_frame_ids"],
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "state_source_key": self.data_config.action_target.pose_source_key,
                "action_representation": self.data_config.action_target.representation,
                "subwindow_latent_start": subwindow["latent_start_index"],
                "subwindow_latent_end": subwindow["latent_end_index"],
                "subwindow_action_start": subwindow["action_start_index"],
                "subwindow_action_end": subwindow["action_end_index"],
                "sampled_chunk_size": subwindow["sampled_chunk_size"],
                "sampled_window_size": subwindow["sampled_window_size"],
                "history_frames": subwindow["history_frames"],
                "current_frames": subwindow["current_frames"],
                "current_start_frame_in_sample": subwindow["current_start_frame_in_sample"],
                "current_end_frame_in_sample": subwindow["current_end_frame_in_sample"],
                "loss_frame_start": subwindow["current_start_frame_in_sample"],
                "loss_frame_end": subwindow["current_end_frame_in_sample"],
                "frame_shift": subwindow["sample_start_frame"],
                **subwindow["action_target_metadata"],
                **self._action_loss_metadata(subwindow["action_mask"]),
                **self._sample_weight_metadata(index),
            },
        )

    def _sample_contextual_subwindow(
        self,
        *,
        video_latents: torch.Tensor,
        rows: list[dict[str, Any]],
        primary_payload: dict[str, Any],
        window: LocalEpisodeWindow,
        index: int,
    ) -> dict[str, Any]:
        sample_cfg = self.data_config.sample_construction
        predict_blocks_per_sample = max(1, int(sample_cfg.predict_blocks_per_sample))
        if sample_cfg.chunk_size <= 0:
            raise ValueError("Contextual subwindow sampling requires positive `sample_construction.chunk_size`.")
        if sample_cfg.window_size <= 0:
            raise ValueError("Contextual subwindow sampling requires positive `sample_construction.window_size`.")

        if self.data_config.action_schema.action_horizon % max(1, self.data_config.num_frames) != 0:
            raise ValueError(
                "Contextual subwindow sampling expects `action_horizon` to divide by `data.num_frames`, "
                f"got action_horizon={self.data_config.action_schema.action_horizon}, "
                f"data.num_frames={self.data_config.num_frames}."
            )

        raw_frame_ids = [int(value) for value in list(primary_payload.get("frame_ids", []))]
        if not raw_frame_ids:
            raw_frame_ids = list(range(window.start_frame, window.end_frame))
        raw_bucket_boundaries = self._build_raw_bucket_boundaries(
            raw_frame_count=len(raw_frame_ids),
            latent_num_frames=int(video_latents.shape[1]),
        )

        if self.data_config.split == DataSplit.TRAIN:
            geometry_rng = random.Random(random.randrange(1 << 30) + index)
        else:
            geometry_rng = random.Random(self.data_config.split_seed + index)
        if sample_cfg.randomize_geometry:
            chunk_size_candidates = list(range(1, int(sample_cfg.chunk_size) + 1))
            if int(sample_cfg.window_size) >= 4:
                window_size_candidates = list(range(4, int(sample_cfg.window_size) + 1))
            else:
                window_size_candidates = [max(1, int(sample_cfg.window_size))]
        else:
            chunk_size_candidates = [int(sample_cfg.chunk_size)]
            window_size_candidates = [max(1, int(sample_cfg.window_size))]

        valid_geometries: list[tuple[int, int, int, int, int]] = []
        for candidate_chunk_size in chunk_size_candidates:
            candidate_current_frames = candidate_chunk_size * predict_blocks_per_sample
            if candidate_current_frames <= 0 or video_latents.shape[1] < candidate_current_frames:
                continue
            for candidate_window_size in window_size_candidates:
                candidate_history_video_chunks = max(1, int(math.ceil(candidate_window_size / 2.0)))
                candidate_history_frames = candidate_history_video_chunks * candidate_chunk_size
                candidate_sample_num_frames = candidate_history_frames + candidate_current_frames
                max_latent_start = int(video_latents.shape[1]) - candidate_sample_num_frames
                if max_latent_start < 0:
                    continue
                valid_geometries.append(
                    (
                        candidate_chunk_size,
                        candidate_window_size,
                        candidate_current_frames,
                        candidate_history_frames,
                        candidate_sample_num_frames,
                    )
                )

        if not valid_geometries:
            raise ValueError(
                "No valid contextual geometry fits inside the source segment. "
                f"source_latent_frames={video_latents.shape[1]}, max_chunk_size={sample_cfg.chunk_size}, "
                f"max_window_size={sample_cfg.window_size}, predict_blocks_per_sample={predict_blocks_per_sample}."
            )

        (
            sampled_chunk_size,
            sampled_window_size,
            current_frames,
            history_frames,
            sample_num_frames,
        ) = valid_geometries[geometry_rng.randrange(len(valid_geometries))]
        max_latent_start = int(video_latents.shape[1]) - sample_num_frames

        valid_latent_starts: list[int] = []
        action_per_video_frame = self.data_config.action_schema.action_horizon // max(1, self.data_config.num_frames)
        for latent_start in range(max_latent_start + 1):
            latent_end = latent_start + sample_num_frames
            raw_start_position = raw_bucket_boundaries[latent_start]
            if raw_start_position >= len(raw_frame_ids):
                continue
            sample_start_frame = raw_frame_ids[raw_start_position]
            # LingBot-style long-window targets can legally run past the end of
            # the parquet episode rows because the target builder pads the
            # missing tail with zeros. We only require a valid anchor inside the
            # available episode rows here.
            if sample_start_frame >= len(rows):
                continue
            valid_latent_starts.append(latent_start)

        if not valid_latent_starts:
            raise ValueError(
                "No valid contextual subwindow could be sampled from the local latent segment. "
                f"episode_index={window.episode_index}, latent_frames={video_latents.shape[1]}, "
                f"history_frames={history_frames}, current_frames={current_frames}, "
                f"action_per_video_frame={action_per_video_frame}."
            )

        if self.data_config.split == DataSplit.TRAIN:
            start_rng = random.Random(random.randrange(1 << 30) + 17 * (index + 1))
        else:
            start_rng = random.Random(self.data_config.split_seed + 97 * (index + 1))
        latent_start = valid_latent_starts[start_rng.randrange(len(valid_latent_starts))]
        latent_end = latent_start + sample_num_frames

        raw_start_position = raw_bucket_boundaries[latent_start]
        raw_end_position = raw_bucket_boundaries[latent_end]
        sample_start_frame = raw_frame_ids[raw_start_position]
        sample_end_frame = raw_frame_ids[max(raw_start_position, raw_end_position - 1)] + 1
        observed_frame_ids = raw_frame_ids[raw_start_position:raw_end_position]
        if not observed_frame_ids:
            observed_frame_ids = [sample_start_frame]
        anchor_frame_index = observed_frame_ids[-1]

        sampled_window = LocalEpisodeWindow(
            repo_root=window.repo_root,
            episode_index=window.episode_index,
            start_frame=sample_start_frame,
            end_frame=sample_end_frame,
        )
        actions, action_mask, action_target_metadata = self._build_lingbot_window_action_targets(
            rows=rows,
            window=sampled_window,
            observed_frame_ids=observed_frame_ids,
            latent_num_frames=sample_num_frames,
        )
        state_start = max(0, anchor_frame_index - self.data_config.action_schema.state_horizon + 1)
        state_rows = rows[state_start : anchor_frame_index + 1]
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=self.data_config.action_target.pose_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=self.data_config.action_schema.state_horizon,
            left_pad=True,
        )

        return {
            "video_latents": video_latents[:, latent_start:latent_end].contiguous(),
            "actions": actions,
            "action_mask": action_mask,
            "action_target_metadata": action_target_metadata,
            "state": state,
            "state_mask": state_mask,
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "anchor_frame_index": anchor_frame_index,
            "observed_frame_ids": observed_frame_ids,
            "latent_start_index": latent_start,
            "latent_end_index": latent_end,
            "action_start_index": sample_start_frame,
            "action_end_index": sample_start_frame + sample_num_frames * action_per_video_frame,
            "sampled_chunk_size": sampled_chunk_size,
            "sampled_window_size": sampled_window_size,
            "history_frames": history_frames,
            "current_frames": current_frames,
            "current_start_frame_in_sample": history_frames,
            "current_end_frame_in_sample": history_frames + current_frames,
        }


class AlignedSubwindowLocalLeRobotLatentDataset(LocalLeRobotLatentWindowDataset):
    """DreamZero-style aligned subwindow view over the shared local latent source."""

    def __getitem__(self, index: int) -> LatentWAMSample:
        window = self.windows[index]
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        latent_payloads = self._load_window_latents(window, repo_bundle.metadata)
        full_video_latents, latent_layout_metadata = self._assemble_canonical_latents(latent_payloads)
        primary_payload = latent_payloads[self.data_config.latent_camera_names[0]]

        subwindow = self._sample_aligned_subwindow(
            video_latents=full_video_latents,
            rows=rows,
            primary_payload=primary_payload,
            window=window,
            index=index,
        )
        task_index = int(rows[min(subwindow["action_start_index"], len(rows) - 1)].get("task_index", 0)) if rows else 0
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
                "window_sampling_mode": WindowSamplingMode.ALIGNED_SUBWINDOW,
                "window_start_frame": subwindow["sample_start_frame"],
                "window_end_frame": subwindow["sample_end_frame"],
                "anchor_frame_index": subwindow["anchor_frame_index"],
                "observed_frame_ids": subwindow["observed_frame_ids"],
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "state_source_key": self.data_config.action_target.pose_source_key,
                "action_representation": self.data_config.action_target.representation,
                "subwindow_latent_start": subwindow["latent_start_index"],
                "subwindow_latent_end": subwindow["latent_end_index"],
                "subwindow_action_start": subwindow["action_start_index"],
                "subwindow_action_end": subwindow["action_end_index"],
                "state_indices": subwindow["state_indices"],
                "subwindow_state_indices": subwindow["state_indices"],
                **self._action_loss_metadata(subwindow["action_mask"]),
                **self._sample_weight_metadata(index),
            },
        )

    def _sample_aligned_subwindow(
        self,
        *,
        video_latents: torch.Tensor,
        rows: list[dict[str, Any]],
        primary_payload: dict[str, Any],
        window: LocalEpisodeWindow,
        index: int,
    ) -> dict[str, Any]:
        sample_cfg = self.data_config.sample_construction
        sample_num_frames = int(sample_cfg.num_frames)
        frame_stride = max(1, int(sample_cfg.frame_stride))
        if video_latents.shape[1] < sample_num_frames:
            raise ValueError(
                "Aligned subwindow sampling requires more latent frames than the source segment provides, "
                f"got source_latent_frames={video_latents.shape[1]}, requested={sample_num_frames}."
            )

        if sample_num_frames < 2:
            raise ValueError("Aligned subwindow sampling requires at least two latent frames.")

        num_blocks = sample_num_frames - 1
        action_horizon = int(sample_cfg.action_horizon)
        state_horizon = int(sample_cfg.state_horizon)
        if action_horizon % num_blocks != 0:
            raise ValueError(
                "Aligned subwindow sampling currently expects action_horizon to divide evenly across future blocks, "
                f"got action_horizon={action_horizon}, num_blocks={num_blocks}."
            )
        if state_horizon % num_blocks != 0:
            raise ValueError(
                "Aligned subwindow sampling currently expects state_horizon to divide evenly across future blocks, "
                f"got state_horizon={state_horizon}, num_blocks={num_blocks}."
            )

        raw_frame_ids = [int(value) for value in list(primary_payload.get("frame_ids", []))]
        if not raw_frame_ids:
            raw_frame_ids = list(range(window.start_frame, window.end_frame))
        raw_bucket_boundaries = self._build_raw_bucket_boundaries(
            raw_frame_count=len(raw_frame_ids),
            latent_num_frames=int(video_latents.shape[1]),
        )
        required_latent_span = 1 + (sample_num_frames - 1) * frame_stride
        max_latent_start = int(video_latents.shape[1]) - required_latent_span

        action_per_block = action_horizon // num_blocks
        state_per_block = state_horizon // num_blocks
        valid_latent_starts: list[int] = []
        for latent_start in range(max_latent_start + 1):
            latent_indices = [latent_start + offset * frame_stride for offset in range(sample_num_frames)]
            raw_start_position = raw_bucket_boundaries[latent_indices[0]]
            raw_end_position = raw_bucket_boundaries[latent_indices[-1] + 1]
            if raw_start_position >= len(raw_frame_ids):
                continue
            sample_start_frame = raw_frame_ids[raw_start_position]
            action_end_index = sample_start_frame + action_horizon
            if action_end_index > len(rows):
                continue
            state_indices = []
            for block_index in range(num_blocks):
                block_state_start = sample_start_frame + block_index * action_per_block
                for offset in range(state_per_block):
                    state_indices.append(block_state_start + offset)
            if state_indices and max(state_indices) >= len(rows):
                continue
            valid_latent_starts.append(latent_start)

        if not valid_latent_starts:
            raise ValueError(
                "No valid aligned subwindow could be sampled from the local latent segment. "
                f"episode_index={window.episode_index}, latent_frames={video_latents.shape[1]}, "
                f"requested_num_frames={sample_num_frames}, action_horizon={action_horizon}, "
                f"state_horizon={state_horizon}."
            )

        rng = random.Random(self.data_config.split_seed + index)
        latent_start = valid_latent_starts[rng.randrange(len(valid_latent_starts))]
        latent_indices = [latent_start + offset * frame_stride for offset in range(sample_num_frames)]
        raw_start_position = raw_bucket_boundaries[latent_indices[0]]
        raw_end_position = raw_bucket_boundaries[latent_indices[-1] + 1]
        sample_start_frame = raw_frame_ids[raw_start_position]
        sample_end_frame = raw_frame_ids[max(raw_start_position, raw_end_position - 1)] + 1
        observed_frame_ids: list[int] = []
        for latent_index in latent_indices:
            bucket_start = raw_bucket_boundaries[latent_index]
            bucket_end = raw_bucket_boundaries[latent_index + 1]
            if bucket_start >= len(raw_frame_ids):
                observed_frame_ids.append(raw_frame_ids[-1])
                continue
            bucket = raw_frame_ids[bucket_start:bucket_end]
            observed_frame_ids.append(bucket[-1] if bucket else raw_frame_ids[bucket_start])
        if not observed_frame_ids:
            observed_frame_ids = [sample_start_frame]

        action_rows = rows[sample_start_frame : sample_start_frame + action_horizon]
        actions, action_mask = self._extract_sequence(
            rows=action_rows,
            key=self.data_config.action_target.source_key,
            target_dim=self.data_config.action_schema.action_dim,
            target_length=self.data_config.action_schema.action_horizon,
        )

        state_indices: list[int] = []
        for block_index in range(num_blocks):
            block_state_start = sample_start_frame + block_index * action_per_block
            for offset in range(state_per_block):
                state_indices.append(block_state_start + offset)
        state_rows = [rows[state_index] for state_index in state_indices]
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=self.data_config.action_target.pose_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=self.data_config.action_schema.state_horizon,
        )

        return {
            "video_latents": video_latents.index_select(
                1,
                torch.tensor(latent_indices, dtype=torch.long, device=video_latents.device),
            ).contiguous(),
            "actions": actions,
            "action_mask": action_mask,
            "state": state,
            "state_mask": state_mask,
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "anchor_frame_index": sample_start_frame,
            "observed_frame_ids": observed_frame_ids,
            "latent_start_index": latent_start,
            "latent_end_index": latent_indices[-1] + 1,
            "action_start_index": sample_start_frame,
            "action_end_index": sample_start_frame + action_horizon,
            "state_indices": tuple(state_indices),
        }


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
        raw_start_position = raw_bucket_boundaries[latent_start]
        raw_end_position = raw_bucket_boundaries[latent_end]
        sample_start_frame = raw_frame_ids[raw_start_position]
        sample_end_frame = raw_frame_ids[max(raw_start_position, raw_end_position - 1)] + 1
        observed_frame_ids = raw_frame_ids[raw_start_position:raw_end_position]
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
            "latent_start_index": latent_start,
            "latent_end_index": latent_end,
            "observed_prefix_frames": int(bucket.observed_frames),
            "future_suffix_frames": int(bucket.future_frames),
            "valid_video_frames": total_frames,
        }


def build_local_lerobot_latent_train_val_datasets(
    data_config: DataConfig,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    bundles = discover_local_lerobot_repo_bundles(data_config.local_root or "")
    train_windows: list[LocalEpisodeWindow] = []
    val_windows: list[LocalEpisodeWindow] = []
    for bundle in bundles:
        repo_windows = scan_local_latent_windows(bundle.root, data_config)
        repo_episodes = [episode.episode_index for episode in bundle.metadata.episodes]
        replay_status_records, replay_status_path = load_replay_status_records(
            bundle.root,
            replay_status_path=data_config.replay_status_path,
            require=data_config.require_replay_status,
        )
        repo_episodes, _ = filter_episode_indices_by_replay_status(
            repo_episodes,
            replay_status_records=replay_status_records,
            policy=data_config.replay_status_policy,
            require_labeled=bool(replay_status_records) or bool(data_config.require_replay_status),
            source_path=replay_status_path,
        )
        train_episodes, repo_val_episodes = split_local_episode_indices(
            episode_indices=repo_episodes,
            train_fraction=data_config.train_fraction,
            split_seed=data_config.split_seed,
            max_train_episodes=data_config.max_train_episodes,
            max_val_episodes=data_config.max_val_episodes,
        )
        train_episode_set = set(train_episodes)
        val_episode_set = set(repo_val_episodes)
        repo_train_windows = [window for window in repo_windows if window.episode_index in train_episode_set]
        repo_val_windows = [window for window in repo_windows if window.episode_index in val_episode_set]
        if not repo_val_windows and repo_train_windows:
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
    elif data_config.sample_construction.mode == WindowSamplingMode.RANDOM_SUBWINDOW:
        dataset_cls = RandomSubwindowLocalLeRobotLatentDataset
    elif data_config.sample_construction.mode == WindowSamplingMode.CONTEXTUAL_SUBWINDOW:
        if data_config.train_batch_size != 1 or data_config.val_batch_size != 1:
            raise ValueError(
                "Contextual subwindow sampling currently requires train/val batch size 1 because "
                "sample lengths vary across examples."
            )
        dataset_cls = ContextualSubwindowLocalLeRobotLatentDataset
    elif data_config.sample_construction.mode == WindowSamplingMode.ALIGNED_SUBWINDOW:
        dataset_cls = AlignedSubwindowLocalLeRobotLatentDataset
    elif data_config.sample_construction.mode == WindowSamplingMode.CAUSAL_PREFIX_SUFFIX:
        dataset_cls = CausalPrefixSuffixLocalLeRobotLatentDataset
    else:
        raise ValueError(
            f"Unsupported sample_construction.mode for local latent datasets: "
            f"{data_config.sample_construction.mode!r}"
        )

    return (
        dataset_cls(data_config=data_config, windows=train_windows),
        dataset_cls(data_config=data_config, windows=val_windows),
    )


def discover_local_lerobot_repo_bundles(local_root: str | Path) -> list[LocalRepoBundle]:
    """Discover one or more local LeRobot-style repo roots."""

    root = Path(local_root).expanduser()
    repo_roots: list[Path] = []
    if (root / "meta" / "info.json").exists():
        repo_roots.append(root)
    else:
        repo_roots.extend(sorted(path.parent.parent for path in root.rglob("meta/info.json")))
    if not repo_roots:
        raise FileNotFoundError(f"No local LeRobot repo roots were discovered under {root}.")

    bundles: list[LocalRepoBundle] = []
    for repo_root in repo_roots:
        metadata = load_lerobot_v2_local_metadata(repo_root)
        bundles.append(
            LocalRepoBundle(
                root=repo_root,
                metadata=metadata,
                episodes_by_index={episode.episode_index: episode for episode in metadata.episodes},
            )
        )
    return bundles


def scan_local_latent_windows(repo_root: Path, data_config: DataConfig) -> list[LocalEpisodeWindow]:
    """Scan a local latent export tree into reusable latent windows."""

    primary_camera = data_config.latent_camera_names[0]
    windows: list[LocalEpisodeWindow] = []
    for camera_dir in sorted((path for path in resolve_latent_root(repo_root, data_config).glob(f"chunk-*/{primary_camera}") if path.is_dir())):
        for latent_file in sorted(camera_dir.glob("episode_*.pth")):
            match = _LATENT_FILE_PATTERN.match(latent_file.name)
            if match is None:
                continue
            payload = torch.load(latent_file, map_location="cpu", weights_only=False)
            observed_frame_ids: tuple[int, ...] = ()
            if isinstance(payload, dict):
                raw_frame_ids = payload.get("frame_ids")
                if isinstance(raw_frame_ids, (list, tuple)):
                    observed_frame_ids = tuple(int(value) for value in raw_frame_ids)
            windows.append(
                LocalEpisodeWindow(
                    repo_root=repo_root,
                    episode_index=int(match.group("episode")),
                    start_frame=int(match.group("start")),
                    end_frame=int(match.group("end")),
                    observed_frame_ids=observed_frame_ids,
                )
            )
    return windows


def split_local_episode_indices(
    *,
    episode_indices: list[int],
    train_fraction: float,
    split_seed: int,
    max_train_episodes: int | None,
    max_val_episodes: int | None,
) -> tuple[list[int], list[int]]:
    shuffled = list(episode_indices)
    rng = random.Random(split_seed)
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * train_fraction)
    train_count = min(max(train_count, 1), len(shuffled))
    train_episodes = shuffled[:train_count]
    val_episodes = shuffled[train_count:]
    if max_train_episodes is not None:
        train_episodes = train_episodes[:max_train_episodes]
    if max_val_episodes is not None:
        val_episodes = val_episodes[:max_val_episodes]
    if not val_episodes and train_episodes:
        val_episodes = train_episodes[:1]
    return train_episodes, val_episodes


def load_lerobot_v2_local_metadata(repo_root: Path) -> LeRobotV2Metadata:
    """Load the self-describing metadata files from one local LeRobot-style repo."""

    info = read_json_local(repo_root / "meta" / "info.json")
    episodes = read_jsonl_local(repo_root / "meta" / "episodes.jsonl")
    tasks = read_jsonl_local(repo_root / "meta" / "tasks.jsonl")
    return LeRobotV2Metadata(
        repo_id=str(repo_root),
        codebase_version=str(info["codebase_version"]),
        fps=int(info["fps"]),
        chunk_size=int(info["chunks_size"]),
        total_episodes=int(info["total_episodes"]),
        data_path_template=str(info["data_path"]),
        features={name: dict(feature) for name, feature in info["features"].items()},
        episodes=tuple(
            LeRobotEpisodeRecord(
                episode_index=int(record["episode_index"]),
                length=int(record["length"]),
                tasks=tuple(record.get("tasks", [])),
            )
            for record in episodes
        ),
        tasks_by_index={int(record["task_index"]): str(record["task"]) for record in tasks},
    )


def resolve_latent_root(repo_root: Path, data_config: DataConfig) -> Path:
    if data_config.latent_root is None:
        return repo_root / data_config.latent_subdir
    configured = Path(data_config.latent_root).expanduser()
    if configured.is_absolute():
        return configured
    return (repo_root / configured).resolve()


def latent_filename(*, episode_index: int, start_frame: int, end_frame: int) -> str:
    return f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"


def read_json_local(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_local(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reshape_latent_payload(payload: dict[str, Any]) -> torch.Tensor:
    latent = payload["latent"]
    if not isinstance(latent, torch.Tensor):
        latent = torch.tensor(latent)
    latent_num_frames = int(payload["latent_num_frames"])
    latent_height = int(payload["latent_height"])
    latent_width = int(payload["latent_width"])
    if latent.ndim == 2:
        return rearrange(
            latent,
            "(f h w) c -> f h w c",
            f=latent_num_frames,
            h=latent_height,
            w=latent_width,
        )
    if latent.ndim == 4:
        if tuple(latent.shape[:3]) != (latent_num_frames, latent_height, latent_width):
            raise ValueError(
                "Latent payload shape does not match metadata. "
                f"shape={tuple(latent.shape)}, expected=({latent_num_frames}, {latent_height}, {latent_width}, C)."
            )
        return latent
    raise ValueError(
        "Unsupported latent payload shape. "
        f"Expected flattened `[F*H*W, C]` or `[F, H, W, C]`, got {tuple(latent.shape)}."
    )
