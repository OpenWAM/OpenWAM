from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from open_wam.configs import (
    DataConfig,
    DataSplit,
    GeneralistDynamicsMixtureConfig,
    PaddedTargetPolicy,
    TailPaddingPolicy,
    WindowSamplingMode,
)
from open_wam.configs.variant_semantics import (
    GENERALIST_TRAINING_BUCKET_METADATA_KEY,
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
)

from .distributed_sampling import PaddedEpochOffsetDistributedSampler
from .latent_contracts import LatentWAMSample
from .latent_temporal import latent_anchor_positions


REAL_DEMO_SOURCE = "real_demo"
COUNTERFACTUAL_DYNAMICS_SOURCE = "counterfactual_dynamics"
JOINT_MODE = "joint"
ACTION_CONDITIONED_VIDEO_MODE = "action_conditioned_video"
VIDEO_CONDITIONED_ACTION_MODE = "video_conditioned_action"
COUNTERFACTUAL_STATE_KEY = "observation.state"
COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY = "next_latent_source_offset"
COUNTERFACTUAL_CONTRACT_T0_PLUS_FUTURE = "t0_observation_plus_future"
COUNTERFACTUAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE = "target_only_t0_observation_plus_future"
GENERALIST_CONDITIONAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE = "target_only_t0_observation_plus_future"
GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON = "t0_singleton"
GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY = "previous_boundary_video_only"


@dataclass(frozen=True)
class GeneralistMixtureBucket:
    name: str
    source: str
    mode: str
    weight: float
    drop_text: bool


@dataclass(frozen=True)
class _CounterfactualWindowSpec:
    transition_index: int
    task_key: str
    start_min: int
    start_max: int
    eligible_start_count: int
    mass_within_task: float
    source_latent_frames: int
    context_frames: int


@dataclass(frozen=True)
class _CounterfactualTaskSpec:
    task_key: str
    eligible_start_count: int
    demo_count: int
    task_mass: float
    windows: tuple[_CounterfactualWindowSpec, ...]
    window_mass_total: float


class EncodedCounterfactualDynamicsLatentDataset(Dataset[LatentWAMSample]):
    """Latent dataset for simulator-rendered counterfactual dynamics samples.

    Each sample uses the counterfactual target payload directly: target latent
    0 is the observed t0 frame, and loss starts at target latent 1. The
    pre-t0 context artifacts remain useful for data provenance but are not
    included in the model-visible sequence.
    """

    def __init__(self, data_config: DataConfig, encoded_root: str | Path, *, split: str) -> None:
        self.data_config = data_config
        self.encoded_root = Path(encoded_root).expanduser().resolve()
        self.split = str(split)
        self.manifest = _read_json(self.encoded_root / "manifest.json")
        self.raw_root = _counterfactual_raw_root_from_manifest(self.manifest, encoded_root=self.encoded_root)
        _validate_counterfactual_condition_latent_manifest(
            self.manifest,
            encoded_root=self.encoded_root,
            source_frame_offset=int(data_config.sample_construction.condition_source_frame_offset),
        )
        self.transition_rows = _read_jsonl(self.encoded_root / "metadata" / "encoded_transitions.jsonl")
        context_rows = _read_jsonl(self.encoded_root / "metadata" / "encoded_contexts.jsonl")
        self.context_rows = {
            _context_key(row): row
            for row in context_rows
        }
        self.empty_text_embedding = _load_empty_text_embedding(data_config.empty_text_embedding_path)
        if not self.transition_rows:
            raise ValueError(f"No encoded counterfactual transitions found under {self.encoded_root}.")
        self._window_specs = self._build_window_specs()
        self._task_specs = self._build_task_specs()
        self._task_weights = tuple(float(task.task_mass) for task in self._task_specs)
        self._task_mass_total = float(sum(self._task_weights))
        self._epoch_sample_count = (
            sum(spec.eligible_start_count for spec in self._window_specs)
            if self._uses_hierarchical_fixed_segment
            else len(self.transition_rows)
        )
        if self._epoch_sample_count <= 0:
            raise ValueError(f"No eligible counterfactual samples found under {self.encoded_root}.")

    def __len__(self) -> int:
        return self._epoch_sample_count

    def build_balanced_source_indices(self) -> tuple[int, ...]:
        if self._uses_hierarchical_fixed_segment:
            return tuple(range(len(self)))
        return _balanced_counterfactual_source_indices(self.transition_rows)

    def __getitem__(self, index: int) -> LatentWAMSample:
        if self._uses_hierarchical_fixed_segment:
            task_spec, window_spec, latent_start = self._draw_hierarchical_sample(index)
            row = self.transition_rows[window_spec.transition_index]
            hierarchical_metadata = self._hierarchical_sample_metadata(
                index=index,
                task_spec=task_spec,
                window_spec=window_spec,
            )
            segment_length = int(self.data_config.sample_construction.segment_frames or window_spec.source_latent_frames)
        else:
            row = self.transition_rows[int(index) % len(self.transition_rows)]
            latent_start = 0
            segment_length = None
            hierarchical_metadata = {}
        target_payload = _load_latent_payload(self._resolve_encoded_path(row, "target_latent_path"))
        target_latents = _payload_latents(target_payload, key="target_video_latents")
        source_video_latents = target_latents.contiguous()
        context_frames = 0
        target_frames = int(target_latents.shape[1])
        source_frames = int(source_video_latents.shape[1])
        if segment_length is None:
            segment_length = source_frames

        source_condition_latents, condition_latents_source = _counterfactual_target_only_condition_latents_from_payload(
            target_payload=target_payload,
            fallback_video_latents=source_video_latents,
            source_frame_offset=int(self.data_config.sample_construction.condition_source_frame_offset),
        )
        sample_npz = np.load(self._resolve_raw_path(row, "sample_path"))
        source_actions = _pack_actions(
            np.asarray(sample_npz["future_actions"], dtype=np.float32),
            target_dim=int(self.data_config.action_schema.action_dim),
        )
        state_dim = int(self.data_config.action_schema.state_dim)
        source_proprio_frames, source_proprio_frames_mask = _counterfactual_latent_state_frames(
            sample_npz,
            latent_frames=target_frames,
            state_dim=state_dim,
            data_config=self.data_config,
        )
        action_per_frame = _counterfactual_action_steps_per_frame(
            source_actions,
            total_frames=source_frames,
            data_config=self.data_config,
        )
        sampled_chunk_size, sampled_window_size = _sample_counterfactual_attention_geometry(
            data_config=self.data_config,
            segment_length=int(segment_length),
        )
        segment = _build_counterfactual_fixed_segment(
            video_latents=source_video_latents,
            condition_latents=source_condition_latents,
            actions=source_actions,
            context_frames=context_frames,
            latent_start=int(latent_start),
            segment_length=int(segment_length),
            action_per_frame=action_per_frame,
            condition_source_frame_offset=int(self.data_config.sample_construction.condition_source_frame_offset),
            mask_leading_zero_action_context=True,
        )
        video_latents = segment["video_latents"]
        condition_latents = segment["condition_latents"]
        actions = segment["actions"]
        action_mask = segment["action_mask"]
        total_frames = int(video_latents.shape[1])
        proprio_context_frames = _slice_counterfactual_frame_tensor_with_edge_hold(
            source_proprio_frames,
            latent_start=int(latent_start),
            segment_length=int(segment_length),
        )
        proprio_context_frames_mask = _slice_counterfactual_frame_tensor_with_edge_hold(
            source_proprio_frames_mask,
            latent_start=int(latent_start),
            segment_length=int(segment_length),
        )
        proprio_context_state = proprio_context_frames.clone()
        proprio_context_state_mask = proprio_context_frames_mask.clone()
        state, state_mask = _counterfactual_state_history_from_frames(
            proprio_context_frames=source_proprio_frames,
            proprio_context_frames_mask=source_proprio_frames_mask,
            anchor_frame=int(segment["prefix_state_source_frame"]),
            state_horizon=int(self.data_config.action_schema.state_horizon),
            state_dim=state_dim,
        )
        proprio_context_source = (
            COUNTERFACTUAL_STATE_KEY
            if float(proprio_context_frames_mask.sum().item()) > 0.0
            else "unavailable_zero_mask"
        )
        text_context = self.empty_text_embedding.clone() if self.empty_text_embedding is not None else None

        observed_frame_ids = _counterfactual_observed_frame_ids(
            context_start_frame=int(row.get("t0_frame", 0)),
            latent_start=int(latent_start),
            segment_length=int(segment_length),
            source_frames=source_frames,
            action_per_frame=action_per_frame,
        )
        valid_frame_ids = observed_frame_ids[: max(0, int(segment["valid_source_frames"]))]
        sample_start_frame = int(valid_frame_ids[0]) if valid_frame_ids else 0
        sample_end_frame = (
            int(valid_frame_ids[-1]) + int(action_per_frame)
            if valid_frame_ids
            else sample_start_frame
        )
        loss_frame_start = int(segment["loss_frame_start"])
        loss_frame_end = int(segment["loss_frame_end"])
        first_loss_frame_id = (
            int(observed_frame_ids[loss_frame_start])
            if 0 <= loss_frame_start < len(observed_frame_ids)
            else sample_end_frame
        )
        last_loss_frame_end = (
            int(observed_frame_ids[loss_frame_end - 1]) + int(action_per_frame)
            if loss_frame_end > loss_frame_start and loss_frame_end - 1 < len(observed_frame_ids)
            else first_loss_frame_id
        )
        target_observation_frame = int(segment["target_observation_frame"])
        target_observation_frame_id = (
            int(observed_frame_ids[target_observation_frame])
            if 0 <= target_observation_frame < len(observed_frame_ids)
            else None
        )
        transition_action_steps_required = max(0, source_frames - 1) * action_per_frame
        extra_source_action_steps = max(0, int(source_actions.shape[0]) - int(transition_action_steps_required))

        metadata = {
            "dataset_id": str(self.encoded_root),
            "dataset_kind": "encoded_counterfactual_dynamics",
            "generalist_conditional_contract": GENERALIST_CONDITIONAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE,
            "generalist_conditional_training_sequence": "target_only",
            "generalist_conditional_context_used_for_training": False,
            "generalist_gjd_chunk_contract": GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON,
            "generalist_conditional_history_policy": (
                GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY
            ),
            "counterfactual_contract": COUNTERFACTUAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE,
            "counterfactual_generation_contract": COUNTERFACTUAL_CONTRACT_T0_PLUS_FUTURE,
            "counterfactual_training_sequence": "target_only",
            "counterfactual_context_used_for_training": False,
            "split": self.split,
            "counterfactual_sample_id": int(row["sample_id"]),
            "counterfactual_context_id": int(row["context_id"]),
            "counterfactual_branch": row.get("branch"),
            "counterfactual_branch_family": row.get("branch_family"),
            "counterfactual_branch_strength": row.get("branch_strength"),
            "counterfactual_branch_is_ood": bool(row.get("branch_is_ood", False)),
            "episode_index": int(row.get("dataset_episode_index", -1)),
            "task_index": int(row.get("task_id", -1)),
            "init_state_index": row.get("init_state_index"),
            "t0_frame": int(row.get("t0_frame", 0)),
            "t0_action_frame": int(row.get("t0_frame", 0)) * action_per_frame,
            "context_start_frame": int(row.get("context_start_frame", 0)),
            "context_start_action_frame": int(row.get("context_start_frame", 0)) * action_per_frame,
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "observation_start": sample_start_frame,
            "observation_frame_indices": observed_frame_ids,
            "window_sampling_mode": self.data_config.sample_construction.mode,
            "window_start_frame": sample_start_frame,
            "window_end_frame": sample_end_frame,
            "anchor_frame_index": int(valid_frame_ids[-1]) if valid_frame_ids else sample_start_frame,
            "segment_length_frames": total_frames,
            "segment_valid_latent_frames": int(segment["valid_latent_frames"]),
            "segment_padded_latent_frames": int(segment["padded_latent_frames"]),
            "tail_padding_mode": "none" if int(segment["padded_latent_frames"]) == 0 else "zero_order_hold",
            "history_frames": loss_frame_start,
            "loss_frame_start": loss_frame_start,
            "loss_frame_end": loss_frame_end,
            "latent_loss_frame_start": loss_frame_start,
            "latent_loss_frame_end": loss_frame_end,
            "action_loss_frame_start": loss_frame_start,
            "action_loss_frame_end": loss_frame_end,
            "chunk_origin_frame": int(segment["chunk_origin_frame"]),
            "target_observation_frame_in_sample": int(segment["target_observation_frame"]),
            "target_observation_frame_index": target_observation_frame_id,
            "first_supervised_future_frame_in_sample": loss_frame_start,
            "first_supervised_future_frame_index": first_loss_frame_id,
            "supervised_future_latent_frames": max(0, loss_frame_end - loss_frame_start),
            "target_frame_start": first_loss_frame_id,
            "target_frame_end": last_loss_frame_end,
            "sampled_chunk_size": int(sampled_chunk_size),
            "counterfactual_gjd_chunk_contract": GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON,
            "singleton_chunk_frame": int(segment["target_observation_frame"]),
            "conditional_history_policy": GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
            "counterfactual_conditional_history_policy": GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
            "sampled_window_size": int(sampled_window_size),
            "has_condition_latents": condition_latents is not None,
            "condition_source_frame_offset": int(self.data_config.sample_construction.condition_source_frame_offset),
            "condition_latents_source": condition_latents_source,
            "proprio_context_source": proprio_context_source,
            "proprio_context_chunk_count": int(proprio_context_state.shape[0]),
            "proprio_context_frame_count": int(proprio_context_frames.shape[0]),
            "state_source_key": COUNTERFACTUAL_STATE_KEY if proprio_context_source == COUNTERFACTUAL_STATE_KEY else None,
            "state_anchor_frame": int(segment["prefix_state_frame"]),
            "state_anchor_source_frame": int(segment["prefix_state_source_frame"]),
            "state_anchor_frame_in_sample": segment["prefix_state_frame_in_sample"],
            "latent_frame_start": int(latent_start),
            "frame_shift": int(latent_start),
            "start_padding_frames": max(0, int(self.data_config.sample_construction.start_padding_frames)),
            "segment_pre_start_frames": int(segment["pre_start_frames"]),
            "start_padding_mode": "repeat_first_latent" if int(segment["pre_start_frames"]) > 0 else "none",
            "subwindow_latent_start": int(latent_start),
            "subwindow_latent_end": int(latent_start) + int(segment_length),
            "subwindow_action_start": max(0, int(latent_start)) * action_per_frame,
            "subwindow_action_end": max(0, int(latent_start)) * action_per_frame + int(actions.shape[0]),
            "source_action_steps": int(source_actions.shape[0]),
            "transition_action_steps_required": int(transition_action_steps_required),
            "extra_source_action_steps": int(extra_source_action_steps),
            "lingbot_window_action_alignment": {
                "latent_num_frames": total_frames,
                "prefix_actions": action_per_frame,
                "required_action_num": int(actions.shape[0]),
                "leading_zero_action_frames": int(segment["leading_zero_action_frames"]),
                "leading_zero_action_steps": int(segment["leading_zero_action_frames"]) * action_per_frame,
                "leading_zero_action_mask": float(segment["leading_zero_action_mask"]),
                "source_action_steps": int(source_actions.shape[0]),
                "transition_action_steps_required": int(transition_action_steps_required),
                "extra_source_action_steps": int(extra_source_action_steps),
            },
            "valid_action_steps": int(action_mask.float().sum(dim=-1).gt(0).sum().item()),
            "valid_action_values": int(action_mask.float().sum().item()),
            "counterfactual_source_row": {
                key: row.get(key)
                for key in (
                    "sample_id",
                    "context_id",
                    "branch",
                    "branch_family",
                    "branch_strength",
                    "action_delta_l2_mean",
                    "target_vs_gt_rgb_mse",
                )
            },
            **hierarchical_metadata,
        }
        return LatentWAMSample(
            video_latents=video_latents,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            proprio_context_state=proprio_context_state,
            proprio_context_state_mask=proprio_context_state_mask,
            proprio_context_frames=proprio_context_frames,
            proprio_context_frames_mask=proprio_context_frames_mask,
            task_text=None,
            text_context=text_context,
            negative_text_context=text_context.clone() if text_context is not None else None,
            condition_latents=condition_latents,
            metadata=metadata,
        )

    @property
    def _uses_hierarchical_fixed_segment(self) -> bool:
        return self.data_config.sample_construction.mode == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT

    def _build_window_specs(self) -> tuple[_CounterfactualWindowSpec, ...]:
        sample_cfg = self.data_config.sample_construction
        if self._uses_hierarchical_fixed_segment:
            if sample_cfg.tail_padding_policy != TailPaddingPolicy.ZERO_ORDER_HOLD:
                raise ValueError("Counterfactual hierarchical sampling requires zero-order-hold tail padding.")
            if sample_cfg.padded_target_policy != PaddedTargetPolicy.MASK_LOSS:
                raise ValueError("Counterfactual hierarchical sampling requires masked padded targets.")
            if sample_cfg.segment_frames is None:
                raise ValueError("Counterfactual hierarchical sampling requires `sample_construction.segment_frames`.")

        specs: list[_CounterfactualWindowSpec] = []
        for transition_index, row in enumerate(self.transition_rows):
            target_frames = _latent_frame_count_from_row_or_payload(
                self._resolve_encoded_path(row, "target_latent_path"),
                row=row,
                shape_key="target_video_latent_shape",
                payload_key="target_video_latents",
            )
            source_frames = int(target_frames)
            # Target-only CF training must keep target latent 0 in the sample:
            # that frame is the rollout-style t0 observation. Do not draw
            # shifted subwindows that crop t0 out of the model-visible sequence.
            start_min = 0
            start_max = 0
            eligible_start_count = max(0, start_max - start_min + 1)
            if eligible_start_count <= 0:
                continue
            task_key = str(row.get("task_text") or f"task:{int(row.get('task_id', -1))}")
            specs.append(
                _CounterfactualWindowSpec(
                    transition_index=int(transition_index),
                    task_key=task_key,
                    start_min=int(start_min),
                    start_max=int(start_max),
                    eligible_start_count=int(eligible_start_count),
                    mass_within_task=float(eligible_start_count) ** float(sample_cfg.trajectory_start_power),
                    source_latent_frames=int(source_frames),
                    context_frames=0,
                )
            )
        return tuple(specs)

    def _build_task_specs(self) -> tuple[_CounterfactualTaskSpec, ...]:
        sample_cfg = self.data_config.sample_construction
        by_task: dict[str, list[_CounterfactualWindowSpec]] = {}
        eligible_by_task: Counter[str] = Counter()
        demos_by_task: dict[str, set[int]] = {}
        for spec in self._window_specs:
            by_task.setdefault(spec.task_key, []).append(spec)
            eligible_by_task[spec.task_key] += int(spec.eligible_start_count)
            episode_index = int(self.transition_rows[spec.transition_index].get("dataset_episode_index", spec.transition_index))
            demos_by_task.setdefault(spec.task_key, set()).add(episode_index)
        task_specs: list[_CounterfactualTaskSpec] = []
        for task_key in sorted(by_task):
            windows = tuple(by_task[task_key])
            eligible_start_count = int(eligible_by_task[task_key])
            demo_count = max(1, len(demos_by_task.get(task_key, ())))
            task_mass = (
                float(eligible_start_count) ** float(sample_cfg.task_start_power)
            ) * (float(demo_count) ** float(sample_cfg.demo_count_power))
            if task_mass <= 0.0:
                task_mass = 1.0
            window_mass_total = float(sum(window.mass_within_task for window in windows))
            if window_mass_total <= 0.0:
                windows = tuple(
                    replace(window, mass_within_task=1.0)
                    for window in windows
                )
                window_mass_total = float(len(windows))
            task_specs.append(
                _CounterfactualTaskSpec(
                    task_key=task_key,
                    eligible_start_count=eligible_start_count,
                    demo_count=demo_count,
                    task_mass=float(task_mass),
                    windows=windows,
                    window_mass_total=window_mass_total,
                )
            )
        return tuple(task_specs)

    def _draw_hierarchical_sample(
        self,
        index: int,
    ) -> tuple[_CounterfactualTaskSpec, _CounterfactualWindowSpec, int]:
        split_salt = 17 if self.split == DataSplit.TRAIN.value else 53
        rng = random.Random(_stable_int_seed(int(self.data_config.split_seed), split_salt, int(index)))
        task_index = _weighted_choice_index(self._task_weights, rng)
        task_spec = self._task_specs[task_index]
        window_weights = tuple(float(window.mass_within_task) for window in task_spec.windows)
        window_index = _weighted_choice_index(window_weights, rng)
        window_spec = task_spec.windows[window_index]
        latent_start = int(rng.randint(window_spec.start_min, window_spec.start_max))
        return task_spec, window_spec, latent_start

    def _hierarchical_sample_metadata(
        self,
        *,
        index: int,
        task_spec: _CounterfactualTaskSpec,
        window_spec: _CounterfactualWindowSpec,
    ) -> dict[str, Any]:
        sample_cfg = self.data_config.sample_construction
        task_probability = float(task_spec.task_mass) / max(1e-12, self._task_mass_total)
        trajectory_probability = float(window_spec.mass_within_task) / max(1e-12, task_spec.window_mass_total)
        return {
            "hierarchical_global_sample_index": int(index),
            "hierarchical_task_text": task_spec.task_key,
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
            "tail_padding_policy": str(sample_cfg.tail_padding_policy),
            "padded_target_policy": str(sample_cfg.padded_target_policy),
        }

    def _resolve_encoded_path(self, row: dict[str, Any], key: str) -> Path:
        relative = Path(str(row[key]))
        direct = self.encoded_root / relative
        if direct.exists():
            return direct
        shard = row.get("shard")
        if shard is not None:
            sharded = self.encoded_root / str(shard) / relative
            if sharded.exists():
                return sharded
        raise FileNotFoundError(f"Missing encoded counterfactual artifact for {key}: {relative}")

    def _resolve_raw_path(self, row: dict[str, Any], key: str) -> Path:
        relative = Path(str(row[key]))
        direct = self.raw_root / relative
        if direct.exists():
            return direct
        shard = row.get("shard")
        if shard is not None:
            sharded = self.raw_root / str(shard) / relative
            if sharded.exists():
                return sharded
        raise FileNotFoundError(f"Missing raw counterfactual artifact for {key}: {relative}")


class GeneralistDynamicsMixtureDataset(Dataset[LatentWAMSample]):
    """Sample-level mixture for the opt-in generalist dynamics paradigm."""

    def __init__(
        self,
        *,
        real_dataset: Dataset[LatentWAMSample],
        counterfactual_dataset: Dataset[LatentWAMSample],
        mixture_config: GeneralistDynamicsMixtureConfig,
        split: str,
    ) -> None:
        if len(real_dataset) <= 0:
            raise ValueError("Generalist dynamics mixture requires a non-empty real-demo dataset.")
        if len(counterfactual_dataset) <= 0:
            raise ValueError("Generalist dynamics mixture requires a non-empty counterfactual dataset.")
        self.real_dataset = real_dataset
        self.counterfactual_dataset = counterfactual_dataset
        self.mixture_config = mixture_config
        self.split = str(split)
        self.buckets = _build_mixture_buckets(mixture_config)
        self._distributed_draw_group_size = 1
        self._distributed_epoch_size = 0
        base_length = max(len(real_dataset), len(counterfactual_dataset))
        self._length = max(1, int(round(base_length * float(mixture_config.length_multiplier))))

    def __len__(self) -> int:
        return self._length

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int]:
        return GeneralistDynamicsMixtureTrainSampler(self, world_size=world_size, rank=rank)

    def set_distributed_draw_group_size(self, world_size: int) -> None:
        self.set_distributed_draw_geometry(world_size=world_size)

    def set_distributed_draw_geometry(self, *, world_size: int, epoch_size: int | None = None) -> None:
        group_size = max(1, int(world_size))
        self._distributed_draw_group_size = group_size
        if epoch_size is None:
            epoch_size = int(math.ceil(len(self) / float(group_size))) * group_size
        self._distributed_epoch_size = max(len(self), int(epoch_size))

    def build_source_view(
        self,
        *,
        source: str,
        mode: str,
        bucket_name: str,
        drop_text: bool,
        spread_indices: bool = False,
    ) -> Dataset[LatentWAMSample]:
        bucket = GeneralistMixtureBucket(
            name=str(bucket_name),
            source=str(source),
            mode=str(mode),
            weight=1.0,
            drop_text=bool(drop_text),
        )
        return GeneralistDynamicsSourceViewDataset(
            self,
            bucket=bucket,
            spread_indices=spread_indices,
        )

    def __getitem__(self, index: int) -> LatentWAMSample:
        index = int(index)
        group_size = max(1, int(getattr(self, "_distributed_draw_group_size", 1)))
        epoch_size = int(getattr(self, "_distributed_epoch_size", 0) or len(self))
        epoch = index // max(1, epoch_size)
        epoch_index = index % max(1, epoch_size)
        if group_size == 1:
            rng = random.Random(int(self.mixture_config.seed) + index * 1_000_003)
            bucket = _sample_bucket(self.buckets, rng)
            source_rng = rng
        else:
            # FSDP requires every rank to enter the same sharded module path in
            # the same order. Coordinate the source/mode bucket per distributed
            # step, then vary the source-row draw by rank for data diversity.
            draw_group = index // group_size
            rank_offset = epoch_index % group_size
            bucket_rng = random.Random(int(self.mixture_config.seed) + draw_group * 1_000_003)
            bucket = _sample_bucket(self.buckets, bucket_rng)
            source_rng = random.Random(
                int(self.mixture_config.seed)
                + draw_group * 1_000_003
                + (rank_offset + 1) * 9176
            )
        if bucket.source == REAL_DEMO_SOURCE:
            sample_index = _draw_source_index(self.real_dataset, rng=source_rng, epoch=epoch)
            sample = self.real_dataset[sample_index]
        elif bucket.source == COUNTERFACTUAL_DYNAMICS_SOURCE:
            sample_index = _draw_source_index(self.counterfactual_dataset, rng=source_rng, epoch=epoch)
            sample = self.counterfactual_dataset[sample_index]
        else:
            raise ValueError(f"Unsupported generalist source bucket {bucket.source!r}.")
        if _uses_real_target_only_conditional_layout(bucket):
            sample = _project_real_conditional_sample_to_target_only(sample)
        return _with_generalist_metadata(
            sample,
            bucket=bucket,
            split=self.split,
            source_index=sample_index,
        )


class GeneralistDynamicsSourceViewDataset(Dataset[LatentWAMSample]):
    """Deterministic source projection that preserves mixture sample transforms."""

    def __init__(
        self,
        mixture_dataset: GeneralistDynamicsMixtureDataset,
        *,
        bucket: GeneralistMixtureBucket,
        spread_indices: bool = False,
    ) -> None:
        self.mixture_dataset = mixture_dataset
        self.bucket = bucket
        self.spread_indices = bool(spread_indices)
        if bucket.source == REAL_DEMO_SOURCE:
            self.source_dataset = mixture_dataset.real_dataset
        elif bucket.source == COUNTERFACTUAL_DYNAMICS_SOURCE:
            self.source_dataset = mixture_dataset.counterfactual_dataset
        else:
            raise ValueError(f"Unsupported generalist source view {bucket.source!r}.")
        self._spread_source_indices = _balanced_source_indices_for_dataset(self.source_dataset) if self.spread_indices else None
        self._uses_balanced_source_indices = (
            self._spread_source_indices is not None and len(self._spread_source_indices) > 0
        )
        self._source_spread_stride = _source_view_spread_stride(len(self.source_dataset))

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> LatentWAMSample:
        source_index = int(index)
        if self._uses_balanced_source_indices:
            source_index = int(self._spread_source_indices[source_index % len(self._spread_source_indices)])
        elif self.spread_indices and len(self.source_dataset) > 1:
            source_index = (source_index * self._source_spread_stride) % len(self.source_dataset)
        sample = self.source_dataset[source_index]
        if _uses_real_target_only_conditional_layout(self.bucket):
            sample = _project_real_conditional_sample_to_target_only(sample)
        sample = _with_generalist_metadata(
            sample,
            bucket=self.bucket,
            split=self.mixture_dataset.split,
            source_index=source_index,
        )
        if self.spread_indices:
            metadata = dict(sample.metadata)
            metadata["generalist_source_view_index"] = int(index)
            metadata["generalist_source_view_order"] = "balanced" if self._uses_balanced_source_indices else "stride"
            metadata["generalist_source_view_stride"] = (
                1 if self._uses_balanced_source_indices else int(self._source_spread_stride)
            )
            sample = replace(sample, metadata=metadata)
        return sample


class GeneralistDynamicsMixtureTrainSampler(PaddedEpochOffsetDistributedSampler):
    """Epoch-offset sampler for mixed real/counterfactual dynamics draws."""

    def __init__(self, dataset: GeneralistDynamicsMixtureDataset, *, world_size: int = 1, rank: int = 0) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message="Generalist dynamics mixture sampling requires a non-empty dataset.",
        )
        self.dataset.set_distributed_draw_geometry(
            world_size=self.world_size,
            epoch_size=self._total_size,
        )


def build_generalist_dynamics_mixture_datasets(
    *,
    data_config: DataConfig,
    train_dataset: Dataset[LatentWAMSample],
    val_dataset: Dataset[LatentWAMSample],
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    mixture_config = data_config.generalist_dynamics_mixture
    if mixture_config.train_latent_root is None:
        raise ValueError(
            "`generalist_training_paradigm = mixed_dynamics` requires "
            "`data.generalist_dynamics_mixture.train_latent_root`."
        )
    train_counterfactual = EncodedCounterfactualDynamicsLatentDataset(
        data_config,
        mixture_config.train_latent_root,
        split="train",
    )
    val_root = mixture_config.val_latent_root
    if val_root is None:
        if not mixture_config.allow_train_latent_root_for_val:
            raise ValueError(
                "`generalist_training_paradigm = mixed_dynamics` requires "
                "`data.generalist_dynamics_mixture.val_latent_root` for validation. "
                "Set `allow_train_latent_root_for_val: true` only for local debug runs."
            )
        val_root = mixture_config.train_latent_root
    val_counterfactual = EncodedCounterfactualDynamicsLatentDataset(
        data_config,
        val_root,
        split="val",
    )
    return (
        GeneralistDynamicsMixtureDataset(
            real_dataset=train_dataset,
            counterfactual_dataset=train_counterfactual,
            mixture_config=mixture_config,
            split="train",
        ),
        GeneralistDynamicsMixtureDataset(
            real_dataset=val_dataset,
            counterfactual_dataset=val_counterfactual,
            mixture_config=mixture_config,
            split="val",
        ),
    )


def _build_mixture_buckets(config: GeneralistDynamicsMixtureConfig) -> tuple[GeneralistMixtureBucket, ...]:
    buckets = (
        GeneralistMixtureBucket(
            name="real_joint",
            source=REAL_DEMO_SOURCE,
            mode=JOINT_MODE,
            weight=float(config.real_joint_weight),
            drop_text=False,
        ),
        GeneralistMixtureBucket(
            name="real_action_conditioned_video",
            source=REAL_DEMO_SOURCE,
            mode=ACTION_CONDITIONED_VIDEO_MODE,
            weight=float(config.real_action_conditioned_video_weight),
            drop_text=True,
        ),
        GeneralistMixtureBucket(
            name="real_video_conditioned_action",
            source=REAL_DEMO_SOURCE,
            mode=VIDEO_CONDITIONED_ACTION_MODE,
            weight=float(config.real_video_conditioned_action_weight),
            drop_text=True,
        ),
        GeneralistMixtureBucket(
            name="counterfactual_action_conditioned_video",
            source=COUNTERFACTUAL_DYNAMICS_SOURCE,
            mode=ACTION_CONDITIONED_VIDEO_MODE,
            weight=float(config.counterfactual_action_conditioned_video_weight),
            drop_text=True,
        ),
        GeneralistMixtureBucket(
            name="counterfactual_video_conditioned_action",
            source=COUNTERFACTUAL_DYNAMICS_SOURCE,
            mode=VIDEO_CONDITIONED_ACTION_MODE,
            weight=float(config.counterfactual_video_conditioned_action_weight),
            drop_text=True,
        ),
    )
    return tuple(bucket for bucket in buckets if bucket.weight > 0.0)


def _sample_bucket(buckets: tuple[GeneralistMixtureBucket, ...], rng: random.Random) -> GeneralistMixtureBucket:
    total = sum(bucket.weight for bucket in buckets)
    draw = rng.random() * total
    cursor = 0.0
    for bucket in buckets:
        cursor += bucket.weight
        if draw <= cursor:
            return bucket
    return buckets[-1]


def _draw_source_index(dataset: Dataset[LatentWAMSample], *, rng: random.Random, epoch: int) -> int:
    local_index = int(rng.randrange(len(dataset)))
    if _dataset_uses_epoch_offset_draw_keys(dataset):
        return int(epoch) * len(dataset) + local_index
    return local_index


def _dataset_uses_epoch_offset_draw_keys(dataset: Dataset[LatentWAMSample]) -> bool:
    explicit = getattr(dataset, "uses_epoch_offset_draw_keys", None)
    if explicit is not None:
        return bool(explicit)
    sample_construction = getattr(getattr(dataset, "data_config", None), "sample_construction", None)
    return (
        getattr(sample_construction, "mode", None) == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT
        and callable(getattr(dataset, "_draw_hierarchical_sample", None))
    )


def _source_view_spread_stride(length: int) -> int:
    if length <= 1:
        return 1
    stride = max(1, int(length) // 10 + 1)
    while math.gcd(stride, int(length)) != 1:
        stride += 1
        if stride >= int(length):
            return 1
    return stride


def _balanced_counterfactual_source_indices(rows: Sequence[dict[str, Any]]) -> tuple[int, ...]:
    indices_by_task_branch: dict[str, dict[str, list[int]]] = {}
    branch_order: list[str] = []
    seen_branches: set[str] = set()
    for index, row in enumerate(rows):
        task_key = _counterfactual_source_task_key(row)
        branch_key = _counterfactual_source_branch_key(row)
        if branch_key not in seen_branches:
            seen_branches.add(branch_key)
            branch_order.append(branch_key)
        task_branches = indices_by_task_branch.setdefault(task_key, {})
        task_branches.setdefault(branch_key, []).append(int(index))
    if not indices_by_task_branch:
        return tuple(range(len(rows)))

    ordered_tasks = sorted(indices_by_task_branch, key=_source_view_label_sort_key)
    ordered_branches = tuple(branch_order) if branch_order else ("unknown",)
    max_depth = max(
        len(indices)
        for task_branches in indices_by_task_branch.values()
        for indices in task_branches.values()
    )
    order: list[int] = []
    for depth in range(max_depth):
        for branch_offset in range(len(ordered_branches)):
            for task_offset, task_key in enumerate(ordered_tasks):
                branch_key = ordered_branches[(task_offset + branch_offset) % len(ordered_branches)]
                indices = indices_by_task_branch.get(task_key, {}).get(branch_key, ())
                if depth < len(indices):
                    order.append(int(indices[depth]))
    if len(order) < len(rows):
        seen = set(order)
        order.extend(index for index in range(len(rows)) if index not in seen)
    return tuple(order)


def _counterfactual_source_task_key(row: dict[str, Any]) -> str:
    for key in ("task_id", "task_key", "task_name"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _counterfactual_source_branch_key(row: dict[str, Any]) -> str:
    for key in ("branch", "counterfactual_branch", "branch_family"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _source_view_label_sort_key(label: str) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, str(label))


def _balanced_source_indices_for_dataset(dataset: Dataset[LatentWAMSample]) -> tuple[int, ...] | None:
    build_indices = getattr(dataset, "build_balanced_source_indices", None)
    if not callable(build_indices):
        return None
    indices = tuple(int(index) for index in build_indices())
    if not indices:
        return None
    return indices


def _with_generalist_metadata(
    sample: LatentWAMSample,
    *,
    bucket: GeneralistMixtureBucket,
    split: str,
    source_index: int,
) -> LatentWAMSample:
    metadata = dict(sample.metadata)
    metadata.update(
        {
            GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY: bucket.mode,
            GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY: bool(bucket.drop_text),
            GENERALIST_TRAINING_SOURCE_METADATA_KEY: bucket.source,
            GENERALIST_TRAINING_BUCKET_METADATA_KEY: bucket.name,
            "generalist_training_split": split,
            "generalist_source_index": int(source_index),
        }
    )
    text_context = sample.text_context
    task_text = sample.task_text
    if bucket.drop_text:
        task_text = None
        if sample.negative_text_context is not None:
            text_context = sample.negative_text_context.clone()
        elif text_context is not None:
            text_context = torch.zeros_like(text_context)
    return replace(
        sample,
        task_text=task_text,
        text_context=text_context,
        metadata=metadata,
    )


def _uses_real_target_only_conditional_layout(bucket: GeneralistMixtureBucket) -> bool:
    return bucket.source == REAL_DEMO_SOURCE and bucket.mode in {
        ACTION_CONDITIONED_VIDEO_MODE,
        VIDEO_CONDITIONED_ACTION_MODE,
    }


def _project_real_conditional_sample_to_target_only(sample: LatentWAMSample) -> LatentWAMSample:
    """Match real conditional FDM/IDM layout to the counterfactual target-only contract."""

    total_frames = int(sample.video_latents.shape[1])
    if total_frames < 2:
        raise ValueError(
            "Real conditional GJD target-only projection requires at least two latent frames, "
            f"got {total_frames}."
        )
    if sample.actions.shape[0] % total_frames != 0:
        raise ValueError(
            "Real conditional GJD target-only projection requires frame-aligned actions, "
            f"actions={sample.actions.shape[0]}, latent_frames={total_frames}."
        )

    boundary, boundary_source = _real_conditional_target_boundary(sample.metadata)
    if boundary is None:
        source_start = 0
        boundary_source = "default_first_frame"
    else:
        boundary = int(boundary)
        if boundary <= 0:
            source_start = 0
        elif boundary >= total_frames:
            raise ValueError(
                "Real conditional GJD target-only projection requires at least one future frame after "
                f"the target boundary, got boundary={boundary}, latent_frames={total_frames}."
            )
        else:
            source_start = boundary - 1

    target_frames = int(total_frames - source_start)
    action_steps_per_frame = int(sample.actions.shape[0] // total_frames)
    video_latents = sample.video_latents[:, source_start:].contiguous()
    actions, action_mask = _target_only_shifted_actions(
        sample.actions,
        sample.action_mask,
        source_start_frame=source_start,
        target_frames=target_frames,
        action_steps_per_frame=action_steps_per_frame,
    )
    condition_latents = _trim_optional_video(
        sample.condition_latents,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    canonical_video = _trim_optional_video(
        sample.canonical_video,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    proprio_context_state = _trim_optional_frame_tensor(
        sample.proprio_context_state,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    proprio_context_state_mask = _trim_optional_frame_tensor(
        sample.proprio_context_state_mask,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    proprio_context_frames = _trim_optional_frame_tensor(
        sample.proprio_context_frames,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    proprio_context_frames_mask = _trim_optional_frame_tensor(
        sample.proprio_context_frames_mask,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    state, state_mask, state_anchor_source_frame = _target_only_prefix_state(
        sample,
        source_start_frame=source_start,
    )
    metadata = _target_only_conditional_metadata(
        sample.metadata,
        source_start_frame=source_start,
        target_frames=target_frames,
        action_steps_per_frame=action_steps_per_frame,
        actions=actions,
        action_mask=action_mask,
        boundary_source=boundary_source,
        state_anchor_source_frame=state_anchor_source_frame,
    )
    return replace(
        sample,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        state=state,
        state_mask=state_mask,
        canonical_video=canonical_video,
        condition_latents=condition_latents,
        proprio_context_state=proprio_context_state,
        proprio_context_state_mask=proprio_context_state_mask,
        proprio_context_frames=proprio_context_frames,
        proprio_context_frames_mask=proprio_context_frames_mask,
        metadata=metadata,
    )


def _target_only_shifted_actions(
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    source_start_frame: int,
    target_frames: int,
    action_steps_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected_actions = torch.zeros(
        int(target_frames) * int(action_steps_per_frame),
        int(actions.shape[-1]),
        dtype=actions.dtype,
        device=actions.device,
    )
    projected_mask = torch.zeros_like(projected_actions, dtype=torch.float32)
    projected_mask[:action_steps_per_frame] = 0.0
    source_mask = (
        action_mask.to(device=actions.device, dtype=torch.float32)
        if action_mask is not None
        else torch.ones_like(actions, dtype=torch.float32)
    )
    for target_frame in range(1, int(target_frames)):
        source_frame = int(source_start_frame) + int(target_frame) - 1
        src_start = source_frame * int(action_steps_per_frame)
        src_end = src_start + int(action_steps_per_frame)
        dst_start = int(target_frame) * int(action_steps_per_frame)
        dst_end = dst_start + int(action_steps_per_frame)
        if src_end > int(actions.shape[0]):
            continue
        projected_actions[dst_start:dst_end] = actions[src_start:src_end]
        projected_mask[dst_start:dst_end] = source_mask[src_start:src_end]
    return projected_actions.contiguous(), projected_mask.contiguous()


def _target_only_prefix_state(
    sample: LatentWAMSample,
    *,
    source_start_frame: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    state_anchor_source_frame = max(0, int(source_start_frame) - 1)
    if sample.condition_latents is None:
        state_anchor_source_frame = max(0, int(source_start_frame))
    if sample.proprio_context_frames is None:
        return sample.state, sample.state_mask, state_anchor_source_frame
    frames = sample.proprio_context_frames
    if frames.ndim != 2:
        return sample.state, sample.state_mask, state_anchor_source_frame
    mask = sample.proprio_context_frames_mask
    if mask is None:
        mask = torch.ones_like(frames, dtype=torch.float32)
    if mask.shape != frames.shape:
        return sample.state, sample.state_mask, state_anchor_source_frame
    state_horizon = int(sample.state.shape[0]) if sample.state is not None and sample.state.ndim == 2 else 1
    anchor = max(0, min(int(state_anchor_source_frame), int(frames.shape[0]) - 1))
    start = max(0, anchor - max(1, state_horizon) + 1)
    state = frames[start : anchor + 1].contiguous()
    state_mask = mask[start : anchor + 1].to(device=frames.device, dtype=torch.float32).contiguous()
    if int(state.shape[0]) < state_horizon:
        pad_count = state_horizon - int(state.shape[0])
        state = torch.cat([state[:1].expand(pad_count, -1), state], dim=0).contiguous()
        state_mask = torch.cat([state_mask[:1].expand(pad_count, -1), state_mask], dim=0).contiguous()
    return state, state_mask, anchor


def _target_only_conditional_metadata(
    metadata: dict[str, Any],
    *,
    source_start_frame: int,
    target_frames: int,
    action_steps_per_frame: int,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    boundary_source: str,
    state_anchor_source_frame: int,
) -> dict[str, Any]:
    updated = dict(metadata)
    original_observed = _metadata_sequence(updated.get("observation_frame_indices")) or _metadata_sequence(
        updated.get("observed_frame_ids")
    )
    observed_frame_ids = None
    if original_observed is not None and len(original_observed) >= int(source_start_frame) + int(target_frames):
        observed_frame_ids = original_observed[int(source_start_frame) : int(source_start_frame) + int(target_frames)]
        if "observation_frame_indices" in updated:
            updated["observation_frame_indices"] = list(observed_frame_ids)
        if "observed_frame_ids" in updated:
            updated["observed_frame_ids"] = list(observed_frame_ids)

    old_frame_start = _metadata_frame_boundary(
        metadata,
        ("sample_start_frame", "observation_start", "window_start_frame", "frame_shift", "latent_frame_start"),
    )
    if observed_frame_ids:
        sample_start_frame = int(observed_frame_ids[0])
        first_future_frame = int(observed_frame_ids[1]) if int(target_frames) > 1 else sample_start_frame
        sample_end_frame = int(observed_frame_ids[-1]) + int(action_steps_per_frame)
    else:
        sample_start_frame = int(old_frame_start or 0) + int(source_start_frame)
        first_future_frame = sample_start_frame + int(action_steps_per_frame)
        sample_end_frame = sample_start_frame + int(target_frames) * int(action_steps_per_frame)

    valid_action_steps, valid_action_values = _action_validity_stats(actions=actions, action_mask=action_mask)
    old_valid_frames = metadata.get("segment_valid_latent_frames")
    if old_valid_frames is None:
        valid_latent_frames = int(target_frames)
    else:
        valid_latent_frames = max(0, min(int(target_frames), int(old_valid_frames) - int(source_start_frame)))
    padded_latent_frames = max(0, int(target_frames) - int(valid_latent_frames))

    updated.update(
        {
            "generalist_conditional_contract": GENERALIST_CONDITIONAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE,
            "generalist_conditional_training_sequence": "target_only",
            "generalist_conditional_context_used_for_training": False,
            "generalist_conditional_boundary_source": str(boundary_source),
            "generalist_conditional_source_t0_frame_in_sample": int(source_start_frame),
            "generalist_conditional_source_future_start_frame_in_sample": int(source_start_frame) + 1,
            "generalist_gjd_chunk_contract": GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON,
            "history_frames": 1,
            "loss_frame_start": 1,
            "loss_frame_end": int(target_frames),
            "latent_loss_frame_start": 1,
            "latent_loss_frame_end": int(target_frames),
            "action_loss_frame_start": 1,
            "action_loss_frame_end": int(target_frames),
            "current_start_frame_in_sample": 1,
            "current_end_frame_in_sample": int(target_frames),
            "supervised_start": 1,
            "supervised_end": int(target_frames),
            "chunk_origin_frame": 1,
            "target_observation_frame_in_sample": 0,
            "target_observation_frame_index": sample_start_frame,
            "conditional_history_policy": GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
            "generalist_conditional_history_policy": GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
            "first_supervised_future_frame_in_sample": 1,
            "first_supervised_future_frame_index": first_future_frame,
            "supervised_future_latent_frames": max(0, int(target_frames) - 1),
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "observation_start": sample_start_frame,
            "window_start_frame": sample_start_frame,
            "window_end_frame": sample_end_frame,
            "anchor_frame_index": sample_start_frame,
            "target_frame_start": first_future_frame,
            "target_frame_end": sample_end_frame,
            "segment_length_frames": int(target_frames),
            "segment_valid_latent_frames": int(valid_latent_frames),
            "segment_padded_latent_frames": int(padded_latent_frames),
            "tail_padding_mode": "none" if padded_latent_frames == 0 else "zero_order_hold",
            "segment_pre_start_frames": 0,
            "start_padding_mode": "none",
            "context_prefix_frames_in_sample": 1,
            "context_prefix_real_frames": 1,
            "context_prefix_truncated_frames": int(max(0, int(metadata.get("context_prefix_frames_requested", 0) or 0) - 1)),
            "singleton_chunk_frame": 0,
            "state_anchor_source_frame": int(state_anchor_source_frame),
            "state_anchor_frame_in_sample": None if int(state_anchor_source_frame) < int(source_start_frame) else 0,
            "valid_action_steps": int(valid_action_steps),
            "valid_action_values": int(valid_action_values),
        }
    )
    for key in ("latent_frame_start", "frame_shift", "effective_start", "effective_frame_start", "logical_frame_start"):
        if key in metadata and metadata[key] is not None:
            updated[key] = int(metadata[key]) + int(source_start_frame)
    for start_key, end_key in (("effective_start", "effective_end"), ("effective_frame_start", "effective_frame_end")):
        if start_key in updated and updated[start_key] is not None:
            updated[end_key] = int(updated[start_key]) + int(target_frames)
    if "logical_frame_start" in updated and updated["logical_frame_start"] is not None:
        updated["logical_frame_end"] = int(updated["logical_frame_start"]) + int(target_frames)
    for key in ("subwindow_latent_start", "virtual_latent_start"):
        updated[key] = sample_start_frame
    updated["subwindow_latent_end"] = sample_end_frame
    original_action_start = metadata.get("subwindow_action_start")
    updated["subwindow_action_start"] = (
        int(original_action_start)
        if original_action_start is not None
        else 0
    ) + int(source_start_frame) * int(action_steps_per_frame)
    updated["subwindow_action_end"] = int(updated["subwindow_action_start"]) + int(actions.shape[0])

    alignment = updated.get("lingbot_window_action_alignment")
    if not isinstance(alignment, dict):
        alignment = {}
    else:
        alignment = dict(alignment)
    alignment.update(
        {
            "latent_num_frames": int(target_frames),
            "prefix_actions": int(action_steps_per_frame),
            "required_action_num": int(actions.shape[0]),
            "leading_zero_action_frames": 1,
            "leading_zero_action_steps": int(action_steps_per_frame),
            "leading_zero_action_mask": 0.0,
        }
    )
    updated["lingbot_window_action_alignment"] = alignment
    return updated


def _metadata_frame_boundary(metadata: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return int(value)
    return None


def _real_conditional_target_boundary(metadata: dict[str, Any]) -> tuple[int | None, str]:
    """Resolve the sampled current/history boundary for real-demo FDM/IDM."""

    for key in ("current_start_frame_in_sample", "history_frames"):
        value = metadata.get(key)
        if value is not None and int(value) > 0:
            return int(value), key
    for key in ("loss_frame_start", "latent_loss_frame_start", "action_loss_frame_start"):
        value = metadata.get(key)
        if value is not None and int(value) > 0:
            return int(value), key
    return None, "default_first_frame"


def _trim_optional_video(
    canonical_video: torch.Tensor | None,
    *,
    crop_frames: int,
    total_frames: int,
) -> torch.Tensor | None:
    if canonical_video is None:
        return None
    if canonical_video.ndim >= 1 and int(canonical_video.shape[0]) == total_frames:
        return canonical_video[crop_frames:].contiguous()
    if canonical_video.ndim >= 2 and int(canonical_video.shape[1]) == total_frames:
        return canonical_video[:, crop_frames:].contiguous()
    return canonical_video


def _trim_optional_frame_tensor(
    tensor: torch.Tensor | None,
    *,
    crop_frames: int,
    total_frames: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim >= 1 and int(tensor.shape[0]) == total_frames:
        return tensor[crop_frames:].contiguous()
    if tensor.ndim >= 2 and int(tensor.shape[1]) == total_frames:
        return tensor[:, crop_frames:].contiguous()
    return tensor


def _metadata_sequence(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return None


def _action_validity_stats(
    *,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> tuple[int, int]:
    if action_mask is None:
        return int(actions.shape[0]), int(actions.numel())
    reduced = action_mask.float().sum(dim=-1)
    return int((reduced > 0).sum().item()), int(action_mask.float().sum().item())


def _context_key(row: dict[str, Any]) -> tuple[str | None, int]:
    shard = row.get("shard")
    return (None if shard is None else str(shard), int(row["context_id"]))


def _load_latent_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected latent payload dict at {path}, got {type(payload).__name__}.")
    return payload


def _payload_latents(payload: dict[str, Any], *, key: str) -> torch.Tensor:
    if key not in payload:
        raise ValueError(f"Expected key {key!r} in latent payload.")
    tensor = payload[key]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError(f"Expected {key!r} tensor [C,T,H,W], got {type(tensor)!r}.")
    return tensor.to(dtype=torch.float32).contiguous()


def _counterfactual_latent_state_frames(
    payload: np.lib.npyio.NpzFile,
    *,
    latent_frames: int,
    state_dim: int,
    data_config: DataConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_frames = int(latent_frames)
    state_dim = int(state_dim)
    if state_dim <= 0:
        empty = torch.zeros(latent_frames, 0, dtype=torch.float32)
        return empty, empty.clone()
    if COUNTERFACTUAL_STATE_KEY not in payload.files:
        state = torch.zeros(latent_frames, state_dim, dtype=torch.float32)
        return state, torch.zeros_like(state)
    raw_state = np.asarray(payload[COUNTERFACTUAL_STATE_KEY], dtype=np.float32)
    if raw_state.ndim != 2:
        raise ValueError(f"Expected {COUNTERFACTUAL_STATE_KEY!r} with shape [T,D], got {raw_state.shape}.")
    if raw_state.shape[0] <= 0:
        state = torch.zeros(latent_frames, state_dim, dtype=torch.float32)
        return state, torch.zeros_like(state)
    anchors = latent_anchor_positions(
        raw_frame_count=int(raw_state.shape[0]),
        latent_num_frames=latent_frames,
        layout=data_config.latent_temporal_layout,
    )
    selected = raw_state[np.asarray(anchors, dtype=np.int64)]
    state = _pack_state(selected, target_dim=state_dim)
    return state, torch.ones_like(state)


def _pack_state(state: np.ndarray, *, target_dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(state, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected state array [T,D], got {tuple(tensor.shape)}.")
    if tensor.shape[1] > target_dim:
        raise ValueError(
            f"Counterfactual state dim {tensor.shape[1]} exceeds configured state_dim={target_dim}."
        )
    if tensor.shape[1] == target_dim:
        return tensor.contiguous()
    padded = torch.zeros(tensor.shape[0], target_dim, dtype=torch.float32)
    padded[:, : tensor.shape[1]] = tensor
    return padded


def _pack_actions(actions: np.ndarray, *, target_dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(actions, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected action array [T,D], got {tuple(tensor.shape)}.")
    if tensor.shape[1] > target_dim:
        raise ValueError(
            f"Counterfactual action dim {tensor.shape[1]} exceeds configured action_dim={target_dim}."
        )
    if tensor.shape[1] == target_dim:
        return tensor.contiguous()
    padded = torch.zeros(tensor.shape[0], target_dim, dtype=torch.float32)
    padded[:, : tensor.shape[1]] = tensor
    return padded


def _slice_counterfactual_frame_tensor_with_edge_hold(
    tensor: torch.Tensor,
    *,
    latent_start: int,
    segment_length: int,
) -> torch.Tensor:
    source_frames = int(tensor.shape[0])
    if source_frames <= 0:
        raise ValueError("Counterfactual frame tensor requires at least one source frame.")
    source_start = max(0, int(latent_start))
    source_end = min(source_frames, int(latent_start) + int(segment_length))
    if source_end <= source_start:
        source_end = min(source_frames, source_start + 1)
    valid_slice = tensor[source_start:source_end]

    parts: list[torch.Tensor] = []
    if int(latent_start) < 0:
        parts.append(tensor[:1].expand(min(-int(latent_start), segment_length), -1))
    parts.append(valid_slice)
    current_frames = sum(int(part.shape[0]) for part in parts)
    if current_frames < segment_length:
        parts.append(tensor[-1:].expand(segment_length - current_frames, -1))
    return torch.cat(parts, dim=0)[:segment_length].contiguous()


def _counterfactual_state_history_from_frames(
    *,
    proprio_context_frames: torch.Tensor,
    proprio_context_frames_mask: torch.Tensor,
    anchor_frame: int,
    state_horizon: int,
    state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_horizon = int(state_horizon)
    state_dim = int(state_dim)
    if state_horizon <= 0 or state_dim <= 0:
        empty = torch.zeros(max(0, state_horizon), max(0, state_dim), dtype=torch.float32)
        return empty, empty.clone()
    if proprio_context_frames.shape[0] <= 0:
        state = torch.zeros(state_horizon, state_dim, dtype=torch.float32)
        return state, torch.zeros_like(state)
    anchor = max(0, min(int(anchor_frame), int(proprio_context_frames.shape[0]) - 1))
    start = max(0, anchor - state_horizon + 1)
    state = proprio_context_frames[start : anchor + 1]
    mask = proprio_context_frames_mask[start : anchor + 1]
    if state.shape[0] < state_horizon:
        pad_count = state_horizon - int(state.shape[0])
        state = torch.cat([state[:1].expand(pad_count, -1), state], dim=0)
        mask = torch.cat([mask[:1].expand(pad_count, -1), mask], dim=0)
    return state[-state_horizon:].contiguous(), mask[-state_horizon:].contiguous()


def _counterfactual_action_steps_per_frame(
    actions: torch.Tensor,
    *,
    total_frames: int,
    data_config: DataConfig,
) -> int:
    if total_frames <= 0:
        raise ValueError("Counterfactual sample must contain at least one latent frame.")
    configured = _configured_action_steps_per_latent_frame(data_config)
    if configured is not None:
        min_required = max(0, int(total_frames) - 1) * int(configured)
        max_legacy = int(total_frames) * int(configured)
        action_steps = int(actions.shape[0])
        if min_required <= action_steps <= max_legacy:
            return int(configured)

    transition_frames = max(1, int(total_frames) - 1)
    if actions.shape[0] % transition_frames == 0:
        action_per_frame = int(actions.shape[0] // transition_frames)
    elif actions.shape[0] % total_frames == 0:
        action_per_frame = int(actions.shape[0] // total_frames)
    else:
        raise ValueError(
            "Counterfactual action count must be transition- or legacy frame-aligned, "
            f"got actions={actions.shape[0]}, latent_frames={total_frames}."
        )
    if action_per_frame <= 0:
        raise ValueError("Counterfactual sample must contain at least one action per latent frame.")
    return action_per_frame


def _configured_action_steps_per_latent_frame(data_config: DataConfig) -> int | None:
    num_frames = int(getattr(data_config, "num_frames", 0) or 0)
    action_horizon = int(getattr(data_config.action_schema, "action_horizon", 0) or 0)
    if num_frames <= 0 or action_horizon <= 0:
        return None
    if action_horizon % num_frames != 0:
        return None
    action_per_frame = int(action_horizon // num_frames)
    return action_per_frame if action_per_frame > 0 else None


def _counterfactual_condition_latents_from_source(
    video_latents: torch.Tensor,
    *,
    source_frame_offset: int,
) -> torch.Tensor | None:
    if int(source_frame_offset) == 0:
        return None
    source_frames = int(video_latents.shape[1])
    if source_frames <= 0:
        raise ValueError("Counterfactual condition latent synthesis requires at least one source latent frame.")
    source_indices = torch.arange(source_frames, dtype=torch.long, device=video_latents.device)
    source_indices = (source_indices + int(source_frame_offset)).clamp_(0, source_frames - 1)
    return video_latents.index_select(dim=1, index=source_indices).contiguous()


def _validate_counterfactual_condition_latent_manifest(
    manifest: dict[str, Any],
    *,
    encoded_root: Path,
    source_frame_offset: int,
) -> None:
    if int(source_frame_offset) == 0:
        return
    if manifest.get("condition_latents") is True:
        return
    raise ValueError(
        "Counterfactual encoded latents are missing explicit condition latents. "
        f"{encoded_root} has manifest condition_latents={manifest.get('condition_latents')!r}, "
        f"but this config uses condition_source_frame_offset={int(source_frame_offset)}. "
        "Re-encode with scripts/encode_libero_fdm_counterfactual_dataset.py without "
        "`--skip-condition-latents`."
    )


def _counterfactual_target_only_condition_latents_from_payload(
    *,
    target_payload: dict[str, Any],
    fallback_video_latents: torch.Tensor,
    source_frame_offset: int,
) -> tuple[torch.Tensor | None, str]:
    if int(source_frame_offset) == 0:
        return None, "disabled_zero_offset"
    target_condition = _optional_counterfactual_condition_latents(
        target_payload,
        key="target_condition_video_latents",
        source_frame_offset=source_frame_offset,
    )
    if target_condition is not None:
        if tuple(target_condition.shape) != tuple(fallback_video_latents.shape):
            raise ValueError(
                "Encoded target-only counterfactual condition_latents must match target video latents, "
                f"got condition={tuple(target_condition.shape)}, video={tuple(fallback_video_latents.shape)}."
            )
        return target_condition, "encoded_target_single_frame"

    fallback = _counterfactual_condition_latents_from_source(
        fallback_video_latents,
        source_frame_offset=source_frame_offset,
    )
    return fallback, "synthesized_target_shift"


def _optional_counterfactual_condition_latents(
    payload: dict[str, Any],
    *,
    key: str,
    source_frame_offset: int,
) -> torch.Tensor | None:
    if key not in payload:
        return None
    payload_offset = payload.get("condition_source_frame_offset")
    if payload_offset is None or int(payload_offset) != int(source_frame_offset):
        raise ValueError(
            "Encoded counterfactual condition_source_frame_offset mismatch: "
            f"payload={payload_offset!r}, expected={int(source_frame_offset)}."
        )
    payload_policy = payload.get("condition_source_frame_policy")
    if payload_policy != COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY:
        raise ValueError(
            "Encoded counterfactual condition_source_frame_policy mismatch: "
            f"payload={payload_policy!r}, expected={COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY!r}."
        )
    return _payload_latents(payload, key=key)


def _slice_counterfactual_latents_with_edge_hold(
    video_latents: torch.Tensor,
    *,
    latent_start: int,
    segment_length: int,
) -> torch.Tensor:
    source_frames = int(video_latents.shape[1])
    source_start = max(0, int(latent_start))
    source_end = min(source_frames, int(latent_start) + int(segment_length))
    if source_end <= source_start:
        source_end = min(source_frames, source_start + 1)
    valid_slice = video_latents[:, source_start:source_end]

    parts: list[torch.Tensor] = []
    if int(latent_start) < 0:
        parts.append(video_latents[:, :1].expand(-1, min(-int(latent_start), segment_length), -1, -1))
    parts.append(valid_slice)
    current_frames = sum(int(part.shape[1]) for part in parts)
    if current_frames < segment_length:
        parts.append(video_latents[:, -1:].expand(-1, segment_length - current_frames, -1, -1))
    return torch.cat(parts, dim=1)[:, :segment_length].contiguous()


def _build_counterfactual_fixed_segment(
    *,
    video_latents: torch.Tensor,
    condition_latents: torch.Tensor | None = None,
    actions: torch.Tensor,
    context_frames: int,
    latent_start: int,
    segment_length: int,
    action_per_frame: int,
    condition_source_frame_offset: int = 0,
    mask_leading_zero_action_context: bool = False,
) -> dict[str, Any]:
    source_frames = int(video_latents.shape[1])
    if source_frames <= 0:
        raise ValueError("Counterfactual segment sampling requires at least one source latent frame.")
    if segment_length <= 0:
        raise ValueError(f"Counterfactual segment_length must be positive, got {segment_length}.")
    if condition_latents is not None and tuple(condition_latents.shape) != tuple(video_latents.shape):
        raise ValueError(
            "Counterfactual condition_latents must match video_latents exactly, "
            f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
        )
    source_start = max(0, int(latent_start))
    source_end = min(source_frames, int(latent_start) + int(segment_length))
    if source_end <= source_start:
        source_end = min(source_frames, source_start + 1)
    pre_start_frames = max(0, min(segment_length, -int(latent_start))) if int(latent_start) < 0 else 0
    valid_latent_frames = max(0, min(segment_length, source_frames - int(latent_start)))
    padded_latent_frames = max(0, segment_length - valid_latent_frames)

    segment_video = _slice_counterfactual_latents_with_edge_hold(
        video_latents,
        latent_start=int(latent_start),
        segment_length=int(segment_length),
    )
    segment_condition = (
        _slice_counterfactual_latents_with_edge_hold(
            condition_latents,
            latent_start=int(latent_start),
            segment_length=int(segment_length),
        )
        if condition_latents is not None
        else None
    )

    segment_actions = torch.zeros(
        segment_length * action_per_frame,
        actions.shape[1],
        dtype=torch.float32,
    )
    action_mask = torch.zeros_like(segment_actions)
    leading_zero_action_frames = int(pre_start_frames) if int(pre_start_frames) > 0 else 1
    leading_zero_action_mask = 0.0 if int(pre_start_frames) > 0 or mask_leading_zero_action_context else 1.0
    for output_frame in range(segment_length):
        dst_start = output_frame * action_per_frame
        dst_end = dst_start + action_per_frame
        if output_frame < leading_zero_action_frames:
            action_mask[dst_start:dst_end] = float(leading_zero_action_mask)
            continue
        source_frame = source_start + output_frame - leading_zero_action_frames
        if source_frame < 0 or source_frame >= source_frames:
            continue
        src_start = source_frame * action_per_frame
        src_end = src_start + action_per_frame
        if src_end > int(actions.shape[0]):
            continue
        segment_actions[dst_start:dst_end] = actions[src_start:src_end]
        action_mask[dst_start:dst_end] = 1.0

    target_observation_frame = int(context_frames) - int(latent_start)
    first_supervised_future_frame = int(target_observation_frame) + 1
    loss_frame_start = max(0, int(pre_start_frames), int(first_supervised_future_frame))
    loss_frame_end = min(int(segment_length), int(valid_latent_frames))
    if loss_frame_end < loss_frame_start:
        loss_frame_end = loss_frame_start
    prefix_state_source_frame = int(source_start)
    if condition_latents is not None:
        prefix_state_source_frame = int(source_start) + int(condition_source_frame_offset)
    prefix_state_source_frame = max(0, min(source_frames - 1, int(prefix_state_source_frame)))
    prefix_state_frame_in_sample: int | None = None
    if int(source_start) <= prefix_state_source_frame < int(source_end):
        prefix_state_frame_in_sample = int(prefix_state_source_frame) - int(source_start) + int(pre_start_frames)
    prefix_state_frame = (
        int(prefix_state_frame_in_sample)
        if prefix_state_frame_in_sample is not None
        else max(0, min(segment_length - 1, int(pre_start_frames)))
    )
    return {
        "video_latents": segment_video,
        "condition_latents": segment_condition,
        "actions": segment_actions.contiguous(),
        "action_mask": action_mask.contiguous(),
        "pre_start_frames": int(pre_start_frames),
        "valid_latent_frames": int(valid_latent_frames),
        "padded_latent_frames": int(padded_latent_frames),
        "valid_source_frames": max(0, int(source_end) - int(source_start)),
        "loss_frame_start": int(loss_frame_start),
        "loss_frame_end": int(loss_frame_end),
        "chunk_origin_frame": int(loss_frame_start),
        "target_observation_frame": int(target_observation_frame),
        "first_supervised_future_frame": int(first_supervised_future_frame),
        "prefix_state_frame": int(prefix_state_frame),
        "prefix_state_source_frame": int(prefix_state_source_frame),
        "prefix_state_frame_in_sample": prefix_state_frame_in_sample,
        "leading_zero_action_frames": int(leading_zero_action_frames),
        "leading_zero_action_mask": float(leading_zero_action_mask),
    }


def _sample_counterfactual_attention_geometry(
    *,
    data_config: DataConfig,
    segment_length: int,
) -> tuple[int, int]:
    """Mirror real uniform-segment chunk/window randomization for counterfactual rows."""

    sample_cfg = data_config.sample_construction
    if sample_cfg.mode != WindowSamplingMode.UNIFORM_SEGMENT:
        return (
            max(1, int(sample_cfg.chunk_size)),
            max(1, int(sample_cfg.window_size)),
        )

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


def _counterfactual_observed_frame_ids(
    *,
    context_start_frame: int,
    latent_start: int,
    segment_length: int,
    source_frames: int,
    action_per_frame: int,
) -> list[int]:
    ids: list[int] = []
    for offset in range(int(segment_length)):
        source_frame = min(max(0, int(latent_start) + offset), int(source_frames) - 1)
        ids.append((int(context_start_frame) + source_frame) * int(action_per_frame))
    return ids


def _latent_frame_count_from_row_or_payload(
    path: Path,
    *,
    row: dict[str, Any],
    shape_key: str,
    payload_key: str,
) -> int:
    shape = row.get(shape_key)
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return int(shape[1])
    return int(_payload_latents(_load_latent_payload(path), key=payload_key).shape[1])


def _stable_int_seed(*values: int) -> int:
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


def _load_empty_text_embedding(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Expected empty text embedding tensor at {path!r}, got {type(payload)!r}.")
    if payload.ndim == 3 and payload.shape[0] == 1:
        payload = payload.squeeze(0)
    return payload.to(dtype=torch.float32).contiguous()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counterfactual_raw_root_from_manifest(manifest: dict[str, Any], *, encoded_root: Path) -> Path:
    for key in ("dataset_root", "source_dataset_root"):
        value = manifest.get(key)
        if value is not None:
            return Path(str(value)).expanduser().resolve()

    for summary_key in ("source_summary", "source_aggregate_summary"):
        summary = manifest.get(summary_key)
        if not isinstance(summary, dict):
            continue
        for key in ("root", "dataset_root", "output_root"):
            value = summary.get(key)
            if value is not None:
                return Path(str(value)).expanduser().resolve()

    available = ", ".join(sorted(str(key) for key in manifest.keys()))
    raise KeyError(
        "Counterfactual latent manifest must include a raw dataset root via "
        "`dataset_root`, `source_dataset_root`, `source_summary.root`, "
        "`source_summary.output_root`, or `source_aggregate_summary.root`; "
        f"encoded_root={encoded_root}, available_keys=[{available}]"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
