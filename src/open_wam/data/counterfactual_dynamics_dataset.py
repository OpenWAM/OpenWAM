"""Encoded counterfactual dynamics dataset adapter and metadata assembly."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from open_wam.configs import (
    DataConfig,
    DataSplit,
    PaddedTargetPolicy,
    TailPaddingPolicy,
    WindowSamplingMode,
)

from .conditional_dynamics_layout import (
    GENERALIST_CONDITIONAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE,
    GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
    GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON,
)
from .counterfactual_dynamics_materialization import (
    COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY,
    COUNTERFACTUAL_CONTRACT_T0_PLUS_FUTURE,
    COUNTERFACTUAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE,
    COUNTERFACTUAL_STATE_KEY,
    _build_counterfactual_fixed_segment,
    _configured_action_steps_per_latent_frame,
    _counterfactual_action_steps_per_frame,
    _counterfactual_condition_latents_from_source,
    _counterfactual_latent_state_frames,
    _counterfactual_observed_frame_ids,
    _counterfactual_state_history_from_frames,
    _counterfactual_target_only_condition_latents_from_payload,
    _load_empty_text_embedding,
    _load_latent_payload,
    _optional_counterfactual_condition_latents,
    _pack_actions,
    _pack_state,
    _payload_latents,
    _sample_counterfactual_attention_geometry,
    _slice_counterfactual_frame_tensor_with_edge_hold,
    _slice_counterfactual_latents_with_edge_hold,
    _validate_counterfactual_condition_latent_manifest,
)
from .counterfactual_source_order import (
    _balanced_counterfactual_source_indices,
    _counterfactual_source_branch_key,
    _counterfactual_source_task_key,
    _source_view_label_sort_key,
)
from .distributed_sampling import draw_hierarchical_sample_index
from .latent_contracts import LatentWAMSample


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
        state_dim = int(self.data_config.action_schema.state_dim)
        with np.load(
            self._resolve_raw_path(row, "sample_path"),
            allow_pickle=False,
        ) as sample_npz:
            source_actions = _pack_actions(
                np.asarray(sample_npz["future_actions"], dtype=np.float32),
                target_dim=int(self.data_config.action_schema.action_dim),
            )
            (
                source_proprio_frames,
                source_proprio_frames_mask,
            ) = _counterfactual_latent_state_frames(
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
        draw = draw_hierarchical_sample_index(
            seed_values=(int(self.data_config.split_seed), split_salt, int(index)),
            task_weights=self._task_weights,
            task_specs=self._task_specs,
        )
        task_spec = self._task_specs[draw.task_index]
        window_spec = task_spec.windows[draw.window_index]
        return task_spec, window_spec, draw.start

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


def _context_key(row: dict[str, Any]) -> tuple[str | None, int]:
    shard = row.get("shard")
    return (None if shard is None else str(shard), int(row["context_id"]))


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


_COUNTERFACTUAL_DATASET_COMPATIBILITY_EXPORTS = (
    _balanced_counterfactual_source_indices,
    _build_counterfactual_fixed_segment,
    _configured_action_steps_per_latent_frame,
    _counterfactual_action_steps_per_frame,
    _counterfactual_condition_latents_from_source,
    _counterfactual_latent_state_frames,
    _counterfactual_observed_frame_ids,
    _counterfactual_source_branch_key,
    _counterfactual_source_task_key,
    _counterfactual_state_history_from_frames,
    _counterfactual_target_only_condition_latents_from_payload,
    _load_empty_text_embedding,
    _load_latent_payload,
    _optional_counterfactual_condition_latents,
    _pack_actions,
    _pack_state,
    _payload_latents,
    _sample_counterfactual_attention_geometry,
    _slice_counterfactual_frame_tensor_with_edge_hold,
    _slice_counterfactual_latents_with_edge_hold,
    _source_view_label_sort_key,
    _validate_counterfactual_condition_latent_manifest,
)

__all__ = [
    "COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY",
    "COUNTERFACTUAL_CONTRACT_T0_PLUS_FUTURE",
    "COUNTERFACTUAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE",
    "COUNTERFACTUAL_STATE_KEY",
    "EncodedCounterfactualDynamicsLatentDataset",
]
