from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
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
    SampleTargetAlignment,
    TailPaddingPolicy,
    WindowSamplingMode,
)
from open_wam.configs.variant_semantics import (
    GENERALIST_TRAINING_BUCKET_METADATA_KEY,
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
)

from .latent_contracts import LatentWAMSample


REAL_DEMO_SOURCE = "real_demo"
COUNTERFACTUAL_DYNAMICS_SOURCE = "counterfactual_dynamics"
JOINT_MODE = "joint"
ACTION_CONDITIONED_VIDEO_MODE = "action_conditioned_video"
VIDEO_CONDITIONED_ACTION_MODE = "video_conditioned_action"


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

    Each sample concatenates the clean context latents/actions and the
    counterfactual future latents/actions. Loss metadata masks out the context
    frames, so FDM/IDM objectives train only the counterfactual future while
    still attending to the observed prefix.
    """

    def __init__(self, data_config: DataConfig, encoded_root: str | Path, *, split: str) -> None:
        self.data_config = data_config
        self.encoded_root = Path(encoded_root).expanduser().resolve()
        self.split = str(split)
        self.manifest = _read_json(self.encoded_root / "manifest.json")
        self.raw_root = Path(self.manifest["dataset_root"]).expanduser().resolve()
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
        context_row = self.context_rows.get(_context_key(row))
        if context_row is None:
            raise KeyError(
                "Missing encoded context row for counterfactual transition, "
                f"shard={row.get('shard')!r}, context_id={row.get('context_id')!r}."
            )

        context_latents = _load_latents(
            self._resolve_encoded_path(context_row, "context_latent_path"),
            key="video_latents",
        )
        target_latents = _load_latents(
            self._resolve_encoded_path(row, "target_latent_path"),
            key="target_video_latents",
        )
        if context_latents.shape[0] != target_latents.shape[0] or context_latents.shape[2:] != target_latents.shape[2:]:
            raise ValueError(
                "Counterfactual context/target latent geometry mismatch, "
                f"context={tuple(context_latents.shape)}, target={tuple(target_latents.shape)}."
            )
        source_video_latents = torch.cat([context_latents, target_latents], dim=1).contiguous()
        context_frames = int(context_latents.shape[1])
        source_frames = int(source_video_latents.shape[1])
        if segment_length is None:
            segment_length = source_frames

        context_npz = np.load(self._resolve_raw_path(context_row, "context_path"))
        sample_npz = np.load(self._resolve_raw_path(row, "sample_path"))
        source_actions = _pack_actions(
            np.concatenate(
                [
                    np.asarray(context_npz["action_context"], dtype=np.float32),
                    np.asarray(sample_npz["future_actions"], dtype=np.float32),
                ],
                axis=0,
            ),
            target_dim=int(self.data_config.action_schema.action_dim),
        )
        action_per_frame = _action_steps_per_frame(source_actions, total_frames=source_frames)
        segment = _build_counterfactual_fixed_segment(
            video_latents=source_video_latents,
            actions=source_actions,
            context_frames=context_frames,
            latent_start=int(latent_start),
            segment_length=int(segment_length),
            action_per_frame=action_per_frame,
            mask_leading_zero_action_context=(
                self.data_config.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
            ),
        )
        video_latents = segment["video_latents"]
        actions = segment["actions"]
        action_mask = segment["action_mask"]
        total_frames = int(video_latents.shape[1])

        state = torch.zeros(
            int(self.data_config.action_schema.state_horizon),
            int(self.data_config.action_schema.state_dim),
            dtype=torch.float32,
        )
        state_mask = torch.zeros_like(state)
        text_context = self.empty_text_embedding.clone() if self.empty_text_embedding is not None else None

        metadata = {
            "dataset_id": str(self.encoded_root),
            "dataset_kind": "encoded_counterfactual_dynamics",
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
            "context_start_frame": int(row.get("context_start_frame", 0)),
            "sample_start_frame": int(row.get("context_start_frame", 0)) + max(0, int(latent_start)),
            "sample_end_frame": int(row.get("context_start_frame", 0)) + max(0, int(latent_start)) + int(segment["valid_source_frames"]),
            "observation_start": int(row.get("context_start_frame", 0)) + max(0, int(latent_start)),
            "observation_frame_indices": _counterfactual_observed_frame_ids(
                context_start_frame=int(row.get("context_start_frame", 0)),
                latent_start=int(latent_start),
                segment_length=int(segment_length),
                source_frames=source_frames,
            ),
            "window_sampling_mode": self.data_config.sample_construction.mode,
            "window_start_frame": int(row.get("context_start_frame", 0)) + max(0, int(latent_start)),
            "window_end_frame": int(row.get("context_start_frame", 0)) + max(0, int(latent_start)) + int(segment["valid_source_frames"]),
            "anchor_frame_index": int(row.get("context_start_frame", 0))
            + min(source_frames - 1, max(0, int(latent_start) + int(segment["valid_source_frames"]) - 1)),
            "segment_length_frames": total_frames,
            "segment_valid_latent_frames": int(segment["valid_latent_frames"]),
            "segment_padded_latent_frames": int(segment["padded_latent_frames"]),
            "tail_padding_mode": "none" if int(segment["padded_latent_frames"]) == 0 else "zero_order_hold",
            "history_frames": int(segment["loss_frame_start"]),
            "loss_frame_start": int(segment["loss_frame_start"]),
            "loss_frame_end": int(segment["loss_frame_end"]),
            "latent_loss_frame_start": int(segment["loss_frame_start"]),
            "latent_loss_frame_end": int(segment["loss_frame_end"]),
            "action_loss_frame_start": int(segment["loss_frame_start"]),
            "action_loss_frame_end": int(segment["loss_frame_end"]),
            "sampled_chunk_size": max(1, int(self.data_config.sample_construction.chunk_size)),
            "sampled_window_size": max(1, int(self.data_config.sample_construction.window_size)),
            "latent_frame_start": int(latent_start),
            "frame_shift": int(latent_start),
            "start_padding_frames": max(0, int(self.data_config.sample_construction.start_padding_frames)),
            "segment_pre_start_frames": int(segment["pre_start_frames"]),
            "start_padding_mode": "repeat_first_latent" if int(segment["pre_start_frames"]) > 0 else "none",
            "subwindow_latent_start": int(latent_start),
            "subwindow_latent_end": int(latent_start) + int(segment_length),
            "subwindow_action_start": max(0, int(latent_start)) * action_per_frame,
            "subwindow_action_end": max(0, int(latent_start)) * action_per_frame + int(actions.shape[0]),
            "lingbot_window_action_alignment": {
                "latent_num_frames": total_frames,
                "prefix_actions": action_per_frame,
                "required_action_num": int(actions.shape[0]),
                "leading_zero_action_frames": int(segment["leading_zero_action_frames"]),
                "leading_zero_action_steps": int(segment["leading_zero_action_frames"]) * action_per_frame,
                "leading_zero_action_mask": float(segment["leading_zero_action_mask"]),
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
            task_text=None,
            text_context=text_context,
            negative_text_context=text_context.clone() if text_context is not None else None,
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
        start_padding_frames = max(0, int(sample_cfg.start_padding_frames))
        for transition_index, row in enumerate(self.transition_rows):
            context_row = self.context_rows.get(_context_key(row))
            if context_row is None:
                continue
            context_frames = _latent_frame_count_from_row_or_payload(
                self._resolve_encoded_path(context_row, "context_latent_path"),
                row=context_row,
                shape_key="context_video_latent_shape",
                payload_key="video_latents",
            )
            target_frames = _latent_frame_count_from_row_or_payload(
                self._resolve_encoded_path(row, "target_latent_path"),
                row=row,
                shape_key="target_video_latent_shape",
                payload_key="target_video_latents",
            )
            source_frames = int(context_frames + target_frames)
            start_min = -start_padding_frames if self._uses_hierarchical_fixed_segment else 0
            # Counterfactual FDM/IDM samples must retain at least one pre-t0
            # context frame. This is the objective-specific validity bound on
            # top of the #101 hierarchical fixed-segment start sampler.
            start_max = max(start_min, min(source_frames - 1, max(0, context_frames - 1)))
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
                    context_frames=int(context_frames),
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
        base_length = max(len(real_dataset), len(counterfactual_dataset))
        self._length = max(1, int(round(base_length * float(mixture_config.length_multiplier))))

    def __len__(self) -> int:
        return self._length

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int]:
        return GeneralistDynamicsMixtureTrainSampler(self, world_size=world_size, rank=rank)

    def build_source_view(
        self,
        *,
        source: str,
        mode: str,
        bucket_name: str,
        drop_text: bool,
    ) -> Dataset[LatentWAMSample]:
        bucket = GeneralistMixtureBucket(
            name=str(bucket_name),
            source=str(source),
            mode=str(mode),
            weight=1.0,
            drop_text=bool(drop_text),
        )
        return GeneralistDynamicsSourceViewDataset(self, bucket=bucket)

    def __getitem__(self, index: int) -> LatentWAMSample:
        index = int(index)
        epoch = index // len(self)
        rng = random.Random(int(self.mixture_config.seed) + index * 1_000_003)
        bucket = _sample_bucket(self.buckets, rng)
        if bucket.source == REAL_DEMO_SOURCE:
            sample_index = _draw_source_index(self.real_dataset, rng=rng, epoch=epoch)
            sample = self.real_dataset[sample_index]
        elif bucket.source == COUNTERFACTUAL_DYNAMICS_SOURCE:
            sample_index = _draw_source_index(self.counterfactual_dataset, rng=rng, epoch=epoch)
            sample = self.counterfactual_dataset[sample_index]
        else:
            raise ValueError(f"Unsupported generalist source bucket {bucket.source!r}.")
        if bucket.drop_text:
            sample = _trim_conditional_history(
                sample,
                max_history_frames=self.mixture_config.conditional_history_frames,
            )
        return _with_generalist_metadata(
            sample,
            bucket=bucket,
            split=self.split,
            source_index=sample_index,
        )


class GeneralistDynamicsSourceViewDataset(Dataset[LatentWAMSample]):
    """Deterministic source projection that preserves mixture sample transforms."""

    def __init__(self, mixture_dataset: GeneralistDynamicsMixtureDataset, *, bucket: GeneralistMixtureBucket) -> None:
        self.mixture_dataset = mixture_dataset
        self.bucket = bucket
        if bucket.source == REAL_DEMO_SOURCE:
            self.source_dataset = mixture_dataset.real_dataset
        elif bucket.source == COUNTERFACTUAL_DYNAMICS_SOURCE:
            self.source_dataset = mixture_dataset.counterfactual_dataset
        else:
            raise ValueError(f"Unsupported generalist source view {bucket.source!r}.")

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> LatentWAMSample:
        source_index = int(index)
        sample = self.source_dataset[source_index]
        if self.bucket.drop_text:
            sample = _trim_conditional_history(
                sample,
                max_history_frames=self.mixture_dataset.mixture_config.conditional_history_frames,
            )
        return _with_generalist_metadata(
            sample,
            bucket=self.bucket,
            split=self.mixture_dataset.split,
            source_index=source_index,
        )


class GeneralistDynamicsMixtureTrainSampler(Sampler[int]):
    """Epoch-offset sampler for mixed real/counterfactual dynamics draws."""

    def __init__(self, dataset: GeneralistDynamicsMixtureDataset, *, world_size: int = 1, rank: int = 0) -> None:
        if len(dataset) <= 0:
            raise ValueError("Generalist dynamics mixture sampling requires a non-empty dataset.")
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
        epoch_offset = int(self.epoch) * len(self.dataset)
        return iter(epoch_offset + global_index for global_index in range(self.rank, self._total_size, self.world_size))


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


def _trim_conditional_history(
    sample: LatentWAMSample,
    *,
    max_history_frames: int | None,
) -> LatentWAMSample:
    """Physically crop excess clean prefix for conditional FDM/IDM samples."""

    if max_history_frames is None:
        return sample
    max_history_frames = int(max_history_frames)
    if max_history_frames <= 0:
        raise ValueError("Conditional generalist history cap must be positive or None.")
    total_frames = int(sample.video_latents.shape[1])
    loss_frame_start = _metadata_frame_boundary(
        sample.metadata,
        ("loss_frame_start", "latent_loss_frame_start", "action_loss_frame_start", "history_frames"),
    )
    if loss_frame_start is None or loss_frame_start <= max_history_frames:
        return sample
    if loss_frame_start >= total_frames:
        raise ValueError(
            "Cannot trim conditional history when the future target is outside the sampled latent segment, "
            f"loss_frame_start={loss_frame_start}, total_frames={total_frames}."
        )
    crop_frames = int(loss_frame_start - max_history_frames)
    if sample.actions.shape[0] % total_frames != 0:
        raise ValueError(
            "Conditional history trimming requires frame-aligned action targets, "
            f"actions={sample.actions.shape[0]}, latent_frames={total_frames}."
        )
    action_steps_per_frame = int(sample.actions.shape[0] // total_frames)
    action_crop = crop_frames * action_steps_per_frame
    pre_start_frames = int(sample.metadata.get("segment_pre_start_frames", 0) or 0)
    source_action_crop = max(0, crop_frames - pre_start_frames) * action_steps_per_frame
    new_total_frames = total_frames - crop_frames

    video_latents = sample.video_latents[:, crop_frames:].contiguous()
    actions = sample.actions[action_crop:].contiguous()
    action_mask = sample.action_mask[action_crop:].contiguous() if sample.action_mask is not None else None
    canonical_video = _trim_optional_video(sample.canonical_video, crop_frames=crop_frames, total_frames=total_frames)
    condition_latents = _trim_optional_video(
        sample.condition_latents,
        crop_frames=crop_frames,
        total_frames=total_frames,
    )
    proprio_context_state = _trim_optional_frame_tensor(
        sample.proprio_context_state,
        crop_frames=crop_frames,
        total_frames=total_frames,
    )
    proprio_context_state_mask = _trim_optional_frame_tensor(
        sample.proprio_context_state_mask,
        crop_frames=crop_frames,
        total_frames=total_frames,
    )
    proprio_context_frames = _trim_optional_frame_tensor(
        sample.proprio_context_frames,
        crop_frames=crop_frames,
        total_frames=total_frames,
    )
    proprio_context_frames_mask = _trim_optional_frame_tensor(
        sample.proprio_context_frames_mask,
        crop_frames=crop_frames,
        total_frames=total_frames,
    )
    metadata = _trim_conditional_history_metadata(
        sample.metadata,
        crop_frames=crop_frames,
        source_action_crop=source_action_crop,
        action_steps_per_frame=action_steps_per_frame,
        new_total_frames=new_total_frames,
        action_mask=action_mask,
        actions=actions,
        max_history_frames=max_history_frames,
    )
    return replace(
        sample,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        canonical_video=canonical_video,
        condition_latents=condition_latents,
        proprio_context_state=proprio_context_state,
        proprio_context_state_mask=proprio_context_state_mask,
        proprio_context_frames=proprio_context_frames,
        proprio_context_frames_mask=proprio_context_frames_mask,
        metadata=metadata,
    )


def _metadata_frame_boundary(metadata: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return int(value)
    return None


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


def _trim_conditional_history_metadata(
    metadata: dict[str, Any],
    *,
    crop_frames: int,
    source_action_crop: int,
    action_steps_per_frame: int,
    new_total_frames: int,
    action_mask: torch.Tensor | None,
    actions: torch.Tensor,
    max_history_frames: int,
) -> dict[str, Any]:
    updated = dict(metadata)
    original_observed = _metadata_sequence(updated.get("observation_frame_indices")) or _metadata_sequence(
        updated.get("observed_frame_ids")
    )
    trimmed_observed = None
    if original_observed is not None and len(original_observed) >= crop_frames:
        trimmed_observed = original_observed[crop_frames : crop_frames + new_total_frames]
        if "observation_frame_indices" in updated:
            updated["observation_frame_indices"] = list(trimmed_observed)
        if "observed_frame_ids" in updated:
            updated["observed_frame_ids"] = list(trimmed_observed)
    new_observation_start = int(trimmed_observed[0]) if trimmed_observed else None
    original_context_prefix_in_sample = max(0, int(metadata.get("context_prefix_frames_in_sample", 0) or 0))
    original_context_prefix_real = max(
        0,
        int(metadata.get("context_prefix_real_frames", original_context_prefix_in_sample) or 0),
    )
    # Cropping can consume real prefix context before it reaches chunk-alignment target frames.
    cropped_context_prefix_frames = min(crop_frames, original_context_prefix_in_sample)
    cropped_real_prefix_frames = min(crop_frames, original_context_prefix_real)

    for key in (
        "loss_frame_start",
        "loss_frame_end",
        "latent_loss_frame_start",
        "latent_loss_frame_end",
        "action_loss_frame_start",
        "action_loss_frame_end",
        "current_start_frame_in_sample",
        "current_end_frame_in_sample",
        "supervised_start",
        "supervised_end",
    ):
        if key in updated and updated[key] is not None:
            updated[key] = min(new_total_frames, max(0, int(updated[key]) - crop_frames))

    loss_frame_start = _metadata_frame_boundary(
        updated,
        ("loss_frame_start", "latent_loss_frame_start", "action_loss_frame_start"),
    )
    if loss_frame_start is None:
        loss_frame_start = min(max_history_frames, new_total_frames - 1)
        updated["loss_frame_start"] = loss_frame_start
    updated["history_frames"] = int(loss_frame_start)

    if "segment_length_frames" in updated:
        updated["segment_length_frames"] = int(new_total_frames)
    if "segment_pre_start_frames" in updated:
        updated["segment_pre_start_frames"] = max(0, int(updated["segment_pre_start_frames"]) - crop_frames)
        updated["start_padding_mode"] = "repeat_first_latent" if int(updated["segment_pre_start_frames"]) > 0 else "none"
    if "segment_valid_latent_frames" in updated:
        pre_start = int(metadata.get("segment_pre_start_frames", 0) or 0)
        valid_removed = max(0, crop_frames - pre_start)
        updated["segment_valid_latent_frames"] = max(0, int(updated["segment_valid_latent_frames"]) - valid_removed)
    if "segment_padded_latent_frames" in updated and "segment_valid_latent_frames" in updated:
        updated["segment_padded_latent_frames"] = max(
            0,
            int(new_total_frames) - int(updated["segment_valid_latent_frames"]),
        )
        updated["tail_padding_mode"] = "none" if int(updated["segment_padded_latent_frames"]) == 0 else "zero_order_hold"
    if "head_padded_frame_count" in updated and updated["head_padded_frame_count"] is not None:
        updated["head_padded_frame_count"] = max(0, int(updated["head_padded_frame_count"]) - crop_frames)
    if "context_prefix_frames_in_sample" in updated and updated["context_prefix_frames_in_sample"] is not None:
        updated["context_prefix_frames_in_sample"] = max(
            0,
            int(updated["context_prefix_frames_in_sample"]) - cropped_context_prefix_frames,
        )
    if "context_prefix_real_frames" in updated and updated["context_prefix_real_frames"] is not None:
        updated["context_prefix_real_frames"] = max(
            0,
            int(updated["context_prefix_real_frames"]) - cropped_real_prefix_frames,
        )
    if "context_prefix_truncated_frames" in updated and updated["context_prefix_truncated_frames"] is not None:
        truncated_prefix = max(0, int(updated["context_prefix_truncated_frames"])) + cropped_context_prefix_frames
        requested_prefix = updated.get("context_prefix_frames_requested")
        if requested_prefix is not None:
            truncated_prefix = min(max(0, int(requested_prefix)), truncated_prefix)
        updated["context_prefix_truncated_frames"] = truncated_prefix

    for key in ("sample_start_frame", "observation_start", "window_start_frame"):
        if key in updated and updated[key] is not None:
            updated[key] = int(new_observation_start) if new_observation_start is not None else int(updated[key]) + crop_frames
    for key in ("latent_frame_start", "frame_shift", "effective_start", "effective_frame_start", "logical_frame_start"):
        if key in updated and updated[key] is not None:
            updated[key] = int(updated[key]) + crop_frames
    for start_key, end_key in (("effective_start", "effective_end"), ("effective_frame_start", "effective_frame_end")):
        if start_key in updated and end_key in updated and updated[start_key] is not None:
            updated[end_key] = int(updated[start_key]) + int(new_total_frames)
    target_start = updated.get("target_frame_start")
    target_end = updated.get("target_frame_end")
    if target_start is not None or target_end is not None:
        adjusted_target_start = int(target_start) if target_start is not None else None
        new_effective_start = _metadata_frame_boundary(
            updated,
            ("effective_frame_start", "effective_start", "frame_shift"),
        )
        if adjusted_target_start is not None and new_effective_start is not None:
            adjusted_target_start = max(adjusted_target_start, int(new_effective_start))
        if adjusted_target_start is not None and target_end is not None:
            adjusted_target_start = min(adjusted_target_start, int(target_end))
        if adjusted_target_start is not None:
            updated["target_frame_start"] = int(adjusted_target_start)
        if target_start is not None:
            for key in ("subwindow_latent_start", "virtual_latent_start"):
                if key in updated and updated[key] is not None:
                    updated[key] = int(adjusted_target_start)
        if target_end is not None and "subwindow_latent_end" in updated and updated["subwindow_latent_end"] is not None:
            updated["subwindow_latent_end"] = int(target_end)
    else:
        for key in ("subwindow_latent_start", "virtual_latent_start"):
            if key in updated and updated[key] is not None:
                updated[key] = int(updated[key]) + crop_frames
    if "subwindow_action_start" in updated and updated["subwindow_action_start"] is not None:
        updated["subwindow_action_start"] = int(updated["subwindow_action_start"]) + int(source_action_crop)

    alignment = updated.get("lingbot_window_action_alignment")
    if isinstance(alignment, dict):
        alignment = dict(alignment)
        alignment["latent_num_frames"] = int(new_total_frames)
        alignment["required_action_num"] = int(actions.shape[0])
        if "leading_zero_action_frames" in alignment:
            leading_frames = max(0, int(alignment["leading_zero_action_frames"]) - crop_frames)
            alignment["leading_zero_action_frames"] = leading_frames
            alignment["leading_zero_action_steps"] = leading_frames * action_steps_per_frame
        updated["lingbot_window_action_alignment"] = alignment

    valid_steps, valid_values = _action_validity_stats(actions=actions, action_mask=action_mask)
    updated["valid_action_steps"] = valid_steps
    updated["valid_action_values"] = valid_values
    updated["generalist_conditional_history_frames"] = int(max_history_frames)
    updated["generalist_history_trimmed_frames"] = int(crop_frames)
    return updated


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


def _load_latents(path: Path, *, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or key not in payload:
        raise ValueError(f"Expected key {key!r} in latent payload at {path}.")
    tensor = payload[key]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError(f"Expected {key!r} tensor [C,T,H,W] at {path}, got {type(tensor)!r}.")
    return tensor.to(dtype=torch.float32).contiguous()


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


def _action_steps_per_frame(actions: torch.Tensor, *, total_frames: int) -> int:
    if total_frames <= 0:
        raise ValueError("Counterfactual sample must contain at least one latent frame.")
    if actions.shape[0] % total_frames != 0:
        raise ValueError(
            "Counterfactual action count must be frame-aligned, "
            f"got actions={actions.shape[0]}, latent_frames={total_frames}."
        )
    action_per_frame = int(actions.shape[0] // total_frames)
    if action_per_frame <= 0:
        raise ValueError("Counterfactual sample must contain at least one action per latent frame.")
    return action_per_frame


def _build_counterfactual_fixed_segment(
    *,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    context_frames: int,
    latent_start: int,
    segment_length: int,
    action_per_frame: int,
    mask_leading_zero_action_context: bool = False,
) -> dict[str, Any]:
    source_frames = int(video_latents.shape[1])
    if source_frames <= 0:
        raise ValueError("Counterfactual segment sampling requires at least one source latent frame.")
    if segment_length <= 0:
        raise ValueError(f"Counterfactual segment_length must be positive, got {segment_length}.")
    source_start = max(0, int(latent_start))
    source_end = min(source_frames, int(latent_start) + int(segment_length))
    if source_end <= source_start:
        source_end = min(source_frames, source_start + 1)
    valid_slice = video_latents[:, source_start:source_end]
    pre_start_frames = max(0, min(segment_length, -int(latent_start))) if int(latent_start) < 0 else 0
    valid_latent_frames = max(0, min(segment_length, source_frames - int(latent_start)))
    padded_latent_frames = max(0, segment_length - valid_latent_frames)

    parts: list[torch.Tensor] = []
    if int(latent_start) < 0:
        parts.append(video_latents[:, :1].expand(-1, min(-int(latent_start), segment_length), -1, -1))
    parts.append(valid_slice)
    current_frames = sum(int(part.shape[1]) for part in parts)
    if current_frames < segment_length:
        parts.append(video_latents[:, -1:].expand(-1, segment_length - current_frames, -1, -1))
    segment_video = torch.cat(parts, dim=1)[:, :segment_length].contiguous()

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
        segment_actions[dst_start:dst_end] = actions[src_start:src_end]
        action_mask[dst_start:dst_end] = 1.0

    future_start = int(context_frames) - int(latent_start)
    loss_frame_start = max(0, int(pre_start_frames), int(future_start))
    loss_frame_end = min(int(segment_length), int(valid_latent_frames))
    if loss_frame_end < loss_frame_start:
        loss_frame_end = loss_frame_start
    return {
        "video_latents": segment_video,
        "actions": segment_actions.contiguous(),
        "action_mask": action_mask.contiguous(),
        "pre_start_frames": int(pre_start_frames),
        "valid_latent_frames": int(valid_latent_frames),
        "padded_latent_frames": int(padded_latent_frames),
        "valid_source_frames": max(0, int(source_end) - int(source_start)),
        "loss_frame_start": int(loss_frame_start),
        "loss_frame_end": int(loss_frame_end),
        "leading_zero_action_frames": int(leading_zero_action_frames),
        "leading_zero_action_mask": float(leading_zero_action_mask),
    }


def _counterfactual_observed_frame_ids(
    *,
    context_start_frame: int,
    latent_start: int,
    segment_length: int,
    source_frames: int,
) -> list[int]:
    ids: list[int] = []
    for offset in range(int(segment_length)):
        source_frame = min(max(0, int(latent_start) + offset), int(source_frames) - 1)
        ids.append(int(context_start_frame) + source_frame)
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
    return int(_load_latents(path, key=payload_key).shape[1])


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
