"""Sampling policy for local LeRobot latent datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass, replace
import random
from typing import Any, Protocol

import torch

from open_wam.configs import (
    DataConfig,
    DataSplit,
    LatentWindowProfile,
    SampleConstructionConfig,
    SampleWeightMode,
    WindowSamplingMode,
)

from .distributed_sampling import (
    EpochOffsetDistributedSampler,
    EpochOrderDistributedSampler,
    EpochOrderSource,
    WeightedReplacementDistributedSampler,
    draw_hierarchical_sample_index,
)
from .latent_temporal import raw_span_for_latent_range
from .lerobot_v2_latent_storage import LocalEpisodeWindow, LocalRepoBundle


__all__ = [
    "HierarchicalFixedSegmentSamplingPlan",
    "HierarchicalFixedSegmentTaskSpec",
    "HierarchicalFixedSegmentTrainSampler",
    "HierarchicalFixedSegmentWindowSpec",
    "LocalLatentEpochOrderSampler",
    "LocalLatentUniformSegmentSamplingPlan",
    "LocalLatentWindowWeightPlan",
    "LocalLatentWeightedTrainSampler",
    "build_hierarchical_fixed_segment_task_specs",
]


class _WeightedLocalLatentSource(Protocol):
    """Dataset fields needed by replacement sampling."""

    data_config: DataConfig
    sample_weights: Sequence[float]

    def __len__(self) -> int: ...


def _build_local_latent_sample_weights(
    *,
    sample_config: SampleConstructionConfig,
    item_count: int,
    dataset_mean_valid_action_steps: float,
    dataset_mean_task_demo_count: float,
    valid_action_steps_for_index: Callable[[int], float],
    task_text_for_index: Callable[[int], str],
    task_demo_counts: Mapping[str, int],
    task_virtual_start_counts: Mapping[str, int] | None = None,
    dataset_mean_task_virtual_start_count: float = 1.0,
) -> tuple[float, ...]:
    """Build normalized weights for physical windows or virtual starts."""

    mode = sample_config.sample_weight_mode
    if mode == SampleWeightMode.UNIFORM:
        return tuple(1.0 for _ in range(item_count))
    reference_steps = max(1.0, float(dataset_mean_valid_action_steps))
    reference_task_count = max(1.0, float(dataset_mean_task_demo_count))
    weights: list[float] = []
    for index in range(item_count):
        weight = 1.0
        if mode in {
            SampleWeightMode.VALID_ACTION_STEPS,
            SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT,
        }:
            weight *= max(
                1.0,
                float(valid_action_steps_for_index(index)),
            ) / reference_steps
        if mode in {
            SampleWeightMode.INVERSE_TASK_DEMO_COUNT,
            SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT,
        }:
            task_count = max(
                1,
                task_demo_counts[task_text_for_index(index)],
            )
            weight *= reference_task_count / float(task_count)
        if (
            mode == SampleWeightMode.TASK_VIRTUAL_START_COUNT_POWER
            and task_virtual_start_counts is not None
        ):
            task_start_count = max(
                1.0,
                float(task_virtual_start_counts[task_text_for_index(index)]),
            )
            reference_start_count = max(
                1.0,
                float(dataset_mean_task_virtual_start_count),
            )
            weight *= (task_start_count / reference_start_count) ** (
                float(sample_config.sample_weight_length_power) - 1.0
            )
        if sample_config.sample_weight_min is not None:
            weight = max(float(sample_config.sample_weight_min), weight)
        if sample_config.sample_weight_max is not None:
            weight = min(float(sample_config.sample_weight_max), weight)
        weights.append(float(weight))
    if not any(weight > 0 for weight in weights):
        return tuple(1.0 for _ in range(item_count))
    return tuple(weights)


@dataclass(frozen=True)
class LocalLatentWindowWeightPlan:
    """Deterministic task statistics and weights for physical latent windows."""

    data_config: DataConfig
    windows: tuple[LocalEpisodeWindow, ...]
    window_valid_action_steps: tuple[int, ...]
    dataset_mean_valid_action_steps: float
    window_task_texts: tuple[str, ...]
    task_demo_counts: Counter[str]
    dataset_mean_task_demo_count: float
    sample_weights: tuple[float, ...]

    @classmethod
    def from_windows(
        cls,
        *,
        data_config: DataConfig,
        windows: Sequence[LocalEpisodeWindow],
        repo_bundles: Mapping[str, LocalRepoBundle],
    ) -> LocalLatentWindowWeightPlan:
        """Resolve all deterministic weighting state for physical windows."""

        window_tuple = tuple(windows)
        valid_action_steps: list[int] = []
        task_texts: list[str] = []
        for window in window_tuple:
            if (
                data_config.sample_construction.mode
                == WindowSamplingMode.FULL_SEGMENT
                and data_config.latent_window_profile
                == LatentWindowProfile.EXACT_CHUNKED_WINDOW
            ):
                prefix_actions = int(
                    data_config.action_schema.action_horizon
                    // max(1, data_config.num_frames)
                )
                window_span = max(0, window.end_frame - window.start_frame)
                valid_steps = prefix_actions + max(
                    len(window.observation_frame_indices),
                    window_span,
                )
            elif (
                data_config.sample_construction.mode
                == WindowSamplingMode.CAUSAL_PREFIX_SUFFIX
            ):
                valid_steps = 0
            else:
                valid_steps = int(data_config.action_schema.action_horizon)
            valid_action_steps.append(max(0, int(valid_steps)))

            repo_bundle = repo_bundles.get(str(window.repo_root))
            episode_record = (
                None
                if repo_bundle is None
                else repo_bundle.episodes_by_index.get(window.episode_index)
            )
            task_texts.append(
                str(episode_record.tasks[0])
                if episode_record is not None and episode_record.tasks
                else f"{window.repo_root}:episode:{window.episode_index}"
            )

        window_valid_action_steps = tuple(valid_action_steps)
        positive_estimates = [value for value in valid_action_steps if value > 0]
        if not positive_estimates:
            dataset_mean_valid_action_steps = float(
                max(1, data_config.action_schema.action_horizon)
            )
        else:
            dataset_mean_valid_action_steps = float(
                sum(positive_estimates) / len(positive_estimates)
            )
        window_task_texts = tuple(task_texts)
        demo_keys_by_task: dict[str, set[tuple[str, int]]] = {}
        for window, task_text in zip(
            window_tuple,
            window_task_texts,
            strict=True,
        ):
            demo_keys_by_task.setdefault(task_text, set()).add(
                (str(window.repo_root), int(window.episode_index))
            )
        task_demo_counts = Counter(
            {
                task_text: len(demo_keys)
                for task_text, demo_keys in demo_keys_by_task.items()
            }
        )
        dataset_mean_task_demo_count = (
            float(sum(task_demo_counts.values()) / len(task_demo_counts))
            if task_demo_counts
            else 1.0
        )
        sample_weights = _build_local_latent_sample_weights(
            sample_config=data_config.sample_construction,
            item_count=len(window_tuple),
            dataset_mean_valid_action_steps=dataset_mean_valid_action_steps,
            dataset_mean_task_demo_count=dataset_mean_task_demo_count,
            valid_action_steps_for_index=(
                lambda index: window_valid_action_steps[index]
            ),
            task_text_for_index=lambda index: window_task_texts[index],
            task_demo_counts=task_demo_counts,
        )
        return cls(
            data_config=data_config,
            windows=window_tuple,
            window_valid_action_steps=window_valid_action_steps,
            dataset_mean_valid_action_steps=dataset_mean_valid_action_steps,
            window_task_texts=window_task_texts,
            task_demo_counts=task_demo_counts,
            dataset_mean_task_demo_count=dataset_mean_task_demo_count,
            sample_weights=sample_weights,
        )

    def sample_weight_metadata(self, index: int) -> dict[str, Any]:
        """Describe weighting inputs for one physical window."""

        task_text = self.window_task_texts[index]
        return {
            "train_sample_weight": self.sample_weights[index],
            "train_sample_weight_mode": (
                self.data_config.sample_construction.sample_weight_mode
            ),
            "eligible_task_demo_count": self.task_demo_counts[task_text],
            "dataset_mean_eligible_task_demo_count": (
                self.dataset_mean_task_demo_count
            ),
        }

    def task_text_for_window_index(self, index: int) -> str:
        """Return the resolved task label for one physical window."""

        return self.window_task_texts[index]


@dataclass(frozen=True)
class HierarchicalFixedSegmentWindowSpec:
    """One eligible trajectory/chunk geometry for hierarchical sampling."""

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


class LocalLatentWeightedTrainSampler(WeightedReplacementDistributedSampler):
    """Replacement train sampler for weighted local latent examples."""

    def __init__(
        self,
        dataset: _WeightedLocalLatentSource,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            weights=dataset.sample_weights,
            base_seed=int(dataset.data_config.split_seed),
            world_size=world_size,
            rank=rank,
            empty_dataset_message=(
                "Weighted local latent sampling requires a non-empty dataset."
            ),
        )


class LocalLatentEpochOrderSampler(EpochOrderDistributedSampler):
    """Sampler backed by a dataset-provided epoch order."""

    def __init__(
        self,
        dataset: EpochOrderSource,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message=(
                "Epoch-order local latent sampling requires a non-empty dataset."
            ),
            empty_order_message=(
                "Epoch-order local latent sampler received an empty order."
            ),
        )


class HierarchicalFixedSegmentTrainSampler(EpochOffsetDistributedSampler):
    """Deterministic sampler for hierarchical fixed-segment draw keys."""

    def __init__(
        self,
        dataset: Sized,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message=(
                "Hierarchical fixed-segment sampling requires a non-empty dataset."
            ),
        )


@dataclass(frozen=True)
class LocalLatentUniformSegmentSamplingPlan:
    """Eligibility, weighting, ordering, and geometry for uniform segments."""

    data_config: DataConfig
    windows: tuple[LocalEpisodeWindow, ...]
    window_task_texts: tuple[str, ...]
    task_demo_counts: dict[str, int]
    dataset_mean_task_demo_count: float
    segment_length_candidates: tuple[int, ...]
    virtual_index: tuple[tuple[int, int], ...]
    virtual_indices_by_window: dict[int, tuple[int, ...]]
    task_virtual_start_counts: dict[str, int]
    dataset_mean_task_virtual_start_count: float
    dataset_mean_valid_action_steps: float
    sample_weights: tuple[float, ...]

    @classmethod
    def from_windows(
        cls,
        *,
        data_config: DataConfig,
        windows: Sequence[LocalEpisodeWindow],
        window_task_texts: Sequence[str],
        task_demo_counts: Mapping[str, int],
        dataset_mean_task_demo_count: float,
    ) -> LocalLatentUniformSegmentSamplingPlan:
        """Resolve all deterministic plan state for a local latent catalog."""

        window_tuple = tuple(windows)
        task_text_tuple = tuple(str(value) for value in window_task_texts)
        if len(task_text_tuple) != len(window_tuple):
            raise ValueError(
                "Uniform segment sampling requires one task label per window, "
                f"got task_labels={len(task_text_tuple)}, windows={len(window_tuple)}."
            )
        segment_length_candidates = cls.resolve_segment_length_candidates(
            data_config
        )
        virtual_index = cls.build_virtual_index(
            data_config=data_config,
            windows=window_tuple,
            segment_length_candidates=segment_length_candidates,
        )
        if not virtual_index:
            raise ValueError(
                "Uniform segment sampling requires at least one latent start."
            )
        virtual_indices_by_window: dict[int, list[int]] = {}
        for virtual_sample_index, (window_index, _) in enumerate(virtual_index):
            virtual_indices_by_window.setdefault(window_index, []).append(
                virtual_sample_index
            )
        task_virtual_start_counts: dict[str, int] = {}
        for window_index, _ in virtual_index:
            task_text = task_text_tuple[window_index]
            task_virtual_start_counts[task_text] = (
                task_virtual_start_counts.get(task_text, 0) + 1
            )
        positive_task_counts = [
            count for count in task_virtual_start_counts.values() if count > 0
        ]
        mean_task_virtual_start_count = (
            float(sum(positive_task_counts) / len(positive_task_counts))
            if positive_task_counts
            else 1.0
        )
        plan = cls(
            data_config=data_config,
            windows=window_tuple,
            window_task_texts=task_text_tuple,
            task_demo_counts={
                str(task_text): int(count)
                for task_text, count in task_demo_counts.items()
            },
            dataset_mean_task_demo_count=float(dataset_mean_task_demo_count),
            segment_length_candidates=segment_length_candidates,
            virtual_index=virtual_index,
            virtual_indices_by_window={
                window_index: tuple(indices)
                for window_index, indices in virtual_indices_by_window.items()
            },
            task_virtual_start_counts=dict(task_virtual_start_counts),
            dataset_mean_task_virtual_start_count=(
                mean_task_virtual_start_count
            ),
            dataset_mean_valid_action_steps=0.0,
            sample_weights=(),
        )
        mean_valid_action_steps = plan.estimate_mean_valid_action_steps()
        plan = replace(
            plan,
            dataset_mean_valid_action_steps=mean_valid_action_steps,
        )
        return replace(plan, sample_weights=plan.build_sample_weights())

    def materialize_virtual_indices_by_window(self) -> dict[int, list[int]]:
        """Return the historical mutable dataset view of grouped indices."""

        return {
            window_index: list(indices)
            for window_index, indices in self.virtual_indices_by_window.items()
        }

    def materialize_task_virtual_start_counts(self) -> dict[str, int]:
        """Return the historical mutable dataset view of task counts."""

        return dict(self.task_virtual_start_counts)

    @staticmethod
    def resolve_segment_length_candidates(
        data_config: DataConfig,
    ) -> tuple[int, ...]:
        """Resolve the inclusive configured length grid."""

        sample_config = data_config.sample_construction
        min_frames = int(sample_config.segment_min_frames or data_config.num_frames)
        max_frames = int(sample_config.segment_max_frames or min_frames)
        stride = max(1, int(sample_config.segment_length_stride))
        if min_frames > max_frames:
            raise ValueError(
                "Uniform segment sampling requires segment_min_frames <= segment_max_frames, "
                f"got min={min_frames}, max={max_frames}."
            )
        candidates = list(range(min_frames, max_frames + 1, stride))
        if candidates[-1] != max_frames:
            candidates.append(max_frames)
        return tuple(candidates)

    @staticmethod
    def resolve_start_padding_frames(
        data_config: DataConfig,
        window: LocalEpisodeWindow,
    ) -> int:
        """Return startup padding only for windows beginning at trajectory zero."""

        padding_frames = max(
            0,
            int(data_config.sample_construction.start_padding_frames),
        )
        if padding_frames <= 0:
            return 0
        return padding_frames if int(window.observation_start) == 0 else 0

    @classmethod
    def build_virtual_index(
        cls,
        *,
        data_config: DataConfig,
        windows: Sequence[LocalEpisodeWindow],
        segment_length_candidates: Sequence[int],
    ) -> tuple[tuple[int, int], ...]:
        """Enumerate trajectory/start frequency slots in canonical order."""

        virtual_index: list[tuple[int, int]] = []
        min_segment_length = min(segment_length_candidates)
        for window_index, window in enumerate(windows):
            source_latent_frames = max(1, int(window.latent_num_frames))
            start_padding_frames = cls.resolve_start_padding_frames(
                data_config,
                window,
            )
            min_latent_start = -start_padding_frames
            logical_source_frames = source_latent_frames + start_padding_frames
            for latent_start in range(min_latent_start, source_latent_frames):
                if data_config.sample_construction.require_full_segment:
                    if (
                        logical_source_frames < min_segment_length
                        and latent_start > min_latent_start
                    ):
                        continue
                    max_length_from_start = source_latent_frames - latent_start
                    if (
                        logical_source_frames >= min_segment_length
                        and max_length_from_start < min_segment_length
                    ):
                        continue
                virtual_index.append((window_index, latent_start))
        return tuple(virtual_index)

    def build_epoch_index_order(self, *, epoch: int) -> list[int]:
        """Build the deterministic locality-aware global order for one epoch."""

        epoch_seed = self.data_config.split_seed + int(epoch) * 1_000_003
        rng = random.Random(epoch_seed)
        if (
            self.data_config.sample_construction.sample_weight_mode
            == SampleWeightMode.UNIFORM
        ):
            per_window = {
                window_index: list(indices)
                for window_index, indices in self.virtual_indices_by_window.items()
            }
            for indices in per_window.values():
                rng.shuffle(indices)
        else:
            weights = torch.tensor(self.sample_weights, dtype=torch.double)
            if float(weights.sum().item()) <= 0:
                weights = torch.ones(len(self.virtual_index), dtype=torch.double)
            generator = torch.Generator()
            generator.manual_seed(epoch_seed & 0x7FFF_FFFF_FFFF_FFFF)
            sampled = torch.multinomial(
                weights,
                num_samples=len(self.virtual_index),
                replacement=True,
                generator=generator,
            ).tolist()
            per_window: dict[int, list[int]] = {}
            for virtual_sample_index in sampled:
                window_index, _ = self.virtual_index[
                    int(virtual_sample_index)
                ]
                per_window.setdefault(window_index, []).append(
                    int(virtual_sample_index)
                )
            for indices in per_window.values():
                rng.shuffle(indices)

        window_order = list(per_window)
        rng.shuffle(window_order)
        block_size = max(
            1,
            int(
                self.data_config.sample_construction.segment_locality_block_size
            ),
        )
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

    def eligible_segment_lengths(
        self,
        *,
        source_latent_frames: int,
        start_padding_frames: int = 0,
    ) -> tuple[int, ...]:
        """Return configured lengths eligible for one logical source span."""

        if not self.data_config.sample_construction.require_full_segment:
            return self.segment_length_candidates
        logical_source_frames = int(source_latent_frames) + max(
            0,
            int(start_padding_frames),
        )
        candidates = tuple(
            length
            for length in self.segment_length_candidates
            if length <= logical_source_frames
        )
        if not candidates and logical_source_frames > 0:
            return (logical_source_frames,)
        if not candidates:
            raise ValueError(
                "Uniform segment sampling with require_full_segment=True found no eligible segment length for "
                f"source_latent_frames={source_latent_frames}; start_padding_frames={start_padding_frames}; "
                f"minimum candidate={min(self.segment_length_candidates)}."
            )
        return candidates

    def estimate_segment_valid_action_steps(
        self,
        *,
        window: LocalEpisodeWindow,
        latent_start: int,
        segment_length: int,
    ) -> int:
        """Estimate valid supervised action rows for one segment geometry."""

        raw_frame_ids = list(window.observation_frame_indices)
        source_latent_frames = len(raw_frame_ids)
        if not raw_frame_ids or source_latent_frames <= 0:
            return 0
        prefix_actions = int(
            self.data_config.action_schema.action_horizon
            // max(1, self.data_config.num_frames)
        )
        source_latent_start = max(0, latent_start)
        valid_latent_end = min(
            source_latent_frames,
            max(0, latent_start + segment_length),
        )
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
        if self.resolve_start_padding_frames(self.data_config, window) > 0 and latent_start <= 0:
            leading_valid_action_steps = 0
        return min(
            required_action_steps,
            leading_valid_action_steps + raw_action_steps,
        )

    def estimate_virtual_valid_action_steps(
        self,
        virtual_sample_index: int,
    ) -> float:
        """Average valid action rows across eligible lengths for one slot."""

        window_index, latent_start = self.virtual_index[virtual_sample_index]
        window = self.windows[window_index]
        source_latent_frames = int(window.latent_num_frames)
        start_padding_frames = self.resolve_start_padding_frames(
            self.data_config,
            window,
        )
        estimates = [
            self.estimate_segment_valid_action_steps(
                window=window,
                latent_start=latent_start,
                segment_length=segment_length,
            )
            for segment_length in self.eligible_segment_lengths(
                source_latent_frames=source_latent_frames,
                start_padding_frames=start_padding_frames,
            )
        ]
        return float(sum(estimates) / len(estimates))

    def estimate_mean_valid_action_steps(self) -> float:
        """Average positive action-row estimates over virtual slots."""

        estimates = [
            self.estimate_virtual_valid_action_steps(virtual_sample_index)
            for virtual_sample_index in range(len(self.virtual_index))
        ]
        positive = [value for value in estimates if value > 0]
        if not positive:
            return float(
                max(1, self.data_config.action_schema.action_horizon)
            )
        return float(sum(positive) / len(positive))

    def build_sample_weights(self) -> tuple[float, ...]:
        """Build one replacement-sampling weight per virtual slot."""

        return _build_local_latent_sample_weights(
            sample_config=self.data_config.sample_construction,
            item_count=len(self.virtual_index),
            dataset_mean_valid_action_steps=(
                self.dataset_mean_valid_action_steps
            ),
            dataset_mean_task_demo_count=self.dataset_mean_task_demo_count,
            valid_action_steps_for_index=(
                self.estimate_virtual_valid_action_steps
            ),
            task_text_for_index=lambda index: self.window_task_texts[
                self.virtual_index[index][0]
            ],
            task_demo_counts=self.task_demo_counts,
            task_virtual_start_counts=self.task_virtual_start_counts,
            dataset_mean_task_virtual_start_count=(
                self.dataset_mean_task_virtual_start_count
            ),
        )

    def sample_weight_metadata(self, index: int) -> dict[str, Any]:
        """Describe weighting inputs for one virtual sample."""

        window_index, _ = self.virtual_index[index]
        task_text = self.window_task_texts[window_index]
        sample_config = self.data_config.sample_construction
        return {
            "train_sample_weight": self.sample_weights[index],
            "train_sample_weight_mode": sample_config.sample_weight_mode,
            "eligible_task_demo_count": self.task_demo_counts[task_text],
            "dataset_mean_eligible_task_demo_count": (
                self.dataset_mean_task_demo_count
            ),
            "eligible_task_virtual_start_count": (
                self.task_virtual_start_counts[task_text]
            ),
            "dataset_mean_eligible_task_virtual_start_count": (
                self.dataset_mean_task_virtual_start_count
            ),
            "sample_weight_length_power": (
                sample_config.sample_weight_length_power
            ),
        }

    def sample_segment_geometry(
        self,
        *,
        index: int,
        source_latent_frames: int,
        virtual_latent_start: int,
        start_padding_frames: int = 0,
    ) -> tuple[int, int]:
        """Draw one segment length/start while preserving legacy RNG order."""

        sample_config = self.data_config.sample_construction
        start_padding_frames = max(0, int(start_padding_frames))
        candidates = self.eligible_segment_lengths(
            source_latent_frames=source_latent_frames,
            start_padding_frames=start_padding_frames,
        )
        if sample_config.randomize_segment_length:
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

        if sample_config.randomize_segment_start:
            min_start = -start_padding_frames
            if sample_config.require_full_segment:
                max_start = max(
                    min_start,
                    int(source_latent_frames) - int(segment_length),
                )
            else:
                max_start = max(min_start, int(source_latent_frames) - 1)
            latent_start = int(random.randint(min_start, max_start))
        else:
            latent_start = int(virtual_latent_start)
            if sample_config.require_full_segment:
                min_start = -start_padding_frames
                max_start = max(
                    min_start,
                    int(source_latent_frames) - int(segment_length),
                )
                latent_start = min(max(latent_start, min_start), max_start)
        return int(segment_length), int(latent_start)

    def sample_attention_geometry(
        self,
        *,
        segment_length: int,
    ) -> tuple[int, int]:
        """Draw chunk/window geometry while preserving legacy RNG order."""

        sample_config = self.data_config.sample_construction
        max_chunk_size = max(
            1,
            min(int(sample_config.chunk_size), int(segment_length)),
        )
        if bool(sample_config.randomize_geometry) and max_chunk_size > 1:
            sampled_chunk_size = int(random.randint(1, max_chunk_size))
        else:
            sampled_chunk_size = max_chunk_size

        max_window_size = max(1, int(sample_config.window_size))
        if bool(sample_config.randomize_geometry) and max_window_size >= 4:
            sampled_window_size = int(random.randint(4, max_window_size))
        else:
            sampled_window_size = max_window_size
        return sampled_chunk_size, sampled_window_size


def build_hierarchical_fixed_segment_task_specs(
    *,
    window_task_texts: Sequence[str],
    window_start_ranges_by_chunk: Sequence[
        Sequence[tuple[int, int, int, int]]
    ],
    task_demo_counts: Mapping[str, int],
    sample_config: SampleConstructionConfig,
) -> tuple[HierarchicalFixedSegmentTaskSpec, ...]:
    """Build deterministic task and trajectory mass tables."""

    window_specs_by_task: dict[
        str,
        list[HierarchicalFixedSegmentWindowSpec],
    ] = {}
    eligible_starts_by_task: Counter[str] = Counter()
    for window_index, task_text in enumerate(window_task_texts):
        for (
            sampled_chunk_size,
            start_min,
            start_max,
            eligible_start_count,
        ) in window_start_ranges_by_chunk[window_index]:
            if eligible_start_count <= 0:
                continue
            trajectory_mass = float(eligible_start_count) ** float(
                sample_config.trajectory_start_power
            )
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
        demo_count = max(1, int(task_demo_counts[task_text]))
        task_mass = (
            float(eligible_start_count) ** float(sample_config.task_start_power)
        ) * (float(demo_count) ** float(sample_config.demo_count_power))
        if task_mass <= 0.0:
            task_mass = 1.0
        windows = tuple(window_specs_by_task[task_text])
        window_mass_total = float(
            sum(window.mass_within_task for window in windows)
        )
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
        raise ValueError(
            "Hierarchical fixed-segment sampling found no eligible "
            "task/window starts."
        )
    return tuple(task_specs)


@dataclass(frozen=True)
class HierarchicalFixedSegmentSamplingPlan:
    """Resolved mass table and deterministic draw policy for one dataset."""

    task_specs: tuple[HierarchicalFixedSegmentTaskSpec, ...]
    task_weights: tuple[float, ...]
    task_mass_total: float
    task_specs_by_text: dict[str, HierarchicalFixedSegmentTaskSpec]
    epoch_sample_count: int

    @classmethod
    def from_task_specs(
        cls,
        task_specs: tuple[HierarchicalFixedSegmentTaskSpec, ...],
    ) -> HierarchicalFixedSegmentSamplingPlan:
        task_weights = tuple(float(task.task_mass) for task in task_specs)
        task_mass_total = float(sum(task_weights))
        task_specs_by_text = {task.task_text: task for task in task_specs}
        epoch_sample_count = sum(
            int(window_spec.eligible_start_count)
            for task_spec in task_specs
            for window_spec in task_spec.windows
        )
        if epoch_sample_count <= 0:
            raise ValueError(
                "Hierarchical fixed-segment sampling requires at least one "
                "eligible start."
            )
        return cls(
            task_specs=task_specs,
            task_weights=task_weights,
            task_mass_total=task_mass_total,
            task_specs_by_text=task_specs_by_text,
            epoch_sample_count=epoch_sample_count,
        )

    def draw(
        self,
        *,
        index: int,
        split_seed: int,
        split: DataSplit,
    ) -> tuple[
        HierarchicalFixedSegmentTaskSpec,
        HierarchicalFixedSegmentWindowSpec,
        int,
        int,
    ]:
        split_salt = 17 if split == DataSplit.TRAIN else 53
        draw = draw_hierarchical_sample_index(
            seed_values=(int(split_seed), split_salt, int(index)),
            task_weights=self.task_weights,
            task_specs=self.task_specs,
        )
        task_spec = self.task_specs[draw.task_index]
        window_spec = task_spec.windows[draw.window_index]
        return (
            task_spec,
            window_spec,
            draw.start,
            int(window_spec.sampled_chunk_size),
        )

    def iter_eligible_start_keys(self) -> Iterator[tuple[int, int, int]]:
        """Yield every trajectory/start/chunk key represented by the plan."""

        for task_spec in self.task_specs:
            for window_spec in task_spec.windows:
                for latent_start in range(
                    int(window_spec.start_min),
                    int(window_spec.start_max) + 1,
                ):
                    yield (
                        int(window_spec.window_index),
                        int(latent_start),
                        int(window_spec.sampled_chunk_size),
                    )

    def sample_metadata(
        self,
        *,
        index: int,
        task_spec: HierarchicalFixedSegmentTaskSpec,
        window_spec: HierarchicalFixedSegmentWindowSpec,
        sample_config: SampleConstructionConfig,
    ) -> dict[str, Any]:
        """Describe one resolved hierarchical draw without dataset payloads."""

        task_probability = float(task_spec.task_mass) / max(
            1e-12,
            self.task_mass_total,
        )
        trajectory_probability = float(window_spec.mass_within_task) / max(
            1e-12,
            task_spec.window_mass_total,
        )
        return {
            "hierarchical_global_sample_index": int(index),
            "hierarchical_task_text": task_spec.task_text,
            "hierarchical_task_start_power": float(
                sample_config.task_start_power
            ),
            "hierarchical_demo_count_power": float(
                sample_config.demo_count_power
            ),
            "hierarchical_trajectory_start_power": float(
                sample_config.trajectory_start_power
            ),
            "hierarchical_task_eligible_start_count": int(
                task_spec.eligible_start_count
            ),
            "hierarchical_task_demo_count": int(task_spec.demo_count),
            "hierarchical_task_mass": float(task_spec.task_mass),
            "hierarchical_task_probability": task_probability,
            "hierarchical_trajectory_eligible_start_count": int(
                window_spec.eligible_start_count
            ),
            "hierarchical_trajectory_mass": float(
                window_spec.mass_within_task
            ),
            "hierarchical_trajectory_probability_within_task": (
                trajectory_probability
            ),
            "hierarchical_start_min": int(window_spec.start_min),
            "hierarchical_start_max": int(window_spec.start_max),
            "hierarchical_start_count": int(window_spec.eligible_start_count),
            "hierarchical_task_count": int(len(self.task_specs)),
            "hierarchical_epoch_sample_count": int(self.epoch_sample_count),
            "context_prefix_policy": str(
                sample_config.context_prefix_policy
            ),
            "context_prefix_config_frames": int(
                sample_config.context_prefix_frames
            ),
            "target_alignment": str(sample_config.target_alignment),
            "rollout_context_policy": str(
                sample_config.rollout_context_policy
            ),
            "rollout_context_config_frames": (
                None
                if sample_config.rollout_context_frames is None
                else int(sample_config.rollout_context_frames)
            ),
            "tail_padding_policy": str(sample_config.tail_padding_policy),
            "padded_target_policy": str(sample_config.padded_target_policy),
        }
