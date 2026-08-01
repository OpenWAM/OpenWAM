"""Uniform-segment sampling policy for local LeRobot latent datasets."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from open_wam.configs import DataConfig, DataSplit, SampleWeightMode

from .latent_temporal import raw_span_for_latent_range
from .lerobot_v2_latent_storage import LocalEpisodeWindow
from .lerobot_v2_latent_weighting import _build_local_latent_sample_weights


__all__ = ["LocalLatentUniformSegmentSamplingPlan"]


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
        segment_length_candidates = cls.resolve_segment_length_candidates(data_config)
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
            dataset_mean_task_virtual_start_count=(mean_task_virtual_start_count),
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
                window_index, _ = self.virtual_index[int(virtual_sample_index)]
                per_window.setdefault(window_index, []).append(
                    int(virtual_sample_index)
                )
            for indices in per_window.values():
                rng.shuffle(indices)

        window_order = list(per_window)
        rng.shuffle(window_order)
        block_size = max(
            1,
            int(self.data_config.sample_construction.segment_locality_block_size),
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
        if (
            self.resolve_start_padding_frames(self.data_config, window) > 0
            and latent_start <= 0
        ):
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
            return float(max(1, self.data_config.action_schema.action_horizon))
        return float(sum(positive) / len(positive))

    def build_sample_weights(self) -> tuple[float, ...]:
        """Build one replacement-sampling weight per virtual slot."""

        return _build_local_latent_sample_weights(
            sample_config=self.data_config.sample_construction,
            item_count=len(self.virtual_index),
            dataset_mean_valid_action_steps=(self.dataset_mean_valid_action_steps),
            dataset_mean_task_demo_count=self.dataset_mean_task_demo_count,
            valid_action_steps_for_index=(self.estimate_virtual_valid_action_steps),
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
            "sample_weight_length_power": (sample_config.sample_weight_length_power),
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
