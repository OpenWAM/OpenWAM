"""Tensor-free causal prefix/suffix planning for encoded latent timelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import random

from open_wam.configs import (
    CausalPrefixSuffixBucketConfig,
    DataConfig,
    DataSplit,
    LatentTemporalLayout,
    SampleConstructionConfig,
)

from .latent_temporal import (
    latent_raw_boundaries,
    observed_frame_ids_for_latent_segment,
    raw_span_for_latent_range,
)


__all__ = [
    "LatentCausalPrefixSuffixCandidate",
    "LatentCausalPrefixSuffixWindowPlan",
    "LatentCausalPrefixSuffixWindowPlanner",
]


@dataclass(frozen=True)
class LatentCausalPrefixSuffixCandidate:
    """One eligible latent start and configured causal bucket."""

    latent_start: int
    bucket_index: int


@dataclass(frozen=True)
class LatentCausalPrefixSuffixWindowPlan:
    """Resolved causal latent window and its raw-frame coordinates."""

    latent_start: int
    latent_end: int
    raw_start_position: int
    raw_end_position: int
    sample_start_frame: int
    sample_end_frame: int
    observed_frame_ids: tuple[int, ...]
    observed_prefix_frames: int
    future_suffix_frames: int
    valid_video_frames: int
    padded_video_frames: int
    latent_temporal_layout: LatentTemporalLayout


@dataclass(frozen=True)
class LatentCausalPrefixSuffixWindowPlanner:
    """Own causal candidate geometry and split-aware draw order."""

    sample_config: SampleConstructionConfig
    latent_temporal_layout: LatentTemporalLayout
    split: DataSplit
    split_seed: int

    @classmethod
    def from_data_config(
        cls,
        data_config: DataConfig,
    ) -> LatentCausalPrefixSuffixWindowPlanner:
        return cls(
            sample_config=data_config.sample_construction,
            latent_temporal_layout=data_config.latent_temporal_layout,
            split=data_config.split,
            split_seed=int(data_config.split_seed),
        )

    @property
    def buckets(self) -> tuple[CausalPrefixSuffixBucketConfig, ...]:
        return tuple(self.sample_config.causal_prefix_suffix_buckets)

    def build_candidates(
        self,
        *,
        raw_frame_ids: Sequence[int],
        source_latent_frames: int,
        row_count: int,
    ) -> tuple[LatentCausalPrefixSuffixCandidate, ...]:
        """Enumerate every bucket/start pair with a valid raw-frame span."""

        buckets = self.buckets
        if not buckets:
            raise ValueError(
                "Causal prefix/suffix sampling requires non-empty "
                "`sample_construction.causal_prefix_suffix_buckets`."
            )
        raw_bucket_boundaries = latent_raw_boundaries(
            raw_frame_count=len(raw_frame_ids),
            latent_num_frames=int(source_latent_frames),
            layout=self.latent_temporal_layout,
        )
        candidates: list[LatentCausalPrefixSuffixCandidate] = []
        for bucket_index, bucket in enumerate(buckets):
            total_frames = int(bucket.total_frames)
            if total_frames > int(source_latent_frames):
                continue
            max_latent_start = int(source_latent_frames) - total_frames
            for latent_start in range(max_latent_start + 1):
                latent_end = latent_start + total_frames
                raw_start_position = raw_bucket_boundaries[latent_start]
                raw_end_position = raw_bucket_boundaries[latent_end]
                if (
                    raw_start_position >= len(raw_frame_ids)
                    or raw_end_position <= raw_start_position
                ):
                    continue
                sample_end_frame = (
                    int(
                        raw_frame_ids[
                            max(raw_start_position, raw_end_position - 1)
                        ]
                    )
                    + 1
                )
                if sample_end_frame > int(row_count):
                    continue
                candidates.append(
                    LatentCausalPrefixSuffixCandidate(
                        latent_start=latent_start,
                        bucket_index=bucket_index,
                    )
                )
        return tuple(candidates)

    def select_candidate(
        self,
        candidates: Sequence[LatentCausalPrefixSuffixCandidate],
        *,
        sample_index: int,
    ) -> LatentCausalPrefixSuffixCandidate:
        """Select one candidate while preserving the historical RNG contract."""

        if not candidates:
            raise ValueError(
                "Causal prefix/suffix candidate selection requires at least "
                "one eligible candidate."
            )
        if self.split == DataSplit.TRAIN:
            rng = random.Random(
                random.randrange(1 << 30) + int(sample_index)
            )
        else:
            rng = random.Random(self.split_seed + int(sample_index))
        return candidates[rng.randrange(len(candidates))]

    def plan(
        self,
        *,
        raw_frame_ids: Sequence[int],
        source_latent_frames: int,
        row_count: int,
        sample_index: int,
    ) -> LatentCausalPrefixSuffixWindowPlan | None:
        """Resolve one causal window, returning `None` when none is eligible."""

        candidates = self.build_candidates(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=source_latent_frames,
            row_count=row_count,
        )
        if not candidates:
            return None
        candidate = self.select_candidate(
            candidates,
            sample_index=sample_index,
        )
        bucket = self.buckets[candidate.bucket_index]
        total_frames = int(bucket.total_frames)
        latent_end = int(candidate.latent_start) + total_frames
        (
            raw_start_position,
            raw_end_position,
            sample_start_frame,
            sample_end_frame,
        ) = raw_span_for_latent_range(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=int(source_latent_frames),
            latent_start=int(candidate.latent_start),
            latent_end=latent_end,
            layout=self.latent_temporal_layout,
        )
        observed_frame_ids = observed_frame_ids_for_latent_segment(
            raw_frame_ids=raw_frame_ids,
            source_latent_frames=int(source_latent_frames),
            latent_start=int(candidate.latent_start),
            segment_length=total_frames,
            layout=self.latent_temporal_layout,
        )
        return LatentCausalPrefixSuffixWindowPlan(
            latent_start=int(candidate.latent_start),
            latent_end=latent_end,
            raw_start_position=int(raw_start_position),
            raw_end_position=int(raw_end_position),
            sample_start_frame=int(sample_start_frame),
            sample_end_frame=int(sample_end_frame),
            observed_frame_ids=tuple(int(value) for value in observed_frame_ids),
            observed_prefix_frames=int(bucket.observed_frames),
            future_suffix_frames=int(bucket.future_frames),
            valid_video_frames=total_frames,
            padded_video_frames=int(self.sample_config.num_frames),
            latent_temporal_layout=self.latent_temporal_layout,
        )
