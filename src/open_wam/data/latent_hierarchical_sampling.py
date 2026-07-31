"""Hierarchical fixed-segment planning for local latent trajectories."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from open_wam.configs import (
    DataConfig,
    RolloutContextPolicy,
    SampleTargetAlignment,
    SegmentContextPolicy,
)

from .latent_segment_geometry import (
    compact_boundary_start_range,
    resolve_compact_boundary_segment,
    resolve_rollout_parity_boundary_segment,
    rollout_parity_start_range,
)
from .lerobot_v2_latent_sampling import (
    HierarchicalFixedSegmentSamplingPlan,
    HierarchicalFixedSegmentTaskSpec,
    HierarchicalFixedSegmentWindowSpec,
    LocalLatentUniformSegmentSamplingPlan,
    build_hierarchical_fixed_segment_task_specs,
)
from .lerobot_v2_latent_storage import LocalEpisodeWindow


__all__ = [
    "LocalLatentHierarchicalSampleKey",
    "LocalLatentHierarchicalSegmentPlan",
]


@dataclass(frozen=True)
class LocalLatentHierarchicalSampleKey:
    """Resolved hierarchy and segment geometry for one sampler index."""

    epoch: int
    epoch_sample_index: int
    task_text: str
    trajectory_window_index: int
    latent_start: int
    start_min: int
    start_max: int
    window_eligible_start_count: int
    logical_frame_start: int
    logical_frame_end: int
    effective_frame_start: int
    effective_frame_end: int
    effective_segment_frames: int
    supervised_frame_start: int
    supervised_frame_end: int
    loss_frame_start: int
    loss_frame_end: int
    head_padded_frame_count: int
    tail_padded_frame_count: int
    context_prefix_policy: SegmentContextPolicy
    target_alignment: SampleTargetAlignment
    rollout_context_policy: RolloutContextPolicy
    context_prefix_frames_requested: int
    context_prefix_frames_in_sample: int
    context_prefix_real_frames: int
    context_prefix_truncated_frames: int
    chunk_size_for_boundary: int
    sampled_chunk_size: int
    sampled_window_size: int

    def as_metadata(self) -> dict[str, Any]:
        """Return the historical dictionary-shaped diagnostics contract."""

        return {
            "epoch": self.epoch,
            "epoch_sample_index": self.epoch_sample_index,
            "task_text": self.task_text,
            "trajectory_window_index": self.trajectory_window_index,
            "latent_start": self.latent_start,
            "start_min": self.start_min,
            "start_max": self.start_max,
            "window_eligible_start_count": self.window_eligible_start_count,
            "logical_frame_start": self.logical_frame_start,
            "logical_frame_end": self.logical_frame_end,
            "effective_frame_start": self.effective_frame_start,
            "effective_frame_end": self.effective_frame_end,
            "effective_segment_frames": self.effective_segment_frames,
            "supervised_frame_start": self.supervised_frame_start,
            "supervised_frame_end": self.supervised_frame_end,
            "loss_frame_start": self.loss_frame_start,
            "loss_frame_end": self.loss_frame_end,
            "head_padded_frame_count": self.head_padded_frame_count,
            "tail_padded_frame_count": self.tail_padded_frame_count,
            "context_prefix_policy": str(self.context_prefix_policy),
            "target_alignment": str(self.target_alignment),
            "rollout_context_policy": str(self.rollout_context_policy),
            "context_prefix_frames_requested": (
                self.context_prefix_frames_requested
            ),
            "context_prefix_frames_in_sample": (
                self.context_prefix_frames_in_sample
            ),
            "context_prefix_real_frames": self.context_prefix_real_frames,
            "context_prefix_truncated_frames": (
                self.context_prefix_truncated_frames
            ),
            "chunk_size_for_boundary": self.chunk_size_for_boundary,
            "sampled_chunk_size": self.sampled_chunk_size,
            "sampled_window_size": self.sampled_window_size,
        }


@dataclass(frozen=True)
class LocalLatentHierarchicalSegmentPlan:
    """Complete geometry and draw plan for local hierarchical segments."""

    data_config: DataConfig
    windows: tuple[LocalEpisodeWindow, ...]
    segment_frames: int
    chunk_size_candidates: tuple[int, ...]
    window_start_ranges_by_chunk: tuple[
        tuple[tuple[int, int, int, int], ...], ...
    ]
    sampling_plan: HierarchicalFixedSegmentSamplingPlan

    @classmethod
    def from_windows(
        cls,
        *,
        data_config: DataConfig,
        windows: Sequence[LocalEpisodeWindow],
        window_task_texts: Sequence[str],
        task_demo_counts: Mapping[str, int],
    ) -> LocalLatentHierarchicalSegmentPlan:
        """Resolve deterministic geometry and mass tables for a catalog."""

        segment_frames = data_config.sample_construction.segment_frames
        if segment_frames is None:
            raise ValueError(
                "Hierarchical fixed-segment sampling requires "
                "`sample_construction.segment_frames`."
            )
        window_tuple = tuple(windows)
        task_text_tuple = tuple(str(value) for value in window_task_texts)
        if len(task_text_tuple) != len(window_tuple):
            raise ValueError(
                "Hierarchical fixed-segment sampling requires one task label "
                f"per window, got task_labels={len(task_text_tuple)}, "
                f"windows={len(window_tuple)}."
            )
        resolved_segment_frames = int(segment_frames)
        chunk_size_candidates = cls.resolve_chunk_size_candidates(data_config)
        ranges_by_window: list[tuple[tuple[int, int, int, int], ...]] = []
        for window in window_tuple:
            source_latent_frames = max(1, int(window.latent_num_frames))
            window_ranges: list[tuple[int, int, int, int]] = []
            for chunk_size in chunk_size_candidates:
                if (
                    data_config.sample_construction.target_alignment
                    == SampleTargetAlignment.NEXT_AFTER_CONTEXT
                ):
                    start_min, start_max, eligible_start_count = (
                        rollout_parity_start_range(
                            source_latent_frames=source_latent_frames,
                        )
                    )
                else:
                    start_min, start_max, eligible_start_count = (
                        compact_boundary_start_range(
                            source_latent_frames=source_latent_frames,
                            segment_length=resolved_segment_frames,
                            start_padding_frames=(
                                LocalLatentUniformSegmentSamplingPlan.resolve_start_padding_frames(
                                    data_config,
                                    window,
                                )
                            ),
                            chunk_size=chunk_size,
                            context_prefix_frames=(
                                cls.resolve_context_prefix_frames(
                                    data_config=data_config,
                                    segment_frames=resolved_segment_frames,
                                    sampled_chunk_size=chunk_size,
                                )
                            ),
                        )
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
        window_start_ranges_by_chunk = tuple(ranges_by_window)
        task_specs = build_hierarchical_fixed_segment_task_specs(
            window_task_texts=task_text_tuple,
            window_start_ranges_by_chunk=window_start_ranges_by_chunk,
            task_demo_counts=task_demo_counts,
            sample_config=data_config.sample_construction,
        )
        return cls(
            data_config=data_config,
            windows=window_tuple,
            segment_frames=resolved_segment_frames,
            chunk_size_candidates=chunk_size_candidates,
            window_start_ranges_by_chunk=window_start_ranges_by_chunk,
            sampling_plan=HierarchicalFixedSegmentSamplingPlan.from_task_specs(
                task_specs
            ),
        )

    @staticmethod
    def resolve_chunk_size_candidates(
        data_config: DataConfig,
    ) -> tuple[int, ...]:
        """Resolve fixed or randomized chunk-size candidates."""

        sample_config = data_config.sample_construction
        max_chunk_size = max(1, int(sample_config.chunk_size))
        if (
            sample_config.target_alignment
            == SampleTargetAlignment.NEXT_AFTER_CONTEXT
        ):
            return (max_chunk_size,)
        if bool(sample_config.randomize_geometry) and max_chunk_size > 1:
            return tuple(range(1, max_chunk_size + 1))
        return (max_chunk_size,)

    @staticmethod
    def resolve_context_prefix_frames(
        *,
        data_config: DataConfig,
        segment_frames: int,
        sampled_chunk_size: int,
    ) -> int:
        """Resolve the clean prefix for one candidate chunk size."""

        sample_config = data_config.sample_construction
        if (
            sample_config.target_alignment
            == SampleTargetAlignment.NEXT_AFTER_CONTEXT
        ):
            if sample_config.rollout_context_frames is not None:
                return max(1, int(sample_config.rollout_context_frames))
            if (
                sample_config.rollout_context_policy
                == RolloutContextPolicy.ONE_FRAME
            ):
                return 1
            if (
                sample_config.rollout_context_policy
                == RolloutContextPolicy.ROLLOUT_HISTORY
            ):
                chunk_size = max(1, int(sampled_chunk_size))
                window_size = max(1, int(sample_config.window_size))
                history_chunks = max(1, int(math.ceil(window_size / 2.0)))
                return max(1, history_chunks * chunk_size)
            raise ValueError(
                "Unsupported rollout_context_policy: "
                f"{sample_config.rollout_context_policy!r}"
            )

        if sample_config.context_prefix_policy == SegmentContextPolicy.NONE:
            return 0
        if sample_config.context_prefix_policy == SegmentContextPolicy.FIXED:
            return max(0, int(sample_config.context_prefix_frames))
        if (
            sample_config.context_prefix_policy
            == SegmentContextPolicy.ROLLOUT_HISTORY
        ):
            chunk_size = max(1, int(sampled_chunk_size))
            window_size = max(1, int(sample_config.window_size))
            history_chunks = max(1, int(math.ceil(window_size / 2.0)))
            return max(
                0,
                min(
                    history_chunks * chunk_size,
                    int(segment_frames) - 1,
                ),
            )
        raise ValueError(
            "Unsupported context_prefix_policy: "
            f"{sample_config.context_prefix_policy!r}"
        )

    def context_prefix_frames(self, sampled_chunk_size: int) -> int:
        """Resolve context under this plan's fixed segment geometry."""

        return self.resolve_context_prefix_frames(
            data_config=self.data_config,
            segment_frames=self.segment_frames,
            sampled_chunk_size=sampled_chunk_size,
        )

    def draw(
        self,
        index: int,
    ) -> tuple[
        HierarchicalFixedSegmentTaskSpec,
        HierarchicalFixedSegmentWindowSpec,
        int,
        int,
    ]:
        """Resolve one deterministic hierarchy/start draw."""

        return self.sampling_plan.draw(
            index=index,
            split_seed=int(self.data_config.split_seed),
            split=self.data_config.split,
        )

    def resolve_sample_key(
        self,
        index: int,
    ) -> LocalLatentHierarchicalSampleKey:
        """Resolve one sampler index without loading trajectory tensors."""

        epoch, epoch_sample_index = divmod(
            int(index),
            self.sampling_plan.epoch_sample_count,
        )
        task_spec, window_spec, latent_start, sampled_chunk_size = self.draw(
            index
        )
        window = self.windows[int(window_spec.window_index)]
        source_latent_frames = max(1, int(window.latent_num_frames))
        context_prefix_frames = self.context_prefix_frames(sampled_chunk_size)
        sample_config = self.data_config.sample_construction
        if (
            sample_config.target_alignment
            == SampleTargetAlignment.NEXT_AFTER_CONTEXT
        ):
            boundary = resolve_rollout_parity_boundary_segment(
                source_latent_frames=source_latent_frames,
                latent_start=latent_start,
                target_frame_count=self.segment_frames,
                context_frames=context_prefix_frames,
                chunk_size=sampled_chunk_size,
            )
        else:
            boundary = resolve_compact_boundary_segment(
                source_latent_frames=source_latent_frames,
                latent_start=latent_start,
                segment_length=self.segment_frames,
                start_padding_frames=(
                    LocalLatentUniformSegmentSamplingPlan.resolve_start_padding_frames(
                        self.data_config,
                        window,
                    )
                ),
                chunk_size=sampled_chunk_size,
                context_prefix_frames=context_prefix_frames,
            )
        return LocalLatentHierarchicalSampleKey(
            epoch=int(epoch),
            epoch_sample_index=int(epoch_sample_index),
            task_text=task_spec.task_text,
            trajectory_window_index=int(window_spec.window_index),
            latent_start=int(latent_start),
            start_min=int(window_spec.start_min),
            start_max=int(window_spec.start_max),
            window_eligible_start_count=int(
                window_spec.eligible_start_count
            ),
            logical_frame_start=int(boundary["logical_frame_start"]),
            logical_frame_end=int(boundary["logical_frame_end"]),
            effective_frame_start=int(boundary["effective_frame_start"]),
            effective_frame_end=int(boundary["effective_frame_end"]),
            effective_segment_frames=int(
                boundary["effective_segment_frames"]
            ),
            supervised_frame_start=int(boundary["supervised_start"]),
            supervised_frame_end=int(boundary["supervised_end"]),
            loss_frame_start=int(boundary["loss_frame_start"]),
            loss_frame_end=int(boundary["loss_frame_end"]),
            head_padded_frame_count=int(boundary["head_padded_frame_count"]),
            tail_padded_frame_count=int(boundary["tail_padded_frame_count"]),
            context_prefix_policy=sample_config.context_prefix_policy,
            target_alignment=sample_config.target_alignment,
            rollout_context_policy=sample_config.rollout_context_policy,
            context_prefix_frames_requested=int(
                boundary["context_prefix_frames_requested"]
            ),
            context_prefix_frames_in_sample=int(
                boundary["context_prefix_frames_in_sample"]
            ),
            context_prefix_real_frames=int(
                boundary["context_prefix_real_frames"]
            ),
            context_prefix_truncated_frames=int(
                boundary["context_prefix_truncated_frames"]
            ),
            chunk_size_for_boundary=int(boundary["chunk_size_for_boundary"]),
            sampled_chunk_size=int(sampled_chunk_size),
            sampled_window_size=max(1, int(sample_config.window_size)),
        )

    def iter_eligible_start_keys(self) -> Iterator[tuple[int, int, int]]:
        """Yield every trajectory/start/chunk key represented by the plan."""

        return self.sampling_plan.iter_eligible_start_keys()

    def sample_metadata(
        self,
        *,
        index: int,
        task_spec: HierarchicalFixedSegmentTaskSpec,
        window_spec: HierarchicalFixedSegmentWindowSpec,
    ) -> dict[str, Any]:
        """Describe one hierarchy draw using the stable metadata contract."""

        return self.sampling_plan.sample_metadata(
            index=index,
            task_spec=task_spec,
            window_spec=window_spec,
            sample_config=self.data_config.sample_construction,
        )
