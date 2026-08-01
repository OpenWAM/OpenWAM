"""Hierarchical mass and draw policy for local LeRobot latent datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from open_wam.configs import DataSplit, SampleConstructionConfig

from .distributed_sampling import draw_hierarchical_sample_index


__all__ = [
    "HierarchicalFixedSegmentSamplingPlan",
    "HierarchicalFixedSegmentTaskSpec",
    "HierarchicalFixedSegmentWindowSpec",
    "build_hierarchical_fixed_segment_task_specs",
]


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


def build_hierarchical_fixed_segment_task_specs(
    *,
    window_task_texts: Sequence[str],
    window_start_ranges_by_chunk: Sequence[Sequence[tuple[int, int, int, int]]],
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
        raise ValueError(
            "Hierarchical fixed-segment sampling found no eligible task/window starts."
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
            "hierarchical_task_start_power": float(sample_config.task_start_power),
            "hierarchical_demo_count_power": float(sample_config.demo_count_power),
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
            "hierarchical_trajectory_mass": float(window_spec.mass_within_task),
            "hierarchical_trajectory_probability_within_task": (trajectory_probability),
            "hierarchical_start_min": int(window_spec.start_min),
            "hierarchical_start_max": int(window_spec.start_max),
            "hierarchical_start_count": int(window_spec.eligible_start_count),
            "hierarchical_task_count": int(len(self.task_specs)),  # noqa: RUF046
            "hierarchical_epoch_sample_count": int(self.epoch_sample_count),
            "context_prefix_policy": str(sample_config.context_prefix_policy),
            "context_prefix_config_frames": int(sample_config.context_prefix_frames),
            "target_alignment": str(sample_config.target_alignment),
            "rollout_context_policy": str(sample_config.rollout_context_policy),
            "rollout_context_config_frames": (
                None
                if sample_config.rollout_context_frames is None
                else int(sample_config.rollout_context_frames)
            ),
            "tail_padding_policy": str(sample_config.tail_padding_policy),
            "padded_target_policy": str(sample_config.padded_target_policy),
        }
