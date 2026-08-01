"""Deterministic episode-selection contracts for sampled evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
import math
from pathlib import Path
import random
from typing import Any

from open_wam.data.replay_status import (
    ReplayStatusFilterReport,
    filter_episode_indices_by_replay_status,
    normalize_replay_status_policy,
)


class SampledEvalMode(StrEnum):
    """Episode-axis policy used to construct a sampled evaluation run."""

    DATASET_DISTRIBUTION = "dataset_distribution"
    TASK_EPISODE_AXIS = "task_episode_axis"
    FULL = "full"


class DistributionEpisodeStrategy(StrEnum):
    """Task-local selection policy after proportional allocation."""

    FIRST = "first"
    RANDOM = "random"
    EVENLY_SPACED = "evenly_spaced"


class TaskAxisInitSource(StrEnum):
    """Source of the simulator init id attached to a sampled episode."""

    AUTO = "auto"
    REPLAY_STATUS = "replay_status"
    TASK_LOCAL = "task_local"


_SAMPLE_MODE_ALIASES = {
    "uniform_task_distribution": SampledEvalMode.DATASET_DISTRIBUTION.value,
}
SAMPLE_MODE_CHOICES = (
    SampledEvalMode.DATASET_DISTRIBUTION.value,
    "uniform_task_distribution",
    SampledEvalMode.TASK_EPISODE_AXIS.value,
    SampledEvalMode.FULL.value,
)


__all__ = [
    "DatasetEpisode",
    "DistributionEpisodeStrategy",
    "SAMPLE_MODE_CHOICES",
    "SampledEvalMode",
    "TaskAxisInitSource",
    "allocate_proportional_counts",
    "attach_replay_status_to_dataset_episodes",
    "build_dataset_episodes",
    "build_replay_status_warnings",
    "build_sample_warnings",
    "evenly_spaced_indices",
    "filter_dataset_episodes_by_replay_status",
    "normalize_sample_mode",
    "parse_int_selector",
    "sample_episodes_by_task_distribution",
    "select_distribution_task_episodes",
    "select_full_task_init_axis",
    "select_sampled_episodes",
    "select_task_episode_axis",
    "uses_replay_resolved_init_ids",
]


@dataclass(frozen=True)
class DatasetEpisode:
    """Dataset episode coordinates resolved onto one benchmark task/init axis."""

    dataset_episode_index: int
    task_text: str
    task_index: int | None
    task_id: int
    task_name: str | None
    episode_idx: int
    length: int
    replay_status: str | None = None
    episode_id: int | None = None
    init_id: int | None = None
    resolved_init_state_index: int | None = None
    init_id_source: str = "task_local_rank"

    def __post_init__(self) -> None:
        if self.episode_id is None:
            object.__setattr__(self, "episode_id", int(self.dataset_episode_index))
        if self.init_id is None:
            object.__setattr__(self, "init_id", int(self.episode_idx))


def normalize_sample_mode(mode: SampledEvalMode | str) -> str:
    """Normalize legacy aliases while preserving the string-facing CLI API."""

    raw_mode = mode.value if isinstance(mode, SampledEvalMode) else str(mode)
    return _SAMPLE_MODE_ALIASES.get(raw_mode, raw_mode)


def build_dataset_episodes(
    episode_records: Sequence[Mapping[str, Any]],
    *,
    task_text_to_index: Mapping[str, int],
    task_text_to_task_id: Mapping[str, int],
    task_text_to_task_name: Mapping[str, str | None],
) -> list[DatasetEpisode]:
    """Resolve raw LeRobot episode records into stable task-local coordinates."""

    task_counts: dict[str, int] = defaultdict(int)
    episodes: list[DatasetEpisode] = []
    for record in sorted(episode_records, key=lambda item: int(item["episode_index"])):
        tasks = record.get("tasks") or ()
        if not tasks:
            raise ValueError(f"Episode {record.get('episode_index')} has no task text.")
        task_text = str(tasks[0])
        if task_text not in task_text_to_task_id:
            raise ValueError(f"Task text {task_text!r} has no resolved LIBERO task id.")
        task_local_rank = task_counts[task_text]
        task_counts[task_text] += 1
        episodes.append(
            DatasetEpisode(
                dataset_episode_index=int(record["episode_index"]),
                task_text=task_text,
                task_index=task_text_to_index.get(task_text),
                task_id=int(task_text_to_task_id[task_text]),
                task_name=task_text_to_task_name.get(task_text),
                episode_idx=task_local_rank,
                length=int(record.get("length", 0)),
            )
        )
    return episodes


def _replay_resolved_init_state_index(record: Any) -> int | None:
    raw = getattr(record, "raw", None)
    if isinstance(raw, Mapping):
        value = raw.get("resolved_init_state_index")
        if value is not None:
            return int(value)
    value = getattr(record, "resolved_init_state_index", None)
    return None if value is None else int(value)


def attach_replay_status_to_dataset_episodes(
    episodes: Sequence[DatasetEpisode],
    replay_status_records: Mapping[int, Any],
    *,
    use_resolved_init_ids: bool = False,
) -> list[DatasetEpisode]:
    """Attach labels and optional simulator init ids without changing episode rank."""

    if not replay_status_records:
        return list(episodes)
    attached: list[DatasetEpisode] = []
    for episode in episodes:
        record = replay_status_records.get(episode.dataset_episode_index)
        if record is None:
            attached.append(replace(episode, replay_status=None))
            continue
        resolved_init_state_index = _replay_resolved_init_state_index(record)
        updates: dict[str, Any] = {
            "replay_status": record.replay_status,
            "resolved_init_state_index": resolved_init_state_index,
        }
        if use_resolved_init_ids and resolved_init_state_index is not None:
            updates["init_id"] = resolved_init_state_index
            updates["init_id_source"] = "replay_status.resolved_init_state_index"
        attached.append(replace(episode, **updates))
    return attached


def uses_replay_resolved_init_ids(
    *,
    sample_mode: SampledEvalMode | str,
    task_axis_init_source: TaskAxisInitSource | str = TaskAxisInitSource.AUTO,
) -> bool:
    """Resolve whether replay metadata overrides task-local simulator init ids."""

    raw_source = (
        task_axis_init_source.value
        if isinstance(task_axis_init_source, TaskAxisInitSource)
        else str(task_axis_init_source)
    )
    try:
        source = TaskAxisInitSource(raw_source)
    except ValueError as exc:
        raise ValueError(f"Unsupported task_axis_init_source={raw_source!r}.") from exc
    if source is TaskAxisInitSource.TASK_LOCAL:
        return False
    if source is TaskAxisInitSource.REPLAY_STATUS:
        return True
    return normalize_sample_mode(sample_mode) != SampledEvalMode.FULL.value


def filter_dataset_episodes_by_replay_status(
    episodes: Sequence[DatasetEpisode],
    *,
    policy: str,
    replay_status_records: Mapping[int, Any],
    require_replay_status: bool,
    source_path: Path | None,
    task_axis_validation: bool = False,
) -> tuple[list[DatasetEpisode], ReplayStatusFilterReport]:
    """Apply one replay-status policy while preserving selected episode order."""

    normalized_policy = normalize_replay_status_policy(policy)
    selected_indices = [episode.dataset_episode_index for episode in episodes]
    kept_indices, report = filter_episode_indices_by_replay_status(
        selected_indices,
        replay_status_records=replay_status_records,
        policy=normalized_policy,
        require_labeled=bool(replay_status_records) or bool(require_replay_status),
        source_path=source_path,
    )
    kept_index_set = set(kept_indices)
    kept_episodes = [episode for episode in episodes if episode.dataset_episode_index in kept_index_set]
    if task_axis_validation and len(kept_episodes) != len(episodes):
        failed = [
            f"task_id={episode.task_id},init_id={episode.init_id},"
            f"dataset_episode_index={episode.dataset_episode_index}"
            for episode in episodes
            if episode.dataset_episode_index not in kept_index_set
        ]
        preview = ", ".join(failed[:10])
        suffix = "" if len(failed) <= 10 else f", ... ({len(failed)} filtered total)"
        raise ValueError(
            f"Requested task/init-axis episodes do not satisfy replay_status_policy={normalized_policy!r}: "
            f"{preview}{suffix}"
        )
    return kept_episodes, report


def select_sampled_episodes(
    episodes: Sequence[DatasetEpisode],
    *,
    mode: SampledEvalMode | str,
    count: int,
    seed: int,
    task_ids: str | None = None,
    episode_indices: str | None = None,
    distribution_episode_strategy: DistributionEpisodeStrategy | str = DistributionEpisodeStrategy.FIRST,
    full_init_counts_by_task_id: Mapping[int, int] | None = None,
) -> tuple[list[DatasetEpisode], dict[str, int]]:
    """Select episodes through one canonical sampled-evaluation mode."""

    normalized_mode = normalize_sample_mode(mode)
    if normalized_mode == SampledEvalMode.DATASET_DISTRIBUTION.value:
        if task_ids is not None or episode_indices is not None:
            raise ValueError("--task-ids/--episode-indices require --sample-mode task_episode_axis or full.")
        return sample_episodes_by_task_distribution(
            episodes,
            count=count,
            seed=seed,
            episode_strategy=distribution_episode_strategy,
        )
    if normalized_mode == SampledEvalMode.TASK_EPISODE_AXIS.value:
        return select_task_episode_axis(
            episodes,
            count=count,
            task_ids=task_ids,
            episode_indices=episode_indices,
        )
    if normalized_mode == SampledEvalMode.FULL.value:
        if full_init_counts_by_task_id is None:
            raise ValueError("--sample-mode full requires benchmark init-state counts.")
        return select_full_task_init_axis(
            episodes,
            init_counts_by_task_id=full_init_counts_by_task_id,
            task_ids=task_ids,
            episode_indices=episode_indices,
        )
    raise ValueError(f"Unsupported sample mode: {mode!r}")


def select_task_episode_axis(
    episodes: Sequence[DatasetEpisode],
    *,
    count: int,
    task_ids: str | None,
    episode_indices: str | None,
) -> tuple[list[DatasetEpisode], dict[str, int]]:
    """Select an init-major task/episode grid, optionally from explicit axes."""

    selected_task_ids = (
        parse_int_selector(task_ids) if task_ids is not None else sorted({int(item.task_id) for item in episodes})
    )
    if not selected_task_ids:
        raise ValueError("--task-ids did not select any task ids.")

    if episode_indices is None:
        by_task: dict[int, list[DatasetEpisode]] = defaultdict(list)
        for episode in episodes:
            by_task[int(episode.task_id)].append(episode)
        missing_task_ids = [task_id for task_id in selected_task_ids if task_id not in by_task]
        if missing_task_ids:
            preview = ", ".join(str(task_id) for task_id in missing_task_ids[:10])
            suffix = "" if len(missing_task_ids) <= 10 else f", ... ({len(missing_task_ids)} missing total)"
            raise ValueError(f"Requested LIBERO task ids are not present in dataset metadata: {preview}{suffix}")
        for task_episodes in by_task.values():
            task_episodes.sort(key=lambda item: (item.episode_idx, item.dataset_episode_index))

        available = sum(len(by_task[task_id]) for task_id in selected_task_ids)
        if count > available:
            raise ValueError(
                f"Cannot select {count} task/init-axis episodes from only {available} eligible episodes "
                "for the selected tasks."
            )

        selected: list[DatasetEpisode] = []
        max_task_episodes = max(len(by_task[task_id]) for task_id in selected_task_ids)
        for task_local_rank in range(max_task_episodes):
            for task_id in selected_task_ids:
                task_episodes = by_task[task_id]
                if task_local_rank >= len(task_episodes):
                    continue
                selected.append(task_episodes[task_local_rank])
                if len(selected) >= count:
                    allocations: dict[str, int] = defaultdict(int)
                    for episode in selected:
                        allocations[episode.task_text] += 1
                    return selected, dict(allocations)

        raise RuntimeError("task/init-axis selection exhausted eligible episodes before reaching requested count.")

    selected_episode_indices = parse_int_selector(episode_indices)
    if not selected_episode_indices:
        raise ValueError("--episode-indices did not select any episode indices.")

    by_key = {(int(episode.task_id), int(episode.episode_idx)): episode for episode in episodes}
    selected: list[DatasetEpisode] = []
    missing: list[str] = []
    for episode_idx in selected_episode_indices:
        for task_id in selected_task_ids:
            episode = by_key.get((int(task_id), int(episode_idx)))
            if episode is None:
                missing.append(f"task_id={task_id},episode_idx={episode_idx}")
                continue
            selected.append(episode)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing total)"
        raise ValueError(f"Requested LIBERO task/episode pairs are not present in dataset metadata: {preview}{suffix}")

    allocations: dict[str, int] = defaultdict(int)
    for episode in selected:
        allocations[episode.task_text] += 1
    selected.sort(key=lambda item: (item.episode_idx, item.task_id, item.dataset_episode_index))
    return selected, dict(allocations)


def select_full_task_init_axis(
    episodes: Sequence[DatasetEpisode],
    *,
    init_counts_by_task_id: Mapping[int, int],
    task_ids: str | None,
    episode_indices: str | None,
) -> tuple[list[DatasetEpisode], dict[str, int]]:
    """Enumerate every dataset-backed init in the requested benchmark grid."""

    if episode_indices is not None:
        raise ValueError("--episode-indices is not used with --sample-mode full; full enumerates every init id.")
    if not init_counts_by_task_id:
        raise ValueError("--sample-mode full requires at least one benchmark task/init count.")
    selected_task_ids = parse_int_selector(task_ids) if task_ids is not None else sorted(init_counts_by_task_id)
    if not selected_task_ids:
        raise ValueError("--task-ids did not select any task ids.")

    for task_id in selected_task_ids:
        if task_id not in init_counts_by_task_id:
            available = ", ".join(str(item) for item in sorted(init_counts_by_task_id))
            raise ValueError(
                f"Task id {task_id} is not available for --sample-mode full; "
                f"available task ids: {available}"
            )
        init_count = int(init_counts_by_task_id[task_id])
        if init_count <= 0:
            raise ValueError(f"Task id {task_id} has no LIBERO init states.")

    by_key = {(int(episode.task_id), int(episode.init_id)): episode for episode in episodes}
    selected: list[DatasetEpisode] = []
    missing: list[str] = []
    max_init_count = max(int(init_counts_by_task_id[task_id]) for task_id in selected_task_ids)
    for init_id in range(max_init_count):
        for task_id in selected_task_ids:
            init_count = int(init_counts_by_task_id[task_id])
            if init_id >= init_count:
                continue
            episode = by_key.get((int(task_id), int(init_id)))
            if episode is None:
                missing.append(f"task_id={task_id},init_id={init_id}")
                continue
            selected.append(episode)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing total)"
        raise ValueError(
            "Full LIBERO task/init grid is not present in dataset metadata: "
            f"{preview}{suffix}. Use task_episode_axis for a partial grid or refresh the dataset metadata."
        )

    allocations: dict[str, int] = defaultdict(int)
    for episode in selected:
        allocations[episode.task_text] += 1
    selected.sort(key=lambda item: (item.init_id, item.task_id, item.dataset_episode_index))
    return selected, dict(allocations)


def parse_int_selector(value: str) -> list[int]:
    """Parse unique non-negative integers and Python-style half-open ranges."""

    selected: list[int] = []
    seen: set[int] = set()
    for raw_piece in value.split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        if ":" in piece:
            parts = piece.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid integer range selector {piece!r}.")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step == 0:
                raise ValueError(f"Invalid integer range selector {piece!r}: step cannot be zero.")
            values = range(start, stop, step)
        else:
            values = (int(piece),)
        for item in values:
            if item < 0:
                raise ValueError(f"Negative indices are not supported: {item}")
            if item in seen:
                continue
            seen.add(item)
            selected.append(item)
    return selected


def sample_episodes_by_task_distribution(
    episodes: Sequence[DatasetEpisode],
    *,
    count: int,
    seed: int,
    episode_strategy: DistributionEpisodeStrategy | str = DistributionEpisodeStrategy.FIRST,
) -> tuple[list[DatasetEpisode], dict[str, int]]:
    """Allocate by task population, then select deterministically within tasks."""

    if count > len(episodes):
        raise ValueError(f"Cannot sample {count} episodes without replacement from only {len(episodes)} episodes.")
    raw_strategy = (
        episode_strategy.value
        if isinstance(episode_strategy, DistributionEpisodeStrategy)
        else str(episode_strategy)
    )
    try:
        strategy = DistributionEpisodeStrategy(raw_strategy)
    except ValueError as exc:
        raise ValueError(f"Unsupported dataset distribution episode strategy: {raw_strategy!r}") from exc

    by_task: dict[str, list[DatasetEpisode]] = defaultdict(list)
    for episode in episodes:
        by_task[episode.task_text].append(episode)
    for task_episodes in by_task.values():
        task_episodes.sort(key=lambda item: item.dataset_episode_index)

    task_order = sorted(by_task, key=lambda text: (by_task[text][0].task_id, text))
    allocations = allocate_proportional_counts(
        {task_text: len(task_episodes) for task_text, task_episodes in by_task.items()},
        total=count,
        tie_break_order=task_order,
    )
    rng = random.Random(seed)
    selected: list[DatasetEpisode] = []
    for task_text in task_order:
        task_count = allocations[task_text]
        if task_count <= 0:
            continue
        selected.extend(
            select_distribution_task_episodes(
                by_task[task_text],
                count=task_count,
                strategy=strategy,
                rng=rng,
            )
        )
    selected.sort(key=lambda item: (item.task_id, item.episode_idx, item.dataset_episode_index))
    return selected, allocations


def select_distribution_task_episodes(
    episodes: Sequence[DatasetEpisode],
    *,
    count: int,
    strategy: DistributionEpisodeStrategy | str,
    rng: random.Random,
) -> list[DatasetEpisode]:
    """Select one task's episodes through the requested local strategy."""

    if count > len(episodes):
        raise ValueError(f"Cannot select {count} task episodes from only {len(episodes)} candidates.")
    raw_strategy = strategy.value if isinstance(strategy, DistributionEpisodeStrategy) else str(strategy)
    try:
        resolved_strategy = DistributionEpisodeStrategy(raw_strategy)
    except ValueError as exc:
        raise ValueError(f"Unsupported dataset distribution episode strategy: {raw_strategy!r}") from exc
    if resolved_strategy is DistributionEpisodeStrategy.FIRST:
        return list(episodes[:count])
    if resolved_strategy is DistributionEpisodeStrategy.RANDOM:
        return sorted(rng.sample(list(episodes), count), key=lambda item: item.episode_idx)
    if resolved_strategy is DistributionEpisodeStrategy.EVENLY_SPACED:
        return [episodes[index] for index in evenly_spaced_indices(len(episodes), count)]
    raise AssertionError(f"Unhandled distribution episode strategy: {resolved_strategy.value}")


def evenly_spaced_indices(population: int, count: int) -> list[int]:
    """Return inclusive endpoint indices spread across one finite population."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    if count > population:
        raise ValueError("count cannot exceed population.")
    if count == 0:
        return []
    if count == 1:
        return [0]
    return sorted({round(index * (population - 1) / (count - 1)) for index in range(count)})


def allocate_proportional_counts(
    group_sizes: Mapping[str, int],
    *,
    total: int,
    tie_break_order: Sequence[str] | None = None,
) -> dict[str, int]:
    """Allocate an exact total by largest remainder without oversampling groups."""

    if total < 0:
        raise ValueError("total must be non-negative.")
    if total > sum(group_sizes.values()):
        raise ValueError("total cannot exceed the sum of group sizes.")
    if any(size < 0 for size in group_sizes.values()):
        raise ValueError("group sizes must be non-negative.")

    usable = {key: size for key, size in group_sizes.items() if size > 0}
    if total and not usable:
        raise ValueError("cannot allocate positive total across empty groups.")
    population = sum(usable.values())
    ideals = {key: (total * size / population) for key, size in usable.items()}
    allocations = {key: min(int(math.floor(ideal)), usable[key]) for key, ideal in ideals.items()}
    remaining = total - sum(allocations.values())
    ordered_keys = tuple(tie_break_order) if tie_break_order is not None else tuple(group_sizes)
    tie_rank = {key: index for index, key in enumerate(ordered_keys)}

    def priority(key: str) -> tuple[float, int, int]:
        return (ideals[key] - math.floor(ideals[key]), usable[key], -tie_rank.get(key, len(tie_rank)))

    while remaining > 0:
        candidates = [key for key in usable if allocations[key] < usable[key]]
        if not candidates:
            raise RuntimeError("proportional allocation exhausted all groups before reaching total.")
        for key in sorted(candidates, key=priority, reverse=True):
            if remaining <= 0:
                break
            allocations[key] += 1
            remaining -= 1

    return {key: allocations.get(key, 0) for key in group_sizes}


def build_sample_warnings(
    *,
    mode: SampledEvalMode | str,
    requested_count: int,
    task_allocations: Mapping[str, int],
    distribution_episode_strategy: DistributionEpisodeStrategy | str,
) -> list[str]:
    """Describe coverage caveats implied by one selected episode set."""

    normalized_mode = normalize_sample_mode(mode)
    if normalized_mode == SampledEvalMode.FULL.value:
        if not task_allocations:
            return []
        return [
            "`--sample-mode full` ignores --num-episodes and enumerates every benchmark task/init pair "
            "before replay-status policy validation."
        ]
    if normalized_mode != SampledEvalMode.DATASET_DISTRIBUTION.value or not task_allocations:
        return []
    warnings: list[str] = []
    task_count = len(task_allocations)
    covered_task_count = sum(1 for count in task_allocations.values() if count > 0)
    if requested_count < task_count:
        warnings.append(
            f"Requested {requested_count} sampled episodes across {task_count} tasks; only "
            f"{covered_task_count} tasks are covered. Increase --num-episodes to at least {task_count} "
            "for a cross-task smoke run."
        )
    raw_strategy = (
        distribution_episode_strategy.value
        if isinstance(distribution_episode_strategy, DistributionEpisodeStrategy)
        else str(distribution_episode_strategy)
    )
    if raw_strategy == DistributionEpisodeStrategy.RANDOM.value:
        warnings.append(
            "`--distribution-episode-strategy random` samples arbitrary task-local init states and is "
            "not directly comparable to #77-style episode_idx prefix parity runs."
        )
    return warnings


def build_replay_status_warnings(report: ReplayStatusFilterReport | None) -> list[str]:
    """Describe replay-label coverage and filtering decisions."""

    if report is None:
        return []
    warnings: list[str] = []
    if report.missing_status_file and report.policy != "include_all":
        warnings.append(
            f"Replay-status policy {report.policy!r} was requested, but no replay-status file was found; "
            "sampling fell back to all dataset episodes. Pass --require-replay-status to make this fatal."
        )
    if (
        not report.missing_status_file
        and report.policy != "include_all"
        and report.total_episodes
        and report.labeled_episodes == 0
    ):
        warnings.append(
            f"Replay-status policy {report.policy!r} was requested, but the replay-status file contains "
            "no labels for the selected dataset episodes; sampling fell back to all selected episodes. "
            "Pass --require-replay-status to make an empty or incomplete status file fatal."
        )
    if report.filtered_episodes:
        warnings.append(
            f"Replay-status policy {report.policy!r} filtered {report.filtered_episodes} of "
            f"{report.total_episodes} candidate dataset episodes."
        )
    return warnings
