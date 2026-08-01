from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from open_wam.evals import sampled_eval_sampling


def _task_grid() -> list[sampled_eval_sampling.DatasetEpisode]:
    return [
        sampled_eval_sampling.DatasetEpisode(
            dataset_episode_index=task_id * 10 + init_id,
            task_text=f"task {task_id}",
            task_index=task_id,
            task_id=task_id,
            task_name=f"task_{task_id}",
            episode_idx=init_id,
            length=100,
        )
        for task_id in (2, 0)
        for init_id in range(3)
    ]


def test_sampled_eval_modes_are_typed_with_stable_cli_values() -> None:
    assert sampled_eval_sampling.SAMPLE_MODE_CHOICES == (
        "dataset_distribution",
        "uniform_task_distribution",
        "task_episode_axis",
        "full",
    )
    assert sampled_eval_sampling.normalize_sample_mode("uniform_task_distribution") == (
        sampled_eval_sampling.SampledEvalMode.DATASET_DISTRIBUTION.value
    )
    assert sampled_eval_sampling.normalize_sample_mode(
        sampled_eval_sampling.SampledEvalMode.FULL
    ) == "full"
    assert sampled_eval_sampling.uses_replay_resolved_init_ids(
        sample_mode=sampled_eval_sampling.SampledEvalMode.TASK_EPISODE_AXIS,
        task_axis_init_source=sampled_eval_sampling.TaskAxisInitSource.AUTO,
    )
    assert not sampled_eval_sampling.uses_replay_resolved_init_ids(
        sample_mode="full",
        task_axis_init_source="auto",
    )
    with pytest.raises(ValueError, match="Unsupported task_axis_init_source='unknown'"):
        sampled_eval_sampling.uses_replay_resolved_init_ids(
            sample_mode="full",
            task_axis_init_source="unknown",
        )


def test_dataset_episode_construction_is_sorted_ranked_and_frozen() -> None:
    episodes = sampled_eval_sampling.build_dataset_episodes(
        [
            {"episode_index": 8, "length": 80, "tasks": ["alpha"]},
            {"episode_index": 1, "length": 10, "tasks": ["beta"]},
            {"episode_index": 3, "length": 30, "tasks": ["alpha"]},
        ],
        task_text_to_index={"alpha": 4, "beta": 2},
        task_text_to_task_id={"alpha": 7, "beta": 1},
        task_text_to_task_name={"alpha": "task_alpha", "beta": "task_beta"},
    )

    assert [
        (item.dataset_episode_index, item.task_id, item.episode_idx, item.init_id)
        for item in episodes
    ] == [
        (1, 1, 0, 0),
        (3, 7, 0, 0),
        (8, 7, 1, 1),
    ]
    with pytest.raises(FrozenInstanceError):
        episodes[0].episode_idx = 9  # type: ignore[misc]


def test_selection_modes_preserve_distribution_and_init_major_order() -> None:
    episodes = _task_grid()

    distributed, distribution = sampled_eval_sampling.select_sampled_episodes(
        episodes,
        mode="uniform_task_distribution",
        count=4,
        seed=17,
        distribution_episode_strategy=sampled_eval_sampling.DistributionEpisodeStrategy.FIRST,
    )
    task_axis, task_allocation = sampled_eval_sampling.select_sampled_episodes(
        episodes,
        mode=sampled_eval_sampling.SampledEvalMode.TASK_EPISODE_AXIS,
        count=4,
        seed=17,
        task_ids="2,0",
    )
    full_axis, full_allocation = sampled_eval_sampling.select_sampled_episodes(
        episodes,
        mode=sampled_eval_sampling.SampledEvalMode.FULL,
        count=1,
        seed=17,
        full_init_counts_by_task_id={0: 3, 2: 3},
    )

    assert distribution == {"task 2": 2, "task 0": 2}
    assert [(item.task_id, item.episode_idx) for item in distributed] == [
        (0, 0),
        (0, 1),
        (2, 0),
        (2, 1),
    ]
    assert task_allocation == {"task 2": 2, "task 0": 2}
    assert [(item.task_id, item.episode_idx) for item in task_axis] == [
        (2, 0),
        (0, 0),
        (2, 1),
        (0, 1),
    ]
    assert full_allocation == {"task 0": 3, "task 2": 3}
    assert [(item.task_id, item.init_id) for item in full_axis] == [
        (0, 0),
        (2, 0),
        (0, 1),
        (2, 1),
        (0, 2),
        (2, 2),
    ]


def test_replay_attachment_and_filter_keep_dataset_order() -> None:
    episodes = _task_grid()[:3]
    records = {
        episode.dataset_episode_index: SimpleNamespace(
            replay_status="failure" if episode.episode_idx == 1 else "success",
            raw={"resolved_init_state_index": 20 + episode.episode_idx},
        )
        for episode in episodes
    }

    attached = sampled_eval_sampling.attach_replay_status_to_dataset_episodes(
        episodes,
        records,
        use_resolved_init_ids=True,
    )
    filtered, report = sampled_eval_sampling.filter_dataset_episodes_by_replay_status(
        attached,
        policy="successful_only",
        replay_status_records=records,
        require_replay_status=True,
        source_path=None,
    )

    assert [item.init_id for item in attached] == [20, 21, 22]
    assert [item.dataset_episode_index for item in filtered] == [20, 22]
    assert report.filtered_episodes == 1
    assert sampled_eval_sampling.build_replay_status_warnings(report) == [
        "Replay-status policy 'successful_only' filtered 1 of 3 candidate dataset episodes."
    ]
