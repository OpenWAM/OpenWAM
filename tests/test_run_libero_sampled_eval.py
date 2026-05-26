from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_libero_sampled_eval.py"
SPEC = importlib.util.spec_from_file_location("run_libero_sampled_eval", SCRIPT_PATH)
assert SPEC is not None
sampled_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sampled_eval
SPEC.loader.exec_module(sampled_eval)


def test_allocate_proportional_counts_uniform_libero10_sample50() -> None:
    group_sizes = {f"task {index}": 50 for index in range(10)}

    allocations = sampled_eval.allocate_proportional_counts(group_sizes, total=50)

    assert allocations == {f"task {index}": 5 for index in range(10)}


def test_allocate_proportional_counts_largest_remainder() -> None:
    allocations = sampled_eval.allocate_proportional_counts({"a": 3, "b": 2, "c": 1}, total=4)

    assert allocations == {"a": 2, "b": 1, "c": 1}


def test_normalize_sample_mode_accepts_legacy_uniform_alias() -> None:
    assert sampled_eval.normalize_sample_mode("uniform_task_distribution") == "dataset_distribution"
    assert sampled_eval.normalize_sample_mode("task_episode_axis") == "task_episode_axis"


def test_build_dataset_episodes_uses_task_local_rank() -> None:
    episode_records = [
        {"episode_index": 0, "length": 10, "tasks": ["task a"]},
        {"episode_index": 1, "length": 10, "tasks": ["task b"]},
        {"episode_index": 2, "length": 10, "tasks": ["task a"]},
        {"episode_index": 3, "length": 10, "tasks": ["task b"]},
        {"episode_index": 4, "length": 10, "tasks": ["task a"]},
    ]

    episodes = sampled_eval.build_dataset_episodes(
        episode_records,
        task_text_to_index={"task a": 0, "task b": 1},
        task_text_to_task_id={"task a": 7, "task b": 8},
        task_text_to_task_name={"task a": "task_a", "task b": "task_b"},
    )

    assert [
        (episode.dataset_episode_index, episode.episode_id, episode.task_id, episode.episode_idx, episode.init_id)
        for episode in episodes
    ] == [
        (0, 0, 7, 0, 0),
        (1, 1, 8, 0, 0),
        (2, 2, 7, 1, 1),
        (3, 3, 8, 1, 1),
        (4, 4, 7, 2, 2),
    ]


def test_filter_dataset_episodes_by_replay_status_filters_distribution_candidates() -> None:
    episodes = [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=index,
            task_text="task",
            task_index=0,
            task_id=0,
            task_name=None,
            episode_idx=index,
            length=10,
        )
        for index in range(3)
    ]
    records = {
        0: SimpleNamespace(replay_status="success"),
        1: SimpleNamespace(replay_status="failure"),
        2: SimpleNamespace(replay_status="success"),
    }

    filtered, report = sampled_eval.filter_dataset_episodes_by_replay_status(
        episodes,
        policy="successful_only",
        replay_status_records=records,
        require_replay_status=True,
        source_path=None,
    )

    assert [episode.dataset_episode_index for episode in filtered] == [0, 2]
    assert report.filtered_episodes == 1


def test_attach_replay_status_can_use_resolved_init_state_for_dataset_eval() -> None:
    episode = sampled_eval.DatasetEpisode(
        dataset_episode_index=12,
        task_text="task",
        task_index=0,
        task_id=3,
        task_name=None,
        episode_idx=4,
        length=100,
    )
    records = {
        12: SimpleNamespace(
            replay_status="success",
            raw={"resolved_init_state_index": 19},
        )
    }

    [attached] = sampled_eval.attach_replay_status_to_dataset_episodes(
        [episode],
        records,
        use_resolved_init_ids=True,
    )

    assert attached.episode_idx == 4
    assert attached.init_id == 19
    assert attached.resolved_init_state_index == 19
    assert attached.init_id_source == "replay_status.resolved_init_state_index"
    assert attached.replay_status == "success"


def test_attach_replay_status_keeps_task_local_init_for_full_grid() -> None:
    episode = sampled_eval.DatasetEpisode(
        dataset_episode_index=12,
        task_text="task",
        task_index=0,
        task_id=3,
        task_name=None,
        episode_idx=4,
        length=100,
    )
    records = {
        12: SimpleNamespace(
            replay_status="success",
            raw={"resolved_init_state_index": 19},
        )
    }

    [attached] = sampled_eval.attach_replay_status_to_dataset_episodes(
        [episode],
        records,
        use_resolved_init_ids=False,
    )

    assert attached.episode_idx == 4
    assert attached.init_id == 4
    assert attached.resolved_init_state_index == 19
    assert attached.init_id_source == "task_local_rank"


def test_task_axis_init_source_can_preserve_task_local_episode_index() -> None:
    args = argparse.Namespace(sample_mode="task_episode_axis", task_axis_init_source="task_local")
    assert sampled_eval.use_replay_resolved_init_ids(args) is False

    args = argparse.Namespace(sample_mode="task_episode_axis", task_axis_init_source="auto")
    assert sampled_eval.use_replay_resolved_init_ids(args) is True


def test_filter_dataset_episodes_by_replay_status_rejects_failed_task_axis_request() -> None:
    episodes = [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=0,
            task_text="task",
            task_index=0,
            task_id=0,
            task_name=None,
            episode_idx=0,
            length=10,
        )
    ]
    records = {0: SimpleNamespace(replay_status="failure")}

    with pytest.raises(ValueError, match="do not satisfy replay_status_policy"):
        sampled_eval.filter_dataset_episodes_by_replay_status(
            episodes,
            policy="successful_only",
            replay_status_records=records,
            require_replay_status=True,
            source_path=None,
            task_axis_validation=True,
        )


def test_resolve_task_ids_uses_requested_benchmark(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []
    fake_module = types.ModuleType("open_wam.integrations.libero_env")

    def fake_resolve_libero_task(task_text, project_root, *, benchmark_name=None):
        del project_root
        calls.append((task_text, benchmark_name))
        return SimpleNamespace(
            benchmark_name=benchmark_name,
            task_id=5,
            task_name="STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        )

    fake_module.resolve_libero_task = fake_resolve_libero_task
    monkeypatch.setitem(sys.modules, "open_wam.integrations.libero_env", fake_module)

    task_ids, task_names, warnings = sampled_eval.resolve_task_ids(
        {"pick up the book and place it in the back compartment of the caddy": 9},
        benchmark="libero_10",
        mode="auto",
    )

    assert calls == [("pick up the book and place it in the back compartment of the caddy", "libero_10")]
    assert task_ids == {"pick up the book and place it in the back compartment of the caddy": 5}
    assert task_names == {
        "pick up the book and place it in the back compartment of the caddy": (
            "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
        )
    }
    assert warnings == []


def test_resolve_task_ids_auto_refuses_metadata_fallback(monkeypatch) -> None:
    fake_module = types.ModuleType("open_wam.integrations.libero_env")

    def fake_resolve_libero_task(task_text, project_root, *, benchmark_name=None):
        del task_text, project_root, benchmark_name
        raise ValueError("duplicate task text")

    fake_module.resolve_libero_task = fake_resolve_libero_task
    monkeypatch.setitem(sys.modules, "open_wam.integrations.libero_env", fake_module)

    with pytest.raises(RuntimeError, match="Refusing to fall back to metadata task_index"):
        sampled_eval.resolve_task_ids(
            {"pick up the book and place it in the back compartment of the caddy": 9},
            benchmark="libero_10",
            mode="auto",
        )


def test_resolve_task_ids_metadata_emits_safety_warning() -> None:
    task_ids, task_names, warnings = sampled_eval.resolve_task_ids(
        {"turn on the stove": 2},
        benchmark="libero_10",
        mode="metadata",
    )

    assert task_ids == {"turn on the stove": 2}
    assert task_names == {"turn on the stove": None}
    assert warnings == [
        "Using LeRobot metadata task_index as LIBERO task_id. This is only valid after verifying "
        "the local metadata order matches the requested upstream LIBERO benchmark order."
    ]


def test_resolve_task_ids_libero_repo_root_overrides_stale_environment(monkeypatch, tmp_path: Path) -> None:
    calls: list[str | None] = []
    fake_module = types.ModuleType("open_wam.integrations.libero_env")

    def fake_resolve_libero_task(task_text, project_root, *, benchmark_name=None):
        del task_text, project_root, benchmark_name
        calls.append(os.environ.get("LIBERO_REPO_ROOT"))
        return SimpleNamespace(
            benchmark_name="libero_10",
            task_id=2,
            task_name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        )

    fake_module.resolve_libero_task = fake_resolve_libero_task
    monkeypatch.setitem(sys.modules, "open_wam.integrations.libero_env", fake_module)
    monkeypatch.setenv("LIBERO_REPO_ROOT", "/stale/libero")

    override_root = tmp_path / "LIBERO"
    task_ids, _, _ = sampled_eval.resolve_task_ids(
        {"turn on the stove and put the moka pot on it": 0},
        benchmark="libero_10",
        mode="auto",
        libero_repo_root=override_root,
    )

    assert task_ids == {"turn on the stove and put the moka pot on it": 2}
    assert calls == [str(override_root)]
    assert os.environ["LIBERO_REPO_ROOT"] == "/stale/libero"


def test_resolve_task_ids_local_paths_overrides_stale_environment(monkeypatch, tmp_path: Path) -> None:
    calls: list[str | None] = []
    fake_module = types.ModuleType("open_wam.integrations.libero_env")

    def fake_resolve_libero_task(task_text, project_root, *, benchmark_name=None):
        del task_text, project_root, benchmark_name
        calls.append(os.environ.get("OPEN_WAM_LOCAL_PATHS"))
        return SimpleNamespace(
            benchmark_name="libero_10",
            task_id=2,
            task_name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        )

    fake_module.resolve_libero_task = fake_resolve_libero_task
    monkeypatch.setitem(sys.modules, "open_wam.integrations.libero_env", fake_module)
    monkeypatch.setenv("OPEN_WAM_LOCAL_PATHS", "/stale/local_paths.yaml")

    local_paths = tmp_path / "local_paths.yaml"
    task_ids, _, _ = sampled_eval.resolve_task_ids(
        {"turn on the stove and put the moka pot on it": 0},
        benchmark="libero_10",
        mode="auto",
        local_paths=local_paths,
    )

    assert task_ids == {"turn on the stove and put the moka pot on it": 2}
    assert calls == [str(local_paths)]
    assert os.environ["OPEN_WAM_LOCAL_PATHS"] == "/stale/local_paths.yaml"


def test_sample_episodes_by_task_distribution_is_deterministic() -> None:
    episodes = [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=index,
            task_text=f"task {index // 10}",
            task_index=index // 10,
            task_id=index // 10,
            task_name=None,
            episode_idx=index % 10,
            length=10,
        )
        for index in range(30)
    ]

    first, first_allocations = sampled_eval.sample_episodes_by_task_distribution(episodes, count=6, seed=123)
    second, second_allocations = sampled_eval.sample_episodes_by_task_distribution(episodes, count=6, seed=123)

    assert first_allocations == {"task 0": 2, "task 1": 2, "task 2": 2}
    assert [episode.dataset_episode_index for episode in first] == [0, 1, 10, 11, 20, 21]
    assert [episode.dataset_episode_index for episode in first] == [episode.dataset_episode_index for episode in second]
    assert first_allocations == second_allocations


def test_sample_episodes_by_task_distribution_random_strategy_preserves_seeded_sampling() -> None:
    episodes = [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=index,
            task_text=f"task {index // 10}",
            task_index=index // 10,
            task_id=index // 10,
            task_name=None,
            episode_idx=index % 10,
            length=10,
        )
        for index in range(30)
    ]

    selected, allocations = sampled_eval.sample_episodes_by_task_distribution(
        episodes,
        count=6,
        seed=123,
        episode_strategy="random",
    )

    assert allocations == {"task 0": 2, "task 1": 2, "task 2": 2}
    assert [episode.dataset_episode_index for episode in selected] == [0, 4, 11, 16, 21, 24]


def test_sample_episodes_by_task_distribution_uses_upstream_task_id_tie_breaks() -> None:
    episodes = []
    task_specs = [
        ("metadata task 0", 2),
        ("metadata task 1", 0),
        ("metadata task 2", 1),
    ]
    for dataset_index, (task_text, task_id) in enumerate(task_specs):
        episodes.append(
            sampled_eval.DatasetEpisode(
                dataset_episode_index=dataset_index,
                task_text=task_text,
                task_index=dataset_index,
                task_id=task_id,
                task_name=None,
                episode_idx=0,
                length=10,
            )
        )

    selected, allocations = sampled_eval.sample_episodes_by_task_distribution(episodes, count=2, seed=0)

    assert allocations == {"metadata task 0": 0, "metadata task 1": 1, "metadata task 2": 1}
    assert [(episode.task_id, episode.dataset_episode_index) for episode in selected] == [(0, 1), (1, 2)]


def test_sample_warnings_flag_undercovered_dataset_distribution() -> None:
    warnings = sampled_eval.build_sample_warnings(
        mode="dataset_distribution",
        requested_count=5,
        task_allocations={f"task {index}": 1 if index < 5 else 0 for index in range(10)},
        distribution_episode_strategy="first",
    )

    assert warnings == [
        "Requested 5 sampled episodes across 10 tasks; only 5 tasks are covered. "
        "Increase --num-episodes to at least 10 for a cross-task smoke run."
    ]


def test_replay_status_warnings_distinguish_empty_present_status_file() -> None:
    report = sampled_eval.ReplayStatusFilterReport(
        source_path="/tmp/replay_status.jsonl",
        policy="successful_only",
        require_replay_status=False,
        total_episodes=2,
        labeled_episodes=0,
        kept_episodes=2,
        filtered_episodes=0,
        status_counts={},
        missing_status_file=False,
    )

    warnings = sampled_eval.build_replay_status_warnings(report)

    assert warnings == [
        "Replay-status policy 'successful_only' was requested, but the replay-status file contains "
        "no labels for the selected dataset episodes; sampling fell back to all selected episodes. "
        "Pass --require-replay-status to make an empty or incomplete status file fatal."
    ]


def test_parse_int_selector_supports_lists_and_half_open_ranges() -> None:
    assert sampled_eval.parse_int_selector("0,2:5,4,8:12:2") == [0, 2, 3, 4, 8, 10]


def test_select_task_episode_axis_matches_upstream_task_ids() -> None:
    episodes = [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=250 + index,
            task_text="tomato",
            task_index=5,
            task_id=0,
            task_name="task_0_tomato",
            episode_idx=index,
            length=10,
        )
        for index in range(3)
    ] + [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=50 + index,
            task_text="black bowl",
            task_index=1,
            task_id=3,
            task_name="task_3_black_bowl",
            episode_idx=index,
            length=10,
        )
        for index in range(3)
    ]

    selected, allocations = sampled_eval.select_task_episode_axis(
        episodes,
        count=2,
        task_ids="0",
        episode_indices=None,
    )

    assert [(episode.task_id, episode.episode_idx, episode.dataset_episode_index) for episode in selected] == [
        (0, 0, 250),
        (0, 1, 251),
    ]
    assert allocations == {"tomato": 2}


def test_select_task_episode_axis_orders_init_major_across_tasks() -> None:
    episodes = [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=task_id * 10 + init_id,
            task_text=f"task {task_id}",
            task_index=task_id,
            task_id=task_id,
            task_name=f"task_{task_id}",
            episode_idx=init_id,
            length=10,
        )
        for task_id in (0, 1)
        for init_id in range(2)
    ]

    selected, allocations = sampled_eval.select_task_episode_axis(
        episodes,
        count=4,
        task_ids="0,1",
        episode_indices=None,
    )

    assert [(episode.task_id, episode.episode_idx) for episode in selected] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]
    assert allocations == {"task 0": 2, "task 1": 2}


def test_select_task_episode_axis_defaults_to_all_tasks_and_truncates_init_major() -> None:
    episodes = [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=task_id * 10 + init_id,
            task_text=f"task {task_id}",
            task_index=task_id,
            task_id=task_id,
            task_name=f"task_{task_id}",
            episode_idx=init_id,
            length=10,
        )
        for task_id in (0, 1)
        for init_id in range(3)
    ]

    selected, allocations = sampled_eval.select_task_episode_axis(
        episodes,
        count=5,
        task_ids=None,
        episode_indices=None,
    )

    assert [(episode.task_id, episode.episode_idx) for episode in selected] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
    ]
    assert allocations == {"task 0": 3, "task 1": 2}


def test_select_full_task_init_axis_enumerates_benchmark_init_ids() -> None:
    episodes = [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=index,
            task_text="task 0",
            task_index=0,
            task_id=0,
            task_name="task_0",
            episode_idx=index,
            length=10,
        )
        for index in range(2)
    ] + [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=10 + index,
            task_text="task 1",
            task_index=1,
            task_id=1,
            task_name="task_1",
            episode_idx=index,
            length=10,
        )
        for index in range(3)
    ]

    selected, allocations = sampled_eval.select_full_task_init_axis(
        episodes,
        init_counts_by_task_id={0: 2, 1: 3},
        task_ids=None,
        episode_indices=None,
    )

    assert [(episode.task_id, episode.init_id, episode.dataset_episode_index) for episode in selected] == [
        (0, 0, 0),
        (1, 0, 10),
        (0, 1, 1),
        (1, 1, 11),
        (1, 2, 12),
    ]
    assert allocations == {"task 0": 2, "task 1": 3}


def test_select_full_task_init_axis_rejects_missing_dataset_pairs() -> None:
    episodes = [
        sampled_eval.DatasetEpisode(
            dataset_episode_index=0,
            task_text="task 0",
            task_index=0,
            task_id=0,
            task_name="task_0",
            episode_idx=0,
            length=10,
        )
    ]

    with pytest.raises(ValueError, match="task_id=0,init_id=1"):
        sampled_eval.select_full_task_init_axis(
            episodes,
            init_counts_by_task_id={0: 2},
            task_ids=None,
            episode_indices=None,
        )


def test_select_full_task_init_axis_rejects_episode_indices_selector() -> None:
    with pytest.raises(ValueError, match="episode-indices is not used"):
        sampled_eval.select_full_task_init_axis(
            [],
            init_counts_by_task_id={0: 1},
            task_ids=None,
            episode_indices="0:1",
        )


def test_select_sampled_episodes_rejects_task_axis_options_in_distribution_mode() -> None:
    with pytest.raises(ValueError, match="require --sample-mode task_episode_axis"):
        sampled_eval.select_sampled_episodes(
            [],
            mode="dataset_distribution",
            count=1,
            seed=0,
            task_ids="0",
        )


def test_parse_target_requests_accepts_method_key_label_and_checkpoint() -> None:
    targets = sampled_eval.parse_target_requests(
        [
            "m2:base=/tmp/m2_base",
            "m5:posttrained:latest checkpoint=/tmp/m5_post",
        ]
    )

    assert targets == [
        sampled_eval.TargetRequest(
            method_key="m2",
            checkpoint_key="base",
            label=None,
            checkpoint="/tmp/m2_base",
        ),
        sampled_eval.TargetRequest(
            method_key="m5",
            checkpoint_key="posttrained",
            label="latest checkpoint",
            checkpoint="/tmp/m5_post",
        ),
    ]


def test_resolve_checkpoint_input_accepts_run_root_and_resolved_config_transformer(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    checkpoint_100 = run_root / "checkpoints" / "checkpoint_step_100"
    checkpoint_200 = run_root / "checkpoints" / "checkpoint_step_200"
    transformer_dir = tmp_path / "shared_transformer"
    checkpoint_100.mkdir(parents=True)
    checkpoint_200.mkdir(parents=True)
    transformer_dir.mkdir()
    (checkpoint_100 / "model_state.pt").write_bytes(b"old")
    (checkpoint_200 / "model_state.pt").write_bytes(b"new")
    (transformer_dir / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_200 / "resolved_config.yaml").write_text(
        f"backbone:\n  transformer_subdir: {transformer_dir}\n",
        encoding="utf-8",
    )

    resolution = sampled_eval.resolve_checkpoint_input(str(run_root))

    assert resolution.checkpoint_file == str((checkpoint_200 / "model_state.pt").resolve())
    assert resolution.checkpoint_dir == str(checkpoint_200.resolve())
    assert resolution.runtime_transformer_dir == str(transformer_dir.resolve())
    assert resolution.runtime_transformer_source == "resolved_config"
    assert resolution.problem is None


def test_resolve_checkpoint_input_accepts_transformer_only_model_root(tmp_path: Path) -> None:
    model_root = tmp_path / "lingbot_va"
    transformer_dir = model_root / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text("{}", encoding="utf-8")
    (transformer_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    resolution = sampled_eval.resolve_checkpoint_input(str(model_root))

    assert resolution.checkpoint_file is None
    assert resolution.checkpoint_dir == str(model_root.resolve())
    assert resolution.runtime_transformer_dir == str(transformer_dir.resolve())
    assert resolution.runtime_transformer_source == "input_transformer_subdir"
    assert resolution.problem is None


def test_transformer_only_model_root_requires_config_and_weights(tmp_path: Path) -> None:
    model_root = tmp_path / "lingbot_va"
    transformer_dir = model_root / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text("{}", encoding="utf-8")

    resolution = sampled_eval.resolve_checkpoint_input(str(model_root))

    assert resolution.checkpoint_file is None
    assert resolution.runtime_transformer_dir is None
    assert resolution.problem == (
        "could not resolve model_state.pt, full_training_state.pt, or transformer export "
        "(config.json plus diffusion_pytorch_model*.safetensors)"
    )


def test_checkpoint_transformer_dir_preserves_nonempty_legacy_detection(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint_step_1"
    transformer_dir = checkpoint_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (checkpoint_dir / "model_state.pt").write_bytes(b"state")
    (transformer_dir / "weights.bin").write_bytes(b"placeholder")

    resolution = sampled_eval.resolve_checkpoint_input(str(checkpoint_dir))

    assert resolution.checkpoint_file == str((checkpoint_dir / "model_state.pt").resolve())
    assert resolution.runtime_transformer_dir == str(transformer_dir.resolve())
    assert resolution.runtime_transformer_source == "checkpoint"
    assert resolution.problem is None


def test_resolve_checkpoint_specs_reports_effective_transformer_only_flags(tmp_path: Path) -> None:
    model_root = tmp_path / "lingbot_va"
    transformer_dir = model_root / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text("{}", encoding="utf-8")
    (transformer_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    method = next(method for method in sampled_eval.METHODS if method.key == "m1")
    args = argparse.Namespace(cfg=None, reference_assets_device_policy=None)

    [spec] = sampled_eval.resolve_checkpoint_specs(
        args=args,
        selected_methods=[method],
        target_requests=[
            sampled_eval.TargetRequest(
                method_key="m1",
                checkpoint_key="lingbot_va",
                label="LingBot-VA transformer",
                checkpoint=str(model_root),
            )
        ],
    )

    assert spec.runtime_transformer_dir == str(transformer_dir.resolve())
    assert spec.extra_args == ()


def test_build_cases_uses_method_config_scheduler_and_device_templates() -> None:
    args = argparse.Namespace(
        python=Path("/venv/bin/python"),
        run_label="matrix",
        eval_profile="libero_10hz_full",
        max_actions=None,
        env_horizon=None,
        target_action_hz=None,
        video_fps=None,
        rollout_artifact_profile="lean",
        deadline_miss_policy=None,
        write_fallback_timeline_video=False,
    )
    episode = sampled_eval.DatasetEpisode(
        dataset_episode_index=12,
        task_text="task",
        task_index=0,
        task_id=3,
        task_name=None,
        episode_idx=4,
        length=100,
        replay_status="success",
    )
    method = next(method for method in sampled_eval.METHODS if method.key == "m5")
    checkpoint = sampled_eval.CheckpointSpec(
        key="m5_posttrained",
        label="M5 posttrained",
        checkpoint="/tmp/model_state.pt",
        method_key=method.key,
        method_label=method.label,
        config=method.config,
        reference_assets_device_policy=method.reference_assets_device_policy,
        extra_args=method.extra_args,
    )
    scheduler = next(scheduler for scheduler in sampled_eval.SCHEDULERS if scheduler.key == "freeze_until_clean_chunk")

    cases = sampled_eval.build_cases(
        [episode],
        checkpoint_specs=[checkpoint],
        output_root=Path("/tmp/out"),
        benchmark="libero_10",
        seed=0,
        scheduler_spec=scheduler,
        args=args,
    )

    command = cases[0].command_template
    assert (
        command[command.index("--cfg") + 1]
        == "configs/evals/mot_libero_full_segment_non_joint_action_only_eval.yaml"
    )
    assert command[command.index("--reference-assets-device-policy") + 1] == "cpu_offload"
    assert command[command.index("--runtime-device") + 1] == "{device}"
    assert "--runtime-devices" not in command
    assert command[command.index("--artifact-profile") + 1] == "lean"
    assert command[command.index("--eval-profile") + 1] == "libero_10hz_full"
    assert command[command.index("--realtime-scheduler-profile") + 1] == "freeze_until_clean_chunk"
    assert "--max-actions" not in command
    assert "--env-horizon" not in command
    assert "--video-fps" not in command
    assert command[command.index("--startup-open-loop-chunks") + 1] == "1"
    assert command[command.index("--episode-idx") + 1] == "4"
    assert cases[0].episode_id == 12
    assert cases[0].init_id == 4
    assert cases[0].replay_status == "success"


def test_build_cases_uses_transformer_dir_without_checkpoint_only_flags() -> None:
    args = argparse.Namespace(
        python=Path("/venv/bin/python"),
        run_label="lingbot_va",
        eval_profile="libero_10hz_full",
        max_actions=None,
        env_horizon=None,
        target_action_hz=None,
        video_fps=None,
        rollout_artifact_profile="lean",
        deadline_miss_policy=None,
        write_fallback_timeline_video=False,
    )
    episode = sampled_eval.DatasetEpisode(
        dataset_episode_index=12,
        task_text="task",
        task_index=0,
        task_id=3,
        task_name=None,
        episode_idx=4,
        length=100,
    )
    method = next(method for method in sampled_eval.METHODS if method.key == "m1")
    checkpoint = sampled_eval.CheckpointSpec(
        key="m1_lingbot_va",
        label="M1 LingBot-VA transformer",
        checkpoint="/models/lingbot_va",
        checkpoint_raw="/models/lingbot_va",
        checkpoint_file=None,
        checkpoint_dir="/models/lingbot_va",
        runtime_transformer_dir="/models/lingbot_va/transformer",
        runtime_transformer_source="input_transformer_subdir",
        method_key=method.key,
        method_label=method.label,
        config=method.config,
        reference_assets_device_policy=method.reference_assets_device_policy,
        extra_args=method.extra_args,
    )
    scheduler = next(scheduler for scheduler in sampled_eval.SCHEDULERS if scheduler.key == "blocking_control")

    cases = sampled_eval.build_cases(
        [episode],
        checkpoint_specs=[checkpoint],
        output_root=Path("/tmp/out"),
        benchmark="libero_10",
        seed=0,
        scheduler_spec=scheduler,
        args=args,
    )

    command = cases[0].command_template
    assert "--transformer-dir" in command
    assert command[command.index("--transformer-dir") + 1] == "/models/lingbot_va/transformer"
    assert "--checkpoint" not in command
    assert "--merge-checkpoint-runtime-config" not in command


def test_build_cases_merges_checkpoint_runtime_config_for_exact_methods() -> None:
    args = argparse.Namespace(
        python=Path("/venv/bin/python"),
        run_label="matrix",
        eval_profile="libero_10hz_full",
        max_actions=None,
        env_horizon=None,
        target_action_hz=None,
        video_fps=None,
        rollout_artifact_profile="lean",
        deadline_miss_policy=None,
        write_fallback_timeline_video=False,
    )
    episode = sampled_eval.DatasetEpisode(
        dataset_episode_index=12,
        task_text="task",
        task_index=0,
        task_id=3,
        task_name=None,
        episode_idx=4,
        length=100,
        replay_status="success",
    )
    scheduler = next(scheduler for scheduler in sampled_eval.SCHEDULERS if scheduler.key == "blocking_control")
    specs = [
        sampled_eval.CheckpointSpec(
            key=f"{method.key}_posttrained",
            label=f"{method.label} posttrained",
            checkpoint="/tmp/model_state.pt",
            method_key=method.key,
            method_label=method.label,
            config=method.config,
            reference_assets_device_policy=method.reference_assets_device_policy,
            extra_args=method.extra_args,
        )
        for method in sampled_eval.METHODS
    ]

    cases = sampled_eval.build_cases(
        [episode],
        checkpoint_specs=specs,
        output_root=Path("/tmp/out"),
        benchmark="libero_10",
        seed=0,
        scheduler_spec=scheduler,
        args=args,
    )

    command_by_method = {case.method_key: case.command_template for case in cases}
    assert "--merge-checkpoint-runtime-config" in command_by_method["m1"]
    assert "--merge-checkpoint-runtime-config" in command_by_method["m2"]
    assert "--merge-checkpoint-runtime-config" not in command_by_method["m5"]


def test_acquire_case_claim_is_exclusive_and_stale_recoverable(tmp_path: Path) -> None:
    case = sampled_eval.EvalCase(
        index=0,
        sample_index=0,
        checkpoint_key="m5_posttrained",
        checkpoint_label="M5",
        checkpoint="/tmp/model_state.pt",
        checkpoint_raw=None,
        checkpoint_file=None,
        checkpoint_dir=None,
        runtime_transformer_dir=None,
        runtime_transformer_source=None,
        method_key="m5",
        method_label="M5",
        config="config.yaml",
        scheduler_key="freeze_until_clean_chunk",
        scheduler_label="freeze",
        benchmark="libero_10",
        task_id=0,
        task_text="task",
        task_name=None,
        dataset_episode_index=0,
        episode_id=0,
        init_id=0,
        episode_idx=0,
        replay_status="success",
        seed=0,
        output_dir=str(tmp_path / "out"),
        suffix="case",
        summary_glob=str(tmp_path / "missing" / "*.json"),
        command_template=[],
    )
    status_dir = tmp_path / "status"

    first = sampled_eval.acquire_case_claim(case, status_dir=status_dir, stale_seconds=60)
    second = sampled_eval.acquire_case_claim(case, status_dir=status_dir, stale_seconds=60)
    assert first is not None
    assert second is None

    old_time = 1
    assert first is not None
    first.touch()
    os.utime(first, (old_time, old_time))
    recovered = sampled_eval.acquire_case_claim(case, status_dir=status_dir, stale_seconds=1)
    assert recovered == first


def test_run_case_releases_claim_after_child_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case = sampled_eval.EvalCase(
        index=0,
        sample_index=0,
        checkpoint_key="m5_posttrained",
        checkpoint_label="M5",
        checkpoint="/tmp/model_state.pt",
        checkpoint_raw=None,
        checkpoint_file=None,
        checkpoint_dir=None,
        runtime_transformer_dir=None,
        runtime_transformer_source=None,
        method_key="m5",
        method_label="M5",
        config="config.yaml",
        scheduler_key="freeze_until_clean_chunk",
        scheduler_label="freeze",
        benchmark="libero_10",
        task_id=0,
        task_text="task",
        task_name=None,
        dataset_episode_index=0,
        episode_id=0,
        init_id=0,
        episode_idx=0,
        replay_status="success",
        seed=0,
        output_dir=str(tmp_path / "out"),
        suffix="case",
        summary_glob=str(tmp_path / "missing" / "*.json"),
        command_template=["noop"],
    )
    args = argparse.Namespace(
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl="egl",
        clear_ld_library_path=False,
        resume=True,
        case_claim_stale_seconds=60,
    )

    def fake_run(*_args, **_kwargs):
        return sampled_eval.subprocess.CompletedProcess(args=["noop"], returncode=1)

    monkeypatch.setattr(sampled_eval.subprocess, "run", fake_run)
    status_dir = tmp_path / "status"

    returncode = sampled_eval.run_case(
        case,
        device="cuda:1",
        args=args,
        status_dir=status_dir,
        logs_dir=tmp_path / "logs",
    )

    assert returncode == 1
    assert not list((status_dir / "claims").glob("*.lock"))
    status = json.loads((status_dir / "0000_m5_posttrained_sample000.json").read_text())
    assert status["state"] == "failed"


def test_run_case_releases_claim_when_summary_appears_after_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = sampled_eval.EvalCase(
        index=0,
        sample_index=0,
        checkpoint_key="m5_posttrained",
        checkpoint_label="M5",
        checkpoint="/tmp/model_state.pt",
        checkpoint_raw=None,
        checkpoint_file=None,
        checkpoint_dir=None,
        runtime_transformer_dir=None,
        runtime_transformer_source=None,
        method_key="m5",
        method_label="M5",
        config="config.yaml",
        scheduler_key="freeze_until_clean_chunk",
        scheduler_label="freeze",
        benchmark="libero_10",
        task_id=0,
        task_text="task",
        task_name=None,
        dataset_episode_index=0,
        episode_id=0,
        init_id=0,
        episode_idx=0,
        replay_status="success",
        seed=0,
        output_dir=str(tmp_path / "out"),
        suffix="case",
        summary_glob=str(tmp_path / "summary.json"),
        command_template=["noop"],
    )
    args = argparse.Namespace(
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl="egl",
        clear_ld_library_path=False,
        resume=True,
        case_claim_stale_seconds=60,
    )
    calls = 0

    def fake_find_summary_paths(_case):
        nonlocal calls
        calls += 1
        return [] if calls == 1 else [tmp_path / "summary.json"]

    monkeypatch.setattr(sampled_eval, "find_summary_paths", fake_find_summary_paths)
    status_dir = tmp_path / "status"

    returncode = sampled_eval.run_case(
        case,
        device="cuda:0",
        args=args,
        status_dir=status_dir,
        logs_dir=tmp_path / "logs",
    )

    assert returncode == 0
    assert not list((status_dir / "claims").glob("*.lock"))
    status = json.loads((status_dir / "0000_m5_posttrained_sample000.json").read_text())
    assert status["state"] == "skipped_existing"


def test_run_cases_retires_device_after_repeated_sigaborts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [
        sampled_eval.EvalCase(
            index=index,
            sample_index=index,
            checkpoint_key="m5_posttrained",
            checkpoint_label="M5",
            checkpoint="/ckpt/model_state.pt",
            checkpoint_raw="/ckpt",
            checkpoint_file="/ckpt/model_state.pt",
            checkpoint_dir="/ckpt",
            runtime_transformer_dir="/ckpt/transformer",
            runtime_transformer_source="checkpoint",
            method_key="m5",
            method_label="M5",
            config="cfg.yaml",
            scheduler_key="freeze_until_clean_chunk",
            scheduler_label="freeze",
            benchmark="libero_10",
            task_id=0,
            task_text="task",
            task_name=None,
            dataset_episode_index=index,
            episode_id=index,
            init_id=index,
            episode_idx=index,
            replay_status="success",
            seed=0,
            output_dir=str(tmp_path / "out"),
            suffix=f"case-{index}",
            summary_glob=str(tmp_path / "missing" / f"{index}" / "*.json"),
            command_template=["noop"],
        )
        for index in range(3)
    ]
    args = argparse.Namespace(
        devices="cuda:0",
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl="egl",
        clear_ld_library_path=False,
        resume=True,
        fail_fast=False,
        max_device_sigaborts=2,
        case_claim_stale_seconds=60,
    )
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return sampled_eval.subprocess.CompletedProcess(args=["noop"], returncode=-6)

    monkeypatch.setattr(sampled_eval.subprocess, "run", fake_run)
    status_dir = tmp_path / "status"

    sampled_eval.run_cases(cases, args=args, status_dir=status_dir, logs_dir=tmp_path / "logs")

    assert calls == 2
    assert json.loads((status_dir / "0000_m5_posttrained_sample000.json").read_text())["state"] == "failed"
    assert json.loads((status_dir / "0001_m5_posttrained_sample001.json").read_text())["state"] == "failed"
    assert not (status_dir / "0002_m5_posttrained_sample002.json").exists()


def test_build_paired_rows_preserves_replay_status() -> None:
    rows = sampled_eval.build_paired_rows(
        [
            {
                "case": {
                    "sample_index": 0,
                    "checkpoint_key": "m1_base",
                    "dataset_episode_index": 10,
                    "episode_id": 10,
                    "task_id": 1,
                    "task_text": "task",
                    "init_id": 2,
                    "episode_idx": 2,
                    "replay_status": "success",
                },
                "status": {"state": "completed", "returncode": 0},
                "summary": {"success": True, "executed_actions": 15, "fallback_actions": 0},
                "summary_path": "/tmp/summary.json",
            }
        ]
    )

    assert rows[0]["replay_status"] == "success"


def test_build_child_env_preserves_ld_library_path_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib")
    args = argparse.Namespace(
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl="egl",
        clear_ld_library_path=False,
    )

    env = sampled_eval.build_child_env(args)

    assert env["LD_LIBRARY_PATH"] == "/usr/local/lib"
    assert env["PYOPENGL_PLATFORM"] == "egl"
    assert env["PYTHONFAULTHANDLER"] == "1"
    assert env["TORCH_SHOW_CPP_STACKTRACES"] == "1"


def test_build_child_env_can_clear_ld_library_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib")
    args = argparse.Namespace(
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl="egl",
        clear_ld_library_path=True,
    )

    env = sampled_eval.build_child_env(args)

    assert "LD_LIBRARY_PATH" not in env


def test_build_child_env_defaults_to_shell_mujoco_gl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    args = argparse.Namespace(
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl=None,
        clear_ld_library_path=False,
    )

    env = sampled_eval.build_child_env(args)

    assert env["MUJOCO_GL"] == "egl"
    assert env["PYOPENGL_PLATFORM"] == "egl"


def test_build_child_env_honors_requested_mujoco_gl_over_shell_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    args = argparse.Namespace(
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl="osmesa",
        clear_ld_library_path=False,
    )

    env = sampled_eval.build_child_env(args)

    assert env["MUJOCO_GL"] == "osmesa"
    assert env["PYOPENGL_PLATFORM"] == "osmesa"


def test_build_child_env_sets_egl_device_for_cuda_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    monkeypatch.delenv("EGL_DEVICE_ID", raising=False)
    args = argparse.Namespace(
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl="egl",
        clear_ld_library_path=False,
    )

    env = sampled_eval.build_child_env(args, device="cuda:1")

    assert env["MUJOCO_EGL_DEVICE_ID"] == "1"
    assert env["EGL_DEVICE_ID"] == "1"


def test_build_child_env_prefers_cuda_visible_devices_for_egl_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "4,7")
    args = argparse.Namespace(
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl="egl",
        clear_ld_library_path=False,
    )

    env = sampled_eval.build_child_env(args, device="cuda:1")

    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "1"
    assert env["EGL_DEVICE_ID"] == "1"
    assert sampled_eval.child_env_report(env, clear_ld_library_path=False)["slurm_job_gpus"] == "4,7"


def test_build_child_env_uses_slurm_gpu_allocation_when_cuda_visibility_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("SLURM_JOB_GPUS", "4,7")
    args = argparse.Namespace(
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=Path("/data/lingbot_data_exp/LIBERO"),
        mujoco_gl="egl",
        clear_ld_library_path=False,
    )

    env = sampled_eval.build_child_env(args, device="cuda:1")

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["MUJOCO_EGL_DEVICE_ID"] == "7"
    assert env["EGL_DEVICE_ID"] == "7"


def test_parse_allocated_gpu_ids_handles_ranges_and_uuid_suffixes() -> None:
    assert sampled_eval.parse_allocated_gpu_ids("2-4") == [2, 3, 4]
    assert sampled_eval.parse_allocated_gpu_ids("gpu0,gpu3") == [0, 3]


def test_results_csv_uses_dynamic_target_columns(tmp_path: Path) -> None:
    summary = {
        "paired_rows": [
            {
                "sample_index": 0,
                "dataset_episode_index": 10,
                "episode_id": 10,
                "task_id": 1,
                "init_id": 2,
                "episode_idx": 2,
                "replay_status": "success",
                "task_text": "task",
                "m2_base_status": "completed",
                "m2_base_returncode": 0,
                "m2_base_success": True,
                "m2_base_executed_actions": 15,
                "m2_base_fallback_actions": 0,
                "m2_base_summary_path": "/tmp/m2.json",
                "m5_posttrained_status": "failed",
                "m5_posttrained_returncode": 1,
                "m5_posttrained_success": None,
                "m5_posttrained_executed_actions": None,
                "m5_posttrained_fallback_actions": None,
                "m5_posttrained_summary_path": None,
            }
        ],
        "target_keys": ["m2_base", "m5_posttrained"],
    }
    path = tmp_path / "results.csv"

    sampled_eval.write_results_csv(path, summary)

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "m2_base_success" in rows[0]
    assert "m5_posttrained_status" in rows[0]
    assert rows[0]["init_id"] == "2"
    assert rows[0]["replay_status"] == "success"
    assert "posttrained_success" not in rows[0]
