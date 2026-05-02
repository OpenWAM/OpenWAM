from __future__ import annotations

import argparse
import csv
import importlib.util
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

    assert [(episode.dataset_episode_index, episode.task_id, episode.episode_idx) for episode in episodes] == [
        (0, 7, 0),
        (1, 8, 0),
        (2, 7, 1),
        (3, 8, 1),
        (4, 7, 2),
    ]


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


def test_build_cases_uses_method_config_scheduler_and_device_templates() -> None:
    args = argparse.Namespace(
        python=Path("/venv/bin/python"),
        run_label="matrix",
        max_actions=3000,
        env_horizon=5000,
        target_action_hz=10.0,
        video_fps=15,
        deadline_miss_policy="hold_state",
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
    assert command[command.index("--runtime-devices") + 1] == "{device}"
    assert command[command.index("--startup-open-loop-chunks") + 1] == "1"


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


def test_results_csv_uses_dynamic_target_columns(tmp_path: Path) -> None:
    summary = {
        "paired_rows": [
            {
                "sample_index": 0,
                "dataset_episode_index": 10,
                "task_id": 1,
                "episode_idx": 2,
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
    assert "posttrained_success" not in rows[0]
