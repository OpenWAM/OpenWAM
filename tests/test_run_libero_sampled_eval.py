from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import sys


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
    assert [episode.dataset_episode_index for episode in first] == [
        episode.dataset_episode_index for episode in second
    ]
    assert first_allocations == second_allocations


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
