from __future__ import annotations

import hashlib
import json
from pathlib import Path

from open_wam.evals import sampled_eval_reporting


def _summary_fixture() -> dict[str, object]:
    return {
        "run_id": "report-contract",
        "benchmark": "libero_10",
        "sample_mode": "task_episode_axis",
        "distribution_episode_strategy": None,
        "eval_profile": "libero_10hz_full",
        "rollout_artifact_profile": "lean",
        "scheduler_profile": "blocking_control",
        "replay_status_policy": "successful_only",
        "replay_status_path": "/dataset/meta/replay_status.jsonl",
        "dataset_root": "/dataset",
        "output_root": "/output",
        "log_root": "/output/_sampled_eval",
        "sample_warnings": ["warning with | separator"],
        "task_allocations": {"lift|mug": 1},
        "target_keys": ["alpha", "beta"],
        "checkpoint_specs": [
            {"key": "alpha", "label": "Alpha | base"},
            {"key": "beta", "label": "Beta"},
        ],
        "by_checkpoint": {
            "alpha": {
                "label": "Alpha | base",
                "method_label": "M1 exact",
                "success": 1,
                "finished": 1,
                "total": 1,
                "failed_cases": 0,
            },
            "beta": {
                "label": "Beta",
                "method_label": "M5 joint",
                "success": 0,
                "finished": 0,
                "total": 1,
                "failed_cases": 1,
            },
        },
        "paired_rows": [
            {
                "sample_index": 0,
                "dataset_episode_index": 10,
                "episode_id": 110,
                "task_id": 2,
                "init_id": 7,
                "episode_idx": 7,
                "resolved_init_state_index": 17,
                "init_id_source": "task_local_rank",
                "replay_status": "success",
                "task_text": "lift|mug",
                "alpha_status": "completed",
                "alpha_returncode": 0,
                "alpha_success": True,
                "alpha_executed_actions": 16,
                "alpha_fallback_actions": 0,
                "alpha_summary_path": "/output/alpha.json",
                "beta_status": "failed",
                "beta_returncode": 1,
                "beta_success": None,
                "beta_executed_actions": None,
                "beta_fallback_actions": None,
                "beta_summary_path": None,
            }
        ],
    }


def test_sampled_eval_renderers_preserve_stable_bytes(tmp_path: Path) -> None:
    summary = _summary_fixture()
    csv_path = tmp_path / "results.csv"
    markdown_path = tmp_path / "summary.md"

    sampled_eval_reporting.write_sampled_eval_results_csv(csv_path, summary)
    sampled_eval_reporting.write_sampled_eval_summary_markdown(markdown_path, summary)

    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == (
        "86b081b012d0316dcc2fcc8aad6a982d65236b6e753c17825a69a29f58bb92af"
    )
    assert hashlib.sha256(markdown_path.read_bytes()).hexdigest() == (
        "8cfa3918ccb6262b7e3a73c04cdcabdb405004944c0e8dfe1b335c6d1b7ef3b0"
    )


def test_collect_sampled_eval_run_joins_status_and_rollout_summary(tmp_path: Path) -> None:
    log_root = tmp_path / "_sampled_eval"
    status_root = log_root / "status"
    rollout_root = tmp_path / "rollouts"
    status_root.mkdir(parents=True)
    rollout_root.mkdir()
    manifest = {
        "run_id": "collect-contract",
        "output_root": str(tmp_path),
        "log_root": str(log_root),
        "dataset_root": "/dataset",
        "benchmark": "libero_10",
        "num_sampled_episodes": 1,
        "task_allocations": {"task": 1},
        "checkpoint_specs": [
            {
                "key": "alpha",
                "label": "Alpha",
                "method_key": "m1",
                "method_label": "M1 exact",
            }
        ],
    }
    case = {
        "index": 0,
        "sample_index": 0,
        "checkpoint_key": "alpha",
        "checkpoint_label": "Alpha",
        "method_key": "m1",
        "method_label": "M1 exact",
        "dataset_episode_index": 10,
        "task_id": 2,
        "task_text": "task",
        "episode_idx": 7,
        "summary_glob": str(rollout_root / "*_summary.json"),
    }
    (log_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (log_root / "cases.json").write_text(json.dumps([case]), encoding="utf-8")
    (status_root / "0000_alpha_sample000.json").write_text(
        json.dumps({"state": "completed", "returncode": 0}),
        encoding="utf-8",
    )
    (rollout_root / "rollout_summary.json").write_text(
        json.dumps({"success": True, "executed_actions": 16, "fallback_actions": 0}),
        encoding="utf-8",
    )

    summary = sampled_eval_reporting.collect_sampled_eval_run(tmp_path)

    assert summary["by_checkpoint"]["alpha"]["success"] == 1
    assert summary["paired_rows"][0]["alpha_executed_actions"] == 16
    assert (log_root / "summary.json").is_file()
    assert (log_root / "results.csv").is_file()
    assert (log_root / "summary.md").is_file()
