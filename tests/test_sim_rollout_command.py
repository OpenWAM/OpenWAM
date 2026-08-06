from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from open_wam.cli import sim_rollout as cli
from open_wam.evals import sim_rollout as runtime
from open_wam.simulators import SimActionCommitMode, SimRolloutResult


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sim_rollout_parser_defaults_match_runtime_contract() -> None:
    parser = cli.build_arg_parser()
    action_commit = next(
        action for action in parser._actions if action.dest == "action_commit_mode"
    )
    assert tuple(action_commit.choices) == tuple(mode.value for mode in SimActionCommitMode)

    args = parser.parse_args(["--cfg", "experiment.yaml", "--benchmark", "calvin"])
    assert vars(args) == {
        "action_commit_mode": SimActionCommitMode.FIRST_ACTION.value,
        "allow_partial_checkpoint": False,
        "benchmark": "calvin",
        "calvin_dataset_root": None,
        "calvin_root": None,
        "calvin_task_text": None,
        "checkpoint": None,
        "config": "experiment.yaml",
        "device": "auto",
        "episode_idx": 0,
        "extension": [],
        "instruction": None,
        "max_steps": 80,
        "output_dir": "outputs/sim_realtime",
        "provenance_mode": "standard",
        "robotwin_action_type": "ee",
        "robotwin_expert_precheck": False,
        "robotwin_instruction_type": "seen",
        "robotwin_root": None,
        "robotwin_task_config": None,
        "robotwin_task_name": None,
        "seed": 0,
        "show_gui": False,
        "sim_option": [],
        "suffix": "rollout",
        "target_action_hz": None,
        "task_id": None,
        "video_fps": 15.0,
        "zero_policy": False,
    }


def test_sim_rollout_cli_lazily_delegates_parsed_arguments(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    fake_runtime = types.ModuleType("open_wam.evals.sim_rollout")
    fake_runtime.run_simulator_rollout_command = lambda args: captured.update(vars(args))
    monkeypatch.setitem(sys.modules, "open_wam.evals.sim_rollout", fake_runtime)

    cli.main(["--cfg", "experiment.yaml", "--benchmark", "calvin", "--max-steps", "9"])

    assert captured["config"] == "experiment.yaml"
    assert captured["benchmark"] == "calvin"
    assert captured["max_steps"] == 9


def test_sim_rollout_cli_reports_missing_simulator_extra(monkeypatch) -> None:
    real_import = builtins.__import__

    def fail_runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "open_wam.evals.sim_rollout":
            raise ModuleNotFoundError("No module named 'imageio'", name="imageio")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_runtime_import)

    with pytest.raises(SystemExit, match=r"open-wam\[sim\].*imageio"):
        cli.main(["--cfg", "experiment.yaml", "--benchmark", "calvin"])


def test_sim_rollout_command_preserves_controls_cleanup_and_result_envelope(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    adapter = _FakeAdapter()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(runtime, "_build_adapter", lambda args: adapter)

    def fake_rollout(**kwargs):
        captured.update(kwargs)
        return SimRolloutResult(
            benchmark="calvin",
            task_text="put the object away",
            success=True,
            steps=2,
            target_action_hz=4.0,
            wall_time_s=0.5,
            mean_policy_step_s=0.01,
            mean_env_step_s=0.02,
            achieved_action_hz=4.0,
            policy_action_shapes=((1, 2, 4),),
            action_records=({"action_index": 0},),
            video_frames=(),
        )

    monkeypatch.setattr(runtime, "run_closed_loop_sim_rollout", fake_rollout)
    config_path = REPO_ROOT / "configs" / "examples" / "public_tiny_synthetic_contract.yaml"
    args = cli.build_arg_parser().parse_args(
        [
            "--cfg",
            str(config_path),
            "--benchmark",
            "calvin",
            "--zero-policy",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--suffix",
            "contract",
            "--action-commit-mode",
            "full_chunk",
            "--max-steps",
            "3",
            "--seed",
            "7",
        ]
    )

    runtime.run_simulator_rollout_command(args)

    assert adapter.closed is True
    assert captured["adapter"] is adapter
    assert captured["action_commit_mode"] == "full_chunk"
    assert captured["max_steps"] == 3
    assert captured["seed"] == 7
    assert captured["task_id"] is None
    assert captured["episode_idx"] == 0
    assert isinstance(captured["rollout_runner"], runtime._ZeroActionRolloutRunner)

    summary_path = tmp_path / "calvin_contract.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "open_wam.result.v1"
    assert summary["command"] == "open-wam-sim-rollout"
    assert summary["benchmark"] == "calvin"
    assert summary["seed"] == 7
    assert summary["metrics"] == {
        "achieved_action_hz": 4.0,
        "steps": 2,
        "success": True,
    }
    assert summary["artifacts"] == {
        "summary_path": str(summary_path),
        "video_path": None,
    }
    assert summary["action_commit_mode"] == "full_chunk"
    assert summary["checkpoint_compatibility"] == "strict"
    assert summary["checkpoint_missing_keys"] == []
    assert summary["checkpoint_unexpected_keys"] == []
    assert summary["zero_policy"] is True
    assert json.loads(capsys.readouterr().out) == summary


def test_checkpoint_backbone_override_is_pure(tmp_path: Path) -> None:
    config = runtime.load_experiment_config(
        REPO_ROOT / "configs" / "examples" / "public_tiny_synthetic_contract.yaml"
    )
    original_transformer = config.backbone.transformer_subdir
    checkpoint_path = tmp_path / "checkpoint_step_4" / "model_state.pt"
    transformer_dir = checkpoint_path.parent / "transformer"
    transformer_dir.mkdir(parents=True)

    resolved = runtime._with_checkpoint_backbone_override(
        config,
        checkpoint_path=checkpoint_path,
    )

    assert resolved.backbone.transformer_subdir == str(transformer_dir.resolve())
    assert config.backbone.transformer_subdir == original_transformer


def test_sim_rollout_command_closes_adapter_when_rollout_fails(monkeypatch, tmp_path: Path) -> None:
    adapter = _FakeAdapter()
    monkeypatch.setattr(runtime, "_build_adapter", lambda args: adapter)

    def fail_rollout(**kwargs):
        del kwargs
        raise RuntimeError("simulator step failed")

    monkeypatch.setattr(runtime, "run_closed_loop_sim_rollout", fail_rollout)
    config_path = REPO_ROOT / "configs" / "examples" / "public_tiny_synthetic_contract.yaml"
    args = cli.build_arg_parser().parse_args(
        [
            "--cfg",
            str(config_path),
            "--benchmark",
            "calvin",
            "--zero-policy",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(RuntimeError, match="simulator step failed"):
        runtime.run_simulator_rollout_command(args)

    assert adapter.closed is True


class _FakeAdapter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True
