from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest

from open_wam.cli import sanity as cli
from open_wam.evals import sanity as runtime

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sanity_parser_defaults_and_deprecated_opt_in() -> None:
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--cfg", "experiment.yaml"])
    assert vars(args) == {
        "allow_deprecated_libero_config": False,
        "batch_size": None,
        "config": "experiment.yaml",
        "device": "cpu",
        "extension": [],
        "max_batches": 1,
        "output_json": None,
        "provenance_mode": "standard",
        "require_gpu": False,
        "rollout_steps": 3,
        "seed": 0,
        "split": "val",
    }
    opted_in = parser.parse_args(
        ["--cfg", "experiment.yaml", "--allow-deprecated-libero-config"]
    )
    assert opted_in.allow_deprecated_libero_config is True


def test_sanity_cli_lazily_delegates_parsed_arguments(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    fake_runtime = types.ModuleType("open_wam.evals.sanity")
    fake_runtime.run_sanity_command = lambda args: captured.update(vars(args))
    monkeypatch.setitem(sys.modules, "open_wam.evals.sanity", fake_runtime)

    cli.main(
        [
            "--cfg",
            "experiment.yaml",
            "--split",
            "train",
            "--max-batches",
            "1",
            "--allow-deprecated-libero-config",
        ]
    )

    assert captured["config"] == "experiment.yaml"
    assert captured["split"] == "train"
    assert captured["max_batches"] == 1
    assert captured["allow_deprecated_libero_config"] is True


def test_sanity_cli_reports_missing_train_extra(monkeypatch) -> None:
    real_import = builtins.__import__

    def fail_runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "open_wam.evals.sanity":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_runtime_import)

    with pytest.raises(SystemExit, match=r"open-wam\[train\].*torch"):
        cli.main(["--cfg", "experiment.yaml"])


def test_sanity_rejects_the_historical_unused_multi_batch_value() -> None:
    with pytest.raises(SystemExit, match=r"exactly one batch.*open-wam-eval"):
        cli.main(["--cfg", "experiment.yaml", "--max-batches", "2"])


def test_sanity_rejects_nonpositive_explicit_batch_size() -> None:
    with pytest.raises(SystemExit, match="--batch-size must be positive"):
        cli.main(["--cfg", "experiment.yaml", "--batch-size", "0"])


def test_sanity_dataset_dispatch_uses_batch_contract_not_dataset_name(monkeypatch) -> None:
    data_config = object()
    latent_result = (object(), object())
    raw_result = (object(), object())
    monkeypatch.setattr(
        runtime,
        "build_train_val_latent_datasets",
        lambda observed: latent_result if observed is data_config else None,
    )
    monkeypatch.setattr(
        runtime,
        "build_train_val_datasets",
        lambda observed: raw_result if observed is data_config else None,
    )

    assert runtime._build_datasets(data_config, latent=True) is latent_result
    assert runtime._build_datasets(data_config, latent=False) is raw_result


def test_sanity_command_public_tiny_numerical_contract(tmp_path: Path, capsys) -> None:
    config_path = REPO_ROOT / "configs" / "examples" / "public_tiny_synthetic_contract.yaml"
    output_path = tmp_path / "sanity.json"
    cli.main(
        [
            "--cfg",
            str(config_path),
            "--device",
            "cpu",
            "--max-batches",
            "1",
            "--rollout-steps",
            "1",
            "--seed",
            "17",
            "--output-json",
            str(output_path),
        ],
    )

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == summary
    assert summary["schema_version"] == "open_wam.result.v1"
    assert summary["command"] == "open-wam-sanity"
    assert summary["benchmark"] == "public_tiny"
    assert summary["seed"] == 17
    assert summary["metrics"]["rollout_steps"] == 1
    assert summary["metrics"]["train_loss"] == pytest.approx(
        2.5530142784118652,
        rel=0.0,
        abs=1e-6,
    )
    assert summary["mapping"] == {
        "action_dim": 4,
        "action_mapping_mode": "none",
    }
    assert summary["load"] == {
        "action_mask_sum": 8.0,
        "actions_shape": [1, 2, 4],
        "canonical_video_shape": [1, 3, 2, 64, 64],
        "expected_canonical_video_shape": [1, 3, 2, 64, 64],
        "expected_video_latents_shape": [1, 48, 2, 4, 4],
        "state_shape": [1, 1, 3],
        "task_text_count": 1,
        "view_shapes": {"camera_0": [1, 2, 64, 64, 3]},
    }
    assert summary["train_forward"]["loss"] == pytest.approx(
        2.5530142784118652,
        rel=0.0,
        abs=1e-6,
    )
    train_metrics = summary["train_forward"]["metrics"]
    assert train_metrics["action_diffusion_loss"] == pytest.approx(
        2.5530142784118652,
        rel=0.0,
        abs=1e-6,
    )
    assert train_metrics["weighted_action_diffusion_loss"] == pytest.approx(
        2.5530142784118652,
        rel=0.0,
        abs=1e-6,
    )
    assert train_metrics["action_mse"] == pytest.approx(
        0.16568462550640106,
        rel=0.0,
        abs=1e-7,
    )
    assert summary["batch_infer"]["action_pred_shape"] == [1, 2, 4]
    assert summary["batch_infer"]["target_action_shape"] == [1, 2, 4]
    assert summary["batch_infer"]["masked_action_mse"] == pytest.approx(
        5.424690246582031,
        rel=0.0,
        # Supported Python 3.11/3.12 CPU wheels differ by one float32 ULP for
        # this reduction. Real-checkpoint parity uses the stricter GPU gate.
        abs=1e-6,
    )
    assert summary["rollout_style_infer"]["steps"] == 1
    assert summary["rollout_style_infer"]["action_pred_shapes"] == [[1, 2, 4]]
