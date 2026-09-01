from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from open_wam.runtime.checkpoint_artifacts import (
    CheckpointOperation,
    CheckpointSearchLayout,
    checkpoint_step,
    find_checkpoint_state_file,
    resolve_checkpoint_artifacts,
    transformer_dir_from_resolved_config,
)


def _write_transformer_export(path: Path, *, sharded: bool = False) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    if not sharded:
        (path / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
        return
    shard_names = (
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    )
    for shard_name in shard_names:
        (path / shard_name).write_bytes(b"weights")
    (path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "block.0.weight": shard_names[0],
                    "block.1.weight": shard_names[1],
                }
            }
        ),
        encoding="utf-8",
    )


def test_resolve_checkpoint_artifacts_accepts_run_root_and_explicit_transformer(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    checkpoint_10 = run_root / "checkpoints" / "checkpoint_step_10"
    checkpoint_20 = run_root / "checkpoints" / "checkpoint_step_20"
    checkpoint_10.mkdir(parents=True)
    checkpoint_20.mkdir(parents=True)
    (checkpoint_10 / "model_state.pt").write_bytes(b"old")
    (checkpoint_20 / "full_training_state.pt").write_bytes(b"new")
    transformer = tmp_path / "shared_transformer"
    _write_transformer_export(transformer)
    (checkpoint_20 / "resolved_config.yaml").write_text(
        "backbone:\n"
        f"  runtime_backbone_artifact_path: {transformer}\n",
        encoding="utf-8",
    )

    resolution = resolve_checkpoint_artifacts(run_root)

    assert resolution.checkpoint_file == str(
        (checkpoint_20 / "full_training_state.pt").resolve()
    )
    assert resolution.raw == str(run_root)
    assert resolution.checkpoint_dir == str(checkpoint_20.resolve())
    assert resolution.runtime_transformer_dir == str(transformer.resolve())
    assert resolution.runtime_transformer_source == "resolved_config"
    assert resolution.problem is None


@pytest.mark.parametrize("incomplete_payload", ["config_only", "missing_index_shard"])
def test_checkpoint_transformer_must_be_a_complete_export(
    tmp_path: Path,
    incomplete_payload: str,
) -> None:
    checkpoint = tmp_path / "checkpoint_step_1"
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    (checkpoint / "model_state.pt").write_bytes(b"state")
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    if incomplete_payload == "missing_index_shard":
        (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "block.weight": "diffusion_pytorch_model-00001-of-00001.safetensors"
                    }
                }
            ),
            encoding="utf-8",
        )

    resolution = resolve_checkpoint_artifacts(checkpoint)

    assert resolution.runtime_transformer_dir is None
    assert resolution.runtime_transformer_source is None
    assert resolution.problem is not None
    assert "unusable" in resolution.problem


@pytest.mark.parametrize("absolute_component", [False, True])
def test_resolved_config_transformer_component_uses_model_root(
    tmp_path: Path,
    absolute_component: bool,
) -> None:
    checkpoint = tmp_path / "checkpoint_step_1"
    checkpoint.mkdir()
    (checkpoint / "model_state.pt").write_bytes(b"state")
    model_root = tmp_path / "model"
    transformer = model_root / "transformer"
    _write_transformer_export(transformer)
    component = str(transformer) if absolute_component else "transformer"
    (checkpoint / "resolved_config.yaml").write_text(
        "backbone:\n"
        f"  pretrained_model_name_or_path: {model_root}\n"
        f"  transformer_subdir: {component}\n",
        encoding="utf-8",
    )

    resolution = resolve_checkpoint_artifacts(checkpoint)

    assert resolution.runtime_transformer_dir == str(transformer.resolve())
    assert resolution.runtime_transformer_source == "resolved_config"
    assert resolution.problem is None


def test_resolve_checkpoint_artifacts_accepts_nested_transformer_only_export(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    transformer = model_root / "transformer"
    _write_transformer_export(transformer, sharded=True)

    resolution = resolve_checkpoint_artifacts(str(model_root))

    assert resolution.checkpoint_file is None
    assert resolution.checkpoint_dir == str(model_root.resolve())
    assert resolution.runtime_transformer_dir == str(transformer.resolve())
    assert resolution.runtime_transformer_source == "input_transformer_subdir"
    assert resolution.problem is None


def test_checkpoint_state_preference_and_search_layout_are_explicit(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    step = run_root / "checkpoints" / "checkpoint_step_2"
    step.mkdir(parents=True)
    model_state = step / "model_state.pt"
    model_state.write_bytes(b"model")
    (step / "full_training_state.pt").write_bytes(b"full")

    assert find_checkpoint_state_file(run_root) == model_state.resolve()
    assert (
        find_checkpoint_state_file(
            run_root,
            layout=CheckpointSearchLayout.STEP_OR_CHILD_STEPS.value,
        )
        is None
    )


def test_run_root_selection_ignores_unmarked_steps_when_completed_steps_exist(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "checkpoint_step_10"
    incomplete = tmp_path / "checkpoint_step_20"
    complete.mkdir()
    incomplete.mkdir()
    (complete / "full_training_state.pt").touch()
    (complete / ".checkpoint_complete").touch()
    (incomplete / "full_training_state.pt").touch()

    assert find_checkpoint_state_file(
        tmp_path,
        operation=CheckpointOperation.RESUME_TRAINING,
    ) == (complete / "full_training_state.pt").resolve()


def test_checkpoint_step_malformed_name_policy_preserves_runtime_compatibility(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "checkpoint_step_2"
    malformed = tmp_path / "checkpoint_step_latest"
    valid.mkdir()
    malformed.mkdir()
    (valid / "model_state.pt").write_bytes(b"valid")
    (malformed / "model_state.pt").write_bytes(b"malformed")

    assert checkpoint_step(malformed) == -1
    assert find_checkpoint_state_file(tmp_path) == (valid / "model_state.pt").resolve()
    with pytest.raises(ValueError, match="invalid literal for int"):
        find_checkpoint_state_file(
            tmp_path,
            layout=CheckpointSearchLayout.STEP_OR_CHILD_STEPS,
        )


@pytest.mark.parametrize(
    ("raw_path", "problem"),
    [
        (None, "checkpoint was not provided"),
        (" ", "checkpoint was not provided"),
    ],
)
def test_resolve_checkpoint_artifacts_reports_missing_input(
    raw_path: str | None,
    problem: str,
) -> None:
    resolution = resolve_checkpoint_artifacts(raw_path)

    assert resolution.problem == problem
    assert resolution.checkpoint_file is None
    assert resolution.runtime_transformer_dir is None


def test_minimal_runtime_backbone_parser_is_scope_aware_and_prefers_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = tmp_path / "explicit-transformer"
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(
        f"""
transformer_subdir: ignored
backbone:
  pretrained_model_name_or_path: base
  transformer_subdir: 'exports/legacy-transformer'  # archived path
  runtime_backbone_artifact_path: '{explicit}'
policy_variant:
  transformer_subdir: ignored_too
""",
        encoding="utf-8",
    )
    _disable_yaml_import(monkeypatch)

    assert transformer_dir_from_resolved_config(config_path) == explicit


def test_minimal_runtime_backbone_parser_resolves_component_under_model_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "model"
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(
        f"""
backbone:
  pretrained_model_name_or_path: '{model_root}'
  runtime_backbone_artifact_path: null
  transformer_subdir: 'exports/transformer'
""",
        encoding="utf-8",
    )
    _disable_yaml_import(monkeypatch)

    assert transformer_dir_from_resolved_config(config_path) == (
        model_root / "exports/transformer"
    )


def _disable_yaml_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)
