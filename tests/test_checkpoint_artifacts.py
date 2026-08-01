from __future__ import annotations

from pathlib import Path

import pytest

from open_wam.runtime.checkpoint_artifacts import (
    CheckpointSearchLayout,
    checkpoint_step,
    find_checkpoint_state_file,
    read_backbone_transformer_subdir_without_yaml,
    resolve_checkpoint_artifacts,
)


def _write_transformer_export(path: Path, *, sharded: bool = False) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    weight_name = (
        "diffusion_pytorch_model-00001-of-00002.safetensors"
        if sharded
        else "diffusion_pytorch_model.safetensors"
    )
    (path / weight_name).write_bytes(b"weights")


def test_resolve_checkpoint_artifacts_accepts_run_root_and_relative_transformer(
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
    transformer.mkdir()
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_20 / "resolved_config.yaml").write_text(
        "backbone:\n  transformer_subdir: ../../../shared_transformer\n",
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


def test_minimal_transformer_subdir_parser_is_scope_aware() -> None:
    text = """
transformer_subdir: ignored
backbone:
  pretrained_model_name_or_path: base
  transformer_subdir: 'exports/transformer'  # portable path
policy_variant:
  transformer_subdir: ignored_too
"""

    assert (
        read_backbone_transformer_subdir_without_yaml(text)
        == "exports/transformer"
    )
