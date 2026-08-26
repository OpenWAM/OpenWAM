from __future__ import annotations

from pathlib import Path

import pytest

from open_wam.utils import (
    resolve_transformer_dir_override,
    validate_positive_step_override,
)


def _write_transformer_export(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")


def test_validate_positive_step_override_preserves_none_and_positive_values() -> None:
    assert validate_positive_step_override("video_num_inference_steps", None) is None
    assert validate_positive_step_override("video_num_inference_steps", 20) == 20


def test_validate_positive_step_override_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="--video-num-inference-steps must be positive"):
        validate_positive_step_override("video_num_inference_steps", 0)


def test_resolve_transformer_dir_override_accepts_export_dir(tmp_path: Path) -> None:
    transformer_dir = tmp_path / "checkpoint_step_400" / "transformer"
    _write_transformer_export(transformer_dir)

    assert resolve_transformer_dir_override(transformer_dir) == transformer_dir.resolve()


def test_resolve_transformer_dir_override_accepts_checkpoint_step_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint_step_400"
    transformer_dir = checkpoint_dir / "transformer"
    _write_transformer_export(transformer_dir)

    assert resolve_transformer_dir_override(checkpoint_dir) == transformer_dir.resolve()


def test_resolve_transformer_dir_override_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="path does not exist"):
        resolve_transformer_dir_override(tmp_path / "missing")


def test_resolve_transformer_dir_override_rejects_non_export_dir(tmp_path: Path) -> None:
    bad_dir = tmp_path / "checkpoint_step_400"
    bad_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="usable transformer export"):
        resolve_transformer_dir_override(bad_dir)


def test_resolve_transformer_dir_override_rejects_config_without_weights(
    tmp_path: Path,
) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="complete safetensors weights"):
        resolve_transformer_dir_override(transformer_dir)
