"""Filesystem contracts for OpenWAM checkpoint and transformer artifacts.

This module deliberately has no tensor, model, or configuration-dataclass
dependencies. Commands can inspect checkpoint layouts before importing a
training or inference runtime, while tensor deserialization remains owned by
``open_wam.runtime.checkpoints``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from open_wam.contracts.paths import (
    resolve_model_component_path,
    validate_model_component_path,
)

CHECKPOINT_FILENAMES = ("model_state.pt", "full_training_state.pt")


class CheckpointOperation(StrEnum):
    """Intent governing checkpoint-state selection."""

    INITIALIZE_WEIGHTS = "initialize_weights"
    RESUME_TRAINING = "resume_training"
    EVALUATE = "evaluate"


class CheckpointSearchLayout(StrEnum):
    """Filesystem layouts accepted while locating checkpoint state."""

    STEP_OR_CHILD_STEPS = "step_or_child_steps"
    RUN_OR_STEP = "run_or_step"


@dataclass(frozen=True)
class CheckpointArtifactResolution:
    """Resolved state and transformer artifacts for one user input path."""

    raw: str | None
    checkpoint_file: str | None
    checkpoint_dir: str | None
    runtime_transformer_dir: str | None
    runtime_transformer_source: str | None
    problem: str | None = None


def resolve_checkpoint_artifacts(
    raw_path: str | Path | None,
) -> CheckpointArtifactResolution:
    """Resolve state and transformer artifacts from a run, step, or export.

    Unlike :func:`open_wam.runtime.checkpoints.resolve_checkpoint_file`, this
    preflight-oriented API reports problems in the returned record and accepts
    transformer-only model exports.
    """

    raw_value = str(raw_path) if raw_path is not None else None
    if raw_value is None or not raw_value.strip():
        return CheckpointArtifactResolution(
            raw=raw_value,
            checkpoint_file=None,
            checkpoint_dir=None,
            runtime_transformer_dir=None,
            runtime_transformer_source=None,
            problem="checkpoint was not provided",
        )

    candidate = Path(raw_value).expanduser()
    if not candidate.exists():
        return CheckpointArtifactResolution(
            raw=raw_value,
            checkpoint_file=None,
            checkpoint_dir=None,
            runtime_transformer_dir=None,
            runtime_transformer_source=None,
            problem="checkpoint path does not exist",
        )

    checkpoint_file = find_checkpoint_state_file(candidate)
    if checkpoint_file is None:
        transformer_dir, transformer_source = resolve_transformer_only_input(candidate)
        if transformer_dir is not None:
            return CheckpointArtifactResolution(
                raw=raw_value,
                checkpoint_file=None,
                checkpoint_dir=str(candidate.resolve()),
                runtime_transformer_dir=str(transformer_dir.resolve()),
                runtime_transformer_source=transformer_source,
                problem=None,
            )
        return CheckpointArtifactResolution(
            raw=raw_value,
            checkpoint_file=None,
            checkpoint_dir=None,
            runtime_transformer_dir=None,
            runtime_transformer_source=None,
            problem=(
                "could not resolve model_state.pt, full_training_state.pt, or transformer export "
                "(config.json plus diffusion_pytorch_model*.safetensors)"
            ),
        )

    checkpoint_dir = checkpoint_file.parent
    transformer_dir, transformer_source, transformer_problem = resolve_runtime_transformer_dir(
        checkpoint_dir
    )
    return CheckpointArtifactResolution(
        raw=raw_value,
        checkpoint_file=str(checkpoint_file.resolve()),
        checkpoint_dir=str(checkpoint_dir.resolve()),
        runtime_transformer_dir=(
            str(transformer_dir.resolve()) if transformer_dir is not None else None
        ),
        runtime_transformer_source=transformer_source,
        problem=transformer_problem,
    )


def find_checkpoint_state_file(
    path: str | Path,
    *,
    layout: CheckpointSearchLayout | str = CheckpointSearchLayout.RUN_OR_STEP,
    operation: CheckpointOperation | str = CheckpointOperation.EVALUATE,
) -> Path | None:
    """Find preferred state in a file, step directory, or supported run root."""

    layout = CheckpointSearchLayout(layout)
    operation = CheckpointOperation(operation)
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        if operation is CheckpointOperation.RESUME_TRAINING:
            return candidate if candidate.name == "full_training_state.pt" else None
        return candidate
    direct = state_file_in_dir(candidate, operation=operation)
    if direct is not None:
        return direct

    if layout is CheckpointSearchLayout.RUN_OR_STEP:
        roots = (candidate / "checkpoints", candidate)
        strict_steps = False
    elif layout is CheckpointSearchLayout.STEP_OR_CHILD_STEPS:
        roots = (candidate,)
        strict_steps = True
    for root in roots:
        if not root.is_dir():
            continue
        for checkpoint_dir in reversed(
            sorted_checkpoint_dirs(root, strict_steps=strict_steps)
        ):
            checkpoint_file = state_file_in_dir(
                checkpoint_dir,
                operation=operation,
            )
            if checkpoint_file is not None:
                return checkpoint_file
    return None


def state_file_in_dir(
    path: Path,
    *,
    operation: CheckpointOperation | str = CheckpointOperation.EVALUATE,
) -> Path | None:
    """Return the preferred state file directly inside ``path``."""

    for filename in _checkpoint_filenames_for_operation(
        CheckpointOperation(operation)
    ):
        checkpoint_file = path / filename
        if checkpoint_file.is_file():
            return checkpoint_file.resolve()
    return None


def _checkpoint_filenames_for_operation(
    operation: CheckpointOperation,
) -> tuple[str, ...]:
    if operation is CheckpointOperation.RESUME_TRAINING:
        return ("full_training_state.pt",)
    return CHECKPOINT_FILENAMES


def sorted_checkpoint_dirs(root: Path, *, strict_steps: bool = False) -> list[Path]:
    """Return ``checkpoint_step_*`` children ordered by numeric step."""

    checkpoint_dirs = [path for path in root.glob("checkpoint_step_*") if path.is_dir()]
    completed_dirs = [
        path for path in checkpoint_dirs if (path / ".checkpoint_complete").is_file()
    ]
    if completed_dirs:
        checkpoint_dirs = completed_dirs
    return sorted(
        checkpoint_dirs,
        key=lambda path: checkpoint_step(path, strict=strict_steps),
    )


def checkpoint_step(path: Path, *, strict: bool = False) -> int:
    """Parse a checkpoint step, treating malformed names as oldest by default."""

    try:
        return int(path.name.rsplit("_", 1)[-1])
    except ValueError:
        if strict:
            raise
        return -1


def resolve_transformer_only_input(path: Path) -> tuple[Path | None, str | None]:
    """Resolve a canonical transformer export supplied without model state."""

    candidate = path.expanduser().resolve()
    if is_usable_transformer_dir(candidate):
        return candidate, "input_transformer_dir"
    nested = candidate / "transformer"
    if is_usable_transformer_dir(nested):
        return nested.resolve(), "input_transformer_subdir"
    return None, None


def resolve_runtime_transformer_dir(
    checkpoint_dir: Path,
) -> tuple[Path | None, str | None, str | None]:
    """Resolve the transformer used alongside one checkpoint state file."""

    local_transformer = checkpoint_dir / "transformer"
    if is_usable_transformer_dir(local_transformer):
        return local_transformer, "checkpoint", None

    config_transformer = transformer_dir_from_resolved_config(
        checkpoint_dir / "resolved_config.yaml"
    )
    if config_transformer is not None and is_usable_transformer_dir(config_transformer):
        return config_transformer, "resolved_config", None

    if local_transformer.is_dir():
        return (
            None,
            None,
            "checkpoint transformer directory exists but is empty or unusable, and resolved_config fallback is missing",
        )
    return (
        None,
        None,
        "missing usable transformer export directory or resolved_config runtime-backbone fallback",
    )


def transformer_dir_from_resolved_config(config_path: Path) -> Path | None:
    """Read and resolve the runtime-backbone locator from checkpoint config."""

    if not config_path.is_file():
        return None
    backbone = _read_backbone_config(config_path)
    if not backbone:
        return None
    try:
        transformer_subdir = backbone.get("transformer_subdir") or "transformer"
        validate_model_component_path(
            transformer_subdir,
            field_name="backbone.transformer_subdir",
        )
        return resolve_model_component_path(
            backbone.get("pretrained_model_name_or_path"),
            transformer_subdir,
            artifact_path=backbone.get("runtime_backbone_artifact_path"),
            field_name="backbone.transformer_subdir",
        )
    except (TypeError, ValueError):
        return None


def _read_backbone_config(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _read_backbone_config_without_yaml(text)
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        return {}
    backbone = raw.get("backbone", {})
    return dict(backbone) if isinstance(backbone, dict) else {}


def _read_backbone_config_without_yaml(text: str) -> dict[str, str]:
    """Read path-valued backbone fields without importing the config stack."""

    in_backbone = False
    backbone_indent: int | None = None
    values: dict[str, str] = {}
    field_names = {
        "pretrained_model_name_or_path",
        "runtime_backbone_artifact_path",
        "transformer_subdir",
    }
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "backbone:":
            in_backbone = True
            backbone_indent = indent
            continue
        if in_backbone and backbone_indent is not None and indent <= backbone_indent:
            in_backbone = False
        if not in_backbone:
            continue
        key, separator, raw_value = stripped.partition(":")
        if separator and key in field_names:
            value = raw_value.strip().strip("'\"")
            if value.lower() not in {"", "null", "~"}:
                values[key] = value
    return values


def is_usable_transformer_dir(path: Path) -> bool:
    """Return whether ``path`` is a complete, loadable transformer export."""

    return (
        path.is_dir()
        and _read_json_mapping(path / "config.json") is not None
        and _has_transformer_weights(path)
    )


def _has_transformer_weights(path: Path) -> bool:
    """Return whether an export contains one complete safetensors payload."""

    single_file = path / "diffusion_pytorch_model.safetensors"
    if _is_nonempty_file(single_file):
        return True

    index_path = path / "diffusion_pytorch_model.safetensors.index.json"
    index = _read_json_mapping(index_path)
    if index is None:
        return False
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        return False
    shard_values = tuple(weight_map.values())
    if not all(isinstance(name, str) and name.strip() for name in shard_values):
        return False
    shard_names = set(shard_values)
    for shard_name in shard_names:
        relative_path = Path(shard_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return False
        if not _is_nonempty_file(path / relative_path):
            return False
    return True


def _read_json_mapping(path: Path) -> Mapping[str, Any] | None:
    if not _is_nonempty_file(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


__all__ = [
    "CHECKPOINT_FILENAMES",
    "CheckpointArtifactResolution",
    "CheckpointSearchLayout",
    "checkpoint_step",
    "find_checkpoint_state_file",
    "is_usable_transformer_dir",
    "resolve_checkpoint_artifacts",
    "resolve_runtime_transformer_dir",
    "resolve_transformer_only_input",
    "sorted_checkpoint_dirs",
    "state_file_in_dir",
    "transformer_dir_from_resolved_config",
]
