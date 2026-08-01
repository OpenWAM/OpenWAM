"""Filesystem contracts for Open-WAM checkpoint and transformer artifacts.

This module deliberately has no tensor, model, or configuration-dataclass
dependencies. Commands can inspect checkpoint layouts before importing a
training or inference runtime, while tensor deserialization remains owned by
``open_wam.runtime.checkpoints``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


CHECKPOINT_FILENAMES = ("model_state.pt", "full_training_state.pt")


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
) -> Path | None:
    """Find preferred state in a file, step directory, or supported run root."""

    layout = CheckpointSearchLayout(layout)
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    direct = state_file_in_dir(candidate)
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
            checkpoint_file = state_file_in_dir(checkpoint_dir)
            if checkpoint_file is not None:
                return checkpoint_file
    return None


def state_file_in_dir(path: Path) -> Path | None:
    """Return the preferred state file directly inside ``path``."""

    for filename in CHECKPOINT_FILENAMES:
        checkpoint_file = path / filename
        if checkpoint_file.is_file():
            return checkpoint_file.resolve()
    return None


def sorted_checkpoint_dirs(root: Path, *, strict_steps: bool = False) -> list[Path]:
    """Return ``checkpoint_step_*`` children ordered by numeric step."""

    checkpoint_dirs = [path for path in root.glob("checkpoint_step_*") if path.is_dir()]
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
    if is_transformer_only_input_dir(candidate):
        return candidate, "input_transformer_dir"
    nested = candidate / "transformer"
    if is_transformer_only_input_dir(nested):
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
        "missing usable transformer export directory or resolved_config transformer_subdir fallback",
    )


def transformer_dir_from_resolved_config(config_path: Path) -> Path | None:
    """Read and resolve ``backbone.transformer_subdir`` from checkpoint config."""

    if not config_path.is_file():
        return None
    transformer_value = read_backbone_transformer_subdir(config_path)
    if not transformer_value:
        return None
    transformer_dir = Path(str(transformer_value)).expanduser()
    if not transformer_dir.is_absolute():
        transformer_dir = (config_path.parent / transformer_dir).resolve()
    return transformer_dir


def read_backbone_transformer_subdir(config_path: Path) -> str | None:
    """Read the configured transformer path with a minimal-parser fallback."""

    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return read_backbone_transformer_subdir_without_yaml(text)
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        return None
    backbone = raw.get("backbone", {})
    if not isinstance(backbone, dict):
        return None
    value = backbone.get("transformer_subdir")
    return str(value) if value else None


def read_backbone_transformer_subdir_without_yaml(text: str) -> str | None:
    """Read the one required YAML key when PyYAML is unavailable."""

    in_backbone = False
    backbone_indent: int | None = None
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
        if not in_backbone or not stripped.startswith("transformer_subdir:"):
            continue
        value = stripped.split(":", 1)[1].strip().strip("'\"")
        return value or None
    return None


def is_usable_transformer_dir(path: Path) -> bool:
    """Return whether a checkpoint-local or configured transformer is nonempty."""

    return path.is_dir() and any(path.iterdir())


def is_transformer_only_input_dir(path: Path) -> bool:
    """Return whether ``path`` is a canonical standalone transformer export."""

    return path.is_dir() and (path / "config.json").is_file() and has_transformer_weights(path)


def has_transformer_weights(path: Path) -> bool:
    """Return whether a transformer export contains supported safetensors."""

    return (
        (path / "diffusion_pytorch_model.safetensors").is_file()
        or (path / "diffusion_pytorch_model.safetensors.index.json").is_file()
        or any(path.glob("diffusion_pytorch_model-*.safetensors"))
    )


__all__ = [
    "CHECKPOINT_FILENAMES",
    "CheckpointArtifactResolution",
    "CheckpointSearchLayout",
    "checkpoint_step",
    "find_checkpoint_state_file",
    "has_transformer_weights",
    "is_transformer_only_input_dir",
    "is_usable_transformer_dir",
    "read_backbone_transformer_subdir",
    "read_backbone_transformer_subdir_without_yaml",
    "resolve_checkpoint_artifacts",
    "resolve_runtime_transformer_dir",
    "resolve_transformer_only_input",
    "sorted_checkpoint_dirs",
    "state_file_in_dir",
    "transformer_dir_from_resolved_config",
]
