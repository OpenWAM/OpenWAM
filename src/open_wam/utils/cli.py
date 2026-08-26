from __future__ import annotations

from pathlib import Path

from open_wam.runtime.checkpoint_artifacts import is_usable_transformer_dir


def validate_positive_step_override(name: str, value: int | None) -> int | None:
    """Validate optional denoising step-count CLI overrides."""
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        cli_name = "--" + name.replace("_", "-")
        raise ValueError(f"{cli_name} must be positive when provided; got {resolved}.")
    return resolved


def resolve_transformer_dir_override(
    path: str | Path,
    *,
    option_name: str = "--transformer-dir",
) -> Path:
    """Resolve a transformer export dir, accepting checkpoint roots with transformer/."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{option_name} path does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(
            f"{option_name} must be a directory or checkpoint directory containing transformer/: {resolved}"
        )

    if is_usable_transformer_dir(resolved):
        return resolved
    transformer_candidate = resolved / "transformer"
    if is_usable_transformer_dir(transformer_candidate):
        return transformer_candidate.resolve()
    raise FileNotFoundError(
        f"{option_name} must point to a usable transformer export (config.json plus "
        "complete safetensors weights), or to a checkpoint directory containing "
        f"one under transformer/: {resolved}"
    )
