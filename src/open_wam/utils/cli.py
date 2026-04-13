from __future__ import annotations

from pathlib import Path


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

    transformer_candidate = resolved / "transformer"
    if not (resolved / "config.json").is_file() and transformer_candidate.is_dir():
        resolved = transformer_candidate.resolve()

    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(
            f"{option_name} must point to a transformer export directory with config.json, "
            f"or to a checkpoint directory containing transformer/config.json: {resolved}"
        )
    return resolved
