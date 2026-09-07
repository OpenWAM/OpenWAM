from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised in RoboTwin's Python 3.10 env.
    import tomli as tomllib


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the active OpenWAM source root, falling back to the current directory.

    Source checkouts are detected by walking upward from ``start`` for an
    OpenWAM ``pyproject.toml`` plus the expected source package layout. Wheel
    installs do not contain those source markers, so repo-relative paths resolve
    from the caller's working directory in that case.
    """

    for root in _candidate_roots(start or Path(__file__)):
        if _is_open_wam_root(root):
            return root

    cwd = Path.cwd().resolve()
    for root in _candidate_roots(cwd):
        if _is_open_wam_root(root):
            return root
    return cwd


def _candidate_roots(start: str | Path) -> tuple[Path, ...]:
    path = Path(start).expanduser().resolve()
    if path.is_file():
        path = path.parent
    return (path, *path.parents)


def _is_open_wam_root(path: Path) -> bool:
    if not _has_open_wam_source_layout(path):
        return False
    pyproject_path = path / "pyproject.toml"
    if not pyproject_path.is_file():
        return False
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return pyproject.get("project", {}).get("name") == "openwam"


def _has_open_wam_source_layout(path: Path) -> bool:
    return (path / "src" / "open_wam").is_dir() or (path / "open_wam").is_dir()


REPO_ROOT = find_repo_root(Path(__file__))


def resolve_repo_path(value: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Resolve a path relative to the active source root while preserving absolute paths."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = find_repo_root() if repo_root is None else Path(repo_root).expanduser().resolve()
    return (root / path).resolve()


def validate_model_component_path(
    value: str | Path,
    *,
    field_name: str = "model component path",
) -> Path:
    """Validate an absolute artifact path or a component beneath a model root."""

    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"`{field_name}` must be a non-empty path.")
    path = Path(value).expanduser()
    if not path.is_absolute() and ".." in path.parts:
        raise ValueError(
            f"Relative `{field_name}` cannot contain `..`; use an absolute path "
            "for an artifact outside the model root."
        )
    return path


def resolve_model_component_path(
    pretrained_model_name_or_path: str | Path | None,
    component_path: str | Path,
    *,
    artifact_path: str | Path | None = None,
    field_name: str = "model component path",
) -> Path | None:
    """Resolve an explicit artifact or a component located under a model root.

    Explicit artifacts take precedence. An absolute component path remains
    usable for historical configs; a relative component path is interpreted
    beneath ``pretrained_model_name_or_path``.
    """

    if not isinstance(component_path, (str, Path)) or not str(
        component_path
    ).strip():
        raise ValueError(f"`{field_name}` must be a non-empty path.")
    component = Path(component_path).expanduser()
    if artifact_path is not None:
        if not isinstance(artifact_path, (str, Path)) or not str(
            artifact_path
        ).strip():
            raise ValueError("`artifact_path` must be a non-empty path when set.")
        return Path(artifact_path).expanduser()
    if component.is_absolute():
        return component
    if pretrained_model_name_or_path is None:
        return None

    root = Path(pretrained_model_name_or_path).expanduser()
    candidate = root / component
    if candidate.exists():
        return candidate
    if (root / "config.json").exists():
        return root
    return candidate
