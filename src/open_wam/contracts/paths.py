from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised in RoboTwin's Python 3.10 env.
    import tomli as tomllib


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the active Open-WAM source root, falling back to the current directory.

    Source checkouts are detected by walking upward from ``start`` for an
    Open-WAM ``pyproject.toml`` plus the expected source package layout. Wheel
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
    return pyproject.get("project", {}).get("name") == "open-wam"


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
