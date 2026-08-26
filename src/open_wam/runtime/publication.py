"""Failure-safe publication helpers for directory-shaped runtime artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_UMASK_LOCK = threading.Lock()


def _current_process_umask() -> int:
    with _UMASK_LOCK:
        current = os.umask(0)
        os.umask(current)
    return current


def _prepare_directory_for_publication(path: Path) -> None:
    special_bits = path.stat().st_mode & 0o7000
    path.chmod(special_bits | (0o777 & ~_current_process_umask()))


def ensure_output_path_available(output_path: str | Path) -> Path:
    """Resolve a create-only output path and reject existing filesystem entries."""

    destination = Path(output_path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Output directory already exists: {destination}. Choose a new path."
        )
    return destination


@contextmanager
def staged_output_directory(output_root: str | Path) -> Iterator[Path]:
    """Yield a sibling staging directory and publish it on successful exit.

    The destination is create-only. Writers never expose a partially populated
    output and never remove or mutate an existing artifact.
    """

    destination = ensure_output_path_available(output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    published = False
    try:
        yield temporary_root
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Output directory was created while publishing: {destination}."
            )
        _prepare_directory_for_publication(temporary_root)
        temporary_root.rename(destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary_root, ignore_errors=True)


__all__ = ["ensure_output_path_available", "staged_output_directory"]
