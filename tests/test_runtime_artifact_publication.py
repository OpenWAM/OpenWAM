from __future__ import annotations

from pathlib import Path

import pytest

from open_wam.runtime import publication
from open_wam.runtime.publication import (
    ensure_output_path_available,
    staged_output_directory,
)


def test_staged_output_directory_publishes_only_complete_outputs(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact"

    with staged_output_directory(destination) as staging:
        assert not destination.exists()
        (staging / "payload.txt").write_text("complete", encoding="utf-8")

    assert (destination / "payload.txt").read_text(encoding="utf-8") == "complete"
    assert not list(tmp_path.glob(".artifact.tmp-*"))


def test_staged_output_directory_applies_normal_permissions_at_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publication, "_current_process_umask", lambda: 0o027)
    destination = tmp_path / "artifact"

    with staged_output_directory(destination) as staging:
        assert staging.stat().st_mode & 0o777 == 0o700

    assert destination.stat().st_mode & 0o777 == 0o750


def test_staged_output_directory_preserves_existing_and_cleans_failures(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with (
        pytest.raises(FileExistsError, match="already exists"),
        staged_output_directory(existing),
    ):
        raise AssertionError("unreachable")
    assert sentinel.read_text(encoding="utf-8") == "keep"

    broken_link = tmp_path / "broken-link"
    broken_link.symlink_to(tmp_path / "missing", target_is_directory=True)
    with (
        pytest.raises(FileExistsError, match="already exists"),
        staged_output_directory(broken_link),
    ):
        raise AssertionError("unreachable")
    assert broken_link.is_symlink()

    failed = tmp_path / "failed"
    with (
        pytest.raises(RuntimeError, match="conversion failed"),
        staged_output_directory(failed) as staging,
    ):
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("conversion failed")

    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


def test_output_path_preflight_uses_the_same_create_only_contract(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact"
    assert ensure_output_path_available(destination) == destination

    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_path_available(destination)
