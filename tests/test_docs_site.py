from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_docs_site.py"


def _load_docs_builder():
    spec = importlib.util.spec_from_file_location("build_docs_site", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_docs_site_stages_curated_public_docs_only(tmp_path: Path) -> None:
    builder = _load_docs_builder()
    output = tmp_path / "docs_site"

    summary = builder.build_docs_site(output)

    assert summary["public_pages"] > 0
    assert summary["notes_published"] is False
    assert (output / "index.md").is_file()
    assert (output / builder.OUTPUT_SENTINEL).is_file()
    assert (output / "quickstart.md").is_file()
    assert (output / "architecture.md").is_file()
    assert (output / "method_families.md").is_file()
    assert (output / "benchmarks.md").is_file()
    assert (output / "running_experiments.md").is_file()
    assert not (output / "engineering-notes").exists()
    assert builder.scan_private_fragments(output) == []


@pytest.mark.unit
def test_docs_site_refuses_to_clear_non_generated_output(tmp_path: Path) -> None:
    builder = _load_docs_builder()
    output = tmp_path / "existing_docs"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep me\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="non-generated docs output"):
        builder.build_docs_site(output)


@pytest.mark.unit
def test_docs_site_refuses_symlinked_output(tmp_path: Path) -> None:
    builder = _load_docs_builder()
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "docs_link"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(SystemExit, match="symlinked docs output"):
        builder.build_docs_site(output)
