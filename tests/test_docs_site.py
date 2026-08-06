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

    assert summary["public_pages"] == len(builder.PUBLIC_MARKDOWN_PATHS) == 26
    assert summary["notes_published"] is False
    assert summary["broken_local_links"] == 0
    assert summary["missing_repository_paths"] == 0
    assert (output / "index.md").is_file()
    assert (output / builder.OUTPUT_SENTINEL).is_file()
    assert (output / "quickstart.md").is_file()
    assert (output / "architecture.md").is_file()
    assert (output / "policy_architectures.md").is_file()
    assert (output / "benchmarks.md").is_file()
    assert (output / "running_experiments.md").is_file()
    assert not (output / "engineering-notes").exists()
    assert not (output / "CHECKPOINT.md").exists()
    assert not (output / "camera_sync_deploy_issue.md").exists()
    assert builder.scan_private_fragments(output) == []
    assert builder.scan_broken_local_links(output) == []
    assert builder.scan_broken_local_links(
        REPO_ROOT,
        (REPO_ROOT / "README.md",),
    ) == []


@pytest.mark.unit
def test_docs_site_rejects_unclassified_source_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _load_docs_builder()
    source = tmp_path / "source"
    source.mkdir()
    (source / "index.md").write_text("# Public\n", encoding="utf-8")
    (source / "private_run.md").write_text("# Internal run\n", encoding="utf-8")
    monkeypatch.setattr(builder, "PUBLIC_DOCS", source)
    monkeypatch.setattr(builder, "PUBLIC_MARKDOWN_PATHS", (Path("index.md"),))

    with pytest.raises(SystemExit, match=r"unclassified=private_run\.md"):
        builder.build_docs_site(tmp_path / "output")


@pytest.mark.unit
def test_docs_site_rejects_broken_or_escaping_local_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_docs_builder()
    source = tmp_path / "source"
    source.mkdir()
    (source / "index.md").write_text("[outside](../outside.md)\n", encoding="utf-8")
    (tmp_path / "outside.md").write_text("not public\n", encoding="utf-8")
    monkeypatch.setattr(builder, "PUBLIC_DOCS", source)
    monkeypatch.setattr(builder, "PUBLIC_MARKDOWN_PATHS", (Path("index.md"),))

    with pytest.raises(SystemExit, match="broken or escaping local links"):
        builder.build_docs_site(tmp_path / "output")


@pytest.mark.unit
def test_docs_site_reports_missing_concrete_repository_path(tmp_path: Path) -> None:
    builder = _load_docs_builder()
    guide = tmp_path / "guide.md"
    guide.write_text("python scripts/removed_command.py\n", encoding="utf-8")

    assert builder.scan_missing_repository_paths((guide,), tmp_path) == [
        ("guide.md", 1, "scripts/removed_command.py")
    ]


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
