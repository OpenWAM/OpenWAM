from __future__ import annotations

from html import unescape
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_docs_site.py"
PAPER_AUTHORS = (
    ("Heng", "Yu"),
    ("David D.", "Yuan"),
    ("Juze", "Zhang"),
    ("Changan", "Chen"),
    ("Yao", "Feng"),
    ("Michelle", "Baldonado"),
    ("Steve", "Cousins"),
    ("Li", "Fei-Fei"),
    ("Jiajun", "Wu"),
    ("Ehsan", "Adeli"),
)


def _load_docs_builder():
    spec = importlib.util.spec_from_file_location("build_docs_site", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_readme_leads_with_paper_title_and_arxiv_badge() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    heading = re.search(r'^<h1 align="center">(.+)</h1>', readme)
    assert heading is not None
    assert heading.group(1).replace("<br>", " ") == (
        "OpenWAM: An Open Framework for Composable World-Action Models"
    )

    first_badge = re.search(
        r'<a href="([^"]+)"><img src="https://img\.shields\.io/[^>]+></a>', readme
    )
    assert first_badge is not None
    assert first_badge.group(1) == "https://arxiv.org/pdf/2610.07922"
    assert 'alt="arXiv: 2610.07922"' in first_badge.group(0)
    assert "img.shields.io/badge/arXiv-2610.07922-b31b1b" in first_badge.group(0)
    assert first_badge.start() < readme.index("Paper (PDF)")


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", ("README.md", "docs/index.md"))
def test_affiliation_logos_include_stai_between_svl_and_src(relative_path: str) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    expected = [
        "https://www.stanford.edu/",
        "https://ai.stanford.edu/",
        "https://svl.stanford.edu/",
        "https://stai.stanford.edu/",
        "https://src.stanford.edu/",
    ]
    image_links = re.findall(r'<a href="([^"]+)"[^>]*><img\b', content)
    assert [url for url in image_links if url in expected] == expected
    assert 'alt="Stanford Translational AI (STAI) Lab"' in content
    assert "assets/affiliations/stanford-stai.png" in content
    assert (REPO_ROOT / "docs/assets/affiliations/stanford-stai.png").is_file()


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", ("README.md", "docs/index.md"))
def test_landing_pages_credit_paper_authors_in_order(relative_path: str) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    header = unescape(content.split("\n##", maxsplit=1)[0]).replace("\xa0", " ")
    byline = re.search(r'<p align="center">\s*(Heng[\s\S]+?)</p>', header)
    assert byline is not None
    names = [f"{given} {family}" for given, family in PAPER_AUTHORS]
    byline_text = " ".join(re.sub(r"<[^>]+>", "", byline.group(1)).split())
    assert byline_text == ", ".join(
        f"{name}*" if index < 3 else name for index, name in enumerate(names)
    )
    assert byline.group(1).count("<sup>*</sup>") == 3
    for name in names[:3]:
        assert f"{name}<sup>*</sup>" in byline.group(1)
    assert "Stanford University" in header
    assert "Equal contribution" in header


@pytest.mark.unit
def test_only_paper_citation_is_published() -> None:
    citation = (REPO_ROOT / "CITATION.bib").read_text(encoding="utf-8").strip()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    entries = re.findall(r"```bibtex\n([\s\S]*?)\n```", readme)
    assert entries == [citation]
    assert re.findall(r"^@\w+\{", citation, re.MULTILINE) == ["@article{"]
    assert "title   = {{OpenWAM}: An Open Framework for Composable World-Action Models}" in citation
    assert "journal = {arXiv preprint arXiv:2610.07922}" in citation
    assert "year    = {2026}" in citation
    assert "url     = {https://arxiv.org/pdf/2610.07922}" in citation
    assert not (REPO_ROOT / "CITATION.cff").exists()
    assert "@software" not in readme
    assert "software record" not in readme
    assert "version-specific" not in readme
    bibtex = re.search(r"@article\{yu2026openwam,[\s\S]*?author\s*=\s*\{([^}]+)\}", citation)
    assert bibtex is not None
    assert " ".join(bibtex.group(1).split()) == " and ".join(
        f"{family}, {given}" for given, family in PAPER_AUTHORS
    )


@pytest.mark.unit
def test_readme_distinguishes_paper_blog_and_technical_docs() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert '<a href="https://arxiv.org/pdf/2610.07922">Paper (PDF)</a>' in readme
    assert '<a href="https://openwam.stanford.edu/">Research blog</a>' in readme
    assert (
        '<a href="https://openwam.github.io/OpenWAM/">Technical documentation</a>'
        in readme
    )


@pytest.mark.unit
def test_readme_links_models_to_released_pretraining_weights() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    model_url = "https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining"
    header, body = readme.split("## Capabilities", maxsplit=1)

    assert f'<a href="{model_url}">Models</a>' in header
    assert f"]({model_url})" in body
    assert "docs/artifacts.md" in body
    assert "docs/pretraining/index.md" in body


@pytest.mark.unit
def test_readme_separates_cpu_example_from_reference_training() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = readme.split("## Quickstart\n", maxsplit=1)[1].split("\n##", maxsplit=1)[0]
    training = readme.split("## Train a policy\n", maxsplit=1)[1].split("\n##", maxsplit=1)[0]

    assert "uv sync --frozen --group dev --extra train --extra eval" in quickstart
    assert "openwam-sanity" in quickstart
    assert "--device cpu --max-batches 1 --rollout-steps 1" in quickstart
    assert "configs/examples/public_tiny_synthetic_contract.yaml" in quickstart
    assert "configs/experiments/dual_expert_libero_joint.yaml" in training
    assert "--nproc-per-node=4" in training
    assert "--expected-world-size 4" in training
    assert "docs/running_experiments.md#data-prerequisites" in training
    assert "robotwin_smoke" not in readme


@pytest.mark.unit
def test_docs_and_package_link_to_research_paper() -> None:
    docs_index = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "[Research paper (PDF)](https://arxiv.org/pdf/2610.07922)" in docs_index
    assert project["project"]["urls"]["Paper"] == "https://arxiv.org/pdf/2610.07922"


@pytest.mark.unit
@pytest.mark.parametrize(
    "relative_path",
    ("docs/index.md", "mkdocs.yml", "pyproject.toml"),
)
def test_project_descriptions_include_extensibility_and_composability(
    relative_path: str,
) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert "extensible and composable" in content.lower()


@pytest.mark.unit
def test_docs_site_stages_curated_public_docs_only(tmp_path: Path) -> None:
    builder = _load_docs_builder()
    output = tmp_path / "docs_site"

    summary = builder.build_docs_site(output)

    assert summary["public_pages"] == len(builder.PUBLIC_MARKDOWN_PATHS)
    assert summary["public_assets"] == len(builder.PUBLIC_ASSET_PATHS)
    assert summary["notes_published"] is False
    assert summary["broken_local_links"] == 0
    assert summary["missing_repository_paths"] == 0
    expected_pages = {
        path.with_name("index.md") if path.name == "README.md" else path
        for path in builder.PUBLIC_MARKDOWN_PATHS
    }
    expected_files = {
        *expected_pages,
        *builder.PUBLIC_ASSET_PATHS,
        Path(builder.OUTPUT_SENTINEL),
    }
    actual_files = {
        path.relative_to(output) for path in output.rglob("*") if path.is_file()
    }
    assert actual_files == expected_files
    for relative in builder.PUBLIC_ASSET_PATHS:
        assert (output / relative).read_bytes() == (
            builder.PUBLIC_DOCS / relative
        ).read_bytes()
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
    monkeypatch.setattr(builder, "PUBLIC_ASSET_PATHS", ())

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
