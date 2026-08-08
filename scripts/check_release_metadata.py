from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import tomllib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SDIST_ENTRIES = frozenset(
    {
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "LICENSES",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "configs",
        "docs",
        "mkdocs.yml",
        "notes/index/lerobot_consortium_hf_dataset_contracts.json",
        "notes/index/lerobot_consortium_hf_dataset_inventory.csv",
        "notes/index/lerobot_consortium_hf_dataset_inventory.md",
        "notes/index/lerobot_consortium_hf_repo_ids.txt",
        "pyproject.toml",
        "src/open_wam",
        "templates",
    }
)
REQUIRED_SDIST_EXCLUDES = frozenset(
    {"/.gitignore", "/.grimp_cache", "/configs/local_paths.yaml", "/notes/*.tmp.md"}
)
PRIVATE_PATH_PREFIXES = ("/afs/", "/hai/", "/sailhome/", "/scr/", "/simurgh")
EXCLUDED_PUBLIC_PATHS = frozenset({"configs/local_paths.yaml"})
PUBLIC_TEXT_SUFFIXES = frozenset(
    {"", ".jinja", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)
REQUIRED_PROJECT_URLS = {
    "Documentation": "https://daivdyuan.github.io/Open-WAM/",
    "Issues": "https://github.com/DaivdYuan/Open-WAM/issues",
    "Repository": "https://github.com/DaivdYuan/Open-WAM",
}
REQUIRED_PROJECT_CLASSIFIERS = frozenset(
    {
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    }
)
REQUIRED_PROJECT_KEYWORDS = frozenset({"robotics", "world action models", "world models"})
_SHA256_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")


def validate_project_metadata(pyproject: dict[str, Any]) -> None:
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ValueError("Project metadata must be a TOML table.")
    expected_scalars = {
        "name": "open-wam",
        "readme": "README.md",
        "license": "MIT",
        "requires-python": ">=3.11,<3.13",
    }
    for key, expected in expected_scalars.items():
        if project.get(key) != expected:
            raise ValueError(
                f"Project metadata {key!r} must be {expected!r}, got {project.get(key)!r}."
            )

    expected_license_files = ["LICENSE", "LICENSES/*", "THIRD_PARTY_NOTICES.md"]
    if project.get("license-files") != expected_license_files:
        raise ValueError(
            "Project metadata `license-files` must preserve project and "
            f"third-party terms: {expected_license_files!r}."
        )
    authors = project.get("authors", ())
    if not authors or not all(isinstance(author, dict) and author.get("name") for author in authors):
        raise ValueError("Project metadata must declare at least one named author.")
    if project.get("urls") != REQUIRED_PROJECT_URLS:
        raise ValueError(
            "Project metadata URL mismatch: "
            f"expected {REQUIRED_PROJECT_URLS!r}, got {project.get('urls')!r}."
        )
    missing_classifiers = REQUIRED_PROJECT_CLASSIFIERS - set(project.get("classifiers", ()))
    if missing_classifiers:
        raise ValueError(
            f"Project metadata is missing classifiers: {sorted(missing_classifiers)!r}."
        )
    missing_keywords = REQUIRED_PROJECT_KEYWORDS - set(project.get("keywords", ()))
    if missing_keywords:
        raise ValueError(
            f"Project metadata is missing keywords: {sorted(missing_keywords)!r}."
        )


def validate_release_build_config(pyproject: dict[str, Any]) -> None:
    sdist_config = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]
    only_include = set(sdist_config.get("only-include", ()))
    if only_include != PUBLIC_SDIST_ENTRIES:
        raise ValueError(
            "Source distribution allowlist mismatch: "
            f"expected {sorted(PUBLIC_SDIST_ENTRIES)}, got {sorted(only_include)}"
        )
    missing_excludes = REQUIRED_SDIST_EXCLUDES - set(sdist_config.get("exclude", ()))
    if missing_excludes:
        raise ValueError(
            f"Source distribution exclusions are incomplete: {sorted(missing_excludes)}"
        )
    notice = REPO_ROOT / "THIRD_PARTY_NOTICES.md"
    apache_license = REPO_ROOT / "LICENSES" / "Apache-2.0.txt"
    if not notice.is_file() or not apache_license.is_file():
        raise ValueError(
            "LingBot third-party notice and Apache-2.0 license must be included."
        )


def private_sdist_path_violations(repo_root: Path) -> tuple[str, ...]:
    violations: list[str] = []
    for entry in sorted(PUBLIC_SDIST_ENTRIES):
        root = repo_root / entry
        paths = (root,) if root.is_file() else tuple(root.rglob("*"))
        for path in paths:
            relative_path = path.relative_to(repo_root).as_posix()
            if relative_path in EXCLUDED_PUBLIC_PATHS:
                continue
            if not path.is_file() or path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            for prefix in PRIVATE_PATH_PREFIXES:
                if prefix in text:
                    violations.append(f"{relative_path}: {prefix}")
    return tuple(violations)


def validate_version_state(
    *,
    version: str,
    changelog: str,
    citation: str,
    require_released: bool,
) -> None:
    citation_version = re.search(r'^version:\s*["\']?([^"\'\s]+)', citation, re.MULTILINE)
    if citation_version is None or citation_version.group(1) != version:
        raise ValueError("CITATION.cff version must match the project version.")
    citation_date = re.search(
        r'^date-released:\s*["\']?(\d{4}-\d{2}-\d{2})',
        citation,
        re.MULTILINE,
    )
    release_heading = re.search(
        rf"^## {re.escape(version)} - (Unreleased|\d{{4}}-\d{{2}}-\d{{2}})$",
        changelog,
        re.MULTILINE,
    )
    if release_heading is None:
        raise ValueError(
            f"CHANGELOG.md must contain `## {version} - Unreleased` or a release date."
        )
    changelog_state = release_heading.group(1)
    if not require_released:
        if changelog_state == "Unreleased" and citation_date is not None:
            raise ValueError(
                "CITATION.cff must omit date-released while the changelog is Unreleased."
            )
        return
    if changelog_state == "Unreleased":
        raise ValueError("Release validation requires a dated changelog heading.")
    if citation_date is None or citation_date.group(1) != changelog_state:
        raise ValueError(
            "CITATION.cff date-released must match the changelog release date."
        )
    if date.fromisoformat(changelog_state) > date.today():
        raise ValueError("Release date cannot be in the future.")


def validate_public_model_artifacts(manifest: dict[str, Any]) -> None:
    """Require one downloadable, checksummed, licensed non-fixture model."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Public artifact manifest must contain an `artifacts` list.")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("architecture") == "fixture":
            continue
        download_url = artifact.get("download_url")
        checksum = artifact.get("checksum")
        license_name = artifact.get("license")
        if (
            isinstance(download_url, str)
            and download_url.startswith("https://")
            and isinstance(checksum, str)
            and _SHA256_PATTERN.fullmatch(checksum)
            and isinstance(license_name, str)
            and bool(license_name.strip())
        ):
            return
    raise ValueError(
        "Release validation requires at least one non-fixture model artifact "
        "with an HTTPS download URL, SHA-256 checksum, and license."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        action="store_true",
        help="Require final, mutually consistent release dates.",
    )
    args = parser.parse_args(argv)
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    version = pyproject["project"]["version"]
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    required_docs = (
        "docs/release.md",
        "docs/experiment_cards.md",
        "docs/artifacts.md",
        "docs/testing.md",
    )
    missing = [path for path in required_docs if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing release-facing docs: {missing}")
    try:
        validate_project_metadata(pyproject)
        validate_release_build_config(pyproject)
        validate_version_state(
            version=version,
            changelog=changelog,
            citation=citation,
            require_released=bool(args.release),
        )
        if args.release:
            import yaml

            manifest = yaml.safe_load(
                (REPO_ROOT / "configs" / "artifacts.sample.yaml").read_text(
                    encoding="utf-8"
                )
            ) or {}
            validate_public_model_artifacts(manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    private_paths = private_sdist_path_violations(REPO_ROOT)
    if private_paths:
        raise SystemExit(
            f"Private paths found in the public sdist surface: {private_paths}"
        )
    print(f"release metadata ok for {version}")


if __name__ == "__main__":
    main()
