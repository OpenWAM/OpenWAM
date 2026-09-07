from __future__ import annotations

import argparse
from collections.abc import Iterable
import csv
from datetime import date
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import tomllib
from typing import Any, BinaryIO
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SDIST_ENTRIES = frozenset(
    {
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "LICENSES",
        "NOTICE",
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
    }
)
REQUIRED_SDIST_EXCLUDES = frozenset(
    {"/.grimp_cache", "/configs/local_paths.yaml"}
)
PRIVATE_PATH_PREFIXES = ("/afs/", "/hai/", "/home/", "/sailhome/", "/scr/", "/simurgh")
EXCLUDED_PUBLIC_PATHS = frozenset({"configs/local_paths.yaml"})
PUBLIC_TEXT_SUFFIXES = frozenset(
    {
        "",
        ".cff",
        ".csv",
        ".jinja",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".typed",
        ".yaml",
        ".yml",
    }
)
REQUIRED_PROJECT_URLS = {
    "Documentation": "https://openwam.github.io/OpenWAM/",
    "Issues": "https://github.com/OpenWAM/OpenWAM/issues",
    "Repository": "https://github.com/OpenWAM/OpenWAM",
}
REQUIRED_PROJECT_CLASSIFIERS = frozenset(
    {
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    }
)
REQUIRED_PROJECT_KEYWORDS = frozenset({"robotics", "world action models", "world models"})
REQUIRED_ATTRIBUTION = (
    "OpenWAM Team, Stanford Vision and Learning Lab (SVL), Stanford University. "
    "OpenWAM, version 0.1.0, 2026."
)
_SHA256_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")


def validate_project_metadata(pyproject: dict[str, Any]) -> None:
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ValueError("Project metadata must be a TOML table.")
    expected_scalars = {
        "name": "openwam",
        "readme": "README.md",
        "license": "AGPL-3.0-only",
        "requires-python": ">=3.11,<3.13",
    }
    for key, expected in expected_scalars.items():
        if project.get(key) != expected:
            raise ValueError(
                f"Project metadata {key!r} must be {expected!r}, got {project.get(key)!r}."
            )

    expected_license_files = [
        "LICENSE",
        "NOTICE",
        "LICENSES/*",
        "THIRD_PARTY_NOTICES.md",
    ]
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
    attribution_notice = REPO_ROOT / "NOTICE"
    project_license = REPO_ROOT / "LICENSE"
    prior_mit_license = REPO_ROOT / "LICENSES" / "MIT.txt"
    apache_license = REPO_ROOT / "LICENSES" / "Apache-2.0.txt"
    required_notice_files = (notice, apache_license, prior_mit_license)
    if not all(path.is_file() for path in required_notice_files):
        raise ValueError(
            "Project and third-party license notices must be included."
        )
    if not project_license.is_file() or "GNU AFFERO GENERAL PUBLIC LICENSE" not in (
        project_license.read_text(encoding="utf-8")
    ):
        raise ValueError("LICENSE must contain the unmodified GNU AGPL v3 text.")
    if not attribution_notice.is_file() or REQUIRED_ATTRIBUTION not in (
        attribution_notice.read_text(encoding="utf-8")
    ):
        raise ValueError("NOTICE must contain the required OpenWAM attribution.")
    validate_public_consortium_snapshot(REPO_ROOT)


def validate_public_consortium_snapshot(repo_root: Path) -> None:
    snapshot_root = repo_root / "notes" / "index"
    repo_list_path = snapshot_root / "lerobot_consortium_hf_repo_ids.txt"
    inventory_path = snapshot_root / "lerobot_consortium_hf_dataset_inventory.csv"
    contracts_path = snapshot_root / "lerobot_consortium_hf_dataset_contracts.json"

    repo_ids = {
        line.rsplit(",", 1)[-1].strip()
        for line in repo_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        inventory = tuple(csv.DictReader(handle))
    contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    contract_rows = tuple(contracts.get("datasets", ()))

    private_ids = {
        str(row.get("repo_id", ""))
        for row in (*inventory, *contract_rows)
        if str(row.get("private", "")).strip().lower() in {"1", "true", "yes"}
    }
    if private_ids:
        raise ValueError(
            "Public consortium snapshots must not disclose private repositories: "
            f"{sorted(private_ids)!r}."
        )

    inventory_ids = {str(row.get("repo_id", "")) for row in inventory}
    contract_ids = {str(row.get("repo_id", "")) for row in contract_rows}
    if repo_ids != inventory_ids or inventory_ids != contract_ids:
        raise ValueError("Public consortium repo, inventory, and contract snapshots disagree.")
    if contracts.get("dataset_count") != len(contract_rows):
        raise ValueError("Public consortium contract dataset_count is stale.")


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


def private_distribution_violations(dist_dir: Path) -> tuple[str, ...]:
    """Inspect the built release artifacts rather than trusting build declarations."""

    source_archives = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    violations: list[str] = []
    if not source_archives:
        violations.append(f"{dist_dir}: no source distribution found")
    if not wheels:
        violations.append(f"{dist_dir}: no wheel found")

    for archive in (*source_archives, *wheels):
        if archive.name.endswith(".tar.gz"):
            with tarfile.open(archive, "r:gz") as source:
                members = (
                    (member.name, source.extractfile(member))
                    for member in source.getmembers()
                    if member.isfile()
                )
                violations.extend(_archive_member_violations(archive.name, members))
        else:
            with zipfile.ZipFile(archive) as source:
                members = (
                    (name, source.open(name))
                    for name in source.namelist()
                    if not name.endswith("/")
                )
                violations.extend(_archive_member_violations(archive.name, members))
    return tuple(violations)


def _archive_member_violations(
    archive_name: str,
    members: Iterable[tuple[str, BinaryIO | None]],
) -> list[str]:
    violations: list[str] = []
    for member_name, stream in members:
        member_path = PurePosixPath(member_name)
        if member_path.is_absolute() or ".." in member_path.parts:
            violations.append(f"{archive_name}: unsafe member path {member_name!r}")
            continue
        if stream is None:
            continue
        with stream:
            payload = stream.read()
        if member_path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
            continue
        text = payload.decode("utf-8", errors="ignore")
        for prefix in PRIVATE_PATH_PREFIXES:
            if prefix in text:
                violations.append(f"{archive_name}:{member_name}: {prefix}")
    return violations


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
    """Validate publication metadata for every model advertised as public."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Public artifact manifest must contain an `artifacts` list.")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("architecture") == "fixture":
            continue
        download_url = artifact.get("download_url")
        checksum = artifact.get("checksum")
        license_name = artifact.get("license")
        publication_fields = (download_url, checksum, license_name)
        if not any(value is not None for value in publication_fields):
            continue
        if not (
            isinstance(download_url, str)
            and download_url.startswith("https://")
            and isinstance(checksum, str)
            and _SHA256_PATTERN.fullmatch(checksum)
            and isinstance(license_name, str)
            and bool(license_name.strip())
        ):
            raise ValueError(
                "A public model artifact requires an HTTPS download URL, "
                "SHA-256 checksum, and license."
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        action="store_true",
        help="Require final, mutually consistent release dates.",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        help="Also inspect built wheel and source-distribution contents.",
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
    if args.dist_dir is not None:
        artifact_violations = private_distribution_violations(args.dist_dir)
        if artifact_violations:
            raise SystemExit(
                f"Built distribution validation failed: {artifact_violations}"
            )
    print(f"release metadata ok for {version}")


if __name__ == "__main__":
    main()
