from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SDIST_ENTRIES = frozenset(
    {
        "CHANGELOG.md",
        "LICENSE",
        "README.md",
        "configs",
        "docs",
        "mkdocs.yml",
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


def validate_project_metadata(pyproject: dict[str, Any]) -> None:
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ValueError("Project metadata must be a TOML table.")
    expected_scalars = {
        "name": "open-wam",
        "readme": "README.md",
        "license": "MIT",
    }
    for key, expected in expected_scalars.items():
        if project.get(key) != expected:
            raise ValueError(
                f"Project metadata {key!r} must be {expected!r}, got {project.get(key)!r}."
            )

    if project.get("license-files") != ["LICENSE"]:
        raise ValueError("Project metadata `license-files` must be exactly ['LICENSE'].")
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


def main() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    version = pyproject["project"]["version"]
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## {version}" not in changelog:
        raise SystemExit(f"CHANGELOG.md is missing a section for version {version}.")
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
