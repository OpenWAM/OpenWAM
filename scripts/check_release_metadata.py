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
