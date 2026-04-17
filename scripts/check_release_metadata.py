from __future__ import annotations

from pathlib import Path
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
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
    print(f"release metadata ok for {version}")


if __name__ == "__main__":
    main()
