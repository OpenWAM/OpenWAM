"""Build the canonical SDK and metadata-only installation aliases."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLATION_ALIASES = ("open-wam", "openwam-sdk", "open-wam-sdk")


def alias_pyproject(pyproject: dict[str, Any], name: str) -> dict[str, Any]:
    """Forward the canonical version and every extra without shipping modules."""
    canonical = pyproject["project"]
    requirement = f"{canonical['name']}=={canonical['version']}"
    project = {
        key: canonical[key]
        for key in ("version", "requires-python", "license", "authors", "urls", "classifiers")
    }
    project.update(
        name=name,
        description=f"Official installation alias for {canonical['name']}; use import open_wam.",
        readme="README.md",
        dependencies=[requirement],
    )
    project["license-files"] = ["LICENSE", "NOTICE"]
    project["optional-dependencies"] = {
        extra: [f"{canonical['name']}[{extra}]=={canonical['version']}"]
        for extra in canonical.get("optional-dependencies", {})
    }
    return {
        "build-system": pyproject["build-system"],
        "project": project,
        "tool": {"hatch": {"build": {"targets": {
            "wheel": {"bypass-selection": True},
            "sdist": {"only-include": ["pyproject.toml", "README.md", "LICENSE", "NOTICE"]},
        }}}},
    }


def validate_release_tag(ref: str, version: str) -> None:
    if ref != f"refs/tags/v{version}":
        raise ValueError(f"Publishing requires refs/tags/v{version}, got {ref!r}.")


def build_distributions(out_dir: Path) -> None:
    import tomli_w

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise ValueError(f"Build output must be empty: {out_dir}")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def build(source: Path, destination: Path) -> None:
        # The build frontend builds each wheel from its sdist, not the checkout.
        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(destination), str(source)],
            check=True,
        )

    build(REPO_ROOT, out_dir / "core")
    with TemporaryDirectory(prefix="openwam-aliases-") as temporary:
        for name in INSTALLATION_ALIASES:
            source = Path(temporary) / name
            source.mkdir()
            (source / "pyproject.toml").write_text(
                tomli_w.dumps(alias_pyproject(pyproject, name)), encoding="utf-8"
            )
            (source / "README.md").write_text(
                f"# {name}\n\n"
                "Official installation alias for [OpenWAM](https://pypi.org/project/openwam/).\n\n"
                "This metapackage installs the matching OpenWAM release and forwards its\n"
                "optional extras. It contains no Python modules or separate implementation.\n"
                "Prefer `pip install openwam` for new environments; all installation names\n"
                "use the same `import open_wam` and `openwam-*` commands.\n\n"
                f"For example, `pip install '{name}[train,eval]'` installs the same runtime\n"
                "as `pip install 'openwam[train,eval]'`. All aliases can coexist.\n\n"
                "[Documentation](https://openwam.github.io/OpenWAM/) | "
                "[Source](https://github.com/OpenWAM/OpenWAM)\n",
                encoding="utf-8",
            )
            for notice in ("LICENSE", "NOTICE"):
                shutil.copyfile(REPO_ROOT / notice, source / notice)
            build(source, out_dir / "aliases")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    parser.add_argument("--check-tag", help="Validate a refs/tags/vVERSION ref without building.")
    args = parser.parse_args(argv)
    if args.check_tag is not None:
        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        validate_release_tag(args.check_tag, project["project"]["version"])
    else:
        build_distributions(args.out_dir)


if __name__ == "__main__":
    main()
