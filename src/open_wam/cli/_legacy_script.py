from __future__ import annotations

import runpy
from pathlib import Path

from open_wam.runtime import REPO_ROOT


def run_legacy_script(script_name: str, *, repo_root: Path = REPO_ROOT) -> None:
    """Run a root script from a package-owned console entrypoint.

    This is a compatibility bridge while large legacy scripts are migrated into
    importable package modules.
    """

    script_path = repo_root / "scripts" / script_name
    if not script_path.is_file():
        raise SystemExit(
            f"Could not find legacy script {script_path}. "
            "Use a source checkout or install a release that packages the new runtime module."
        )
    runpy.run_path(str(script_path), run_name="__main__")
