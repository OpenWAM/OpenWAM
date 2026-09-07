from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_entrypoint(relative_path: str, *, allow_deprecated: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.pop("OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG", None)
    command = [sys.executable, str(REPO_ROOT / relative_path)]
    if allow_deprecated:
        command.append("--allow-deprecated-libero-config")

    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    "relative_path,replacement",
    (
        ("scripts/run_libero_exact_realtime_sandbox.py", "scripts/run_libero_realtime_sandbox.py"),
        ("scripts/run_libero_exact_visualization.py", "scripts/run_libero_realtime_sandbox.py"),
    ),
)
@pytest.mark.parametrize("allow_deprecated", (False, True))
def test_removed_python_entrypoint_stubs_always_fail_closed(
    relative_path: str,
    replacement: str,
    allow_deprecated: bool,
) -> None:
    result = _run_entrypoint(relative_path, allow_deprecated=allow_deprecated)

    assert result.returncode == 2
    assert "was removed from the maintained OpenWAM runtime" in result.stderr
    assert replacement in result.stderr
