from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    (
        "scripts/run_libero_exact_visualization.py",
        "scripts/run_libero_exact_realtime_sandbox.py",
        "scripts/run_libero_realtime_ablation.py",
    ),
)
def test_top_level_deprecated_python_entrypoint_stubs_fail_closed(relative_path: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.pop("OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG", None)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / relative_path)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "deprecated for current LIBERO M1/M5 launch paths" in result.stderr
