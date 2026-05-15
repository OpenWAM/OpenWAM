from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from .types import LaunchJob


def resolve_wrapper_train_argv(job: LaunchJob, *, repo_root: Path) -> list[str]:
    """Ask the launcher wrapper for the exact argv passed to open_wam.training.train."""

    env = os.environ.copy()
    env.update(job.env)
    env["OPEN_WAM_PRINT_TRAIN_ARGV"] = "1"
    completed = subprocess.run(
        job.command,
        cwd=str(repo_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Wrapper argv dry-run failed for {job.spec.name} with exit {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Wrapper argv dry-run for {job.spec.name} produced no stdout.")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Wrapper argv dry-run for {job.spec.name} did not end with JSON argv: {lines[-1]!r}"
        ) from exc
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise RuntimeError(f"Wrapper argv dry-run for {job.spec.name} produced invalid argv payload: {payload!r}")
    return payload
