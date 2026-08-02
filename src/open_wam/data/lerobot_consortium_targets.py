"""Deterministic repository-target parsing and serialization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .lerobot_consortium_inventory_contracts import LeRobotConsortiumRepoTarget


def _prefer_repo_target(
    existing: tuple[LeRobotConsortiumRepoTarget, bool] | None,
    candidate: LeRobotConsortiumRepoTarget,
    *,
    explicit_source_group: bool,
) -> tuple[LeRobotConsortiumRepoTarget, bool]:
    if existing is None:
        return candidate, explicit_source_group
    _, existing_explicit = existing
    if explicit_source_group and not existing_explicit:
        return candidate, True
    if explicit_source_group == existing_explicit:
        return candidate, explicit_source_group
    return existing


def infer_lerobot_consortium_source_group(
    repo_id: str, *, default_source_group: str = "manual"
) -> str:
    repo_lower = repo_id.lower()
    if repo_lower.startswith("lerobot/"):
        return "official_lerobot"
    if repo_lower.startswith("daivdyuan/") and repo_lower.endswith("-lerobot"):
        return "nmotion_current"
    return default_source_group


def load_lerobot_consortium_repo_targets(
    path: Path,
    *,
    default_source_group: str = "manual",
) -> tuple[LeRobotConsortiumRepoTarget, ...]:
    """Load repo targets from a plain-text or CSV file.

    Supported formats:

    - `.txt` / `.lst`: one repo per line, or `source_group,repo_id`
    - `.csv`: `repo_id` column with optional `source_group`
    """

    deduped: dict[str, tuple[LeRobotConsortiumRepoTarget, bool]] = {}
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                repo_id = str(raw.get("repo_id", "")).strip()
                if not repo_id:
                    continue
                raw_source_group = str(raw.get("source_group", "")).strip()
                source_group = (
                    raw_source_group
                    or infer_lerobot_consortium_source_group(
                        repo_id,
                        default_source_group=default_source_group,
                    )
                )
                target = LeRobotConsortiumRepoTarget(
                    repo_id=repo_id, source_group=source_group
                )
                deduped[repo_id] = _prefer_repo_target(
                    deduped.get(repo_id),
                    target,
                    explicit_source_group=bool(raw_source_group),
                )
        return tuple(target for target, _ in deduped.values())

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            maybe_group, maybe_repo = [part.strip() for part in line.split(",", 1)]
            if "/" in maybe_group and "/" not in maybe_repo:
                repo_id = maybe_group
                source_group = infer_lerobot_consortium_source_group(
                    repo_id,
                    default_source_group=default_source_group,
                )
            else:
                repo_id = maybe_repo
                source_group = maybe_group or infer_lerobot_consortium_source_group(
                    repo_id,
                    default_source_group=default_source_group,
                )
        else:
            repo_id = line
            source_group = infer_lerobot_consortium_source_group(
                repo_id,
                default_source_group=default_source_group,
            )
        target = LeRobotConsortiumRepoTarget(repo_id=repo_id, source_group=source_group)
        deduped[repo_id] = _prefer_repo_target(
            deduped.get(repo_id),
            target,
            explicit_source_group="," in line and "/" not in maybe_group
            if "," in line
            else False,
        )
    return tuple(target for target, _ in deduped.values())


def write_lerobot_consortium_repo_targets(
    path: Path,
    repo_targets: Iterable[LeRobotConsortiumRepoTarget],
) -> None:
    """Write repo targets in a format understood by `load_*_repo_targets`.

    - `.csv`: writes `repo_id,source_group`
    - other suffixes: writes one `source_group,repo_id` pair per line
    """

    targets = sorted(
        {target.repo_id: target for target in repo_targets}.values(),
        key=lambda target: (target.source_group, target.repo_id),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("repo_id", "source_group"))
            writer.writeheader()
            for target in targets:
                writer.writerow(
                    {"repo_id": target.repo_id, "source_group": target.source_group}
                )
        return

    with path.open("w", encoding="utf-8") as handle:
        for target in targets:
            handle.write(f"{target.source_group},{target.repo_id}\n")


__all__ = [
    "infer_lerobot_consortium_source_group",
    "load_lerobot_consortium_repo_targets",
    "write_lerobot_consortium_repo_targets",
]
