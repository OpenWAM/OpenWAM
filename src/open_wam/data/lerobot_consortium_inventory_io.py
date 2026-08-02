"""Inventory CSV/JSON/Markdown parsing, rendering, and persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .lerobot_consortium_inventory_contracts import (
    LeRobotConsortiumInventoryRow,
    _to_bool,
    _to_float,
    _to_int,
)


def load_lerobot_consortium_inventory_rows(
    path: Path,
) -> list[LeRobotConsortiumInventoryRow]:
    rows: list[LeRobotConsortiumInventoryRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                LeRobotConsortiumInventoryRow(
                    source_group=raw.get("source_group", ""),
                    repo_id=raw.get("repo_id", ""),
                    private=_to_bool(raw.get("private")),
                    domain_type=raw.get("domain_type", "unknown") or "unknown",
                    total_size_mb=_to_float(raw.get("total_size_mb")),
                    data_size_mb=_to_float(raw.get("data_size_mb")),
                    video_size_mb=_to_float(raw.get("video_size_mb")),
                    total_episodes=_to_int(raw.get("total_episodes")),
                    total_frames=_to_int(raw.get("total_frames")),
                    total_tasks=_to_int(raw.get("total_tasks")),
                    total_hours=_to_float(raw.get("total_hours")),
                    avg_seconds_per_episode=_to_float(
                        raw.get("avg_seconds_per_episode")
                    ),
                    fps=_to_float(raw.get("fps")),
                    observation_fps=_to_float(raw.get("observation_fps")),
                    action_fps=_to_float(raw.get("action_fps")),
                    robot_type=raw.get("robot_type") or None,
                    embodiment_type=raw.get("embodiment_type", "unknown") or "unknown",
                    embodiment_confidence=raw.get("embodiment_confidence", "low")
                    or "low",
                    embodiment_reason=raw.get("embodiment_reason", ""),
                    action_dim=_to_int(raw.get("action_dim")),
                    action_shape=raw.get("action_shape") or None,
                    state_dim=_to_int(raw.get("state_dim")),
                    state_shape=raw.get("state_shape") or None,
                    visual_stream_count=_to_int(raw.get("visual_stream_count")) or 0,
                    visual_stream_keys=raw.get("visual_stream_keys", ""),
                    visual_dimensions=raw.get("visual_dimensions", ""),
                    visual_dtypes=raw.get("visual_dtypes", ""),
                    text_annotation_extent=raw.get("text_annotation_extent", "none")
                    or "none",
                    task_text_present=bool(_to_bool(raw.get("task_text_present"))),
                    task_text_count=_to_int(raw.get("task_text_count")),
                    task_text_examples=raw.get("task_text_examples", ""),
                    temporal_dense_present=bool(
                        _to_bool(raw.get("temporal_dense_present"))
                    ),
                    temporal_sparse_present=bool(
                        _to_bool(raw.get("temporal_sparse_present"))
                    ),
                    language_feature_keys=raw.get("language_feature_keys", ""),
                    readme_url=raw.get("readme_url", ""),
                    dataset_url=raw.get("dataset_url", ""),
                    generation_error=raw.get("generation_error") or None,
                )
            )
    return rows


def render_lerobot_consortium_inventory_markdown(
    rows: Iterable[LeRobotConsortiumInventoryRow],
) -> str:
    rows_list = list(rows)
    by_group: dict[str, int] = {}
    group_episodes: dict[str, int] = {}
    group_hours: dict[str, float] = {}
    annotated_counts: dict[str, int] = {}
    errored = [row for row in rows_list if row.generation_error]
    for row in rows_list:
        by_group[row.source_group] = by_group.get(row.source_group, 0) + 1
        if row.total_episodes is not None:
            group_episodes[row.source_group] = (
                group_episodes.get(row.source_group, 0) + row.total_episodes
            )
        if row.total_hours is not None:
            group_hours[row.source_group] = (
                group_hours.get(row.source_group, 0.0) + row.total_hours
            )
        if (
            row.task_text_present
            or row.temporal_dense_present
            or row.temporal_sparse_present
        ):
            annotated_counts[row.source_group] = (
                annotated_counts.get(row.source_group, 0) + 1
            )

    lines = [
        "# LeRobot Consortium HF Dataset Inventory",
        "",
        f"- Total repos: `{len(rows_list)}`",
    ]
    for group, count in sorted(by_group.items()):
        lines.append(
            f"- {group}: `{count}` repos, `{group_episodes.get(group, 0)}` episodes, "
            f"`{group_hours.get(group, 0.0):.2f}` hours, `{annotated_counts.get(group, 0)}` with text annotations"
        )
    if errored:
        lines.append(f"- Rows with incomplete metadata: `{len(errored)}`")
        for row in errored[:10]:
            lines.append(f"  - `{row.repo_id}`: {row.generation_error}")
    lines.extend(
        [
            "",
            "| Source | Repo | Domain | Size (GB) | Episodes | Hours | Obs FPS | Action FPS | Avg sec/ep | Embodiment | Action dim | Cameras | Visual dims | Text annotations |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows_list:
        size_gb = (
            f"{(row.total_size_mb or 0.0) / 1024.0:.2f}"
            if row.total_size_mb is not None
            else ""
        )
        hours = f"{row.total_hours:.2f}" if row.total_hours is not None else ""
        obs_fps = (
            f"{row.observation_fps:.1f}" if row.observation_fps is not None else ""
        )
        action_fps = f"{row.action_fps:.1f}" if row.action_fps is not None else ""
        avg_seconds = (
            f"{row.avg_seconds_per_episode:.1f}"
            if row.avg_seconds_per_episode is not None
            else ""
        )
        visual_dimensions = (
            row.visual_dimensions.replace(" | ", "<br>")
            if row.visual_dimensions
            else ""
        )
        lines.append(
            f"| {row.source_group} | {row.repo_id} | {row.domain_type} | {size_gb} | {row.total_episodes or ''} | "
            f"{hours} | {obs_fps} | {action_fps} | {avg_seconds} | "
            f"{row.embodiment_type} ({row.embodiment_confidence}) | {row.action_dim or ''} | "
            f"{row.visual_stream_count} | {visual_dimensions} | {row.text_annotation_extent} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_lerobot_consortium_inventory_csv(
    path: Path, rows: Iterable[LeRobotConsortiumInventoryRow]
) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows_list[0].to_dict().keys()) if rows_list else []
        )
        if not rows_list:
            return
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row.to_dict())


def write_lerobot_consortium_inventory_json(
    path: Path, rows: Iterable[LeRobotConsortiumInventoryRow]
) -> None:
    payload = {"rows": [row.to_dict() for row in rows]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_lerobot_consortium_inventory_markdown(
    path: Path, rows: Iterable[LeRobotConsortiumInventoryRow]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_lerobot_consortium_inventory_markdown(rows), encoding="utf-8"
    )


__all__ = [
    "load_lerobot_consortium_inventory_rows",
    "render_lerobot_consortium_inventory_markdown",
    "write_lerobot_consortium_inventory_csv",
    "write_lerobot_consortium_inventory_json",
    "write_lerobot_consortium_inventory_markdown",
]
