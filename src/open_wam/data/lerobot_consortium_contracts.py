from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lerobot_consortium_index import LeRobotConsortiumInventoryRow, load_lerobot_consortium_inventory_rows


def _split_pipe(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def _parse_visual_dimensions(value: str | None) -> dict[str, dict[str, int | None]]:
    out: dict[str, dict[str, int | None]] = {}
    for item in _split_pipe(value):
        if ":" not in item:
            continue
        key, shape_text = item.split(":", 1)
        digits = [int(part) for part in shape_text.split("x") if part.isdigit()]
        payload = {"height": None, "width": None, "channels": None}
        if len(digits) == 3:
            payload = {"height": digits[0], "width": digits[1], "channels": digits[2]}
        elif len(digits) == 2:
            payload = {"height": digits[0], "width": digits[1], "channels": None}
        out[key.strip()] = payload
    return out


def _build_visual_stream_contracts(row: LeRobotConsortiumInventoryRow) -> list[dict[str, Any]]:
    keys = _split_pipe(row.visual_stream_keys)
    dims = _parse_visual_dimensions(row.visual_dimensions)
    dtypes = _split_pipe(row.visual_dtypes)
    streams: list[dict[str, Any]] = []
    for stream_index, key in enumerate(keys):
        dim = dims.get(key, {"height": None, "width": None, "channels": None})
        streams.append(
            {
                "stream_index": stream_index,
                "stream_key": key,
                "dtype": dtypes[stream_index] if stream_index < len(dtypes) else None,
                "height": dim.get("height"),
                "width": dim.get("width"),
                "channels": dim.get("channels"),
            }
        )
    return streams


def build_lerobot_consortium_contract_catalog_from_inventory_rows(
    rows: list[LeRobotConsortiumInventoryRow],
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item.source_group, item.repo_id)):
        manifest_total_stream_rows = None
        if row.total_episodes is not None:
            manifest_total_stream_rows = row.total_episodes * max(0, int(row.visual_stream_count))
        datasets.append(
            {
                "repo_id": row.repo_id,
                "source_group": row.source_group,
                "private": row.private,
                "dataset_url": row.dataset_url,
                "readme_url": row.readme_url,
                "domain_type": row.domain_type,
                "embodiment": {
                    "type": row.embodiment_type,
                    "confidence": row.embodiment_confidence,
                    "reason": row.embodiment_reason,
                    "robot_type": row.robot_type,
                },
                "temporal": {
                    "total_episodes": row.total_episodes,
                    "total_frames": row.total_frames,
                    "total_hours": row.total_hours,
                    "avg_seconds_per_episode": row.avg_seconds_per_episode,
                    "fps": row.fps,
                    "observation_fps": row.observation_fps,
                    "action_fps": row.action_fps,
                },
                "control": {
                    "action_dim": row.action_dim,
                    "action_shape": row.action_shape,
                    "state_dim": row.state_dim,
                    "state_shape": row.state_shape,
                },
                "modalities": {
                    "total_size_mb": row.total_size_mb,
                    "data_size_mb": row.data_size_mb,
                    "video_size_mb": row.video_size_mb,
                    "visual_stream_count": row.visual_stream_count,
                    "visual_streams": _build_visual_stream_contracts(row),
                },
                "text_annotations": {
                    "extent": row.text_annotation_extent,
                    "task_text_present": row.task_text_present,
                    "task_text_count": row.task_text_count,
                    "task_text_examples": _split_pipe(row.task_text_examples),
                    "temporal_dense_present": row.temporal_dense_present,
                    "temporal_sparse_present": row.temporal_sparse_present,
                    "language_feature_keys": _split_pipe(row.language_feature_keys),
                },
                "video_contract": {
                    "episode_routing_available": row.generation_error is None and row.visual_stream_count > 0,
                    "manifest_total_episodes": row.total_episodes,
                    "manifest_total_stream_rows": manifest_total_stream_rows,
                    "manifest_visual_stream_count": row.visual_stream_count,
                    "manifest_total_hours": row.total_hours,
                    "manifest_observation_fps": row.observation_fps,
                    "manifest_action_fps": row.action_fps,
                    "example_stream_keys": _split_pipe(row.visual_stream_keys),
                },
                "generation_error": row.generation_error,
            }
        )
    return {
        "contract_version": "hf_dataset_contracts.v1",
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def build_lerobot_consortium_contract_catalog(inventory_csv: Path) -> dict[str, Any]:
    return build_lerobot_consortium_contract_catalog_from_inventory_rows(
        load_lerobot_consortium_inventory_rows(inventory_csv)
    )


def write_lerobot_consortium_contract_catalog(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
