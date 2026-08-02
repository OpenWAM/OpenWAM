"""Remote Hugging Face inspection for LeRobot consortium inventories."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

from .lerobot_consortium_inventory_contracts import (
    LeRobotConsortiumInventoryRow as LeRobotConsortiumInventoryRow,
    LeRobotConsortiumRepoTarget as LeRobotConsortiumRepoTarget,
    _to_bool,
    _to_float,
    _to_int,
)
from .lerobot_consortium_inventory_io import (
    load_lerobot_consortium_inventory_rows as load_lerobot_consortium_inventory_rows,
    render_lerobot_consortium_inventory_markdown as render_lerobot_consortium_inventory_markdown,
    write_lerobot_consortium_inventory_csv as write_lerobot_consortium_inventory_csv,
    write_lerobot_consortium_inventory_json as write_lerobot_consortium_inventory_json,
    write_lerobot_consortium_inventory_markdown as write_lerobot_consortium_inventory_markdown,
)
from .lerobot_consortium_targets import (
    _prefer_repo_target,
    infer_lerobot_consortium_source_group as infer_lerobot_consortium_source_group,
    load_lerobot_consortium_repo_targets as load_lerobot_consortium_repo_targets,
    write_lerobot_consortium_repo_targets as write_lerobot_consortium_repo_targets,
)


_VISUAL_DTYPES = {"image", "video"}
_TEXT_DTYPES = {"string", "large_string"}


def _shape_product(shape: Any) -> int | None:
    if shape is None:
        return None
    if isinstance(shape, int):
        return int(shape)
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    out = 1
    for dim in shape:
        if dim is None:
            return None
        out *= int(dim)
    return int(out)


def _shape_text(shape: Any) -> str | None:
    if shape is None:
        return None
    if isinstance(shape, int):
        return str(int(shape))
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    return "x".join(str(int(dim)) for dim in shape)


def _load_downloaded_json(
    repo_id: str, filename: str, *, token: str | None = None
) -> dict[str, Any] | None:
    try:
        path = Path(
            hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=filename, token=token
            )
        )
    except Exception:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_downloaded_text(
    repo_id: str, filename: str, *, token: str | None = None
) -> str | None:
    try:
        path = Path(
            hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=filename, token=token
            )
        )
    except Exception:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _load_task_texts(repo_id: str, *, token: str | None = None) -> list[str]:
    parquet_candidates = ("meta/tasks.parquet",)
    jsonl_candidates = ("meta/tasks.jsonl",)
    for filename in parquet_candidates:
        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id, repo_type="dataset", filename=filename, token=token
                )
            )
        except Exception:
            continue
        try:
            rows = pq.read_table(path).to_pylist()
        except Exception:
            continue
        return _extract_task_texts(rows)
    for filename in jsonl_candidates:
        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id, repo_type="dataset", filename=filename, token=token
                )
            )
        except Exception:
            continue
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception:
            continue
        return _extract_task_texts(rows)
    return []


def _extract_task_texts(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if key.endswith("index"):
                continue
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
    return ordered


def _remote_file_exists(repo_info: Any, filename: str) -> bool:
    for sibling in getattr(repo_info, "siblings", None) or []:
        if getattr(sibling, "rfilename", None) == filename:
            return True
    return False


def _sum_repo_sizes_mb(repo_info: Any) -> float | None:
    total_bytes = 0
    saw_size = False
    for sibling in getattr(repo_info, "siblings", None) or []:
        size = getattr(sibling, "size", None)
        if size is None:
            continue
        total_bytes += int(size)
        saw_size = True
    if not saw_size:
        return None
    return round(total_bytes / (1024.0 * 1024.0), 2)


def _sum_prefixed_sizes_mb(repo_info: Any, *, prefixes: Iterable[str]) -> float | None:
    total_bytes = 0
    saw_size = False
    for sibling in getattr(repo_info, "siblings", None) or []:
        path = getattr(sibling, "rfilename", None) or ""
        if not any(path.startswith(prefix) for prefix in prefixes):
            continue
        size = getattr(sibling, "size", None)
        if size is None:
            continue
        total_bytes += int(size)
        saw_size = True
    if not saw_size:
        return None
    return round(total_bytes / (1024.0 * 1024.0), 2)


def _find_feature(
    feats: dict[str, Any], candidates: Iterable[str]
) -> dict[str, Any] | None:
    for key in candidates:
        value = feats.get(key)
        if isinstance(value, dict):
            return value
    return None


def _find_visual_features(info: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    feats = info.get("features") or {}
    out: list[tuple[str, dict[str, Any]]] = []
    for key, value in feats.items():
        if not isinstance(value, dict):
            continue
        if str(value.get("dtype") or "").lower() not in _VISUAL_DTYPES:
            continue
        out.append((str(key), value))
    return out


def _find_language_feature_keys(info: dict[str, Any]) -> list[str]:
    feats = info.get("features") or {}
    keys: list[str] = []
    for key, value in feats.items():
        if not isinstance(value, dict):
            continue
        key_text = str(key)
        key_lower = key_text.lower()
        if key_lower == "task_index" or key_lower.endswith("_index"):
            continue
        dtype = str(value.get("dtype") or "").lower()
        if dtype in _TEXT_DTYPES or any(
            token in key_lower
            for token in ("task", "language", "instruction", "caption", "text")
        ):
            keys.append(key_text)
    return sorted(dict.fromkeys(keys))


def _infer_domain_type(repo_id: str, readme_text: str | None) -> str:
    repo_lower = repo_id.lower()
    readme_lower = (readme_text or "").lower()
    joined = f"{repo_lower} {readme_lower}"
    sim_signals = ("_sim", "/sim_", " simulation", " simulated", "in simulation")
    real_signals = (
        "_real",
        "/real_",
        " real-world",
        " real world",
        "real robot",
        "physical robot",
    )
    has_sim = any(signal in joined for signal in sim_signals)
    has_real = any(signal in joined for signal in real_signals)
    if has_sim and not has_real:
        return "sim"
    if has_real and not has_sim:
        return "real"
    if any(
        token in repo_lower
        for token in ("umi", "exumi", "dexumi", "touchwild", "dexwild")
    ):
        return "real"
    return "unknown"


def _infer_embodiment(
    *,
    repo_id: str,
    robot_type: str | None,
    action_dim: int | None,
    state_dim: int | None,
    readme_text: str | None,
) -> tuple[str, str, str]:
    repo_lower = repo_id.lower()
    robot_lower = str(robot_type or "").lower()
    readme_lower = (readme_text or "").lower()
    joined = f"{repo_lower} {robot_lower} {readme_lower}"

    if any(
        token in joined
        for token in ("dexumi", "dexterous", "xhand", "inspire hand", "inspire_hand")
    ):
        return (
            "dexterous_hand",
            "high",
            "repo or README indicates dexterous hand manipulation",
        )
    if "aloha_mobile" in repo_lower or "mobile manipulator" in readme_lower:
        return (
            "mobile_manipulator",
            "high",
            "repo or README indicates mobile manipulator",
        )
    if robot_lower == "aloha" or "bimanual" in joined:
        return (
            "dual_arm",
            "high",
            "ALOHA or bimanual signals indicate dual-arm embodiment",
        )
    if any(token in joined for token in ("mobile robot", "navigation", "gnm")):
        return "mobile_robot", "medium", "repo or README indicates mobile robot"
    if robot_lower in {"panda", "franka", "xarm", "koch", "so100", "sawyer", "ur5"}:
        return "single_arm", "high", f"robot_type={robot_lower}"
    if action_dim == 14 or state_dim == 14:
        return (
            "dual_arm",
            "medium",
            "14D action/state commonly indicates paired-arm control",
        )
    if action_dim in {6, 7, 8, 9, 10} or state_dim in {6, 7, 8, 9, 10}:
        return (
            "single_arm",
            "medium",
            "action/state dimensions match common single-arm control",
        )
    return "unknown", "low", "no strong embodiment signal found"


def _text_annotation_extent(
    *,
    tasks: list[str],
    temporal_dense_present: bool,
    temporal_sparse_present: bool,
    language_feature_keys: list[str],
) -> str:
    tags: list[str] = []
    if tasks:
        tags.append(
            "single_task_instruction" if len(tasks) == 1 else "multi_task_instruction"
        )
    if temporal_dense_present:
        tags.append("dense_temporal")
    if temporal_sparse_present:
        tags.append("sparse_temporal")
    if language_feature_keys:
        tags.append("language_fields")
    return "+".join(tags) if tags else "none"


def _inventory_error_row(
    *,
    repo_id: str,
    source_group: str,
    error: str,
    private: bool | None = None,
    readme_url: str | None = None,
    dataset_url: str | None = None,
) -> LeRobotConsortiumInventoryRow:
    return LeRobotConsortiumInventoryRow(
        source_group=source_group,
        repo_id=repo_id,
        private=private,
        domain_type="unknown",
        total_size_mb=None,
        data_size_mb=None,
        video_size_mb=None,
        total_episodes=None,
        total_frames=None,
        total_tasks=None,
        total_hours=None,
        avg_seconds_per_episode=None,
        fps=None,
        observation_fps=None,
        action_fps=None,
        robot_type=None,
        embodiment_type="unknown",
        embodiment_confidence="low",
        embodiment_reason="inventory generation failed",
        action_dim=None,
        action_shape=None,
        state_dim=None,
        state_shape=None,
        visual_stream_count=0,
        visual_stream_keys="",
        visual_dimensions="",
        visual_dtypes="",
        text_annotation_extent="unknown",
        task_text_present=False,
        task_text_count=0,
        task_text_examples="",
        temporal_dense_present=False,
        temporal_sparse_present=False,
        language_feature_keys="",
        readme_url=readme_url
        or f"https://huggingface.co/datasets/{repo_id}/blob/main/README.md",
        dataset_url=dataset_url or f"https://huggingface.co/datasets/{repo_id}",
        generation_error=error,
    )


def build_lerobot_consortium_inventory_row(
    *,
    api: HfApi,
    repo_id: str,
    source_group: str,
    token: str | None = None,
) -> LeRobotConsortiumInventoryRow:
    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    readme_url = f"{dataset_url}/blob/main/README.md"
    try:
        repo_info = api.repo_info(
            repo_id=repo_id, repo_type="dataset", token=token, files_metadata=True
        )
    except Exception as exc:  # pragma: no cover - defensive network guard
        return _inventory_error_row(
            repo_id=repo_id,
            source_group=source_group,
            error=str(exc),
            readme_url=readme_url,
            dataset_url=dataset_url,
        )

    readme_text = _load_downloaded_text(repo_id, "README.md", token=token)
    info = _load_downloaded_json(repo_id, "meta/info.json", token=token)
    if info is None:
        return _inventory_error_row(
            repo_id=repo_id,
            source_group=source_group,
            private=getattr(repo_info, "private", None),
            error="missing meta/info.json",
            readme_url=readme_url,
            dataset_url=dataset_url,
        )

    features = info.get("features") or {}
    action_feature = _find_feature(features, ("action", "actions"))
    state_feature = _find_feature(
        features, ("observation.state", "state", "observation.states")
    )
    action_shape = (
        action_feature.get("shape") if isinstance(action_feature, dict) else None
    )
    state_shape = (
        state_feature.get("shape") if isinstance(state_feature, dict) else None
    )
    action_dim = _shape_product(action_shape)
    state_dim = _shape_product(state_shape)

    visual_features = _find_visual_features(info)
    visual_keys: list[str] = []
    visual_dimensions: list[str] = []
    visual_dtypes: list[str] = []
    for key, meta in visual_features:
        visual_keys.append(key)
        visual_dtypes.append(str(meta.get("dtype") or ""))
        visual_dimensions.append(f"{key}:{_shape_text(meta.get('shape')) or '?'}")

    task_texts = _load_task_texts(repo_id, token=token)
    language_feature_keys = _find_language_feature_keys(info)
    temporal_dense_present = _remote_file_exists(
        repo_info, "meta/temporal_proportions_dense.json"
    )
    temporal_sparse_present = _remote_file_exists(
        repo_info, "meta/temporal_proportions_sparse.json"
    )
    text_annotation_extent = _text_annotation_extent(
        tasks=task_texts,
        temporal_dense_present=temporal_dense_present,
        temporal_sparse_present=temporal_sparse_present,
        language_feature_keys=language_feature_keys,
    )

    fps = _to_float(info.get("fps"))
    observation_fps = (
        _to_float(info.get("observation_fps"))
        or _to_float(info.get("video_fps"))
        or fps
    )
    action_fps = (
        _to_float(info.get("action_fps")) or _to_float(info.get("control_fps")) or fps
    )
    total_episodes = _to_int(info.get("total_episodes"))
    total_frames = _to_int(info.get("total_frames"))
    total_tasks = _to_int(info.get("total_tasks")) or (
        len(task_texts) if task_texts else None
    )
    total_hours = None
    avg_seconds_per_episode = None
    if total_frames is not None and observation_fps and observation_fps > 0:
        total_hours = round(total_frames / observation_fps / 3600.0, 3)
        if total_episodes:
            avg_seconds_per_episode = round(
                total_frames / observation_fps / total_episodes, 2
            )

    data_size_mb = _to_float(info.get("data_files_size_in_mb"))
    if data_size_mb is None:
        data_size_mb = _sum_prefixed_sizes_mb(repo_info, prefixes=("data/", "meta/"))
    video_size_mb = _to_float(info.get("video_files_size_in_mb"))
    if video_size_mb is None:
        video_size_mb = _sum_prefixed_sizes_mb(repo_info, prefixes=("videos/",))
    total_size_mb = _sum_repo_sizes_mb(repo_info)
    if total_size_mb is None and (
        data_size_mb is not None or video_size_mb is not None
    ):
        total_size_mb = round((data_size_mb or 0.0) + (video_size_mb or 0.0), 2)

    embodiment_type, embodiment_confidence, embodiment_reason = _infer_embodiment(
        repo_id=repo_id,
        robot_type=str(info.get("robot_type") or "") or None,
        action_dim=action_dim,
        state_dim=state_dim,
        readme_text=readme_text,
    )

    return LeRobotConsortiumInventoryRow(
        source_group=source_group,
        repo_id=repo_id,
        private=getattr(repo_info, "private", None),
        domain_type=_infer_domain_type(repo_id, readme_text),
        total_size_mb=total_size_mb,
        data_size_mb=data_size_mb,
        video_size_mb=video_size_mb,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=total_tasks,
        total_hours=total_hours,
        avg_seconds_per_episode=avg_seconds_per_episode,
        fps=fps,
        observation_fps=observation_fps,
        action_fps=action_fps,
        robot_type=str(info.get("robot_type") or "") or None,
        embodiment_type=embodiment_type,
        embodiment_confidence=embodiment_confidence,
        embodiment_reason=embodiment_reason,
        action_dim=action_dim,
        action_shape=_shape_text(action_shape),
        state_dim=state_dim,
        state_shape=_shape_text(state_shape),
        visual_stream_count=len(visual_features),
        visual_stream_keys=" | ".join(visual_keys),
        visual_dimensions=" | ".join(visual_dimensions),
        visual_dtypes=" | ".join(visual_dtypes),
        text_annotation_extent=text_annotation_extent,
        task_text_present=bool(task_texts),
        task_text_count=len(task_texts) if task_texts else 0,
        task_text_examples=" | ".join(task_texts[:3]),
        temporal_dense_present=temporal_dense_present,
        temporal_sparse_present=temporal_sparse_present,
        language_feature_keys=" | ".join(language_feature_keys),
        readme_url=readme_url,
        dataset_url=dataset_url,
        generation_error=None,
    )


def build_lerobot_consortium_inventory(
    repo_targets: Iterable[LeRobotConsortiumRepoTarget],
    *,
    token: str | None = None,
    workers: int = 8,
) -> list[LeRobotConsortiumInventoryRow]:
    api = HfApi(token=token)
    targets = list(repo_targets)
    rows: list[LeRobotConsortiumInventoryRow] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(
                build_lerobot_consortium_inventory_row,
                api=api,
                repo_id=target.repo_id,
                source_group=target.source_group,
                token=token,
            ): target
            for target in targets
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row.source_group, row.repo_id))
    return rows


_COMPATIBILITY_EXPORTS = (
    _prefer_repo_target,
    _to_bool,
    _to_float,
    _to_int,
)


# Preserve the historical wildcard-import surface.
__all__ = [
    "Any",
    "HfApi",
    "Iterable",
    "LeRobotConsortiumInventoryRow",
    "LeRobotConsortiumRepoTarget",
    "Path",
    "ThreadPoolExecutor",
    "annotations",
    "as_completed",
    "asdict",
    "build_lerobot_consortium_inventory",
    "build_lerobot_consortium_inventory_row",
    "csv",
    "dataclass",
    "hf_hub_download",
    "infer_lerobot_consortium_source_group",
    "json",
    "load_lerobot_consortium_inventory_rows",
    "load_lerobot_consortium_repo_targets",
    "pq",
    "render_lerobot_consortium_inventory_markdown",
    "write_lerobot_consortium_inventory_csv",
    "write_lerobot_consortium_inventory_json",
    "write_lerobot_consortium_inventory_markdown",
    "write_lerobot_consortium_repo_targets",
]
