from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq


_VISUAL_DTYPES = {"image", "video"}
_TEXT_DTYPES = {"string", "large_string"}


@dataclass(frozen=True)
class LeRobotConsortiumRepoTarget:
    """One input Hugging Face dataset repo selected for consortium indexing."""

    repo_id: str
    source_group: str


@dataclass(frozen=True)
class LeRobotConsortiumInventoryRow:
    """Spreadsheet-style inventory row for one HF LeRobot-format dataset."""

    source_group: str
    repo_id: str
    private: bool | None
    domain_type: str
    total_size_mb: float | None
    data_size_mb: float | None
    video_size_mb: float | None
    total_episodes: int | None
    total_frames: int | None
    total_tasks: int | None
    total_hours: float | None
    avg_seconds_per_episode: float | None
    fps: float | None
    observation_fps: float | None
    action_fps: float | None
    robot_type: str | None
    embodiment_type: str
    embodiment_confidence: str
    embodiment_reason: str
    action_dim: int | None
    action_shape: str | None
    state_dim: int | None
    state_shape: str | None
    visual_stream_count: int
    visual_stream_keys: str
    visual_dimensions: str
    visual_dtypes: str
    text_annotation_extent: str
    task_text_present: bool
    task_text_count: int | None
    task_text_examples: str
    temporal_dense_present: bool
    temporal_sparse_present: bool
    language_feature_keys: str
    readme_url: str
    dataset_url: str
    generation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    return None


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


def _split_pipe(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(" | ") if item.strip()]


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


def infer_lerobot_consortium_source_group(repo_id: str, *, default_source_group: str = "manual") -> str:
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
                source_group = raw_source_group or infer_lerobot_consortium_source_group(
                    repo_id,
                    default_source_group=default_source_group,
                )
                target = LeRobotConsortiumRepoTarget(repo_id=repo_id, source_group=source_group)
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
            explicit_source_group="," in line and "/" not in maybe_group if "," in line else False,
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
                writer.writerow({"repo_id": target.repo_id, "source_group": target.source_group})
        return

    with path.open("w", encoding="utf-8") as handle:
        for target in targets:
            handle.write(f"{target.source_group},{target.repo_id}\n")


def _load_downloaded_json(repo_id: str, filename: str, *, token: str | None = None) -> dict[str, Any] | None:
    try:
        path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=token))
    except Exception:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_downloaded_text(repo_id: str, filename: str, *, token: str | None = None) -> str | None:
    try:
        path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=token))
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
            path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=token))
        except Exception:
            continue
        try:
            rows = pq.read_table(path).to_pylist()
        except Exception:
            continue
        return _extract_task_texts(rows)
    for filename in jsonl_candidates:
        try:
            path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=token))
        except Exception:
            continue
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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


def _find_feature(feats: dict[str, Any], candidates: Iterable[str]) -> dict[str, Any] | None:
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
        if dtype in _TEXT_DTYPES or any(token in key_lower for token in ("task", "language", "instruction", "caption", "text")):
            keys.append(key_text)
    return sorted(dict.fromkeys(keys))


def _infer_domain_type(repo_id: str, readme_text: str | None) -> str:
    repo_lower = repo_id.lower()
    readme_lower = (readme_text or "").lower()
    joined = f"{repo_lower} {readme_lower}"
    sim_signals = ("_sim", "/sim_", " simulation", " simulated", "in simulation")
    real_signals = ("_real", "/real_", " real-world", " real world", "real robot", "physical robot")
    has_sim = any(signal in joined for signal in sim_signals)
    has_real = any(signal in joined for signal in real_signals)
    if has_sim and not has_real:
        return "sim"
    if has_real and not has_sim:
        return "real"
    if any(token in repo_lower for token in ("umi", "exumi", "dexumi", "touchwild", "dexwild")):
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

    if any(token in joined for token in ("dexumi", "dexterous", "xhand", "inspire hand", "inspire_hand")):
        return "dexterous_hand", "high", "repo or README indicates dexterous hand manipulation"
    if "aloha_mobile" in repo_lower or "mobile manipulator" in readme_lower:
        return "mobile_manipulator", "high", "repo or README indicates mobile manipulator"
    if robot_lower == "aloha" or "bimanual" in joined:
        return "dual_arm", "high", "ALOHA or bimanual signals indicate dual-arm embodiment"
    if any(token in joined for token in ("mobile robot", "navigation", "gnm")):
        return "mobile_robot", "medium", "repo or README indicates mobile robot"
    if robot_lower in {"panda", "franka", "xarm", "koch", "so100", "sawyer", "ur5"}:
        return "single_arm", "high", f"robot_type={robot_lower}"
    if action_dim == 14 or state_dim == 14:
        return "dual_arm", "medium", "14D action/state commonly indicates paired-arm control"
    if action_dim in {6, 7, 8, 9, 10} or state_dim in {6, 7, 8, 9, 10}:
        return "single_arm", "medium", "action/state dimensions match common single-arm control"
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
        tags.append("single_task_instruction" if len(tasks) == 1 else "multi_task_instruction")
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
        readme_url=readme_url or f"https://huggingface.co/datasets/{repo_id}/blob/main/README.md",
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
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", token=token, files_metadata=True)
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
    state_feature = _find_feature(features, ("observation.state", "state", "observation.states"))
    action_shape = action_feature.get("shape") if isinstance(action_feature, dict) else None
    state_shape = state_feature.get("shape") if isinstance(state_feature, dict) else None
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
    temporal_dense_present = _remote_file_exists(repo_info, "meta/temporal_proportions_dense.json")
    temporal_sparse_present = _remote_file_exists(repo_info, "meta/temporal_proportions_sparse.json")
    text_annotation_extent = _text_annotation_extent(
        tasks=task_texts,
        temporal_dense_present=temporal_dense_present,
        temporal_sparse_present=temporal_sparse_present,
        language_feature_keys=language_feature_keys,
    )

    fps = _to_float(info.get("fps"))
    observation_fps = _to_float(info.get("observation_fps")) or _to_float(info.get("video_fps")) or fps
    action_fps = _to_float(info.get("action_fps")) or _to_float(info.get("control_fps")) or fps
    total_episodes = _to_int(info.get("total_episodes"))
    total_frames = _to_int(info.get("total_frames"))
    total_tasks = _to_int(info.get("total_tasks")) or (len(task_texts) if task_texts else None)
    total_hours = None
    avg_seconds_per_episode = None
    if total_frames is not None and observation_fps and observation_fps > 0:
        total_hours = round(total_frames / observation_fps / 3600.0, 3)
        if total_episodes:
            avg_seconds_per_episode = round(total_frames / observation_fps / total_episodes, 2)

    data_size_mb = _to_float(info.get("data_files_size_in_mb"))
    if data_size_mb is None:
        data_size_mb = _sum_prefixed_sizes_mb(repo_info, prefixes=("data/", "meta/"))
    video_size_mb = _to_float(info.get("video_files_size_in_mb"))
    if video_size_mb is None:
        video_size_mb = _sum_prefixed_sizes_mb(repo_info, prefixes=("videos/",))
    total_size_mb = _sum_repo_sizes_mb(repo_info)
    if total_size_mb is None and (data_size_mb is not None or video_size_mb is not None):
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


def load_lerobot_consortium_inventory_rows(path: Path) -> list[LeRobotConsortiumInventoryRow]:
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
                    avg_seconds_per_episode=_to_float(raw.get("avg_seconds_per_episode")),
                    fps=_to_float(raw.get("fps")),
                    observation_fps=_to_float(raw.get("observation_fps")),
                    action_fps=_to_float(raw.get("action_fps")),
                    robot_type=raw.get("robot_type") or None,
                    embodiment_type=raw.get("embodiment_type", "unknown") or "unknown",
                    embodiment_confidence=raw.get("embodiment_confidence", "low") or "low",
                    embodiment_reason=raw.get("embodiment_reason", ""),
                    action_dim=_to_int(raw.get("action_dim")),
                    action_shape=raw.get("action_shape") or None,
                    state_dim=_to_int(raw.get("state_dim")),
                    state_shape=raw.get("state_shape") or None,
                    visual_stream_count=_to_int(raw.get("visual_stream_count")) or 0,
                    visual_stream_keys=raw.get("visual_stream_keys", ""),
                    visual_dimensions=raw.get("visual_dimensions", ""),
                    visual_dtypes=raw.get("visual_dtypes", ""),
                    text_annotation_extent=raw.get("text_annotation_extent", "none") or "none",
                    task_text_present=bool(_to_bool(raw.get("task_text_present"))),
                    task_text_count=_to_int(raw.get("task_text_count")),
                    task_text_examples=raw.get("task_text_examples", ""),
                    temporal_dense_present=bool(_to_bool(raw.get("temporal_dense_present"))),
                    temporal_sparse_present=bool(_to_bool(raw.get("temporal_sparse_present"))),
                    language_feature_keys=raw.get("language_feature_keys", ""),
                    readme_url=raw.get("readme_url", ""),
                    dataset_url=raw.get("dataset_url", ""),
                    generation_error=raw.get("generation_error") or None,
                )
            )
    return rows


def render_lerobot_consortium_inventory_markdown(rows: Iterable[LeRobotConsortiumInventoryRow]) -> str:
    rows_list = list(rows)
    by_group: dict[str, int] = {}
    group_episodes: dict[str, int] = {}
    group_hours: dict[str, float] = {}
    annotated_counts: dict[str, int] = {}
    errored = [row for row in rows_list if row.generation_error]
    for row in rows_list:
        by_group[row.source_group] = by_group.get(row.source_group, 0) + 1
        if row.total_episodes is not None:
            group_episodes[row.source_group] = group_episodes.get(row.source_group, 0) + row.total_episodes
        if row.total_hours is not None:
            group_hours[row.source_group] = group_hours.get(row.source_group, 0.0) + row.total_hours
        if row.task_text_present or row.temporal_dense_present or row.temporal_sparse_present:
            annotated_counts[row.source_group] = annotated_counts.get(row.source_group, 0) + 1

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
        size_gb = f"{(row.total_size_mb or 0.0) / 1024.0:.2f}" if row.total_size_mb is not None else ""
        hours = f"{row.total_hours:.2f}" if row.total_hours is not None else ""
        obs_fps = f"{row.observation_fps:.1f}" if row.observation_fps is not None else ""
        action_fps = f"{row.action_fps:.1f}" if row.action_fps is not None else ""
        avg_seconds = f"{row.avg_seconds_per_episode:.1f}" if row.avg_seconds_per_episode is not None else ""
        visual_dimensions = row.visual_dimensions.replace(" | ", "<br>") if row.visual_dimensions else ""
        lines.append(
            f"| {row.source_group} | {row.repo_id} | {row.domain_type} | {size_gb} | {row.total_episodes or ''} | "
            f"{hours} | {obs_fps} | {action_fps} | {avg_seconds} | "
            f"{row.embodiment_type} ({row.embodiment_confidence}) | {row.action_dim or ''} | "
            f"{row.visual_stream_count} | {visual_dimensions} | {row.text_annotation_extent} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_lerobot_consortium_inventory_csv(path: Path, rows: Iterable[LeRobotConsortiumInventoryRow]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_list[0].to_dict().keys()) if rows_list else [])
        if not rows_list:
            return
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row.to_dict())


def write_lerobot_consortium_inventory_json(path: Path, rows: Iterable[LeRobotConsortiumInventoryRow]) -> None:
    payload = {"rows": [row.to_dict() for row in rows]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_lerobot_consortium_inventory_markdown(path: Path, rows: Iterable[LeRobotConsortiumInventoryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_lerobot_consortium_inventory_markdown(rows), encoding="utf-8")
