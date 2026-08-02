"""Manifest, path, FPS, and source-stream parsing for mixed video."""

from __future__ import annotations

import csv
from pathlib import Path

import imageio.v2 as imageio

from open_wam.configs import (
    MixedVideoDataConfig,
    MixedVideoSourceConfig,
    MixedVideoSourceFormat,
)
from open_wam.contracts import (
    ResolvedVideoClip,
    normalized_video_frame_count,
    resolve_video_source_fps,
)

from .mixed_video_catalog_contracts import MixedVideoStreamRecord


def _load_source_streams(
    source: MixedVideoSourceConfig,
    data_config: MixedVideoDataConfig,
) -> list[MixedVideoStreamRecord]:
    manifest_path = _resolve_manifest_path(source)
    rows = _read_manifest_csv(manifest_path)
    if not rows:
        raise ValueError(f"Mixed-video manifest is empty: {manifest_path}")
    streams: list[MixedVideoStreamRecord] = []
    for row in rows:
        stream_key = _string_field(row, "stream_key") or _string_field(
            row,
            "video_key",
        )
        stream_index = _int_field(row, "stream_index", default=0)
        if not stream_key:
            stream_key = f"stream_{stream_index}"
        if (
            source.include_streams
            and stream_key not in source.include_streams
            and str(stream_index) not in source.include_streams
        ):
            continue
        target_slot = _target_slot_for_stream(
            row,
            source,
            data_config,
            stream_key,
            stream_index,
        )
        configured_slots = set(data_config.camera_names) | set(
            data_config.latent_camera_names
        )
        for combination in data_config.latent_view_combinations:
            if combination.enabled:
                configured_slots.update(combination.slots)
        if target_slot not in configured_slots:
            continue
        length_frames = _int_field(row, "length_frames", default=0)
        if length_frames <= 0:
            length_frames = _int_field(row, "num_frames", default=0)
        if length_frames <= 0:
            raise ValueError(
                "Mixed-video manifest row must include positive length_frames: "
                f"{manifest_path}, source={source.source_id}, "
                f"stream={stream_key}."
            )
        local_path = _local_video_path(row, source, manifest_path)
        latent_path = _local_latent_path(row, source, manifest_path)
        shard_relative_path = _string_field(row, "shard_relative_path")
        latent_shard_relative_path = (
            _string_field(row, "latent_shard_relative_path")
            or _string_field(row, "video_latents_shard_relative_path")
        )
        latent_length_frames = (
            _optional_int_field(row, "latent_length_frames")
            or _optional_int_field(row, "video_latent_frames")
        )
        if (
            latent_length_frames is None
            and source.source_format == MixedVideoSourceFormat.LATENT
        ):
            latent_length_frames = length_frames
        repo_id = _string_field(row, "repo_id") or source.repo_id
        dataset_id = (
            _string_field(row, "dataset_id")
            or repo_id
            or source.source_id
        )
        episode_index = _int_field(row, "episode_index", default=0)
        clip_id = (
            _string_field(row, "clip_id")
            or _string_field(row, "clip_key")
            or "default"
        )
        observation_fps = _float_field(row, "observation_fps")
        container_fps = None
        source_has_rgb = source.source_format in {
            MixedVideoSourceFormat.RGB,
            MixedVideoSourceFormat.RGB_AND_LATENT,
        }
        if observation_fps is None and source_has_rgb:
            container_fps = _probe_video_observation_fps(local_path)
        if (
            observation_fps is None
            and container_fps is None
            and source_has_rgb
            and data_config.target_observation_fps is not None
            and local_path is None
            and repo_id is not None
            and shard_relative_path is not None
        ):
            raise ValueError(
                "Remote mixed-video RGB rows must include `observation_fps` "
                "when `target_observation_fps` is enabled. Container FPS "
                "probing is only performed for local files; either add "
                "manifest observation_fps, disable FPS normalization, or "
                "materialize the video locally. "
                f"manifest={manifest_path}, source={source.source_id}, "
                f"dataset_id={dataset_id}, episode_index={episode_index}, "
                f"clip_id={clip_id}, stream_key={stream_key}, "
                f"shard_relative_path={shard_relative_path!r}."
            )
        resolved_fps = resolve_video_source_fps(
            observation_fps,
            container_fps=container_fps,
            missing_observation_fps=data_config.missing_observation_fps,
        )
        normalized_length = normalized_video_frame_count(
            length_frames,
            source_fps=resolved_fps.value,
            target_fps=data_config.target_observation_fps,
        )
        clip = ResolvedVideoClip(
            clip_id=clip_id,
            source_id=source.source_id,
            dataset_id=dataset_id,
            episode_index=episode_index,
            stream_key=stream_key,
            target_slot=target_slot,
            path_key=_stream_path_key(
                local_path=local_path,
                latent_path=latent_path,
                repo_id=repo_id,
                shard_relative_path=shard_relative_path,
                latent_shard_relative_path=latent_shard_relative_path,
            ),
            native_length_frames=length_frames,
            source_fps=resolved_fps.value,
            source_fps_source=resolved_fps.source,
            target_fps=data_config.target_observation_fps,
            normalized_length_frames=normalized_length,
            from_timestamp=_float_field(row, "from_timestamp"),
            to_timestamp=_float_field(row, "to_timestamp"),
            width=_optional_int_field(row, "width"),
            height=_optional_int_field(row, "height"),
        )
        streams.append(
            MixedVideoStreamRecord(
                source_id=source.source_id,
                source_group=_string_field(row, "source_group")
                or source.source_group,
                repo_id=repo_id,
                dataset_id=dataset_id,
                episode_index=episode_index,
                clip_id=clip_id,
                stream_index=stream_index,
                stream_key=stream_key,
                target_slot=target_slot,
                source_format=source.source_format,
                manifest_path=manifest_path,
                local_path=local_path,
                latent_path=latent_path,
                shard_relative_path=shard_relative_path,
                latent_shard_relative_path=latent_shard_relative_path,
                latent_key=_string_field(row, "latent_key")
                or source.latent_key,
                length_frames=length_frames,
                latent_length_frames=latent_length_frames,
                observation_fps=resolved_fps.value,
                action_fps=_float_field(row, "action_fps"),
                from_timestamp=clip.from_timestamp,
                to_timestamp=clip.to_timestamp,
                width=clip.width,
                height=clip.height,
                channels=_optional_int_field(row, "channels"),
                tasks=_parse_tasks(row),
                clip=clip,
            )
        )
    return streams


def _resolve_manifest_path(source: MixedVideoSourceConfig) -> Path:
    manifest = Path(source.manifest_csv).expanduser()
    if manifest.exists():
        return manifest
    if source.local_root is not None:
        candidate = Path(source.local_root).expanduser() / source.manifest_csv
        if candidate.exists():
            return candidate
    return manifest


def _read_manifest_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing mixed-video manifest CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _probe_video_observation_fps(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps", 0.0) or 0.0)
    finally:
        reader.close()
    return fps if fps > 0.0 else None


def _stream_path_key(
    *,
    local_path: Path | None,
    latent_path: Path | None,
    repo_id: str | None,
    shard_relative_path: str | None,
    latent_shard_relative_path: str | None,
) -> str:
    if local_path is not None:
        return str(local_path)
    if latent_path is not None:
        return str(latent_path)
    if repo_id is not None and shard_relative_path is not None:
        return f"{repo_id}:{shard_relative_path}"
    if repo_id is not None and latent_shard_relative_path is not None:
        return f"{repo_id}:{latent_shard_relative_path}"
    return "<unresolved>"


def _target_slot_for_stream(
    row: dict[str, str],
    source: MixedVideoSourceConfig,
    data_config: MixedVideoDataConfig,
    stream_key: str,
    stream_index: int,
) -> str:
    source_mapping = {
        mapping.source_name: mapping.target_slot
        for mapping in source.channel_mappings
    }
    if stream_key in source_mapping:
        return source_mapping[stream_key]
    row_target = _string_field(row, "target_slot") or _string_field(
        row,
        "target_slot_key",
    )
    if row_target:
        return row_target
    default_slots = (
        data_config.latent_camera_names
        if source.source_format == MixedVideoSourceFormat.LATENT
        and data_config.latent_camera_names
        else data_config.camera_names
    )
    if stream_index < len(default_slots):
        return default_slots[stream_index]
    return stream_key


def _local_video_path(
    row: dict[str, str],
    source: MixedVideoSourceConfig,
    manifest_path: Path,
) -> Path | None:
    raw_path = _string_field(row, "local_path") or _string_field(
        row,
        "video_path",
    )
    if raw_path is None and source.local_root is not None:
        raw_path = _string_field(row, "shard_relative_path")
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if source.local_root is not None:
        candidates.append(Path(source.local_root).expanduser() / path)
    candidates.append(manifest_path.parent / path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _local_latent_path(
    row: dict[str, str],
    source: MixedVideoSourceConfig,
    manifest_path: Path,
) -> Path | None:
    raw_path = (
        _string_field(row, "latent_path")
        or _string_field(row, "video_latents_path")
        or _string_field(row, "latent_local_path")
    )
    if raw_path is None and source.latent_root is not None:
        raw_path = _string_field(
            row,
            "latent_shard_relative_path",
        ) or _string_field(row, "video_latents_shard_relative_path")
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if source.latent_root is not None:
        candidates.append(Path(source.latent_root).expanduser() / path)
    candidates.append(manifest_path.parent / path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parse_tasks(row: dict[str, str]) -> tuple[str, ...]:
    raw = (
        _string_field(row, "tasks")
        or _string_field(row, "task")
        or _string_field(row, "language")
    )
    if raw is None:
        return ()
    cleaned = raw.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    tasks = [
        item.strip().strip("'\"")
        for item in cleaned.replace("|", ",").replace(";", ",").split(",")
    ]
    return tuple(item for item in tasks if item)


def _string_field(
    row: dict[str, str],
    key: str,
) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _int_field(
    row: dict[str, str],
    key: str,
    *,
    default: int,
) -> int:
    value = _string_field(row, key)
    if value is None:
        return default
    return int(float(value))


def _optional_int_field(
    row: dict[str, str],
    key: str,
) -> int | None:
    value = _string_field(row, key)
    if value is None:
        return None
    return int(float(value))


def _float_field(
    row: dict[str, str],
    key: str,
) -> float | None:
    value = _string_field(row, key)
    if value is None:
        return None
    return float(value)


__all__: list[str] = []
