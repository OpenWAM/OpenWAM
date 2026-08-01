"""Mixed-video RGB-to-latent encoding and artifact generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import queue
import threading
from typing import Any, Protocol, TypedDict

import torch

from open_wam.configs import (
    ExperimentConfig,
    MixedVideoDataConfig,
    MixedVideoDecodeSizeMode,
    MixedVideoEncodingSplit,
    MixedVideoFrameFitMode,
    MixedVideoLatentEncodingMode,
    MixedVideoResizeBinConfig,
    MixedVideoSourceFormat,
    ViewLayoutConfig,
)
from open_wam.configs.enums import serialize_enum_values
from open_wam.contracts import ViewPlacement
from open_wam.data.mixed_video_decode import (
    iter_mixed_video_stream_frame_chunks,
)
from open_wam.data.mixed_video_encoding_artifacts import (
    LATENT_KEY,
    _encoded_latent_target_slot,
    _encoded_record_episode_count,
    _encoded_record_report_key,
    _latent_path_for_episode,
    _latent_path_for_episode_view,
    _manifest_row_for_encoded_episode,
    _safe_path_part,
    _validate_encoded_records_for_backbone,
    _write_latent_source_config_patch,
    _write_latent_training_config,
    _write_source_manifests,
)
from open_wam.data.mixed_video_catalog import (
    MixedVideoCatalog,
    MixedVideoEpisodeRecord,
    MixedVideoStreamRecord,
    load_mixed_video_catalog,
    split_mixed_video_episodes,
)
from open_wam.data.raw_video import (
    ConfiguredCanonicalVideoPreprocessor,
    build_canonical_video_preprocessor,
)
from open_wam.models.common.video_geometry import wan_raw_frame_count_to_latent_count
MAX_PENDING_LATENT_SAVE_FUTURES = 1


class MixedVideoLatentEncoder(Protocol):
    """Capability required by the mixed-video encoding engine."""

    def encode_video(
        self,
        canonical_video: torch.Tensor,
        *,
        placements: Sequence[ViewPlacement] | None = None,
        reset_cache: bool = True,
    ) -> torch.Tensor:
        del canonical_video, placements, reset_cache
        raise NotImplementedError


class MixedVideoEncodingReport(TypedDict):
    """Serialized artifact and progress report returned by one encode call."""

    output_root: str
    encoded_episodes: int
    encoded_targets: int
    manifest_encoded_episodes: int
    manifest_encoded_targets: int
    newly_encoded_episodes: int
    newly_encoded_targets: int
    reused_episodes: int
    reused_targets: int
    selected_episodes: int
    shard_episodes: int
    split: str
    shard_count: int
    shard_index: int
    source_ids: list[str]
    decode_size_mode: str
    decode_fit_mode: str
    manifest_paths: dict[str, str]
    config_patch_path: str | None
    latent_training_config_path: str | None
    latent_shapes: dict[str, list[int]]


def _mixed_video_transform_signature(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord | None = None,
) -> dict[str, Any]:
    resize_bins = serialize_enum_values(asdict(data_config))["decode_resize_bins"]
    view_layout = [
        {
            "source_name": layout.source_name,
            "canonical_name": layout.canonical_name,
            "top": int(layout.top),
            "left": int(layout.left),
            "height": int(layout.height),
            "width": int(layout.width),
        }
        for layout in data_config.view_layout
    ]
    signature: dict[str, Any] = {
        "canonical_height": int(data_config.canonical_height),
        "canonical_width": int(data_config.canonical_width),
        "camera_names": list(data_config.camera_names),
        "latent_camera_names": list(data_config.latent_camera_names),
        "view_layout": view_layout,
        "decode_size_mode": data_config.decode_size_mode.value,
        "decode_fit_mode": data_config.decode_fit_mode.value,
        "decode_allow_upscale": bool(data_config.decode_allow_upscale),
        "decode_height": int(data_config.decode_height),
        "decode_width": int(data_config.decode_width),
        "decode_resize_bins": resize_bins,
        "target_observation_fps": data_config.target_observation_fps,
        "missing_observation_fps": float(data_config.missing_observation_fps),
    }
    if episode is not None:
        signature["streams"] = [
            {
                "target_slot": stream.target_slot,
                "stream_key": stream.stream_key,
                "clip_id": stream.clip_id,
                "length_frames": int(stream.length_frames),
                "latent_length_frames": stream.latent_length_frames,
                "observation_fps": stream.observation_fps,
                "resolved_observation_fps": stream.clip.source_fps,
                "source_fps_source": stream.clip.source_fps_source,
                "from_timestamp": stream.from_timestamp,
                "to_timestamp": stream.to_timestamp,
                "width": stream.width,
                "height": stream.height,
                "source_format": stream.source_format.value,
            }
            for stream in sorted(episode.streams, key=lambda item: (item.target_slot, item.stream_index, item.stream_key))
            if stream.target_slot in data_config.camera_names
        ]
    return signature


def _mixed_video_transform_signature_hash(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord | None = None,
) -> str:
    payload = json.dumps(_mixed_video_transform_signature(data_config, episode), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MixedVideoEncodedEpisode:
    """One encoded sidecar and its manifest-facing metadata."""

    source_id: str
    dataset_id: str
    episode_index: int
    clip_id: str
    latent_path: Path
    latent_shape: tuple[int, int, int, int]
    raw_length_frames: int
    latent_length_frames: int
    tasks: tuple[str, ...]
    target_slot: str = "observation.images.slot0"
    encoded_slots: tuple[str, ...] = ()
    encoding_mode: MixedVideoLatentEncodingMode = MixedVideoLatentEncodingMode.CANONICAL

    def __post_init__(self) -> None:
        latent_shape = tuple(int(value) for value in self.latent_shape)
        tasks = (self.tasks,) if isinstance(self.tasks, str) else self.tasks
        encoded_slots = (
            (self.encoded_slots,)
            if isinstance(self.encoded_slots, str)
            else self.encoded_slots
        )
        if len(latent_shape) != 4:
            raise ValueError(
                "Mixed-video encoded episodes require a [C, T, H, W] latent shape."
            )
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "dataset_id", str(self.dataset_id))
        object.__setattr__(self, "episode_index", int(self.episode_index))
        object.__setattr__(self, "clip_id", str(self.clip_id))
        object.__setattr__(self, "latent_path", Path(self.latent_path))
        object.__setattr__(self, "latent_shape", latent_shape)
        object.__setattr__(self, "raw_length_frames", int(self.raw_length_frames))
        object.__setattr__(self, "latent_length_frames", int(self.latent_length_frames))
        object.__setattr__(self, "tasks", tuple(str(value) for value in tasks))
        object.__setattr__(self, "target_slot", str(self.target_slot))
        object.__setattr__(self, "encoded_slots", tuple(str(value) for value in encoded_slots))
        object.__setattr__(
            self,
            "encoding_mode",
            MixedVideoLatentEncodingMode(self.encoding_mode),
        )


@dataclass(frozen=True)
class MixedVideoEncodingTarget:
    """One canonical or per-view latent sidecar planned for an episode."""

    name: str
    mode: MixedVideoLatentEncodingMode
    target_slot: str
    source_slots: tuple[str, ...]
    latent_path: Path
    compatible_existing_paths: tuple[Path, ...] = ()
    include_in_training_manifest: bool = True

    def __post_init__(self) -> None:
        raw_source_slots = (
            (self.source_slots,)
            if isinstance(self.source_slots, str)
            else self.source_slots
        )
        raw_existing_paths = (
            (self.compatible_existing_paths,)
            if isinstance(self.compatible_existing_paths, (str, Path))
            else self.compatible_existing_paths
        )
        source_slots = tuple(str(value) for value in raw_source_slots)
        if not source_slots:
            raise ValueError("Mixed-video encoding targets require at least one source slot.")
        if not self.name:
            raise ValueError("Mixed-video encoding targets require a non-empty name.")
        if not self.target_slot:
            raise ValueError("Mixed-video encoding targets require a non-empty target slot.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "mode", MixedVideoLatentEncodingMode(self.mode))
        object.__setattr__(self, "target_slot", str(self.target_slot))
        object.__setattr__(self, "source_slots", source_slots)
        object.__setattr__(self, "latent_path", Path(self.latent_path))
        object.__setattr__(
            self,
            "compatible_existing_paths",
            tuple(Path(path) for path in raw_existing_paths),
        )
        object.__setattr__(
            self,
            "include_in_training_manifest",
            bool(self.include_in_training_manifest),
        )


@dataclass(frozen=True)
class MixedVideoEncodingSelection:
    """Deterministic episode subset assigned to one encoding worker."""

    split: MixedVideoEncodingSplit = MixedVideoEncodingSplit.ALL
    source_ids: tuple[str, ...] = ()
    episode_indices: tuple[int, ...] = ()
    max_episodes: int | None = None
    shard_count: int = 1
    shard_index: int = 0

    def __post_init__(self) -> None:
        source_ids = (
            (self.source_ids,)
            if isinstance(self.source_ids, str)
            else self.source_ids
        )
        object.__setattr__(self, "split", MixedVideoEncodingSplit(self.split))
        object.__setattr__(self, "source_ids", tuple(str(value) for value in source_ids))
        object.__setattr__(self, "episode_indices", tuple(int(value) for value in self.episode_indices))
        if self.max_episodes is not None:
            object.__setattr__(self, "max_episodes", int(self.max_episodes))
        object.__setattr__(self, "shard_count", int(self.shard_count))
        object.__setattr__(self, "shard_index", int(self.shard_index))


# Private compatibility names used by the historical checkout command.
EncodedEpisode = MixedVideoEncodedEpisode
EncodingTarget = MixedVideoEncodingTarget
EncoderSelection = MixedVideoEncodingSelection


def _wait_for_latent_save(latent_path: Path, save_future: Future) -> None:
    try:
        save_future.result()
    except Exception as exc:
        raise RuntimeError(f"Failed to write mixed-video latent sidecar: {latent_path}") from exc


def _submit_latent_save(
    *,
    save_executor: ThreadPoolExecutor,
    save_futures: list[tuple[Path, Future]],
    latent_path: Path,
    payload: dict[str, Any],
    max_pending: int = MAX_PENDING_LATENT_SAVE_FUTURES,
) -> None:
    pending_limit = max(1, int(max_pending))
    while len(save_futures) >= pending_limit:
        _wait_for_latent_save(*save_futures.pop(0))
    save_futures.append((latent_path, save_executor.submit(torch.save, payload, latent_path)))


def resolve_mixed_video_encoding_config(
    data_config: MixedVideoDataConfig,
    *,
    decode_size_mode: MixedVideoDecodeSizeMode | str | None = None,
    decode_fit_mode: MixedVideoFrameFitMode | str | None = None,
    decode_height: int | None = None,
    decode_width: int | None = None,
    decode_resize_bins: Sequence[MixedVideoResizeBinConfig] | None = None,
) -> MixedVideoDataConfig:
    """Apply typed offline-encoding overrides to a mixed-video data config."""

    updates: dict[str, Any] = {}
    if decode_size_mode is not None:
        updates["decode_size_mode"] = MixedVideoDecodeSizeMode(decode_size_mode)
    if decode_fit_mode is not None:
        updates["decode_fit_mode"] = MixedVideoFrameFitMode(decode_fit_mode)
    if decode_height is not None:
        updates["decode_height"] = int(decode_height)
    if decode_width is not None:
        updates["decode_width"] = int(decode_width)
    if decode_resize_bins is not None:
        updates["decode_resize_bins"] = tuple(decode_resize_bins)
    if updates.get("decode_size_mode", data_config.decode_size_mode) == MixedVideoDecodeSizeMode.ASPECT_RATIO_BINS:
        # The encoder itself is single-episode, but the data config validator
        # also protects training-time collation. Normalize these fields so a
        # fixed-size training config can still be reused for offline encoding.
        updates.setdefault("train_batch_size", 1)
        updates.setdefault("val_batch_size", 1)
    return replace(data_config, **updates)


def encode_mixed_video_latent_sources(
    *,
    data_config: MixedVideoDataConfig,
    assets: MixedVideoLatentEncoder | None,
    output_root: Path,
    device: torch.device,
    split: MixedVideoEncodingSplit | str = MixedVideoEncodingSplit.ALL,
    source_ids: tuple[str, ...] = (),
    selection: MixedVideoEncodingSelection | None = None,
    experiment_config: ExperimentConfig | None = None,
    max_episodes: int | None = None,
    chunk_frames: int = 65,
    overwrite: bool = False,
    skip_existing: bool = False,
    write_manifests: bool = True,
) -> MixedVideoEncodingReport:
    if selection is None:
        selection = MixedVideoEncodingSelection(
            split=split,
            source_ids=tuple(source_ids),
            max_episodes=max_episodes,
        )
    if chunk_frames != 0 and chunk_frames < 5:
        raise ValueError("--chunk-frames must be 0 or at least 5.")
    output_root = output_root.expanduser().resolve()
    latents_root = output_root / "latents"
    manifests_root = output_root / "manifests"
    latents_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    all_selected_episodes = _select_encoder_episodes(data_config, selection=selection, apply_shard=False)
    episodes = _select_encoder_episodes(data_config, selection=selection, apply_shard=True)
    _preflight_output_paths(
        episodes,
        data_config=data_config,
        output_root=output_root,
        latents_root=latents_root,
        manifests_root=manifests_root,
        overwrite=overwrite,
        skip_existing=skip_existing,
        write_manifests=write_manifests,
    )

    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    encoded: list[MixedVideoEncodedEpisode] = []
    manifest_records: list[MixedVideoEncodedEpisode] = []
    newly_encoded_records: list[MixedVideoEncodedEpisode] = []
    reused_records: list[MixedVideoEncodedEpisode] = []
    save_executor: ThreadPoolExecutor | None = None
    save_futures: list[tuple[Path, Future]] = []
    try:
        for episode in episodes:
            targets = _encoding_targets_for_episode(latents_root, episode, data_config)
            if not targets:
                continue
            for target in targets:
                latent_path = target.latent_path
                latent_path.parent.mkdir(parents=True, exist_ok=True)
                existing_path = _resolve_existing_target_path(target) if skip_existing else None
                target_data_config = _data_config_for_encoding_target(data_config, target)
                target_episode = _episode_for_encoding_target(episode, target)
                if existing_path is not None:
                    record = _encoded_episode_from_existing_sidecar(
                        existing_path,
                        target_episode,
                        data_config=target_data_config,
                        target=target,
                    )
                    reused_records.append(record)
                else:
                    if assets is None:
                        raise FileNotFoundError(
                            f"Missing encoded sidecar for source={episode.source_id!r}, "
                            f"episode={episode.episode_index}, target={target.name}: {latent_path}"
                        )
                    canonicalizer = build_canonical_video_preprocessor(target_data_config)
                    latents, encoding_metadata = _encode_episode_latents_streaming(
                        target_data_config,
                        target_episode,
                        canonicalizer=canonicalizer,
                        assets=assets,
                        device=device,
                        chunk_frames=chunk_frames,
                    )
                    latents = latents.detach().cpu().contiguous()[0]
                    payload = {
                        LATENT_KEY: latents,
                        "metadata": {
                            "source_id": target_episode.source_id,
                            "dataset_id": target_episode.dataset_id,
                            "episode_index": target_episode.episode_index,
                            "clip_id": target_episode.clip_id,
                            "raw_length_frames": int(target_episode.length_frames),
                            "native_length_frames": int(target_episode.native_length_frames),
                            "target_observation_fps": target_data_config.target_observation_fps,
                            "missing_observation_fps": float(target_data_config.missing_observation_fps),
                            "latent_length_frames": int(latents.shape[1]),
                            "latent_shape": list(latents.shape),
                            "target_slot": target.target_slot,
                            "encoded_slots": list(target.source_slots),
                            "encoding_mode": target.mode.value,
                            "decode_size_mode": target_data_config.decode_size_mode.value,
                            "decode_fit_mode": target_data_config.decode_fit_mode.value,
                            "decode_allow_upscale": bool(target_data_config.decode_allow_upscale),
                            "decode_height": int(target_data_config.decode_height),
                            "decode_width": int(target_data_config.decode_width),
                            "decode_resize_bins": _mixed_video_transform_signature(target_data_config)["decode_resize_bins"],
                            "stream_transform_signature": _mixed_video_transform_signature(
                                target_data_config,
                                target_episode,
                            ).get("streams", []),
                            "transform_signature_hash": _mixed_video_transform_signature_hash(
                                target_data_config,
                                target_episode,
                            ),
                            **encoding_metadata,
                            "tasks": list(target_episode.tasks),
                        },
                    }
                    # WHY async save: torch.save serializes to disk synchronously which
                    # blocks the next episode's decode. Keep only one pending sidecar so
                    # large latent tensors cannot accumulate across the whole encode run.
                    if save_executor is None:
                        save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mixed-video-save")
                    _submit_latent_save(
                        save_executor=save_executor,
                        save_futures=save_futures,
                        latent_path=latent_path,
                        payload=payload,
                    )
                    record = MixedVideoEncodedEpisode(
                        source_id=target_episode.source_id,
                        dataset_id=target_episode.dataset_id,
                        episode_index=target_episode.episode_index,
                        clip_id=target_episode.clip_id,
                        latent_path=latent_path,
                        latent_shape=tuple(int(value) for value in latents.shape),
                        raw_length_frames=int(target_episode.length_frames),
                        latent_length_frames=int(latents.shape[1]),
                        tasks=target_episode.tasks,
                        target_slot=target.target_slot,
                        encoded_slots=target.source_slots,
                        encoding_mode=target.mode,
                    )
                    newly_encoded_records.append(record)
                encoded.append(record)
                if target.include_in_training_manifest:
                    manifest_records.append(record)
                    manifest_path = manifests_root / f"{_safe_path_part(episode.source_id)}.csv"
                    rows_by_source.setdefault(episode.source_id, []).append(
                        _manifest_row_for_encoded_episode(
                            record,
                            manifest_path=manifest_path,
                        )
                    )
    finally:
        if save_executor is not None:
            save_executor.shutdown(wait=True)
            save_executor = None

    # WHY drain before manifests: a manifest row must never point at a sidecar
    # whose background torch.save failed.
    for latent_path, save_future in save_futures:
        _wait_for_latent_save(latent_path, save_future)
    manifest_paths: dict[str, Path] = {}
    config_patch_path: Path | None = None
    latent_training_config_path: Path | None = None
    if write_manifests:
        if not manifest_records:
            raise ValueError(
                "Mixed-video latent encoding produced no trainable manifest records. "
                "Check source filters, episode filters, camera_names, and latent_encoding_mode."
            )
        if experiment_config is not None:
            _validate_encoded_records_for_backbone(manifest_records, experiment_config=experiment_config)
        manifest_paths = _write_source_manifests(rows_by_source, manifests_root=manifests_root, overwrite=True)
        config_patch_path = _write_latent_source_config_patch(
            data_config,
            encoded_records=manifest_records,
            manifest_paths=manifest_paths,
            output_root=output_root,
        )
        if experiment_config is not None:
            latent_training_config_path = _write_latent_training_config(
                experiment_config,
                data_config=data_config,
                encoded_records=manifest_records,
                manifest_paths=manifest_paths,
                output_root=output_root,
            )
    report: MixedVideoEncodingReport = {
        "output_root": str(output_root),
        "encoded_episodes": _encoded_record_episode_count(encoded),
        "encoded_targets": len(encoded),
        "manifest_encoded_episodes": _encoded_record_episode_count(manifest_records),
        "manifest_encoded_targets": len(manifest_records),
        "newly_encoded_episodes": _encoded_record_episode_count(newly_encoded_records),
        "newly_encoded_targets": len(newly_encoded_records),
        "reused_episodes": _encoded_record_episode_count(reused_records),
        "reused_targets": len(reused_records),
        "selected_episodes": len(all_selected_episodes),
        "shard_episodes": len(episodes),
        "split": selection.split.value,
        "shard_count": int(selection.shard_count),
        "shard_index": int(selection.shard_index),
        "source_ids": sorted(rows_by_source),
        "decode_size_mode": data_config.decode_size_mode.value,
        "decode_fit_mode": data_config.decode_fit_mode.value,
        "manifest_paths": {source_id: str(path) for source_id, path in manifest_paths.items()},
        "config_patch_path": None if config_patch_path is None else str(config_patch_path),
        "latent_training_config_path": (
            None if latent_training_config_path is None else str(latent_training_config_path)
        ),
        "latent_shapes": {
            _encoded_record_report_key(record): list(record.latent_shape)
            for record in encoded
        },
    }
    if write_manifests:
        report_path = output_root / "encode_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _selected_episode_keys(
    data_config: MixedVideoDataConfig,
    catalog: MixedVideoCatalog,
    *,
    split: MixedVideoEncodingSplit,
) -> set[str]:
    if split == MixedVideoEncodingSplit.ALL:
        return {episode.key for episode in catalog.episodes}
    train_keys, val_keys = split_mixed_video_episodes(data_config, catalog)
    return set(train_keys if split == MixedVideoEncodingSplit.TRAIN else val_keys)


def _select_encoder_episodes(
    data_config: MixedVideoDataConfig,
    *,
    selection: MixedVideoEncodingSelection,
    apply_shard: bool,
) -> list[MixedVideoEpisodeRecord]:
    if selection.shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {selection.shard_count}.")
    if selection.shard_index < 0 or selection.shard_index >= selection.shard_count:
        raise ValueError(
            f"shard_index must be in [0, {selection.shard_count}), got {selection.shard_index}."
        )
    catalog = load_mixed_video_catalog(data_config)
    selected_keys = _selected_episode_keys(data_config, catalog, split=selection.split)
    source_filter = set(selection.source_ids)
    episode_filter = {int(index) for index in selection.episode_indices}
    episodes = [
        episode
        for episode in catalog.episodes
        if episode.key in selected_keys
        and (not source_filter or episode.source_id in source_filter)
        and (not episode_filter or int(episode.episode_index) in episode_filter)
        and _episode_has_rgb_streams(episode)
    ]
    if selection.max_episodes is not None:
        episodes = episodes[: int(selection.max_episodes)]
    if apply_shard and selection.shard_count > 1:
        episodes = episodes[int(selection.shard_index) :: int(selection.shard_count)]
    return episodes


def _episode_has_rgb_streams(episode: MixedVideoEpisodeRecord) -> bool:
    return any(
        stream.source_format in {MixedVideoSourceFormat.RGB, MixedVideoSourceFormat.RGB_AND_LATENT}
        for stream in episode.streams
    )


def _encoding_targets_for_episode(
    latents_root: Path,
    episode: MixedVideoEpisodeRecord,
    data_config: MixedVideoDataConfig,
) -> tuple[MixedVideoEncodingTarget, ...]:
    streams_by_slot = {stream.target_slot: stream for stream in episode.streams}
    configured_slots = tuple(slot for slot in data_config.camera_names if slot in streams_by_slot)
    rgb_slots = tuple(
        slot
        for slot in configured_slots
        if streams_by_slot[slot].source_format in {MixedVideoSourceFormat.RGB, MixedVideoSourceFormat.RGB_AND_LATENT}
    )
    targets: list[MixedVideoEncodingTarget] = []
    mode = data_config.latent_encoding_mode
    if mode in {
        MixedVideoLatentEncodingMode.CANONICAL,
        MixedVideoLatentEncodingMode.CANONICAL_AND_PER_VIEW,
    }:
        canonical_slots = tuple(data_config.camera_names)
        missing_canonical_slots = tuple(slot for slot in canonical_slots if slot not in rgb_slots)
        if missing_canonical_slots and mode == MixedVideoLatentEncodingMode.CANONICAL:
            raise KeyError(
                f"Cannot encode canonical mixed-video episode {episode.key!r}: missing RGB streams for "
                f"configured slots {list(missing_canonical_slots)!r}."
            )
        if canonical_slots and not missing_canonical_slots:
            targets.append(
                MixedVideoEncodingTarget(
                    name="canonical",
                    mode=MixedVideoLatentEncodingMode.CANONICAL,
                    target_slot=_encoded_latent_target_slot(data_config),
                    source_slots=canonical_slots,
                    latent_path=_latent_path_for_episode(latents_root, episode),
                    include_in_training_manifest=mode == MixedVideoLatentEncodingMode.CANONICAL,
                )
            )
    if mode in {
        MixedVideoLatentEncodingMode.PER_VIEW,
        MixedVideoLatentEncodingMode.CANONICAL_AND_PER_VIEW,
    }:
        for slot in rgb_slots:
            target_path = _latent_path_for_episode_view(latents_root, episode, slot)
            compatible = ()
            if mode == MixedVideoLatentEncodingMode.PER_VIEW and slot == _encoded_latent_target_slot(data_config):
                compatible = (_latent_path_for_episode(latents_root, episode),)
            targets.append(
                MixedVideoEncodingTarget(
                    name=f"per_view:{slot}",
                    mode=MixedVideoLatentEncodingMode.PER_VIEW,
                    target_slot=slot,
                    source_slots=(slot,),
                    latent_path=target_path,
                    compatible_existing_paths=compatible,
                )
            )
    return tuple(targets)


def _resolve_existing_target_path(target: MixedVideoEncodingTarget) -> Path | None:
    if target.latent_path.exists():
        return target.latent_path
    for path in target.compatible_existing_paths:
        if path.exists():
            return path
    return None


def _data_config_for_encoding_target(
    data_config: MixedVideoDataConfig,
    target: MixedVideoEncodingTarget,
) -> MixedVideoDataConfig:
    if target.mode == MixedVideoLatentEncodingMode.CANONICAL:
        return data_config
    if len(target.source_slots) != 1:
        raise ValueError(f"Per-view encoding target expects exactly one source slot, got {target.source_slots!r}.")
    slot = target.source_slots[0]
    return replace(
        data_config,
        camera_names=(slot,),
        latent_camera_names=(slot,),
        canonical_height=int(data_config.decode_height),
        canonical_width=int(data_config.decode_width),
        view_layout=(
            ViewLayoutConfig(
                source_name=slot,
                canonical_name=slot,
                top=0,
                left=0,
                height=int(data_config.decode_height),
                width=int(data_config.decode_width),
            ),
        ),
    )


def _episode_for_encoding_target(
    episode: MixedVideoEpisodeRecord,
    target: MixedVideoEncodingTarget,
) -> MixedVideoEpisodeRecord:
    selected_streams = tuple(stream for stream in episode.streams if stream.target_slot in set(target.source_slots))
    if not selected_streams:
        raise ValueError(f"Episode {episode.key!r} has no streams for encoding target {target.name!r}.")
    native_length = min(stream.length_frames for stream in selected_streams if stream.length_frames > 0)
    length = min(int(stream.clip.normalized_length_frames) for stream in selected_streams)
    latent_lengths = [
        int(stream.latent_length_frames)
        for stream in selected_streams
        if stream.latent_length_frames is not None and stream.latent_length_frames > 0
    ]
    return replace(
        episode,
        native_length_frames=native_length,
        length_frames=length,
        latent_length_frames=min(latent_lengths) if latent_lengths else None,
        streams=selected_streams,
    )


def _preflight_output_paths(
    episodes: list[MixedVideoEpisodeRecord],
    *,
    data_config: MixedVideoDataConfig,
    output_root: Path,
    latents_root: Path,
    manifests_root: Path,
    overwrite: bool,
    skip_existing: bool = False,
    write_manifests: bool = True,
) -> None:
    latent_paths = [
        target.latent_path
        for episode in episodes
        for target in _encoding_targets_for_episode(latents_root, episode, data_config)
    ]
    source_manifest_paths = [
        manifests_root / f"{_safe_path_part(source_id)}.csv"
        for source_id in sorted({episode.source_id for episode in episodes})
    ]
    duplicate_paths = sorted(
        {
            path
            for paths in (latent_paths, source_manifest_paths)
            for path, count in Counter(paths).items()
            if count > 1
        }
    )
    if duplicate_paths:
        formatted = "\n".join(f"- {path}" for path in duplicate_paths)
        raise FileExistsError(f"Preflight failed: multiple selected episodes would write the same path:\n{formatted}")
    if overwrite:
        return
    checked_paths = [] if skip_existing else list(latent_paths)
    if write_manifests and not skip_existing:
        checked_paths.extend(
            [
                output_root / "encode_report.json",
                output_root / "latent_training_sources.yaml",
                output_root / "latent_training_config.yaml",
            ]
        )
        checked_paths.extend(source_manifest_paths)
    existing_paths = [path for path in checked_paths if path.exists()]
    if existing_paths:
        formatted = "\n".join(f"- {path}" for path in existing_paths)
        raise FileExistsError(
            "Preflight failed: output paths already exist. Pass --overwrite to replace them "
            f"or --skip-existing to resume sidecars:\n{formatted}"
        )


def _encoded_episode_from_existing_sidecar(
    latent_path: Path,
    episode: MixedVideoEpisodeRecord,
    *,
    data_config: MixedVideoDataConfig,
    target: MixedVideoEncodingTarget | None = None,
) -> MixedVideoEncodedEpisode:
    payload = torch.load(latent_path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        latents = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict) and LATENT_KEY in payload:
        latents = payload[LATENT_KEY]
        raw_metadata = payload.get("metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    else:
        raise KeyError(f"Encoded sidecar must contain {LATENT_KEY!r}: {latent_path}")
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4:
        raise ValueError(f"Expected sidecar {latent_path} to contain [C, T, H, W] latents.")
    latent_shape = tuple(int(value) for value in latents.shape)
    _validate_existing_sidecar_metadata(
        latent_path,
        metadata=metadata,
        episode=episode,
        data_config=data_config,
        latent_shape=latent_shape,
        target=target,
    )
    target_slot = _encoded_latent_target_slot(data_config) if target is None else target.target_slot
    encoded_slots = tuple(data_config.camera_names) if target is None else target.source_slots
    encoding_mode = MixedVideoLatentEncodingMode.CANONICAL if target is None else target.mode
    return MixedVideoEncodedEpisode(
        source_id=episode.source_id,
        dataset_id=episode.dataset_id,
        episode_index=int(episode.episode_index),
        clip_id=episode.clip_id,
        latent_path=latent_path,
        latent_shape=latent_shape,
        raw_length_frames=int(episode.length_frames),
        latent_length_frames=int(latents.shape[1]),
        tasks=episode.tasks,
        target_slot=target_slot,
        encoded_slots=encoded_slots,
        encoding_mode=encoding_mode,
    )


def _validate_existing_sidecar_metadata(
    latent_path: Path,
    *,
    metadata: dict[str, Any],
    episode: MixedVideoEpisodeRecord,
    data_config: MixedVideoDataConfig,
    latent_shape: tuple[int, int, int, int],
    target: MixedVideoEncodingTarget | None = None,
) -> None:
    expected_fields = {
        "source_id": str(episode.source_id),
        "dataset_id": str(episode.dataset_id),
        "clip_id": str(episode.clip_id),
        "decode_size_mode": data_config.decode_size_mode.value,
        "decode_fit_mode": data_config.decode_fit_mode.value,
        "decode_allow_upscale": str(bool(data_config.decode_allow_upscale)),
        "target_observation_fps": str(data_config.target_observation_fps),
        "missing_observation_fps": str(float(data_config.missing_observation_fps)),
        "transform_signature_hash": _mixed_video_transform_signature_hash(data_config, episode),
    }
    for field_name, expected in expected_fields.items():
        if field_name not in metadata:
            raise ValueError(
                f"Existing sidecar metadata missing required field {field_name!r} in {latent_path}. "
                "Re-run without --skip-existing or pass --overwrite to regenerate it."
            )
        if str(metadata[field_name]) != expected:
            raise ValueError(
                f"Existing sidecar metadata mismatch for {latent_path}: "
                f"{field_name}={metadata[field_name]!r}, expected {expected!r}."
            )
    int_fields = {
        "episode_index": int(episode.episode_index),
        "raw_length_frames": int(episode.length_frames),
        "native_length_frames": int(episode.native_length_frames),
        "latent_length_frames": int(latent_shape[1]),
    }
    for field_name, expected in {
        "decode_height": int(data_config.decode_height),
        "decode_width": int(data_config.decode_width),
    }.items():
        if field_name in metadata:
            int_fields[field_name] = expected
    for field_name, expected in int_fields.items():
        if field_name not in metadata:
            continue
        try:
            actual = int(metadata[field_name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Existing sidecar metadata field {field_name!r} is not an integer in {latent_path}: "
                f"{metadata[field_name]!r}."
            ) from error
        if actual != expected:
            raise ValueError(
                f"Existing sidecar metadata mismatch for {latent_path}: "
                f"{field_name}={actual}, expected {expected}."
            )
    if "latent_shape" in metadata:
        try:
            metadata_shape = tuple(int(value) for value in metadata["latent_shape"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Existing sidecar metadata field 'latent_shape' is invalid in {latent_path}: "
                f"{metadata['latent_shape']!r}."
            ) from error
        if metadata_shape != latent_shape:
            raise ValueError(
                f"Existing sidecar latent_shape metadata mismatch for {latent_path}: "
                f"{metadata_shape}, actual {latent_shape}."
            )
    if "decode_resize_bins" in metadata:
        expected_bins = _mixed_video_transform_signature(data_config)["decode_resize_bins"]
        if metadata["decode_resize_bins"] != expected_bins:
            raise ValueError(
                f"Existing sidecar decode_resize_bins metadata mismatch for {latent_path}; "
                "re-run without --skip-existing or pass --overwrite to regenerate it."
            )
    if target is not None:
        compatible_legacy_sidecar = latent_path in target.compatible_existing_paths
        optional_expected = {"target_slot": target.target_slot}
        for field_name, expected in optional_expected.items():
            if field_name in metadata and str(metadata[field_name]) != str(expected):
                raise ValueError(
                    f"Existing sidecar metadata mismatch for {latent_path}: "
                    f"{field_name}={metadata[field_name]!r}, expected {expected!r}."
                )
        if "encoding_mode" in metadata:
            allowed_modes = {target.mode.value}
            if compatible_legacy_sidecar:
                allowed_modes.add(MixedVideoLatentEncodingMode.CANONICAL.value)
            if str(metadata["encoding_mode"]) not in allowed_modes:
                raise ValueError(
                    f"Existing sidecar metadata mismatch for {latent_path}: "
                    f"encoding_mode={metadata['encoding_mode']!r}, expected one of {sorted(allowed_modes)!r}."
                )
        if "encoded_slots" in metadata:
            raw_encoded_slots = metadata["encoded_slots"]
            if isinstance(raw_encoded_slots, str):
                encoded_slots = tuple(slot for slot in raw_encoded_slots.split("|") if slot)
            else:
                encoded_slots = tuple(str(value) for value in raw_encoded_slots)
            if encoded_slots != target.source_slots:
                raise ValueError(
                    f"Existing sidecar metadata mismatch for {latent_path}: "
                    f"encoded_slots={encoded_slots!r}, expected {target.source_slots!r}."
                )


def _encode_episode_latents_streaming(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    *,
    canonicalizer: ConfiguredCanonicalVideoPreprocessor,
    assets: MixedVideoLatentEncoder,
    device: torch.device,
    chunk_frames: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode one episode with CPU/GPU overlap double buffering.

    WHY double buffer: CPU video decode and GPU VAE encode are independent
    workloads. By decoding chunk N+1 on CPU while chunk N encodes on GPU,
    we overlap ~60% of decode time with encode time, yielding ~1.5-2x speedup.
    """
    latent_chunks: list[torch.Tensor] = []
    placements_metadata: list[dict[str, Any]] | None = None
    canonical_shape: list[int] | None = None
    raw_chunk_ranges = _streaming_chunk_ranges(int(episode.length_frames), max_chunk_frames=chunk_frames)
    chunk_iterator = enumerate(
        _iter_episode_view_chunks(data_config, episode, raw_chunk_ranges=raw_chunk_ranges)
    )

    # WHY queue maxsize=2: one slot for the chunk being GPU-encoded, one for
    # the prefetched next chunk. Larger queues waste CPU memory on decoded
    # frames without additional GPU overlap benefit.
    prefetch_queue: queue.Queue = queue.Queue(maxsize=2)
    prefetch_error: list[Exception] = []

    def _prefetch_worker():
        """Background thread: decode + canonicalize chunks on CPU, push to queue."""
        try:
            for chunk_index, views in chunk_iterator:
                canonical = canonicalizer(views)
                # WHY keep on CPU here: CUDA init from a background thread fails
                # on nodes with older drivers. H2D transfer happens in main thread.
                prefetch_queue.put((chunk_index, canonical.video, canonical.placements))
        except Exception as exc:
            prefetch_error.append(exc)
        finally:
            prefetch_queue.put(None)  # sentinel

    # WHY daemon=True: if main thread crashes, prefetch thread dies immediately
    worker = threading.Thread(target=_prefetch_worker, daemon=True)
    worker.start()

    while True:
        item = prefetch_queue.get()
        if item is None:
            break
        chunk_index, video_cpu, placements = item
        # WHY .to(device) in main thread: avoids CUDA init in background thread
        # which fails on nodes with old drivers (CUDA 12030)
        video_on_device = video_cpu.to(device=device)
        if placements_metadata is None:
            placements_metadata = [asdict(placement) for placement in placements]
            canonical_shape = list(video_on_device.shape[1:])
        latent_chunks.append(
            assets.encode_video(
                video_on_device,
                placements=placements,
                reset_cache=chunk_index == 0,
            ).detach()
        )

    worker.join()
    if prefetch_error:
        raise prefetch_error[0]

    if not latent_chunks:
        raise ValueError(f"No latent chunks were produced for mixed-video episode {episode.key!r}.")
    metadata = {
        "canonical_shape": canonical_shape,
        "placements": placements_metadata or [],
        "raw_chunk_ranges": [list(item) for item in raw_chunk_ranges],
        "native_length_frames": int(episode.native_length_frames),
        "normalized_length_frames": int(episode.length_frames),
        "target_observation_fps": data_config.target_observation_fps,
        "chunk_frames": int(chunk_frames),
    }
    return torch.cat(latent_chunks, dim=2), metadata


def _iter_episode_view_chunks(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    *,
    raw_chunk_ranges: tuple[tuple[int, int], ...],
) -> Iterator[dict[str, torch.Tensor]]:
    streams_by_slot: dict[str, MixedVideoStreamRecord] = {}
    for stream in sorted(episode.streams, key=lambda item: item.stream_index):
        streams_by_slot.setdefault(stream.target_slot, stream)

    stream_iterators: dict[str, Iterator[torch.Tensor]] = {}
    for camera_name in data_config.camera_names:
        stream = streams_by_slot.get(camera_name)
        if stream is None:
            raise KeyError(
                f"Cannot encode mixed-video episode {episode.key!r}: missing RGB stream for slot {camera_name!r}."
            )
        if stream.source_format not in {MixedVideoSourceFormat.RGB, MixedVideoSourceFormat.RGB_AND_LATENT}:
            raise ValueError(
                f"Cannot encode source={stream.source_id!r}, episode={stream.episode_index}, "
                f"stream={stream.stream_key}: source_format={stream.source_format.value!r} has no RGB input."
            )
        stream_iterators[camera_name] = iter_mixed_video_stream_frame_chunks(
            data_config,
            stream,
            raw_chunk_ranges=raw_chunk_ranges,
        )

    for _ in raw_chunk_ranges:
        yield {
            camera_name: next(stream_iterator)
            for camera_name, stream_iterator in stream_iterators.items()
        }


def _decode_episode_view_chunk(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    *,
    start_frame: int,
    end_frame: int,
) -> dict[str, torch.Tensor]:
    iterator = _iter_episode_view_chunks(
        data_config,
        episode,
        raw_chunk_ranges=((int(start_frame), int(end_frame)),),
    )
    return next(iterator)


def _streaming_chunk_ranges(num_frames: int, *, max_chunk_frames: int) -> tuple[tuple[int, int], ...]:
    if num_frames <= 0:
        raise ValueError(f"Cannot encode an empty video: num_frames={num_frames}.")
    latent_frames = wan_raw_frame_count_to_latent_count(num_frames)
    encoded_raw_frames = 1 + 4 * (latent_frames - 1)
    if max_chunk_frames == 0 or encoded_raw_frames <= max_chunk_frames:
        return ((0, encoded_raw_frames),)
    if max_chunk_frames < 5:
        raise ValueError("max_chunk_frames must be 0 or at least 5.")
    first_capacity = 1 + 4 * ((max_chunk_frames - 1) // 4)
    stream_capacity = 4 * (max_chunk_frames // 4)
    ranges: list[tuple[int, int]] = []
    first_end = min(encoded_raw_frames, first_capacity)
    ranges.append((0, first_end))
    start = first_end
    while start < encoded_raw_frames:
        end = min(encoded_raw_frames, start + stream_capacity)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


# Public planning names make the data-layer boundary explicit while the
# established private names remain available to the legacy command adapter.
plan_mixed_video_episode_encoding_targets = _encoding_targets_for_episode
plan_mixed_video_streaming_chunks = _streaming_chunk_ranges
preflight_mixed_video_encoding_outputs = _preflight_output_paths
resolve_existing_mixed_video_encoding_target = _resolve_existing_target_path
select_mixed_video_encoding_episodes = _select_encoder_episodes


__all__ = [
    "MixedVideoEncodedEpisode",
    "MixedVideoEncodingReport",
    "MixedVideoEncodingSelection",
    "MixedVideoEncodingTarget",
    "MixedVideoLatentEncoder",
    "encode_mixed_video_latent_sources",
    "plan_mixed_video_episode_encoding_targets",
    "plan_mixed_video_streaming_chunks",
    "preflight_mixed_video_encoding_outputs",
    "resolve_existing_mixed_video_encoding_target",
    "resolve_mixed_video_encoding_config",
    "select_mixed_video_encoding_episodes",
]
