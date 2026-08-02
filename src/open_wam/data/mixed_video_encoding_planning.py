"""Deterministic episode and target planning for mixed-video encoding."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from open_wam.configs import (
    MixedVideoDataConfig,
    MixedVideoDecodeSizeMode,
    MixedVideoEncodingSplit,
    MixedVideoFrameFitMode,
    MixedVideoLatentEncodingMode,
    MixedVideoResizeBinConfig,
    MixedVideoSourceFormat,
    ViewLayoutConfig,
)
from open_wam.data.mixed_video_catalog_assembly import load_mixed_video_catalog
from open_wam.data.mixed_video_catalog_contracts import (
    MixedVideoCatalog,
    MixedVideoEpisodeRecord,
)
from open_wam.data.mixed_video_catalog_split import split_mixed_video_episodes
from open_wam.data.mixed_video_encoding_artifacts import (
    _encoded_latent_target_slot,
    _latent_path_for_episode,
    _latent_path_for_episode_view,
    _safe_path_part,
)
from open_wam.data.mixed_video_encoding_contracts import (
    MixedVideoEncodingSelection,
    MixedVideoEncodingTarget,
)


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
    if (
        updates.get("decode_size_mode", data_config.decode_size_mode)
        == MixedVideoDecodeSizeMode.ASPECT_RATIO_BINS
    ):
        # The encoder itself is single-episode, but the data config validator
        # also protects training-time collation. Normalize these fields so a
        # fixed-size training config can still be reused for offline encoding.
        updates.setdefault("train_batch_size", 1)
        updates.setdefault("val_batch_size", 1)
    return replace(data_config, **updates)


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
        stream.source_format
        in {MixedVideoSourceFormat.RGB, MixedVideoSourceFormat.RGB_AND_LATENT}
        for stream in episode.streams
    )


def _encoding_targets_for_episode(
    latents_root: Path,
    episode: MixedVideoEpisodeRecord,
    data_config: MixedVideoDataConfig,
) -> tuple[MixedVideoEncodingTarget, ...]:
    streams_by_slot = {stream.target_slot: stream for stream in episode.streams}
    configured_slots = tuple(
        slot for slot in data_config.camera_names if slot in streams_by_slot
    )
    rgb_slots = tuple(
        slot
        for slot in configured_slots
        if streams_by_slot[slot].source_format
        in {MixedVideoSourceFormat.RGB, MixedVideoSourceFormat.RGB_AND_LATENT}
    )
    targets: list[MixedVideoEncodingTarget] = []
    mode = data_config.latent_encoding_mode
    if mode in {
        MixedVideoLatentEncodingMode.CANONICAL,
        MixedVideoLatentEncodingMode.CANONICAL_AND_PER_VIEW,
    }:
        canonical_slots = tuple(data_config.camera_names)
        missing_canonical_slots = tuple(
            slot for slot in canonical_slots if slot not in rgb_slots
        )
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
                    include_in_training_manifest=mode
                    == MixedVideoLatentEncodingMode.CANONICAL,
                )
            )
    if mode in {
        MixedVideoLatentEncodingMode.PER_VIEW,
        MixedVideoLatentEncodingMode.CANONICAL_AND_PER_VIEW,
    }:
        for slot in rgb_slots:
            target_path = _latent_path_for_episode_view(latents_root, episode, slot)
            compatible = ()
            if (
                mode == MixedVideoLatentEncodingMode.PER_VIEW
                and slot == _encoded_latent_target_slot(data_config)
            ):
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
        raise ValueError(
            f"Per-view encoding target expects exactly one source slot, got {target.source_slots!r}."
        )
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
    selected_streams = tuple(
        stream
        for stream in episode.streams
        if stream.target_slot in set(target.source_slots)
    )
    if not selected_streams:
        raise ValueError(
            f"Episode {episode.key!r} has no streams for encoding target {target.name!r}."
        )
    native_length = min(
        stream.length_frames for stream in selected_streams if stream.length_frames > 0
    )
    length = min(
        int(stream.clip.normalized_length_frames) for stream in selected_streams
    )
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
        raise FileExistsError(
            f"Preflight failed: multiple selected episodes would write the same path:\n{formatted}"
        )
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


plan_mixed_video_episode_encoding_targets = _encoding_targets_for_episode
preflight_mixed_video_encoding_outputs = _preflight_output_paths
resolve_existing_mixed_video_encoding_target = _resolve_existing_target_path
select_mixed_video_encoding_episodes = _select_encoder_episodes


__all__ = [
    "plan_mixed_video_episode_encoding_targets",
    "preflight_mixed_video_encoding_outputs",
    "resolve_existing_mixed_video_encoding_target",
    "resolve_mixed_video_encoding_config",
    "select_mixed_video_encoding_episodes",
]
