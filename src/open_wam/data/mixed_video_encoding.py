"""Public orchestration facade for offline mixed-video latent encoding."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import torch

from open_wam.configs import (
    ExperimentConfig,
    MixedVideoDataConfig,
    MixedVideoEncodingSplit,
)
from open_wam.data.mixed_video_encoding_artifacts import (
    LATENT_KEY,
    _encoded_record_episode_count,
    _encoded_record_report_key,
    _manifest_row_for_encoded_episode,
    _safe_path_part,
    _validate_encoded_records_for_backbone,
    _write_latent_source_config_patch,
    _write_latent_training_config,
    _write_source_manifests,
)
from open_wam.data.mixed_video_encoding_contracts import (
    MixedVideoEncodedEpisode,
    MixedVideoEncodingReport,
    MixedVideoEncodingSelection,
    MixedVideoEncodingTarget,
    MixedVideoLatentEncoder,
)
from open_wam.data.mixed_video_encoding_planning import (
    _data_config_for_encoding_target,
    _encoding_targets_for_episode,
    _episode_for_encoding_target,
    _episode_has_rgb_streams as _planning_episode_has_rgb_streams,
    _preflight_output_paths,
    _resolve_existing_target_path,
    _select_encoder_episodes,
    _selected_episode_keys as _planning_selected_episode_keys,
    resolve_mixed_video_encoding_config,
)
from open_wam.data.mixed_video_encoding_runtime import (
    _decode_episode_view_chunk as _runtime_decode_episode_view_chunk,
    _encode_episode_latents_streaming,
    _iter_episode_view_chunks as _runtime_iter_episode_view_chunks,
    _streaming_chunk_ranges,
)
from open_wam.data.mixed_video_encoding_sidecars import (
    _encoded_episode_from_existing_sidecar,
    _mixed_video_transform_signature,
    _mixed_video_transform_signature_hash,
    _validate_existing_sidecar_metadata as _sidecars_validate_metadata,
)
from open_wam.data.raw_video import build_canonical_video_preprocessor

MAX_PENDING_LATENT_SAVE_FUTURES = 1

# Private compatibility names used by the historical checkout command.
EncodedEpisode = MixedVideoEncodedEpisode
EncodingTarget = MixedVideoEncodingTarget
EncoderSelection = MixedVideoEncodingSelection
_episode_has_rgb_streams = _planning_episode_has_rgb_streams
_selected_episode_keys = _planning_selected_episode_keys
_decode_episode_view_chunk = _runtime_decode_episode_view_chunk
_iter_episode_view_chunks = _runtime_iter_episode_view_chunks
_validate_existing_sidecar_metadata = _sidecars_validate_metadata


def _wait_for_latent_save(latent_path: Path, save_future: Future) -> None:
    try:
        save_future.result()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write mixed-video latent sidecar: {latent_path}"
        ) from exc


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
    save_futures.append(
        (latent_path, save_executor.submit(torch.save, payload, latent_path))
    )


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

    all_selected_episodes = _select_encoder_episodes(
        data_config, selection=selection, apply_shard=False
    )
    episodes = _select_encoder_episodes(
        data_config, selection=selection, apply_shard=True
    )
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
                existing_path = (
                    _resolve_existing_target_path(target) if skip_existing else None
                )
                target_data_config = _data_config_for_encoding_target(
                    data_config, target
                )
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
                    canonicalizer = build_canonical_video_preprocessor(
                        target_data_config
                    )
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
                            "native_length_frames": int(
                                target_episode.native_length_frames
                            ),
                            "target_observation_fps": target_data_config.target_observation_fps,
                            "missing_observation_fps": float(
                                target_data_config.missing_observation_fps
                            ),
                            "latent_length_frames": int(latents.shape[1]),
                            "latent_shape": list(latents.shape),
                            "target_slot": target.target_slot,
                            "encoded_slots": list(target.source_slots),
                            "encoding_mode": target.mode.value,
                            "decode_size_mode": target_data_config.decode_size_mode.value,
                            "decode_fit_mode": target_data_config.decode_fit_mode.value,
                            "decode_allow_upscale": bool(
                                target_data_config.decode_allow_upscale
                            ),
                            "decode_height": int(target_data_config.decode_height),
                            "decode_width": int(target_data_config.decode_width),
                            "decode_resize_bins": _mixed_video_transform_signature(
                                target_data_config
                            )["decode_resize_bins"],
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
                        save_executor = ThreadPoolExecutor(
                            max_workers=1, thread_name_prefix="mixed-video-save"
                        )
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
                    manifest_path = (
                        manifests_root / f"{_safe_path_part(episode.source_id)}.csv"
                    )
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
            _validate_encoded_records_for_backbone(
                manifest_records, experiment_config=experiment_config
            )
        manifest_paths = _write_source_manifests(
            rows_by_source, manifests_root=manifests_root, overwrite=True
        )
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
        "manifest_paths": {
            source_id: str(path) for source_id, path in manifest_paths.items()
        },
        "config_patch_path": None
        if config_patch_path is None
        else str(config_patch_path),
        "latent_training_config_path": (
            None
            if latent_training_config_path is None
            else str(latent_training_config_path)
        ),
        "latent_shapes": {
            _encoded_record_report_key(record): list(record.latent_shape)
            for record in encoded
        },
    }
    if write_manifests:
        report_path = output_root / "encode_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    return report


# Public planning names keep the established facade stable while the
# implementation owners remain independently reusable.
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
