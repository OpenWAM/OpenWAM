"""Deterministic artifacts produced by mixed-video latent encoding.

This module owns filesystem naming, manifests, generated latent-training
configuration, and backbone compatibility checks. The encoding runtime remains
in :mod:`open_wam.data.mixed_video_encoding` and supplies completed records to
these helpers.
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from open_wam.configs import (
    CausalPrefixSuffixBucketConfig,
    ExperimentConfig,
    MixedVideoDataConfig,
    MixedVideoLatentEncodingMode,
)
from open_wam.configs.enums import serialize_enum_values
from open_wam.contracts import VideoFrameMapping, wan_raw_frame_count_to_latent_count

if TYPE_CHECKING:
    from open_wam.data.mixed_video_catalog import MixedVideoEpisodeRecord
    from open_wam.data.mixed_video_encoding import MixedVideoEncodedEpisode


LATENT_KEY = "video_latents"


def _latent_path_for_episode(
    latents_root: Path,
    episode: MixedVideoEpisodeRecord,
) -> Path:
    source = _safe_path_part(episode.source_id)
    dataset = _safe_path_part(episode.dataset_id)
    suffix = "" if episode.clip_id == "default" else f"_{_safe_path_part(episode.clip_id)}"
    return latents_root / source / dataset / f"episode_{int(episode.episode_index):06d}{suffix}.pt"


def _latent_path_for_episode_view(
    latents_root: Path,
    episode: MixedVideoEpisodeRecord,
    target_slot: str,
) -> Path:
    source = _safe_path_part(episode.source_id)
    dataset = _safe_path_part(episode.dataset_id)
    suffix = "" if episode.clip_id == "default" else f"_{_safe_path_part(episode.clip_id)}"
    slot_suffix = _safe_path_part(target_slot)
    return latents_root / source / dataset / f"episode_{int(episode.episode_index):06d}{suffix}__{slot_suffix}.pt"


def _manifest_row_for_encoded_episode(
    record: MixedVideoEncodedEpisode,
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "source_id": record.source_id,
        "dataset_id": record.dataset_id,
        "episode_index": int(record.episode_index),
        "clip_id": record.clip_id,
        "stream_index": 0,
        "stream_key": _encoded_stream_key(record),
        "target_slot_key": record.target_slot,
        "latent_path": os.path.relpath(record.latent_path, manifest_path.parent),
        "latent_length_frames": int(record.latent_length_frames),
        "length_frames": int(record.latent_length_frames),
        "raw_length_frames": int(record.raw_length_frames),
        "latent_key": LATENT_KEY,
        "width": int(record.latent_shape[-1]),
        "height": int(record.latent_shape[-2]),
        "channels": int(record.latent_shape[0]),
        "encoding_mode": record.encoding_mode,
        "encoded_slots": "|".join(record.encoded_slots),
        "tasks": "|".join(record.tasks),
    }


def _encoded_record_report_key(record: MixedVideoEncodedEpisode) -> str:
    return (
        f"{record.source_id}:{record.dataset_id}:{record.episode_index}:"
        f"{record.clip_id}:{record.encoding_mode}:{record.target_slot}"
    )


def _encoded_record_episode_count(records: Sequence[MixedVideoEncodedEpisode]) -> int:
    return len(
        {
            (record.source_id, record.dataset_id, int(record.episode_index), record.clip_id)
            for record in records
        }
    )


def _encoded_stream_key(record: MixedVideoEncodedEpisode) -> str:
    if record.encoding_mode == MixedVideoLatentEncodingMode.PER_VIEW and record.encoded_slots:
        return f"encoded_video_latents:{record.encoded_slots[0]}"
    return "encoded_video_latents"


def _yaml_inline_str_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(json.dumps(str(value)) for value in values) + "]"


def _encoded_latent_target_slot(data_config: MixedVideoDataConfig) -> str:
    if not data_config.camera_names:
        raise ValueError("Mixed-video latent encoding requires at least one configured camera slot.")
    return data_config.camera_names[0]


def _encoded_latent_target_slots(
    data_config: MixedVideoDataConfig,
    encoded_records: list[MixedVideoEncodedEpisode],
) -> tuple[str, ...]:
    slots = tuple(dict.fromkeys(record.target_slot for record in encoded_records))
    if slots:
        return slots
    return (_encoded_latent_target_slot(data_config),)


def _encoded_records_latent_encoding_mode(
    encoded_records: list[MixedVideoEncodedEpisode],
    *,
    fallback: MixedVideoLatentEncodingMode,
) -> str:
    modes = {record.encoding_mode for record in encoded_records}
    if modes == {MixedVideoLatentEncodingMode.CANONICAL}:
        return MixedVideoLatentEncodingMode.CANONICAL.value
    if modes == {MixedVideoLatentEncodingMode.PER_VIEW}:
        return MixedVideoLatentEncodingMode.PER_VIEW.value
    if modes == {
        MixedVideoLatentEncodingMode.CANONICAL,
        MixedVideoLatentEncodingMode.PER_VIEW,
    }:
        return MixedVideoLatentEncodingMode.CANONICAL_AND_PER_VIEW.value
    return fallback.value


def _encoded_latent_view_layouts(
    data_config: MixedVideoDataConfig,
    target_slots: Sequence[str],
) -> list[dict[str, Any]]:
    layouts: list[dict[str, Any]] = []
    for slot in target_slots:
        layouts.append(
            {
                "source_name": slot,
                "canonical_name": slot,
                "top": 0,
                "left": 0,
                "height": int(data_config.canonical_height),
                "width": int(data_config.canonical_width),
            }
        )
    return layouts


def _write_source_manifests(
    rows_by_source: dict[str, list[dict[str, Any]]],
    *,
    manifests_root: Path,
    overwrite: bool,
) -> dict[str, Path]:
    manifest_paths: dict[str, Path] = {}
    fieldnames = (
        "source_id",
        "dataset_id",
        "episode_index",
        "clip_id",
        "stream_index",
        "stream_key",
        "target_slot_key",
        "latent_path",
        "latent_length_frames",
        "length_frames",
        "raw_length_frames",
        "latent_key",
        "width",
        "height",
        "channels",
        "encoding_mode",
        "encoded_slots",
        "tasks",
    )
    for source_id, rows in sorted(rows_by_source.items()):
        manifest_path = manifests_root / f"{_safe_path_part(source_id)}.csv"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"Manifest already exists: {manifest_path}. Pass --overwrite to replace it.")
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        manifest_paths[source_id] = manifest_path
    return manifest_paths


def _write_latent_source_config_patch(
    data_config: MixedVideoDataConfig,
    *,
    encoded_records: list[MixedVideoEncodedEpisode],
    manifest_paths: dict[str, Path],
    output_root: Path,
) -> Path:
    source_by_id = {source.source_id: source for source in data_config.video_sources}
    latent_num_frames = wan_raw_frame_count_to_latent_count(int(data_config.num_frames))
    latent_buckets = _latent_causal_bucket_specs(data_config)
    target_slots = _encoded_latent_target_slots(data_config, encoded_records)
    latent_view_layouts = _encoded_latent_view_layouts(data_config, target_slots)
    manifest_encoding_mode = _encoded_records_latent_encoding_mode(
        encoded_records,
        fallback=data_config.latent_encoding_mode,
    )
    lines = [
        "# Include this block in a mixed-video latent-first training config.",
        "# The generated manifests use latent-frame units for length_frames.",
        "# These num_frames/bucket values are converted from the RGB/WAN raw-frame config.",
        "data:",
        f"  camera_names: {_yaml_inline_str_list(target_slots)}",
        f"  latent_camera_names: {_yaml_inline_str_list(target_slots)}",
        f"  latent_encoding_mode: {manifest_encoding_mode}",
        f"  canonical_height: {int(data_config.canonical_height)}",
        f"  canonical_width: {int(data_config.canonical_width)}",
        "  view_layout:",
    ]
    for latent_view_layout in latent_view_layouts:
        lines.extend(
            [
                f"    - source_name: {json.dumps(latent_view_layout['source_name'])}",
                f"      canonical_name: {json.dumps(latent_view_layout['canonical_name'])}",
                f"      top: {latent_view_layout['top']}",
                f"      left: {latent_view_layout['left']}",
                f"      height: {latent_view_layout['height']}",
                f"      width: {latent_view_layout['width']}",
            ]
        )
    if data_config.latent_view_combinations:
        lines.append("  latent_view_combinations:")
        for combination in data_config.latent_view_combinations:
            lines.extend(
                [
                    f"    - name: {json.dumps(combination.name)}",
                    f"      slots: {_yaml_inline_str_list(combination.slots)}",
                    f"      sampling_weight: {float(combination.sampling_weight)}",
                ]
            )
            if combination.source_ids:
                lines.append(f"      source_ids: {_yaml_inline_str_list(combination.source_ids)}")
            if not combination.enabled:
                lines.append("      enabled: false")
    lines.extend(
        [
            f"  num_frames: {latent_num_frames}",
            "  frame_stride: 1",
            "  sample_stride: 1",
            "  video_sources:",
        ]
    )
    for source_id, manifest_path in sorted(manifest_paths.items()):
        source = source_by_id.get(source_id)
        sampling_weight = None if source is None else source.sampling_weight
        lines.extend(
            [
                f"    - source_id: {source_id}",
                f"      manifest_csv: {manifest_path}",
                "      source_format: latent",
                f"      latent_key: {LATENT_KEY}",
            ]
        )
        if sampling_weight is not None:
            lines.append(f"      sampling_weight: {float(sampling_weight)}")
    lines.extend(
        [
            "  sample_construction:",
            "    mode: causal_prefix_suffix",
            f"    num_frames: {latent_num_frames}",
            "    action_horizon: 0",
            "    state_horizon: 0",
            "    frame_stride: 1",
            "    causal_prefix_suffix_buckets:",
        ]
    )
    for bucket in latent_buckets:
        lines.extend(
            [
                f"      # raw {bucket['raw_observed_frames']} + {bucket['raw_future_frames']} "
                f"frames -> latent {bucket['observed_frames']} + {bucket['future_frames']} frames",
                f"      - observed_frames: {bucket['observed_frames']}",
                f"        future_frames: {bucket['future_frames']}",
            ]
        )
    lines.extend(
        [
            "trainer:",
            "  batch_adapter: latents",
            "",
        ]
    )
    patch_path = output_root / "latent_training_sources.yaml"
    patch_path.write_text("\n".join(lines), encoding="utf-8")
    return patch_path


def _write_latent_training_config(
    experiment_config: ExperimentConfig,
    *,
    data_config: MixedVideoDataConfig,
    encoded_records: list[MixedVideoEncodedEpisode],
    manifest_paths: dict[str, Path],
    output_root: Path,
) -> Path:
    import yaml

    payload = serialize_enum_values(asdict(experiment_config))
    payload["name"] = f"{payload.get('name', 'mixed_video')}_latent_encoded"

    data_payload = dict(payload.get("data", {}))
    latent_num_frames = wan_raw_frame_count_to_latent_count(int(data_config.num_frames))
    target_slots = _encoded_latent_target_slots(data_config, encoded_records)
    manifest_encoding_mode = _encoded_records_latent_encoding_mode(
        encoded_records,
        fallback=data_config.latent_encoding_mode,
    )
    data_payload.update(
        {
            "dataset_name": data_config.dataset_name,
            "dataset_type": data_config.dataset_type,
            "camera_names": list(target_slots),
            "latent_camera_names": list(target_slots),
            "latent_encoding_mode": manifest_encoding_mode,
            "latent_view_combinations": [
                {
                    "name": combination.name,
                    "slots": list(combination.slots),
                    "sampling_weight": float(combination.sampling_weight),
                    **({"source_ids": list(combination.source_ids)} if combination.source_ids else {}),
                    **({"enabled": False} if not combination.enabled else {}),
                }
                for combination in data_config.latent_view_combinations
            ],
            "canonical_height": int(data_config.canonical_height),
            "canonical_width": int(data_config.canonical_width),
            "view_layout": _encoded_latent_view_layouts(data_config, target_slots),
            "num_frames": latent_num_frames,
            "frame_stride": 1,
            "sample_stride": 1,
            "train_batch_size": 1,
            "val_batch_size": 1,
            "video_sources": _latent_video_source_config_entries(
                data_config,
                manifest_paths=manifest_paths,
            ),
        }
    )
    sample_construction = dict(data_payload.get("sample_construction", {}))
    sample_construction.update(
        {
            "mode": "causal_prefix_suffix",
            "num_frames": latent_num_frames,
            "action_horizon": 0,
            "state_horizon": 0,
            "frame_stride": 1,
            "causal_prefix_suffix_buckets": [
                {
                    "observed_frames": int(bucket["observed_frames"]),
                    "future_frames": int(bucket["future_frames"]),
                }
                for bucket in _latent_causal_bucket_specs(data_config)
            ],
        }
    )
    data_payload["sample_construction"] = sample_construction
    payload["data"] = data_payload

    trainer_payload = dict(payload.get("trainer", {}))
    trainer_payload["batch_adapter"] = "latents"
    payload["trainer"] = trainer_payload

    backbone_payload = dict(payload.get("backbone", {}))
    # Latent-first training enters after VAE encoding, so loading the VAE again
    # is unnecessary and can fail on machines that only have precomputed sidecars.
    backbone_payload["load_wan_vae_frontend"] = False
    payload["backbone"] = backbone_payload

    config_path = output_root / "latent_training_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _latent_video_source_config_entries(
    data_config: MixedVideoDataConfig,
    *,
    manifest_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    source_by_id = {source.source_id: source for source in data_config.video_sources}
    entries: list[dict[str, Any]] = []
    for source_id, manifest_path in sorted(manifest_paths.items()):
        source = source_by_id.get(source_id)
        entry: dict[str, Any] = {
            "source_id": source_id,
            "manifest_csv": str(manifest_path),
            "source_format": "latent",
            "latent_key": LATENT_KEY,
        }
        if source is not None and source.sampling_weight is not None:
            entry["sampling_weight"] = float(source.sampling_weight)
        entries.append(entry)
    return entries


def _validate_encoded_records_for_backbone(
    encoded: list[MixedVideoEncodedEpisode],
    *,
    experiment_config: ExperimentConfig,
) -> None:
    patch_t = int(getattr(experiment_config.backbone, "patch_size_t", 1))
    patch_h = int(getattr(experiment_config.backbone, "patch_size_h", 1))
    patch_w = int(getattr(experiment_config.backbone, "patch_size_w", 1))
    errors: list[str] = []
    for record in encoded:
        _, latent_frames, latent_height, latent_width = record.latent_shape
        if (
            int(latent_frames) % patch_t != 0
            or int(latent_height) % patch_h != 0
            or int(latent_width) % patch_w != 0
        ):
            errors.append(
                f"{record.source_id}:{record.dataset_id}:{record.episode_index}:{record.clip_id} "
                f"shape={record.latent_shape} patch={(patch_t, patch_h, patch_w)}"
            )
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(
            "Encoded mixed-video latents are not compatible with the configured shared-transformer patch size. "
            "Re-encode with patch-compatible resize bins, or change backbone.patch_size_* before training:\n"
            f"{formatted}"
        )


def _latent_causal_bucket_specs(data_config: MixedVideoDataConfig) -> list[dict[str, int]]:
    raw_buckets = data_config.sample_construction.causal_prefix_suffix_buckets
    if not raw_buckets:
        raw_observed = max(1, int(data_config.num_frames) // 2)
        raw_buckets = (
            CausalPrefixSuffixBucketConfig(
                observed_frames=raw_observed,
                future_frames=int(data_config.num_frames) - raw_observed,
            ),
        )
    specs: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for bucket in raw_buckets:
        raw_observed = int(bucket.observed_frames)
        raw_future = int(bucket.future_frames)
        try:
            mapping = VideoFrameMapping.wan_causal_prefix_suffix(
                raw_observed_frames=raw_observed,
                raw_future_frames=raw_future,
            )
        except ValueError:
            continue
        observed = mapping.observed_frames
        future = mapping.future_frames
        key = (observed, future)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "raw_observed_frames": raw_observed,
                "raw_future_frames": raw_future,
                "observed_frames": observed,
                "future_frames": future,
            }
        )
    if not specs:
        raise ValueError("No causal prefix/suffix buckets remain valid after raw-to-WAN-latent conversion.")
    return specs


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "unknown"


__all__: list[str] = []
