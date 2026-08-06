"""Persistence metadata and strict resume validation for encoded sidecars."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from open_wam.artifacts import load_tensor_artifact
from open_wam.configs import MixedVideoDataConfig, MixedVideoLatentEncodingMode
from open_wam.configs.enums import serialize_enum_values
from open_wam.data.mixed_video_catalog_contracts import MixedVideoEpisodeRecord
from open_wam.data.mixed_video_encoding_artifacts import (
    LATENT_KEY,
    _encoded_latent_target_slot,
)
from open_wam.data.mixed_video_encoding_contracts import (
    MixedVideoEncodedEpisode,
    MixedVideoEncodingTarget,
)


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
            for stream in sorted(
                episode.streams,
                key=lambda item: (item.target_slot, item.stream_index, item.stream_key),
            )
            if stream.target_slot in data_config.camera_names
        ]
    return signature


def _mixed_video_transform_signature_hash(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord | None = None,
) -> str:
    payload = json.dumps(
        _mixed_video_transform_signature(data_config, episode),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encoded_episode_from_existing_sidecar(
    latent_path: Path,
    episode: MixedVideoEpisodeRecord,
    *,
    data_config: MixedVideoDataConfig,
    target: MixedVideoEncodingTarget | None = None,
) -> MixedVideoEncodedEpisode:
    payload = load_tensor_artifact(latent_path)
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
        raise ValueError(
            f"Expected sidecar {latent_path} to contain [C, T, H, W] latents."
        )
    latent_shape = tuple(int(value) for value in latents.shape)
    _validate_existing_sidecar_metadata(
        latent_path,
        metadata=metadata,
        episode=episode,
        data_config=data_config,
        latent_shape=latent_shape,
        target=target,
    )
    target_slot = (
        _encoded_latent_target_slot(data_config)
        if target is None
        else target.target_slot
    )
    encoded_slots = (
        tuple(data_config.camera_names) if target is None else target.source_slots
    )
    encoding_mode = (
        MixedVideoLatentEncodingMode.CANONICAL if target is None else target.mode
    )
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
        "transform_signature_hash": _mixed_video_transform_signature_hash(
            data_config, episode
        ),
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
        expected_bins = _mixed_video_transform_signature(data_config)[
            "decode_resize_bins"
        ]
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
                encoded_slots = tuple(
                    slot for slot in raw_encoded_slots.split("|") if slot
                )
            else:
                encoded_slots = tuple(str(value) for value in raw_encoded_slots)
            if encoded_slots != target.source_slots:
                raise ValueError(
                    f"Existing sidecar metadata mismatch for {latent_path}: "
                    f"encoded_slots={encoded_slots!r}, expected {target.source_slots!r}."
                )
