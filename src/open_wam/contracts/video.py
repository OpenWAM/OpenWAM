from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


WAN_TEMPORAL_CHUNK_SIZE = 4

FpsSource = Literal["manifest", "container", "fallback"]


@dataclass(frozen=True)
class ViewPlacement:
    """Placement of one resized camera view inside a canonical RGB canvas."""

    source_name: str
    canonical_name: str
    top: int
    left: int
    height: int
    width: int


@dataclass(frozen=True)
class ResolvedSourceFps:
    """Source FPS after applying manifest, container, then fallback precedence."""

    value: float
    source: FpsSource


@dataclass(frozen=True)
class ResolvedVideoClip:
    """Typed identity and timeline for one decoded video clip."""

    clip_id: str
    source_id: str
    dataset_id: str
    episode_index: int
    stream_key: str
    target_slot: str
    path_key: str
    native_length_frames: int
    source_fps: float
    source_fps_source: FpsSource
    target_fps: float | None
    normalized_length_frames: int
    from_timestamp: float | None = None
    to_timestamp: float | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class VideoFrameMapping:
    """Mapping from raw video-frame supervision windows into model frame units."""

    kind: str
    raw_observed_frames: int
    raw_future_frames: int
    raw_total_frames: int
    observed_frames: int
    future_frames: int
    total_frames: int

    @classmethod
    def wan_causal_prefix_suffix(
        cls,
        *,
        raw_observed_frames: int,
        raw_future_frames: int,
        available_frames: int | None = None,
    ) -> VideoFrameMapping:
        raw_observed = int(raw_observed_frames)
        raw_future = int(raw_future_frames)
        raw_total = raw_observed + raw_future
        observed = wan_fully_observed_latent_count(raw_observed)
        total = wan_raw_frame_count_to_latent_count(raw_total)
        if available_frames is not None:
            total = min(total, int(available_frames))
        future = total - observed
        if future <= 0:
            raise ValueError(
                "WAN causal prefix/suffix mapping has no future latent targets, "
                f"raw_observed_frames={raw_observed}, raw_future_frames={raw_future}, "
                f"available_frames={available_frames}, observed_latent_frames={observed}, "
                f"total_latent_frames={total}, future_latent_frames={future}."
            )
        return cls(
            kind="wan_temporal_downsample",
            raw_observed_frames=raw_observed,
            raw_future_frames=raw_future,
            raw_total_frames=raw_total,
            observed_frames=observed,
            future_frames=future,
            total_frames=total,
        )


def wan_safe_temporal_frame_count(num_frames: int, *, cache_initialized: bool) -> int:
    """Return raw frames consumed by Diffusers Wan VAE temporal chunking.

    AutoencoderKLWan encodes a fresh clip as frame 0 plus complete 4-frame
    groups after it. In streaming mode, every emitted latent comes from one
    complete 4-frame group. Incomplete tail frames are not encoded into a
    latent by the reference implementation.
    """

    if num_frames <= 0:
        raise ValueError(f"Wan VAE encoding requires at least one frame, got num_frames={num_frames}.")
    if cache_initialized:
        return WAN_TEMPORAL_CHUNK_SIZE * (num_frames // WAN_TEMPORAL_CHUNK_SIZE)
    return 1 + WAN_TEMPORAL_CHUNK_SIZE * ((num_frames - 1) // WAN_TEMPORAL_CHUNK_SIZE)


def wan_raw_frame_count_to_latent_count(num_frames: int) -> int:
    """Map a fresh Wan VAE raw-frame span to Diffusers' latent-frame count."""

    if num_frames <= 0:
        raise ValueError(f"Wan VAE encoding requires at least one frame, got num_frames={num_frames}.")
    return 1 + (num_frames - 1) // WAN_TEMPORAL_CHUNK_SIZE


def wan_fully_observed_latent_count(raw_observed_frames: int) -> int:
    """Count Wan latent frames whose full raw support lies inside the observed prefix."""

    if raw_observed_frames <= 0:
        raise ValueError(
            f"Wan observed-prefix mapping requires at least one raw frame, got {raw_observed_frames}."
        )
    return 1 + max(0, (raw_observed_frames - 1) // WAN_TEMPORAL_CHUNK_SIZE)


def resolve_video_source_fps(
    observation_fps: float | None,
    *,
    container_fps: float | None = None,
    missing_observation_fps: float = 30.0,
) -> ResolvedSourceFps:
    if observation_fps is not None and float(observation_fps) > 0:
        return ResolvedSourceFps(value=float(observation_fps), source="manifest")
    if container_fps is not None and float(container_fps) > 0:
        return ResolvedSourceFps(value=float(container_fps), source="container")
    if float(missing_observation_fps) <= 0:
        raise ValueError("`missing_observation_fps` must be positive.")
    return ResolvedSourceFps(value=float(missing_observation_fps), source="fallback")


def normalized_video_frame_count(
    length_frames: int,
    *,
    source_fps: float,
    target_fps: float | None,
) -> int:
    length = int(length_frames)
    if length <= 0:
        return 0
    if target_fps is None:
        return length
    source = float(source_fps)
    target = float(target_fps)
    if source <= 0:
        raise ValueError("`source_fps` must be positive.")
    if target <= 0:
        raise ValueError("`target_fps` must be positive or None.")
    return max(1, int(math.ceil((float(length) * target) / source)))
