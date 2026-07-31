"""Dependency-free contracts shared across Open-WAM subsystems."""

from .paths import REPO_ROOT, find_repo_root, resolve_repo_path
from .video import (
    FpsSource,
    ResolvedSourceFps,
    ResolvedVideoClip,
    VideoFrameMapping,
    WAN_TEMPORAL_CHUNK_SIZE,
    normalized_video_frame_count,
    resolve_video_source_fps,
    wan_fully_observed_latent_count,
    wan_raw_frame_count_to_latent_count,
    wan_safe_temporal_frame_count,
)

__all__ = [
    "FpsSource",
    "REPO_ROOT",
    "ResolvedSourceFps",
    "ResolvedVideoClip",
    "VideoFrameMapping",
    "WAN_TEMPORAL_CHUNK_SIZE",
    "find_repo_root",
    "normalized_video_frame_count",
    "resolve_repo_path",
    "resolve_video_source_fps",
    "wan_fully_observed_latent_count",
    "wan_raw_frame_count_to_latent_count",
    "wan_safe_temporal_frame_count",
]
