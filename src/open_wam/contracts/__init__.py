"""Dependency-free contracts shared across Open-WAM subsystems."""

from .paths import REPO_ROOT, find_repo_root, resolve_repo_path
from .sample_metadata import (
    GENERALIST_TRAINING_BUCKET_METADATA_KEY,
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
    GeneralistTrainingSampleMetadata,
    SampleConstructionMetadata,
    single_sample_metadata_mapping,
)
from .video import (
    FpsSource,
    ResolvedSourceFps,
    ResolvedVideoClip,
    ViewPlacement,
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
    "GENERALIST_TRAINING_BUCKET_METADATA_KEY",
    "GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY",
    "GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY",
    "GENERALIST_TRAINING_SOURCE_METADATA_KEY",
    "GeneralistTrainingSampleMetadata",
    "REPO_ROOT",
    "ResolvedSourceFps",
    "ResolvedVideoClip",
    "SampleConstructionMetadata",
    "ViewPlacement",
    "VideoFrameMapping",
    "WAN_TEMPORAL_CHUNK_SIZE",
    "find_repo_root",
    "normalized_video_frame_count",
    "resolve_repo_path",
    "resolve_video_source_fps",
    "single_sample_metadata_mapping",
    "wan_fully_observed_latent_count",
    "wan_raw_frame_count_to_latent_count",
    "wan_safe_temporal_frame_count",
]
