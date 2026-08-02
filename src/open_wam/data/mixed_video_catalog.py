"""Stable import facade for mixed-video catalog construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
from pathlib import Path
import random

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

from .mixed_video_catalog_assembly import (
    _merge_tasks,
    _stream_normalized_length_frames,
    _validate_unique_episode_target_slots,
    load_mixed_video_catalog as load_mixed_video_catalog,
)
from .mixed_video_catalog_contracts import (
    MixedVideoCatalog as MixedVideoCatalog,
    MixedVideoEpisodeRecord as MixedVideoEpisodeRecord,
    MixedVideoStreamRecord as MixedVideoStreamRecord,
)
from .mixed_video_catalog_split import (
    _physical_episode_group_key,
    split_mixed_video_episodes as split_mixed_video_episodes,
)
from .mixed_video_manifest import (
    _float_field,
    _int_field,
    _load_source_streams,
    _local_latent_path,
    _local_video_path,
    _optional_int_field,
    _parse_tasks,
    _probe_video_observation_fps,
    _read_manifest_csv,
    _resolve_manifest_path,
    _stream_path_key,
    _string_field,
    _target_slot_for_stream,
)


_COMPATIBILITY_EXPORTS = (
    _float_field,
    _int_field,
    _load_source_streams,
    _local_latent_path,
    _local_video_path,
    _merge_tasks,
    _optional_int_field,
    _parse_tasks,
    _physical_episode_group_key,
    _probe_video_observation_fps,
    _read_manifest_csv,
    _resolve_manifest_path,
    _stream_normalized_length_frames,
    _stream_path_key,
    _string_field,
    _target_slot_for_stream,
    _validate_unique_episode_target_slots,
)


# Preserve the historical wildcard-import surface.
__all__ = [
    "Iterable",
    "MixedVideoCatalog",
    "MixedVideoDataConfig",
    "MixedVideoEpisodeRecord",
    "MixedVideoSourceConfig",
    "MixedVideoSourceFormat",
    "MixedVideoStreamRecord",
    "Path",
    "ResolvedVideoClip",
    "Sequence",
    "annotations",
    "csv",
    "dataclass",
    "defaultdict",
    "imageio",
    "load_mixed_video_catalog",
    "normalized_video_frame_count",
    "random",
    "resolve_video_source_fps",
    "split_mixed_video_episodes",
]
