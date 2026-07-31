"""Compatibility exports for shared video timeline contracts.

New code should import these contracts from :mod:`open_wam.contracts`.
"""

from open_wam.contracts.video import (
    FpsSource,
    ResolvedSourceFps,
    ResolvedVideoClip,
    VideoFrameMapping,
    normalized_video_frame_count,
    resolve_video_source_fps,
)

__all__ = [
    "FpsSource",
    "ResolvedSourceFps",
    "ResolvedVideoClip",
    "VideoFrameMapping",
    "normalized_video_frame_count",
    "resolve_video_source_fps",
]
