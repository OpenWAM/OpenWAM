from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

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
class CanonicalViewLayout:
    """Validated view placements in one canonical image or latent canvas."""

    SCHEMA_VERSION: ClassVar[str] = "open_wam.canonical_view_layout.v1"

    canvas_height: int
    canvas_width: int
    placements: tuple[ViewPlacement, ...]

    def __post_init__(self) -> None:
        if not _is_metadata_int(self.canvas_height) or not _is_metadata_int(
            self.canvas_width
        ):
            raise TypeError("Canonical view layout canvas dimensions must be integers.")
        if self.canvas_height <= 0 or self.canvas_width <= 0:
            raise ValueError(
                "Canonical view layout requires positive canvas dimensions, "
                f"got {(self.canvas_height, self.canvas_width)}."
            )
        if not self.placements:
            raise ValueError("Canonical view layout requires at least one placement.")
        for placement in self.placements:
            if not isinstance(placement, ViewPlacement):
                raise TypeError(
                    "Canonical view layout placements must be ViewPlacement values."
                )
            if not isinstance(placement.source_name, str) or not placement.source_name:
                raise ValueError("Canonical view layout requires non-empty source names.")
            if (
                not isinstance(placement.canonical_name, str)
                or not placement.canonical_name
            ):
                raise ValueError(
                    "Canonical view layout requires non-empty canonical names."
                )
            if not all(
                _is_metadata_int(value)
                for value in (
                    placement.top,
                    placement.left,
                    placement.height,
                    placement.width,
                )
            ):
                raise TypeError("Canonical view placement coordinates must be integers.")
        source_names = tuple(placement.source_name for placement in self.placements)
        canonical_names = tuple(
            placement.canonical_name for placement in self.placements
        )
        if len(set(source_names)) != len(source_names):
            raise ValueError("Canonical view layout source names must be unique.")
        if len(set(canonical_names)) != len(canonical_names):
            raise ValueError("Canonical view layout canonical names must be unique.")
        for placement in self.placements:
            if (
                placement.top < 0
                or placement.left < 0
                or placement.height <= 0
                or placement.width <= 0
            ):
                raise ValueError(f"Invalid canonical view placement: {placement}.")
            if (
                placement.top + placement.height > self.canvas_height
                or placement.left + placement.width > self.canvas_width
            ):
                raise ValueError(
                    f"Canonical view placement {placement.source_name!r} exceeds "
                    f"canvas {(self.canvas_height, self.canvas_width)}."
                )
        for index, placement in enumerate(self.placements):
            for other in self.placements[index + 1 :]:
                if _placements_overlap(placement, other):
                    raise ValueError(
                        "Canonical view placements overlap: "
                        f"{placement.source_name!r} and {other.source_name!r}."
                    )

    @classmethod
    def from_metadata(cls, raw: Mapping[str, Any]) -> CanonicalViewLayout:
        """Parse the one canonical layout metadata representation."""

        schema_version = raw.get("schema_version")
        if schema_version != cls.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported canonical view layout schema: "
                f"expected {cls.SCHEMA_VERSION!r}, got {schema_version!r}."
            )
        raw_placements = raw.get("placements")
        if not isinstance(raw_placements, list) or not raw_placements:
            raise ValueError(
                "Canonical latent layout requires a non-empty `placements` list."
            )
        canvas_height = _required_metadata_int(raw, "canvas_height", scope="layout")
        canvas_width = _required_metadata_int(raw, "canvas_width", scope="layout")
        placements: list[ViewPlacement] = []
        for index, item in enumerate(raw_placements):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"Canonical latent layout placement #{index} must be a mapping."
                )
            source_name = item.get("source_name")
            canonical_name = item.get("canonical_name")
            if not isinstance(source_name, str) or not source_name:
                raise ValueError(
                    f"Canonical latent layout placement #{index} requires a source name."
                )
            if not isinstance(canonical_name, str) or not canonical_name:
                raise ValueError(
                    f"Canonical latent layout placement #{index} requires a canonical name."
                )
            placements.append(
                ViewPlacement(
                    source_name=source_name,
                    canonical_name=canonical_name,
                    top=_required_metadata_int(item, "top", scope=f"placement #{index}"),
                    left=_required_metadata_int(
                        item,
                        "left",
                        scope=f"placement #{index}",
                    ),
                    height=_required_metadata_int(
                        item,
                        "height",
                        scope=f"placement #{index}",
                    ),
                    width=_required_metadata_int(
                        item,
                        "width",
                        scope=f"placement #{index}",
                    ),
                )
            )
        return cls(
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            placements=tuple(placements),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "canvas_height": self.canvas_height,
            "canvas_width": self.canvas_width,
            "placements": [
                {
                    "source_name": placement.source_name,
                    "canonical_name": placement.canonical_name,
                    "top": placement.top,
                    "left": placement.left,
                    "height": placement.height,
                    "width": placement.width,
                }
                for placement in self.placements
            ],
        }


def _required_metadata_int(
    raw: Mapping[str, Any],
    key: str,
    *,
    scope: str,
) -> int:
    value = raw.get(key)
    if not _is_metadata_int(value):
        raise TypeError(
            f"Canonical latent {scope} requires integer `{key}`, got {value!r}."
        )
    assert isinstance(value, int)
    return value


def _is_metadata_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _placements_overlap(left: ViewPlacement, right: ViewPlacement) -> bool:
    return not (
        left.left + left.width <= right.left
        or right.left + right.width <= left.left
        or left.top + left.height <= right.top
        or right.top + right.height <= left.top
    )


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
    return max(1, math.ceil((float(length) * target) / source))
