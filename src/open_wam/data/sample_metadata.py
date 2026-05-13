from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from open_wam.configs.variant_semantics import (
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
)


@dataclass(frozen=True)
class GeneralistTrainingSampleMetadata:
    """Typed view over optional generalist-denoising metadata."""

    mode_override: Any | None = None
    drop_text_conditioning: bool | None = None
    source: str | None = None


@dataclass(frozen=True)
class SampleConstructionMetadata:
    """Typed adapter for per-sample construction metadata.

    Dataset samples still carry plain dict metadata for serialization and
    compatibility. Runtime code should use this adapter rather than repeating
    string-key parsing across policy variants.
    """

    raw: Mapping[str, Any]
    sampled_chunk_size: int | None = None
    sampled_window_size: int | None = None
    history_frames: int | None = None
    frame_shift: int | None = None
    generalist: GeneralistTrainingSampleMetadata = GeneralistTrainingSampleMetadata()

    @classmethod
    def from_mapping(cls, metadata: Mapping[str, Any] | None) -> "SampleConstructionMetadata | None":
        if metadata is None:
            return None
        raw_source = metadata.get(GENERALIST_TRAINING_SOURCE_METADATA_KEY)
        drop_text_conditioning = (
            bool(metadata[GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY])
            if GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY in metadata
            else None
        )
        return cls(
            raw=metadata,
            sampled_chunk_size=_optional_positive_int(metadata.get("sampled_chunk_size")),
            sampled_window_size=_optional_positive_int(metadata.get("sampled_window_size")),
            history_frames=_optional_int(metadata.get("history_frames")),
            frame_shift=_optional_int(metadata.get("frame_shift")),
            generalist=GeneralistTrainingSampleMetadata(
                mode_override=metadata.get(GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY),
                drop_text_conditioning=drop_text_conditioning,
                source=None if raw_source is None else str(raw_source),
            ),
        )

    @classmethod
    def from_batch_metadata(cls, metadata: object) -> "SampleConstructionMetadata | None":
        mapping = single_sample_metadata_mapping(metadata)
        return cls.from_mapping(mapping)

    def optional_frame_range(
        self,
        *,
        observed_num_frames: int,
        start_key: str = "loss_frame_start",
        end_key: str = "loss_frame_end",
        fallback_to_generic: bool = True,
        error_label: str = "train loss-frame metadata",
    ) -> tuple[int, int] | None:
        metadata_start = self.raw.get(start_key)
        metadata_end = self.raw.get(end_key)
        if (
            metadata_start is None
            and metadata_end is None
            and fallback_to_generic
            and (start_key, end_key) != ("loss_frame_start", "loss_frame_end")
        ):
            metadata_start = self.raw.get("loss_frame_start")
            metadata_end = self.raw.get("loss_frame_end")
        if metadata_start is None and metadata_end is None:
            return None
        start = 0 if metadata_start is None else int(metadata_start)
        end = int(observed_num_frames) if metadata_end is None else int(metadata_end)
        _validate_frame_range(
            start=start,
            end=end,
            observed_num_frames=observed_num_frames,
            error_label=error_label,
            start_key=start_key,
            end_key=end_key,
        )
        return start, end

    def frame_range_or_default(
        self,
        *,
        observed_num_frames: int,
        start_key: str = "loss_frame_start",
        end_key: str = "loss_frame_end",
        default_start: int = 0,
        default_end: int | None = None,
        fallback_to_generic: bool = True,
        error_label: str = "train loss-frame metadata",
    ) -> tuple[int, int]:
        frame_range = self.optional_frame_range(
            observed_num_frames=observed_num_frames,
            start_key=start_key,
            end_key=end_key,
            fallback_to_generic=fallback_to_generic,
            error_label=error_label,
        )
        if frame_range is not None:
            return frame_range
        end = int(observed_num_frames) if default_end is None else int(default_end)
        start = int(default_start)
        _validate_frame_range(
            start=start,
            end=end,
            observed_num_frames=observed_num_frames,
            error_label=error_label,
            start_key=start_key,
            end_key=end_key,
        )
        return start, end

    def sampled_chunk_size_for(self, observed_num_frames: int) -> int | None:
        if self.sampled_chunk_size is None:
            return None
        return min(self.sampled_chunk_size, int(observed_num_frames))


def single_sample_metadata_mapping(metadata: object) -> Mapping[str, Any] | None:
    """Return one sample metadata mapping from a collated metadata object."""

    if isinstance(metadata, tuple) and len(metadata) == 1 and isinstance(metadata[0], Mapping):
        return metadata[0]
    if isinstance(metadata, list) and len(metadata) == 1 and isinstance(metadata[0], Mapping):
        return metadata[0]
    if isinstance(metadata, Mapping):
        return metadata
    return None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    return resolved if resolved > 0 else None


def _validate_frame_range(
    *,
    start: int,
    end: int,
    observed_num_frames: int,
    error_label: str,
    start_key: str,
    end_key: str,
) -> None:
    if start < 0 or end < start or end > int(observed_num_frames):
        raise ValueError(
            f"Invalid {error_label}, keys=({start_key!r}, {end_key!r}), "
            f"got start={start}, end={end}, observed_num_frames={observed_num_frames}."
        )
