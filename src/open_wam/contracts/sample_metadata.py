from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# These wire keys are intentionally stable: existing encoded datasets and
# immutable parity fixtures already contain them. Public Python names describe
# the current generic routing contract without rewriting stored metadata.
DYNAMICS_ROUTING_MODE_METADATA_KEY = "generalist_training_mode_override"
DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY = "generalist_drop_text_conditioning"
DYNAMICS_ROUTING_SOURCE_METADATA_KEY = "generalist_training_source"
DYNAMICS_ROUTING_BUCKET_METADATA_KEY = "generalist_training_bucket"
DYNAMICS_CONDITIONAL_LAYOUT_METADATA_KEY = "generalist_conditional_contract"
DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_METADATA_KEY = "generalist_gjd_chunk_contract"
DYNAMICS_CONDITIONAL_HISTORY_POLICY_METADATA_KEY = (
    "generalist_conditional_history_policy"
)
DYNAMICS_CONDITIONAL_LAYOUT_TARGET_ONLY_T0_PLUS_FUTURE = (
    "target_only_t0_observation_plus_future"
)
DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_T0_SINGLETON = "t0_singleton"
DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY = (
    "previous_boundary_video_only"
)
_TARGET_ALIGNMENT_NEXT_AFTER_CONTEXT = "next_after_context"


@dataclass(frozen=True)
class DynamicsRoutingSampleMetadata:
    """Typed view over optional dynamics-routing metadata."""

    mode_override: str | None = None
    drop_text_conditioning: bool | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ConditionalDynamicsSequenceLayout:
    """Canonical frame and attention layout for conditional dynamics."""

    history_frames: int = field(default=1, init=False)
    loss_frame_start: int = field(default=1, init=False)
    chunk_origin_frame: int = field(default=1, init=False)
    singleton_chunk_frame: int = field(default=0, init=False)
    context_prefix_frames: int = field(default=1, init=False)
    history_policy: str = field(
        default=DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
        init=False,
    )

    def loss_frame_range(self, *, observed_num_frames: int) -> tuple[int, int]:
        if int(observed_num_frames) <= self.loss_frame_start:
            raise ValueError(
                "Conditional dynamics requires one observed t0 frame and at "
                "least one future frame, got "
                f"observed_num_frames={observed_num_frames}."
            )
        return self.loss_frame_start, int(observed_num_frames)

    def contract_metadata(self) -> dict[str, Any]:
        """Serialize the invariant part of the target-only sequence contract."""

        return {
            DYNAMICS_CONDITIONAL_LAYOUT_METADATA_KEY: (
                DYNAMICS_CONDITIONAL_LAYOUT_TARGET_ONLY_T0_PLUS_FUTURE
            ),
            DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_METADATA_KEY: (
                DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_T0_SINGLETON
            ),
            DYNAMICS_CONDITIONAL_HISTORY_POLICY_METADATA_KEY: self.history_policy,
            "history_frames": self.history_frames,
            "loss_frame_start": self.loss_frame_start,
            "latent_loss_frame_start": self.loss_frame_start,
            "action_loss_frame_start": self.loss_frame_start,
            "chunk_origin_frame": self.chunk_origin_frame,
            "target_observation_frame_in_sample": self.singleton_chunk_frame,
            "singleton_chunk_frame": self.singleton_chunk_frame,
            "context_prefix_frames_in_sample": self.context_prefix_frames,
        }

    def to_metadata(self, *, observed_num_frames: int) -> dict[str, Any]:
        """Serialize the complete contract for a materialized sample length."""

        _, loss_frame_end = self.loss_frame_range(
            observed_num_frames=observed_num_frames
        )
        return {
            **self.contract_metadata(),
            "loss_frame_end": loss_frame_end,
            "latent_loss_frame_end": loss_frame_end,
            "action_loss_frame_end": loss_frame_end,
        }


@dataclass(frozen=True)
class SampleConstructionMetadata:
    """Typed view of serialized sample metadata consumed by model runtimes."""

    raw: Mapping[str, Any]
    sampled_chunk_size: int | None = None
    sampled_window_size: int | None = None
    history_frames: int | None = None
    context_prefix_frames_in_sample: int | None = None
    frame_shift: int | None = None
    dynamics_routing: DynamicsRoutingSampleMetadata = DynamicsRoutingSampleMetadata()

    @classmethod
    def from_mapping(
        cls,
        metadata: Mapping[str, Any] | None,
    ) -> SampleConstructionMetadata | None:
        if metadata is None:
            return None
        raw_mode = metadata.get(DYNAMICS_ROUTING_MODE_METADATA_KEY)
        raw_source = metadata.get(DYNAMICS_ROUTING_SOURCE_METADATA_KEY)
        drop_text_conditioning = (
            bool(metadata[DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY])
            if DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY in metadata
            else None
        )
        return cls(
            raw=metadata,
            sampled_chunk_size=_optional_positive_int(
                metadata.get("sampled_chunk_size")
            ),
            sampled_window_size=_optional_positive_int(
                metadata.get("sampled_window_size")
            ),
            history_frames=_optional_int(metadata.get("history_frames")),
            context_prefix_frames_in_sample=_optional_nonnegative_int(
                metadata.get("context_prefix_frames_in_sample")
            ),
            frame_shift=_optional_int(metadata.get("frame_shift")),
            dynamics_routing=DynamicsRoutingSampleMetadata(
                mode_override=(
                    None
                    if raw_mode is None
                    else str(getattr(raw_mode, "value", raw_mode))
                ),
                drop_text_conditioning=drop_text_conditioning,
                source=None if raw_source is None else str(raw_source),
            ),
        )

    @classmethod
    def from_batch_metadata(
        cls,
        metadata: object,
    ) -> SampleConstructionMetadata | None:
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

    def chunk_origin_frame_for(self, *, observed_num_frames: int) -> int:
        """Resolve the shared chunk-coordinate origin for one sample."""

        explicit = self.raw.get("chunk_origin_frame")
        if explicit is not None:
            return int(explicit)
        if self.raw.get("target_alignment") != _TARGET_ALIGNMENT_NEXT_AFTER_CONTEXT:
            return 0
        loss_start, _ = self.frame_range_or_default(
            observed_num_frames=observed_num_frames,
            error_label="sample chunk-origin metadata",
        )
        return int(loss_start)

    def singleton_chunk_frame_for(
        self,
        *,
        observed_num_frames: int,
    ) -> int | None:
        """Resolve an optional singleton frame in the shared chunk layout."""

        if (
            self.raw.get(DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_METADATA_KEY)
            != DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_T0_SINGLETON
        ):
            return None
        raw_frame = self.raw.get(
            "singleton_chunk_frame",
            self.raw.get("target_observation_frame_in_sample"),
        )
        if raw_frame is None:
            return None
        frame = int(raw_frame)
        if frame < 0 or frame >= int(observed_num_frames):
            raise ValueError(
                "Invalid singleton chunk frame, "
                f"got {frame} for observed_num_frames={observed_num_frames}."
            )
        return frame

    @property
    def conditional_history_policy(self) -> str | None:
        value = self.raw.get(DYNAMICS_CONDITIONAL_HISTORY_POLICY_METADATA_KEY)
        return None if value is None else str(value)

    @property
    def is_target_only_conditional_layout(self) -> bool:
        return (
            self.raw.get(DYNAMICS_CONDITIONAL_LAYOUT_METADATA_KEY)
            == DYNAMICS_CONDITIONAL_LAYOUT_TARGET_ONLY_T0_PLUS_FUTURE
        )

    def require_target_only_conditional_layout(
        self,
    ) -> ConditionalDynamicsSequenceLayout:
        """Validate and return the canonical one-t0 dynamics sequence."""

        layout = ConditionalDynamicsSequenceLayout()
        expected = layout.contract_metadata()
        mismatches = {
            key: (self.raw.get(key), value)
            for key, value in expected.items()
            if self.raw.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (expected {expected_value!r})"
                for key, (actual, expected_value) in sorted(mismatches.items())
            )
            raise ValueError(
                "Conditional dynamics training requires the canonical target-only "
                f"t0-plus-future sample contract; {details}."
            )
        return layout


def single_sample_metadata_mapping(metadata: object) -> Mapping[str, Any] | None:
    """Return one sample metadata mapping from a collated metadata object."""

    if (
        isinstance(metadata, tuple)
        and len(metadata) == 1
        and isinstance(metadata[0], Mapping)
    ):
        return metadata[0]
    if (
        isinstance(metadata, list)
        and len(metadata) == 1
        and isinstance(metadata[0], Mapping)
    ):
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


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 0:
        raise ValueError(
            f"Sample metadata integer must be non-negative, got {resolved}."
        )
    return resolved


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
