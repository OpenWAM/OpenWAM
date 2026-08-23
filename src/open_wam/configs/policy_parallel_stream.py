"""Typed parallel-stream policy configuration and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import (
    ActionDecoderName,
    ActionNormMethod,
    AttachSite,
    ParallelActionAttentionScope,
    ParallelActionConditionSource,
    ParallelCacheMode,
    ParallelMaskMode,
    ParallelRuntimeMode,
    ParallelSequenceComponent,
    PolicyVariantName,
    TemporalPositionMode,
    VideoActionProgram,
    coerce_fields,
)
from .policy_video_action import VideoActionPolicyConfig

_PARALLEL_RUNTIME_MODE_BY_PROGRAM = {
    VideoActionProgram.VIDEO_THEN_ACTION: ParallelRuntimeMode.LINGBOT_EXACT,
    VideoActionProgram.ACTION_THEN_VIDEO: ParallelRuntimeMode.LINGBOT_EXACT,
    VideoActionProgram.DECOUPLED_SAME_STEP: ParallelRuntimeMode.LINGBOT_EXACT,
    VideoActionProgram.JOINT: ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
    VideoActionProgram.VIDEO_NOISY_TO_ACTION: (
        ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    ),
    VideoActionProgram.ACTION_NOISY_TO_VIDEO: (
        ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    ),
    VideoActionProgram.GENERALIST_JOINT_DENOISING: (
        ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    ),
    VideoActionProgram.FORWARD_DYNAMICS: (
        ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    ),
    VideoActionProgram.INVERSE_DYNAMICS: (
        ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    ),
}


def parallel_runtime_mode_for_program(
    program: VideoActionProgram | str,
) -> ParallelRuntimeMode:
    """Return the Parallel Stream backend required by one semantic program."""

    resolved_program = VideoActionProgram(program)
    try:
        return _PARALLEL_RUNTIME_MODE_BY_PROGRAM[resolved_program]
    except KeyError as exc:
        raise ValueError(
            f"Parallel Stream does not implement program {resolved_program.value!r}."
        ) from exc


def parallel_video_conditioning_for_program(
    program: VideoActionProgram | str,
) -> bool:
    """Return whether the Parallel Stream backend consumes action conditioning."""

    return (
        parallel_runtime_mode_for_program(program)
        == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    )


@dataclass(frozen=True)
class ParallelStreamPolicyConfig(VideoActionPolicyConfig):
    name: PolicyVariantName = PolicyVariantName.PARALLEL_STREAM
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.WITHIN_VISUAL_CORE
    reference_profile: str | None = None
    frame_chunk_size: int = 2
    action_per_frame: int = 1
    attn_window: int = 8
    sequence_order: tuple[ParallelSequenceComponent, ...] = field(
        default_factory=lambda: (
            ParallelSequenceComponent.VIDEO_NOISY,
            ParallelSequenceComponent.VIDEO_CONDITION,
            ParallelSequenceComponent.ACTION_NOISY,
            ParallelSequenceComponent.ACTION_CONDITION,
        )
    )
    mask_mode: ParallelMaskMode = ParallelMaskMode.LINGBOT_CHUNKED
    cache_mode: ParallelCacheMode = ParallelCacheMode.METADATA_ONLY
    video_action_condition_source: ParallelActionConditionSource = (
        ParallelActionConditionSource.NOISY_ACTION
    )
    video_action_attention_scope: ParallelActionAttentionScope = (
        ParallelActionAttentionScope.BLOCK_LOCAL
    )
    temporal_position_mode: TemporalPositionMode = TemporalPositionMode.GLOBAL_SHIFTED
    used_action_channel_ids: tuple[int, ...] = field(default_factory=tuple)
    inverse_used_action_channel_ids: tuple[int, ...] = field(default_factory=tuple)
    action_norm_method: ActionNormMethod = ActionNormMethod.NONE
    norm_q01: tuple[float, ...] = field(default_factory=tuple)
    norm_q99: tuple[float, ...] = field(default_factory=tuple)

    @property
    def default_action_decoder(self) -> ActionDecoderName:
        return ActionDecoderName.PARALLEL_STREAM

    @property
    def runtime_mode(self) -> ParallelRuntimeMode:
        """Numerical backend derived from the public program."""

        if self.program is None:  # guarded by ``__post_init__``
            raise RuntimeError("Parallel Stream program has not been resolved.")
        return parallel_runtime_mode_for_program(self.program)

    @property
    def video_condition_on_action(self) -> bool:
        """Whether the program's Parallel backend consumes action conditioning."""

        if self.program is None:  # guarded by ``__post_init__``
            raise RuntimeError("Parallel Stream program has not been resolved.")
        return parallel_video_conditioning_for_program(self.program)

    def normalize_config_override_values(
        self,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reject direct coupling and keep program-owned backend fields aligned."""

        normalized = super().normalize_config_override_values(values)
        derived_fields = {
            "runtime_mode",
            "current_block_coupling",
            "variant_profile",
            "video_condition_on_action",
        }.intersection(normalized)
        if derived_fields:
            fields = ", ".join(
                f"policy_variant.{field_name}" for field_name in sorted(derived_fields)
            )
            raise ValueError(
                f"{fields} cannot be set for Parallel Stream; select "
                "`policy_variant.program` instead."
            )
        return normalized

    def __post_init__(self) -> None:
        super().__post_init__()
        parallel_runtime_mode_for_program(self.program)
        coerce_fields(
            self,
            enum_fields={
                "mask_mode": ParallelMaskMode,
                "cache_mode": ParallelCacheMode,
                "video_action_condition_source": ParallelActionConditionSource,
                "video_action_attention_scope": ParallelActionAttentionScope,
                "temporal_position_mode": TemporalPositionMode,
                "action_norm_method": ActionNormMethod,
            },
            enum_tuple_fields={"sequence_order": ParallelSequenceComponent},
        )


__all__ = [
    "ParallelStreamPolicyConfig",
]
