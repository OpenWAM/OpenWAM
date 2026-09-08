"""Typed dual-expert policy configuration and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .backbone import SharedVideoTransformerConfig
from .enums import (
    ActionDecoderName,
    AttachSite,
    BatchingMode,
    DualExpertActionExpertInitMode,
    DualExpertActionExpertSize,
    DualExpertConditionMode,
    DualExpertPreset,
    PolicyVariantName,
    coerce_fields,
)
from .policy_video_action import VideoActionPolicyConfig


@dataclass(frozen=True)
class DualExpertPolicyConfig(VideoActionPolicyConfig):
    """Configuration for separate video and action transformer experts.

    ``program`` is the sole authored execution semantic. The low-level
    same-chunk coupling is derived and cannot conflict with it.
    """

    name: PolicyVariantName = PolicyVariantName.DUAL_EXPERT
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    preset: DualExpertPreset | None = None
    action_expert_size: DualExpertActionExpertSize = DualExpertActionExpertSize.CONFIGURED
    condition_mode: DualExpertConditionMode = DualExpertConditionMode.FIRST_FRAME
    action_expert_init_mode: DualExpertActionExpertInitMode = (
        DualExpertActionExpertInitMode.VIDEO_WEIGHT_COPY
    )
    video_prefix_frames: int = 1
    teacher_forcing_video_noise_prob: float = 0.5
    num_action_layers: int = 30
    action_hidden_size: int | None = None
    action_ffn_dim: int | None = None
    # Trade forward compute for activation memory by recomputing each
    # (video, action) block pair during backward instead of storing its
    # activations. Only affects two-stream train paths that run through
    # `forward_joint_video_action_denoise`.
    use_activation_checkpointing: bool = False

    @property
    def default_action_decoder(self) -> ActionDecoderName:
        return ActionDecoderName.DUAL_EXPERT

    @property
    def supported_batching_modes(self) -> tuple[BatchingMode, ...]:
        return (BatchingMode.STRICT, BatchingMode.PADDED, BatchingMode.PACKED)

    def normalize_config_override_values(
        self, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Reject removed Dual Expert controls at the typed override boundary."""

        removed = {
            "runtime_mode",
            "current_block_coupling",
            "video_can_attend_action",
        }.intersection(values)
        if removed:
            fields = ", ".join(f"policy_variant.{name}" for name in sorted(removed))
            raise ValueError(
                f"{fields} cannot be set for Dual Expert; select "
                "`policy_variant.program` instead."
            )
        return super().normalize_config_override_values(values)

    def validate_backbone_config(
        self, backbone: SharedVideoTransformerConfig
    ) -> None:
        """Keep the named 500M profile tied to its measured video geometry."""

        if self.action_expert_size is not DualExpertActionExpertSize.SMALL_500M:
            return
        head_dim = backbone.attention_head_dim or (
            backbone.hidden_size // backbone.num_heads
        )
        if (
            self.num_action_layers != backbone.num_layers
            or backbone.num_layers != 30
            or backbone.hidden_size != 3072
            or backbone.num_heads * head_dim != 3072
        ):
            raise ValueError(
                "`action_expert_size=small_500m` requires the 30-layer, "
                "3072-wide video backbone with attention width 3072 and exactly "
                "one action layer per video layer. Use `configured` for other "
                "backbone geometries and measure their parameter count separately."
            )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attach_site != AttachSite.POST_VISUAL_CORE:
            raise ValueError(
                "DualExpert policy requires `attach_site = post_visual_core`, "
                f"got attach_site={self.attach_site!r}."
            )
        if int(self.video_prefix_frames) <= 0:
            raise ValueError(
                "DualExpert policy requires `video_prefix_frames > 0`, "
                f"got video_prefix_frames={self.video_prefix_frames!r}."
            )
        if not (0.0 <= float(self.teacher_forcing_video_noise_prob) <= 1.0):
            raise ValueError(
                "DualExpert policy requires `0 <= teacher_forcing_video_noise_prob <= 1`, "
                f"got teacher_forcing_video_noise_prob={self.teacher_forcing_video_noise_prob!r}."
            )
        if int(self.num_action_layers) <= 0:
            raise ValueError(
                "DualExpert policy requires `num_action_layers > 0`, "
                f"got num_action_layers={self.num_action_layers!r}."
            )
        if self.action_hidden_size is not None and int(self.action_hidden_size) <= 0:
            raise ValueError(
                "DualExpert policy requires `action_hidden_size > 0` when provided, "
                f"got action_hidden_size={self.action_hidden_size!r}."
            )
        if self.action_ffn_dim is not None and int(self.action_ffn_dim) <= 0:
            raise ValueError(
                "DualExpert policy requires `action_ffn_dim > 0` when provided, "
                f"got action_ffn_dim={self.action_ffn_dim!r}."
            )
        coerce_fields(
            self,
            enum_fields={
                "condition_mode": DualExpertConditionMode,
                "action_expert_init_mode": DualExpertActionExpertInitMode,
                "action_expert_size": DualExpertActionExpertSize,
            },
            optional_enum_fields={"preset": DualExpertPreset},
        )
        if self.action_expert_size is DualExpertActionExpertSize.SMALL_500M:
            for name, expected in (
                ("action_hidden_size", 576),
                ("action_ffn_dim", 2048),
                ("num_action_layers", 30),
            ):
                actual = getattr(self, name)
                if actual is not None and actual != expected:
                    raise ValueError(
                        f"`action_expert_size=small_500m` requires `{name}={expected}`; "
                        "remove the conflicting override or choose `configured`."
                    )
            if self.action_expert_init_mode is DualExpertActionExpertInitMode.VIDEO_WEIGHT_COPY:
                raise ValueError(
                    "The smaller action expert cannot copy wider video weights exactly. "
                    "Set `action_expert_init_mode=video_weight_interpolate` or `random`."
                )
            coerce_fields(
                self,
                transforms={
                    "action_hidden_size": lambda value: 576 if value is None else value,
                    "action_ffn_dim": lambda value: 2048 if value is None else value,
                },
            )


__all__ = [
    "DualExpertPolicyConfig",
]
