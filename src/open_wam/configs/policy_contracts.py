"""Shared and attachment-specific policy configuration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import (
    ActionChunkAnchorMode,
    AttachSite,
    DecodeFeatureMode,
    PolicyVariantName,
    PoolingMode,
    TemporalProjection,
    VideoConditionInputSpace,
    VideoConditionSource,
    coerce_fields,
)
from .visual_readout import VisualReadoutConfig


@dataclass(frozen=True)
class PolicyVariantConfig:
    """Base config shared by all policy variants."""

    name: PolicyVariantName
    hidden_size: int
    attach_site: AttachSite

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "name": PolicyVariantName,
                "attach_site": AttachSite,
            },
        )


@dataclass(frozen=True)
class ExtensionPolicyConfig(PolicyVariantConfig):
    """Config envelope for an application-owned policy variant."""

    name: PolicyVariantName = PolicyVariantName.EXTENSION
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    extension_type: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name != PolicyVariantName.EXTENSION:
            raise ValueError("Extension policy requires `name = extension`.")
        if not isinstance(self.extension_type, str) or not self.extension_type.strip():
            raise ValueError("Extension policy requires a non-empty `extension_type` string.")
        if self.extension_type != self.extension_type.strip():
            raise ValueError("Extension policy `extension_type` must not have surrounding whitespace.")
        if not isinstance(self.options, Mapping):
            raise ValueError("Extension policy `options` must be a mapping.")
        if not all(isinstance(key, str) for key in self.options):
            raise ValueError("Extension policy `options` keys must be strings.")
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class PostLatentPolicyConfig(PolicyVariantConfig):
    name: PolicyVariantName = PolicyVariantName.POST_LATENT
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    pooling_mode: PoolingMode = PoolingMode.PER_FRAME_MEAN
    query_count: int = 0
    temporal_projection: TemporalProjection = TemporalProjection.INTERPOLATE
    use_state_projection: bool = True
    compatibility_mode: bool = False
    video_condition_input_space: VideoConditionInputSpace = VideoConditionInputSpace.VIDEO_LATENT
    train_video_condition_source: VideoConditionSource = VideoConditionSource.LOCAL_WINDOW
    action_chunk_anchor_mode: ActionChunkAnchorMode = ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
    local_video_window_frames: int = 4
    current_video_frame_index: int = 0
    visual_readout: VisualReadoutConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attach_site != AttachSite.POST_VISUAL_CORE:
            raise ValueError(
                "Post-latent policy now requires `attach_site = post_visual_core` so all variants share "
                f"the same visual backbone path, got attach_site={self.attach_site!r}."
            )
        coerce_fields(
            self,
            enum_fields={
                "pooling_mode": PoolingMode,
                "temporal_projection": TemporalProjection,
                "video_condition_input_space": VideoConditionInputSpace,
                "train_video_condition_source": VideoConditionSource,
                "action_chunk_anchor_mode": ActionChunkAnchorMode,
            },
        )
        if int(self.local_video_window_frames) <= 0:
            raise ValueError(
                "Post-latent policy requires `local_video_window_frames > 0`, "
                f"got local_video_window_frames={self.local_video_window_frames!r}."
            )
        if not (0 <= int(self.current_video_frame_index) < int(self.local_video_window_frames)):
            raise ValueError(
                "Post-latent policy requires `0 <= current_video_frame_index < local_video_window_frames`, "
                f"got current_video_frame_index={self.current_video_frame_index!r}, "
                f"local_video_window_frames={self.local_video_window_frames!r}."
            )


@dataclass(frozen=True)
class PostDecodedPolicyConfig(PolicyVariantConfig):
    name: PolicyVariantName = PolicyVariantName.POST_DECODED
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_DECODE
    decode_feature_mode: DecodeFeatureMode = DecodeFeatureMode.FRAME_TOKEN_SEQUENCE
    pooling_mode: PoolingMode = PoolingMode.PER_FRAME_MEAN
    temporal_projection: TemporalProjection = TemporalProjection.INTERPOLATE
    use_state_projection: bool = True
    video_condition_input_space: VideoConditionInputSpace = VideoConditionInputSpace.RGB_VIDEO
    train_video_condition_source: VideoConditionSource = VideoConditionSource.LOCAL_WINDOW
    action_chunk_anchor_mode: ActionChunkAnchorMode = ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
    local_video_window_frames: int = 4
    current_video_frame_index: int = 0
    visual_readout: VisualReadoutConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attach_site != AttachSite.POST_VISUAL_DECODE:
            raise ValueError(
                "Post-decoded policy requires `attach_site = post_visual_decode`, "
                f"got attach_site={self.attach_site!r}."
            )
        coerce_fields(
            self,
            enum_fields={
                "decode_feature_mode": DecodeFeatureMode,
                "pooling_mode": PoolingMode,
                "temporal_projection": TemporalProjection,
                "video_condition_input_space": VideoConditionInputSpace,
                "train_video_condition_source": VideoConditionSource,
                "action_chunk_anchor_mode": ActionChunkAnchorMode,
            },
        )
        if int(self.local_video_window_frames) <= 0:
            raise ValueError(
                "Post-decoded policy requires `local_video_window_frames > 0`, "
                f"got local_video_window_frames={self.local_video_window_frames!r}."
            )
        if not (0 <= int(self.current_video_frame_index) < int(self.local_video_window_frames)):
            raise ValueError(
                "Post-decoded policy requires `0 <= current_video_frame_index < local_video_window_frames`, "
                f"got current_video_frame_index={self.current_video_frame_index!r}, "
                f"local_video_window_frames={self.local_video_window_frames!r}."
            )


@dataclass(frozen=True)
class CausalVideoPredictionPolicyConfig(PolicyVariantConfig):
    """Standalone causal video-only pretraining variant."""

    name: PolicyVariantName = PolicyVariantName.CAUSAL_VIDEO_PREDICTION
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attach_site != AttachSite.POST_VISUAL_CORE:
            raise ValueError(
                "Causal video prediction requires `attach_site = post_visual_core`, "
                f"got attach_site={self.attach_site!r}."
            )


__all__ = [
    "PolicyVariantConfig",
    "ExtensionPolicyConfig",
    "PostLatentPolicyConfig",
    "PostDecodedPolicyConfig",
    "CausalVideoPredictionPolicyConfig",
]
