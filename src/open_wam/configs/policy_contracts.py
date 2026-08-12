"""Shared and attachment-specific policy configuration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import AttachSite, PolicyVariantName, coerce_fields


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
    "CausalVideoPredictionPolicyConfig",
]
