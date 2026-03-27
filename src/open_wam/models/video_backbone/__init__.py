"""Shared video-transformer backbone boundary."""

from typing import TYPE_CHECKING

from .config import (
    LingbotCompatibleVideoBackboneConfig,
    SharedVideoTransformerConfig,
    normalize_backbone_implementation,
    resolve_stage_attention_mode,
)
from .contracts import BackboneOutput, CacheState, ChunkMetadata, ConditioningState, TokenGridMetadata

if TYPE_CHECKING:
    from .lingbot_compatible import LingbotCompatibleVideoBackbone, SharedVideoTransformerBackbone

__all__ = [
    "BackboneOutput",
    "CacheState",
    "ChunkMetadata",
    "ConditioningState",
    "LingbotCompatibleVideoBackbone",
    "LingbotCompatibleVideoBackboneConfig",
    "SharedVideoTransformerBackbone",
    "SharedVideoTransformerConfig",
    "TokenGridMetadata",
    "normalize_backbone_implementation",
    "resolve_stage_attention_mode",
]


def __getattr__(name: str):
    if name in {"LingbotCompatibleVideoBackbone", "SharedVideoTransformerBackbone"}:
        from .lingbot_compatible import LingbotCompatibleVideoBackbone, SharedVideoTransformerBackbone

        if name == "SharedVideoTransformerBackbone":
            return SharedVideoTransformerBackbone
        return LingbotCompatibleVideoBackbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
