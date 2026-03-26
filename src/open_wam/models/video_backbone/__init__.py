"""Shared LingBot-compatible video backbone boundary."""

from typing import TYPE_CHECKING

from .config import LingbotCompatibleVideoBackboneConfig
from .contracts import BackboneOutput, CacheState, ChunkMetadata, ConditioningState, TokenGridMetadata

if TYPE_CHECKING:
    from .lingbot_compatible import LingbotCompatibleVideoBackbone

__all__ = [
    "BackboneOutput",
    "CacheState",
    "ChunkMetadata",
    "ConditioningState",
    "LingbotCompatibleVideoBackbone",
    "LingbotCompatibleVideoBackboneConfig",
    "TokenGridMetadata",
]


def __getattr__(name: str):
    if name == "LingbotCompatibleVideoBackbone":
        from .lingbot_compatible import LingbotCompatibleVideoBackbone

        return LingbotCompatibleVideoBackbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
