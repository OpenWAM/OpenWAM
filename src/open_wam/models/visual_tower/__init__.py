"""Stage-aware visual tower shared across policy variants."""

from .contracts import (
    DecodedFeatureLayout,
    VisualCoreInput,
    VisualCoreOutput,
    VisualDecodeOutput,
    VisualFrontendOutput,
    VisualStageOutputs,
)
from .tower import VisualTower

__all__ = [
    "DecodedFeatureLayout",
    "VisualCoreInput",
    "VisualCoreOutput",
    "VisualDecodeOutput",
    "VisualFrontendOutput",
    "VisualStageOutputs",
    "VisualTower",
]
