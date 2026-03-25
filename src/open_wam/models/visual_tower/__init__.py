"""Stage-aware visual tower shared across policy variants."""

from .contracts import (
    DecodedFeatureLayout,
    VisualCoreInput,
    VisualCoreOutput,
    VisualDecodeOutput,
    VisualFrontendOutput,
    VisualStageOutputs,
)
from .reference_transformer import build_reference_transformer, preferred_reference_dtype
from .tower import VisualTower

__all__ = [
    "build_reference_transformer",
    "DecodedFeatureLayout",
    "preferred_reference_dtype",
    "VisualCoreInput",
    "VisualCoreOutput",
    "VisualDecodeOutput",
    "VisualFrontendOutput",
    "VisualStageOutputs",
    "VisualTower",
]
