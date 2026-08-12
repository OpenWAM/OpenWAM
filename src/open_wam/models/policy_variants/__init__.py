"""Policy variants used by the stage-aware WAM pipeline."""

from typing import TYPE_CHECKING

from .base import PolicyVariant
from .contracts import (
    DecoderArtifactEnvelope,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    RolloutCursor,
)

if TYPE_CHECKING:
    from .causal_video_prediction import CausalVideoPredictionPolicyVariant
    from .dual_expert import DualExpertPolicyVariant
    from .parallel_stream import ParallelStreamPolicyVariant

__all__ = [
    "CausalVideoPredictionPolicyVariant",
    "DecoderArtifactEnvelope",
    "DualExpertPolicyVariant",
    "MoTPolicyVariant",
    "ParallelStreamPolicyVariant",
    "PolicyInferContext",
    "PolicyInferOutput",
    "PolicyInferState",
    "PolicyObservedHistory",
    "PolicyObservedHistoryOutput",
    "PolicyPreparedInputs",
    "PolicyTrainBatch",
    "PolicyTrainOutput",
    "PolicyVariant",
    "RolloutCursor",
]


def __getattr__(name: str):
    if name == "CausalVideoPredictionPolicyVariant":
        from .causal_video_prediction import CausalVideoPredictionPolicyVariant

        return CausalVideoPredictionPolicyVariant
    if name == "DualExpertPolicyVariant":
        from .dual_expert import DualExpertPolicyVariant

        return DualExpertPolicyVariant
    if name == "MoTPolicyVariant":
        from .dual_expert import DualExpertPolicyVariant

        return DualExpertPolicyVariant
    if name == "ParallelStreamPolicyVariant":
        from .parallel_stream import ParallelStreamPolicyVariant

        return ParallelStreamPolicyVariant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
