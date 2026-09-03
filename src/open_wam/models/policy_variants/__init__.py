"""Policy variants used by the stage-aware WAM pipeline."""

from typing import TYPE_CHECKING

from .base import PolicyVariant, VideoActionPolicyVariant
from .contracts import (
    DecoderArtifactEnvelope,
    DynamicsRolloutRequest,
    PolicyCompositionCapability,
    PolicyCompositionRngPolicy,
    PolicyExecutionCommit,
    PolicyGeneratedVideo,
    PolicyGenerationActionOrigin,
    PolicyInferContext,
    PolicyInferenceCapabilities,
    PolicyInferenceOutputRequest,
    PolicyInferOutput,
    PolicyInferState,
    PolicyModuleTopology,
    PolicyObservationWindowSessionPolicy,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
    PolicyOutputModality,
    PolicyPipelineRequirements,
    PolicyPreparedInputs,
    PolicyRecurrentHistoryPolicy,
    PolicyRolloutContract,
    PolicyStateDictOverlay,
    PolicyTemporalSpan,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVideoConditionedActionRequest,
    PolicyVideoGenerationRequest,
    PolicyVisualStage,
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
    "DynamicsRolloutRequest",
    "MoTPolicyVariant",
    "ParallelStreamPolicyVariant",
    "PolicyCompositionCapability",
    "PolicyCompositionRngPolicy",
    "PolicyExecutionCommit",
    "PolicyGeneratedVideo",
    "PolicyGenerationActionOrigin",
    "PolicyInferContext",
    "PolicyInferOutput",
    "PolicyInferState",
    "PolicyInferenceCapabilities",
    "PolicyInferenceOutputRequest",
    "PolicyModuleTopology",
    "PolicyObservationWindowSessionPolicy",
    "PolicyObservedHistory",
    "PolicyObservedHistoryOutput",
    "PolicyOutputModality",
    "PolicyPipelineRequirements",
    "PolicyPreparedInputs",
    "PolicyRecurrentHistoryPolicy",
    "PolicyRolloutContract",
    "PolicyStateDictOverlay",
    "PolicyTemporalSpan",
    "PolicyTrainBatch",
    "PolicyTrainOutput",
    "PolicyVariant",
    "PolicyVideoConditionedActionRequest",
    "PolicyVideoGenerationRequest",
    "PolicyVisualStage",
    "RolloutCursor",
    "VideoActionPolicyVariant",
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
