"""Policy variants used by the stage-aware WAM pipeline."""

from typing import TYPE_CHECKING

from .base import PolicyVariant
from .contracts import (
    DecoderArtifactEnvelope,
    DecoderSequenceContext,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    RolloutCursor,
    VideoConditionWindowContext,
)

if TYPE_CHECKING:
    from .causal_video_prediction import CausalVideoPredictionPolicyVariant
    from .dual_expert import DualExpertPolicyVariant
    from .parallel_stream import ParallelStreamPolicyVariant
    from .post_decoded import PostDecodedPolicyVariant
    from .post_latent import PostLatentPolicyVariant

__all__ = [
    "CausalVideoPredictionPolicyVariant",
    "DecoderArtifactEnvelope",
    "DecoderSequenceContext",
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
    "PostDecodedPolicyVariant",
    "PostLatentPolicyVariant",
    "RolloutCursor",
    "VideoConditionWindowContext",
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
    if name == "PostDecodedPolicyVariant":
        from .post_decoded import PostDecodedPolicyVariant

        return PostDecodedPolicyVariant
    if name == "PostLatentPolicyVariant":
        from .post_latent import PostLatentPolicyVariant

        return PostLatentPolicyVariant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
