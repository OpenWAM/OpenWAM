"""Policy variants used by the stage-aware WAM pipeline."""

from typing import TYPE_CHECKING

from .base import PolicyVariant
from .contracts import (
    DecoderSequenceContext,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    RolloutCursor,
    VideoConditionWindowContext,
)

if TYPE_CHECKING:
    from .causal_video_prediction import CausalVideoPredictionPolicyVariant
    from .mot import MoTPolicyVariant
    from .parallel_stream import ParallelStreamPolicyVariant
    from .post_decoded import PostDecodedPolicyVariant
    from .post_latent import PostLatentPolicyVariant
    from .register_attached import RegisterAttachedPolicyVariant
    from .video_sequence_policy import VideoSequencePolicyVariant

__all__ = [
    "ParallelStreamPolicyVariant",
    "CausalVideoPredictionPolicyVariant",
    "DecoderSequenceContext",
    "MoTPolicyVariant",
    "PolicyInferContext",
    "PolicyInferOutput",
    "PolicyInferState",
    "PolicyPreparedInputs",
    "PolicyTrainBatch",
    "PolicyTrainOutput",
    "RolloutCursor",
    "VideoConditionWindowContext",
    "PolicyVariant",
    "PostDecodedPolicyVariant",
    "PostLatentPolicyVariant",
    "RegisterAttachedPolicyVariant",
    "VideoSequencePolicyVariant",
]


def __getattr__(name: str):
    if name == "CausalVideoPredictionPolicyVariant":
        from .causal_video_prediction import CausalVideoPredictionPolicyVariant

        return CausalVideoPredictionPolicyVariant
    if name == "MoTPolicyVariant":
        from .mot import MoTPolicyVariant

        return MoTPolicyVariant
    if name == "ParallelStreamPolicyVariant":
        from .parallel_stream import ParallelStreamPolicyVariant

        return ParallelStreamPolicyVariant
    if name == "PostDecodedPolicyVariant":
        from .post_decoded import PostDecodedPolicyVariant

        return PostDecodedPolicyVariant
    if name == "PostLatentPolicyVariant":
        from .post_latent import PostLatentPolicyVariant

        return PostLatentPolicyVariant
    if name == "RegisterAttachedPolicyVariant":
        from .register_attached import RegisterAttachedPolicyVariant

        return RegisterAttachedPolicyVariant
    if name == "VideoSequencePolicyVariant":
        from .video_sequence_policy import VideoSequencePolicyVariant

        return VideoSequencePolicyVariant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
