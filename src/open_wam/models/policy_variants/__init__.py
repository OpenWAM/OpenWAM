"""Policy variants used by the stage-aware WAM pipeline."""

from typing import TYPE_CHECKING

from .base import PolicyVariant
from .contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    RolloutCursor,
)

if TYPE_CHECKING:
    from .parallel_stream import ParallelStreamPolicyVariant
    from .post_decoded import PostDecodedPolicyVariant
    from .post_latent import PostLatentPolicyVariant
    from .register_attached import RegisterAttachedPolicyVariant

__all__ = [
    "ParallelStreamPolicyVariant",
    "PolicyInferContext",
    "PolicyInferOutput",
    "PolicyInferState",
    "PolicyPreparedInputs",
    "PolicyTrainBatch",
    "PolicyTrainOutput",
    "RolloutCursor",
    "PolicyVariant",
    "PostDecodedPolicyVariant",
    "PostLatentPolicyVariant",
    "RegisterAttachedPolicyVariant",
]


def __getattr__(name: str):
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
