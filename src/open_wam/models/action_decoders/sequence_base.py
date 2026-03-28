from __future__ import annotations

from .base import ActionDecoder
from open_wam.models.policy_variants.contracts import DecoderSequenceContext, PolicyInferOutput, PolicyTrainOutput


class SequenceActionDecoder(ActionDecoder):
    """Base class for decoders that consume rich visual sequence context."""

    @staticmethod
    def require_train_sequence_context(policy_output: PolicyTrainOutput) -> DecoderSequenceContext:
        if policy_output.decoder_sequence_context is None:
            raise ValueError("SequenceActionDecoder requires `decoder_sequence_context` on PolicyTrainOutput.")
        return policy_output.decoder_sequence_context

    @staticmethod
    def require_infer_sequence_context(policy_output: PolicyInferOutput) -> DecoderSequenceContext:
        if policy_output.decoder_sequence_context is None:
            raise ValueError("SequenceActionDecoder requires `decoder_sequence_context` on PolicyInferOutput.")
        return policy_output.decoder_sequence_context
