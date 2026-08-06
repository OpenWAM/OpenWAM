"""Stable policy, decoder, and registration contracts."""

from open_wam.models.action_decoders.base import (
    ActionDecoder,
    ActionDecoderInferOutput,
    ActionDecoderTrainOutput,
)
from open_wam.models.policy_variants.base import PolicyVariant
from open_wam.models.policy_variants.contracts import (
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
    VideoConditionWindowContext,
)
from open_wam.models.common.attention_contracts import (
    AttentionProfileSpec,
    PreparedAttentionProfile,
)
from open_wam.models.visual_tower.contracts import (
    VisualCoreInput,
    VisualCoreOutput,
    VisualStageOutputs,
)
from open_wam.models.visual_tower.runtime_programs import (
    RuntimeProgramSpec,
    RuntimeStepInput,
    RuntimeStepOutput,
    build_dense_runtime_program,
)
from open_wam.models.visual_tower.tower import VisualTower
from open_wam.pipelines.registries import (
    register_action_decoder,
    register_policy_variant,
    registered_action_decoders,
    registered_policy_variants,
)

__all__ = [
    "ActionDecoder",
    "ActionDecoderInferOutput",
    "ActionDecoderTrainOutput",
    "AttentionProfileSpec",
    "DecoderArtifactEnvelope",
    "DecoderSequenceContext",
    "PolicyInferContext",
    "PolicyInferOutput",
    "PolicyInferState",
    "PolicyObservedHistory",
    "PolicyObservedHistoryOutput",
    "PolicyPreparedInputs",
    "PolicyTrainBatch",
    "PolicyTrainOutput",
    "PolicyVariant",
    "PreparedAttentionProfile",
    "RuntimeProgramSpec",
    "RuntimeStepInput",
    "RuntimeStepOutput",
    "VideoConditionWindowContext",
    "VisualCoreInput",
    "VisualCoreOutput",
    "VisualStageOutputs",
    "VisualTower",
    "build_dense_runtime_program",
    "register_action_decoder",
    "register_policy_variant",
    "registered_action_decoders",
    "registered_policy_variants",
]
