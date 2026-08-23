"""Public policy-variant configuration facade.

Concrete contracts live in role-specific sibling modules. This facade keeps
the established ``open_wam.configs.policy_variant`` import path stable for
callers and old serialized configuration objects.
"""

from .backbone import SharedVideoTransformerConfig
from .data_contracts import DataConfig
from .enums import (
    ActionNormMethod,
    AttachSite,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DualExpertActionExpertInitMode,
    DualExpertConditionMode,
    DualExpertPreset,
    GeneralistDenoisingMode,
    GeneralistTrainingParadigm,
    HistoryStreamVisibility,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    MoTActionExpertInitMode,
    MoTConditionMode,
    MoTGeneralistTrainingMode,
    MoTPreset,
    ParallelActionAttentionScope,
    ParallelActionConditionSource,
    ParallelCacheMode,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelMaskMode,
    ParallelRuntimeMode,
    ParallelSequenceComponent,
    ParallelSequenceContract,
    ParallelStreamVariantProfile,
    PolicyVariantName,
    ProprioContextMode,
    TemporalPositionMode,
    VideoActionSequenceContract,
    coerce_fields,
)
from .inference import InferenceConfig
from .policy_contracts import (
    CausalVideoPredictionPolicyConfig,
    ExtensionPolicyConfig,
    PolicyVariantConfig,
)
from .policy_dual_expert import (
    DualExpertPolicyConfig,
    _coerce_mot_generalist_training_mode_probs,
)
from .policy_parallel_stream import (
    ParallelStreamPolicyConfig,
    _coerce_joint_denoise_training_mode_probs,
    _default_joint_denoise_training_mode_probs,
)
from .policy_parsing import parse_policy_variant_config
from .policy_video_action import (
    VideoActionPolicyConfig,
    current_block_coupling_for_program,
    resolve_video_action_program_semantics,
)
from .training import TrainingConfig
from .variant_semantics import (
    coerce_probability_map,
    default_video_action_conditioning_mode_probs,
)

MoTPolicyConfig = DualExpertPolicyConfig


_POLICY_COMPATIBILITY_EXPORTS = (
    ActionNormMethod,
    AttachSite,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DataConfig,
    GeneralistTrainingParadigm,
    HistoryStreamVisibility,
    InferenceConfig,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    DualExpertActionExpertInitMode,
    DualExpertConditionMode,
    GeneralistDenoisingMode,
    DualExpertPreset,
    MoTActionExpertInitMode,
    MoTConditionMode,
    MoTGeneralistTrainingMode,
    MoTPreset,
    ParallelActionAttentionScope,
    ParallelActionConditionSource,
    ParallelCacheMode,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelMaskMode,
    ParallelRuntimeMode,
    ParallelSequenceComponent,
    ParallelSequenceContract,
    ParallelStreamVariantProfile,
    PolicyVariantName,
    ProprioContextMode,
    SharedVideoTransformerConfig,
    TemporalPositionMode,
    TrainingConfig,
    VideoActionSequenceContract,
    coerce_fields,
    coerce_probability_map,
    default_video_action_conditioning_mode_probs,
    _coerce_joint_denoise_training_mode_probs,
    _coerce_mot_generalist_training_mode_probs,
    _default_joint_denoise_training_mode_probs,
)

__all__ = [
    "CausalVideoPredictionPolicyConfig",
    "DualExpertPolicyConfig",
    "ExtensionPolicyConfig",
    "ParallelStreamPolicyConfig",
    "PolicyVariantConfig",
    "VideoActionPolicyConfig",
    "current_block_coupling_for_program",
    "parse_policy_variant_config",
    "resolve_video_action_program_semantics",
]
