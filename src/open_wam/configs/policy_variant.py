"""Public policy-variant configuration facade.

Concrete contracts live in role-specific sibling modules. This facade keeps
the established ``open_wam.configs.policy_variant`` import path stable for
callers and old serialized configuration objects.
"""

from .backbone import SharedVideoTransformerConfig
from .data_contracts import DataConfig
from .enums import (
    ActionChunkAnchorMode,
    ActionNormMethod,
    AttachSite,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DecodeFeatureMode,
    DualExpertActionExpertInitMode,
    DualExpertConditionMode,
    DualExpertPreset,
    DualExpertRuntimeMode,
    GeneralistDenoisingMode,
    GeneralistTrainingParadigm,
    HistoryStreamVisibility,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    MoTActionExpertInitMode,
    MoTConditionMode,
    MoTGeneralistTrainingMode,
    MoTPreset,
    MoTRuntimeMode,
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
    PoolingMode,
    ProprioContextMode,
    TemporalPositionMode,
    TemporalProjection,
    VideoActionSequenceContract,
    VideoConditionInputSpace,
    VideoConditionSource,
    coerce_fields,
)
from .inference import InferenceConfig
from .policy_contracts import (
    CausalVideoPredictionPolicyConfig,
    ExtensionPolicyConfig,
    PolicyVariantConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
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
    resolve_video_action_program_semantics,
)
from .training import TrainingConfig
from .variant_semantics import (
    coerce_probability_map,
    default_video_action_conditioning_mode_probs,
)
from .visual_readout import VisualReadoutConfig, parse_visual_readout_config

MoTPolicyConfig = DualExpertPolicyConfig


_POLICY_COMPATIBILITY_EXPORTS = (
    ActionChunkAnchorMode,
    ActionNormMethod,
    AttachSite,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DataConfig,
    DecodeFeatureMode,
    GeneralistTrainingParadigm,
    HistoryStreamVisibility,
    InferenceConfig,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    DualExpertActionExpertInitMode,
    DualExpertConditionMode,
    GeneralistDenoisingMode,
    DualExpertPreset,
    DualExpertRuntimeMode,
    MoTActionExpertInitMode,
    MoTConditionMode,
    MoTGeneralistTrainingMode,
    MoTPreset,
    MoTRuntimeMode,
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
    PoolingMode,
    ProprioContextMode,
    SharedVideoTransformerConfig,
    TemporalPositionMode,
    TemporalProjection,
    TrainingConfig,
    VideoConditionInputSpace,
    VideoConditionSource,
    VideoActionSequenceContract,
    VisualReadoutConfig,
    coerce_fields,
    coerce_probability_map,
    default_video_action_conditioning_mode_probs,
    parse_visual_readout_config,
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
    "PostDecodedPolicyConfig",
    "PostLatentPolicyConfig",
    "VideoActionPolicyConfig",
    "parse_policy_variant_config",
    "resolve_video_action_program_semantics",
]
