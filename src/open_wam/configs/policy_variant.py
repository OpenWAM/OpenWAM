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
    CurrentBlockCoupling,
    DecodeFeatureMode,
    GeneralistTrainingParadigm,
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
from .policy_mot import MoTPolicyConfig, _coerce_mot_generalist_training_mode_probs
from .policy_parallel_stream import (
    ParallelStreamPolicyConfig,
    _coerce_joint_denoise_training_mode_probs,
    _default_joint_denoise_training_mode_probs,
)
from .policy_parsing import parse_policy_variant_config
from .training import TrainingConfig
from .variant_semantics import coerce_probability_map, default_video_action_conditioning_mode_probs
from .visual_readout import VisualReadoutConfig, parse_visual_readout_config


_POLICY_COMPATIBILITY_EXPORTS = (
    ActionChunkAnchorMode,
    ActionNormMethod,
    AttachSite,
    CurrentBlockCoupling,
    DataConfig,
    DecodeFeatureMode,
    GeneralistTrainingParadigm,
    InferenceConfig,
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
    SharedVideoTransformerConfig,
    TemporalPositionMode,
    TemporalProjection,
    TrainingConfig,
    VideoConditionInputSpace,
    VideoConditionSource,
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
    "ExtensionPolicyConfig",
    "MoTPolicyConfig",
    "ParallelStreamPolicyConfig",
    "PolicyVariantConfig",
    "PostDecodedPolicyConfig",
    "PostLatentPolicyConfig",
    "parse_policy_variant_config",
]
