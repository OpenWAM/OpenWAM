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
    DynamicsObjective,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    MoTActionExpertInitMode,
    MoTConditionMode,
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
    PolicyConditioningRequirements,
    PolicyVariantConfig,
)
from .policy_dual_expert import DualExpertPolicyConfig
from .policy_parallel_stream import ParallelStreamPolicyConfig
from .policy_parsing import parse_policy_variant_config
from .policy_video_action import (
    VideoActionPolicyConfig,
    current_block_coupling_for_program,
)
from .training import TrainingConfig

MoTPolicyConfig = DualExpertPolicyConfig


_POLICY_COMPATIBILITY_EXPORTS = (
    ActionNormMethod,
    AttachSite,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DataConfig,
    HistoryStreamVisibility,
    InferenceConfig,
    JointTimestepCoupling,
    DualExpertActionExpertInitMode,
    DualExpertConditionMode,
    DynamicsObjective,
    DualExpertPreset,
    MoTActionExpertInitMode,
    MoTConditionMode,
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
)

__all__ = [
    "CausalVideoPredictionPolicyConfig",
    "DualExpertPolicyConfig",
    "ExtensionPolicyConfig",
    "ParallelStreamPolicyConfig",
    "PolicyConditioningRequirements",
    "PolicyVariantConfig",
    "VideoActionPolicyConfig",
    "current_block_coupling_for_program",
    "parse_policy_variant_config",
]
