"""Training and inference pipelines for the new WAM framework."""

from .backbone_only import BackboneOnlyPipeline
from .factory import build_action_decoder, build_policy_variant, build_variant_pipeline_from_config
from .unified_wam import UnifiedWAMInferOutput, UnifiedWAMPipeline, UnifiedWAMTrainOutput
from .variant_pipeline import VariantPipeline, VariantPipelineInferOutput, VariantPipelineTrainOutput

__all__ = [
    "BackboneOnlyPipeline",
    "UnifiedWAMInferOutput",
    "UnifiedWAMPipeline",
    "UnifiedWAMTrainOutput",
    "VariantPipeline",
    "VariantPipelineInferOutput",
    "VariantPipelineTrainOutput",
    "build_action_decoder",
    "build_policy_variant",
    "build_variant_pipeline_from_config",
]
