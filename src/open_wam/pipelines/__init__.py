"""Training and inference pipelines for the new WAM framework.

Pipeline exports are lazy so minimal package installs can import registry
surfaces without importing Torch-backed model code.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "ACTION_DECODER_BUILDERS": "open_wam.pipelines.registries",
    "POLICY_VARIANT_BUILDERS": "open_wam.pipelines.registries",
    "register_action_decoder": "open_wam.pipelines.registries",
    "register_policy_variant": "open_wam.pipelines.registries",
    "registered_action_decoders": "open_wam.pipelines.registries",
    "registered_policy_variants": "open_wam.pipelines.registries",
    "VariantPipeline": "open_wam.pipelines.variant_pipeline",
    "VariantPipelineInferOutput": "open_wam.pipelines.variant_pipeline",
    "VariantPipelineTrainOutput": "open_wam.pipelines.variant_pipeline",
    "VariantRolloutRunner": "open_wam.pipelines.rollout",
    "VariantRolloutHistoryOutput": "open_wam.pipelines.rollout",
    "VariantRolloutSession": "open_wam.pipelines.rollout",
    "VariantRolloutStepOutput": "open_wam.pipelines.rollout",
    "PolicyVideoActionConsumerPlan": "open_wam.pipelines.video_action_composition",
    "PolicyVideoProducerPlan": "open_wam.pipelines.video_action_composition",
    "build_video_conditioned_action_context": "open_wam.pipelines.video_action_composition",
    "build_video_conditioned_action_request": "open_wam.pipelines.video_action_composition",
    "build_action_decoder": "open_wam.pipelines.factory",
    "build_policy_variant": "open_wam.pipelines.factory",
    "build_variant_pipeline_from_config": "open_wam.pipelines.factory",
    "require_generated_video": "open_wam.pipelines.video_action_composition",
    "require_compatible_video_latent_spaces": "open_wam.pipelines.video_action_composition",
    "resolve_policy_video_action_consumer_plan": "open_wam.pipelines.video_action_composition",
    "resolve_policy_video_producer_plan": "open_wam.pipelines.video_action_composition",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
