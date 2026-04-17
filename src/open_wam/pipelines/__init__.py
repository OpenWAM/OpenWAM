"""Training and inference pipelines for the new WAM framework.

Pipeline exports are lazy so minimal package installs can import registry
surfaces without importing Torch-backed model code.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "BackboneOnlyPipeline": "open_wam.pipelines.backbone_only",
    "ACTION_DECODER_BUILDERS": "open_wam.pipelines.registries",
    "POLICY_VARIANT_BUILDERS": "open_wam.pipelines.registries",
    "LingbotExactArtifactBundle": "open_wam.pipelines.lingbot_exact",
    "LingbotExactChunkOutput": "open_wam.pipelines.lingbot_exact",
    "LingbotExactRunner": "open_wam.pipelines.lingbot_exact",
    "LingbotExactSession": "open_wam.pipelines.lingbot_exact",
    "LingbotExactWarmupOutput": "open_wam.pipelines.lingbot_exact",
    "VariantPipeline": "open_wam.pipelines.variant_pipeline",
    "VariantPipelineInferOutput": "open_wam.pipelines.variant_pipeline",
    "VariantPipelineTrainOutput": "open_wam.pipelines.variant_pipeline",
    "VariantRolloutRunner": "open_wam.pipelines.rollout",
    "VariantRolloutSession": "open_wam.pipelines.rollout",
    "VariantRolloutStepOutput": "open_wam.pipelines.rollout",
    "build_action_decoder": "open_wam.pipelines.factory",
    "build_exact_runtime_runner_from_config": "open_wam.pipelines.factory",
    "build_lingbot_exact_runner_from_config": "open_wam.pipelines.factory",
    "build_policy_variant": "open_wam.pipelines.factory",
    "build_variant_pipeline_from_config": "open_wam.pipelines.factory",
    "load_lingbot_exact_artifact_bundle": "open_wam.pipelines.lingbot_exact",
    "save_lingbot_exact_artifact_bundle": "open_wam.pipelines.lingbot_exact",
}

_ALIASES: dict[str, str] = {
    "ExactRuntimeArtifactBundle": "LingbotExactArtifactBundle",
    "ExactRuntimeChunkOutput": "LingbotExactChunkOutput",
    "ExactRuntimeRunner": "LingbotExactRunner",
    "ExactRuntimeSession": "LingbotExactSession",
    "ExactRuntimeWarmupOutput": "LingbotExactWarmupOutput",
}

__all__ = sorted((*_EXPORTS, *_ALIASES))


def __getattr__(name: str) -> Any:
    resolved_name = _ALIASES.get(name, name)
    try:
        module_name = _EXPORTS[resolved_name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, resolved_name)
    globals()[resolved_name] = value
    globals()[name] = value
    return value
