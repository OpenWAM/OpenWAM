"""Runtime validation and shared dimension resolution for pipeline construction."""

from __future__ import annotations

from open_wam.configs import ExperimentConfig
from open_wam.data.action_mapping import validate_action_mapping_preflight
from open_wam.models.video_backbone import normalize_backbone_implementation


def validate_experiment_config(config: ExperimentConfig) -> None:
    action_schema = config.data.action_schema
    validate_action_mapping_preflight(
        config.data.action_mapping,
        action_schema_dim=action_schema.action_dim,
    )
    supported_backbones = config.policy_variant.supported_backbone_implementations
    backbone = normalize_backbone_implementation(config.backbone.implementation)
    if supported_backbones and backbone not in supported_backbones:
        supported = ", ".join(item.value for item in supported_backbones)
        raise ValueError(
            f"Policy {config.policy_variant.name.value!r} requires one of the "
            f"following backbone implementations: {supported}; got {backbone.value!r}."
        )


__all__ = ["validate_experiment_config"]
