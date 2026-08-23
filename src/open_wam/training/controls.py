from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from open_wam.configs import TrainingConfig
from open_wam.configs.enums import TrainingComponentSelector, TrainingObjective
from open_wam.configs.training import normalize_enabled_objectives

COMPONENT_ALIASES = {
    "all": TrainingComponentSelector.ALL,
    "visual": TrainingComponentSelector.VISUAL_TOWER,
    "visual_tower": TrainingComponentSelector.VISUAL_TOWER,
    "frontend": TrainingComponentSelector.VISUAL_TOWER_FRONTEND,
    "visual_tower.frontend": TrainingComponentSelector.VISUAL_TOWER_FRONTEND,
    "core": TrainingComponentSelector.VISUAL_TOWER_CORE,
    "backbone": TrainingComponentSelector.VISUAL_TOWER_CORE,
    "visual_tower.core": TrainingComponentSelector.VISUAL_TOWER_CORE,
    "runtime_backbone": TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE,
    "visual_tower.runtime_backbone": TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE,
    "proprio_context_encoder": TrainingComponentSelector.VISUAL_TOWER_PROPRIO_CONTEXT_ENCODER,
    "visual_tower.proprio_context_encoder": TrainingComponentSelector.VISUAL_TOWER_PROPRIO_CONTEXT_ENCODER,
    "generalist_mode_context_encoder": TrainingComponentSelector.VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER,
    "visual_tower.generalist_mode_context_encoder": TrainingComponentSelector.VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER,
    "policy": TrainingComponentSelector.POLICY_VARIANT,
    "variant": TrainingComponentSelector.POLICY_VARIANT,
    "policy_variant": TrainingComponentSelector.POLICY_VARIANT,
    "policy_variant.action_expert": TrainingComponentSelector.POLICY_VARIANT_ACTION_EXPERT,
    "head": TrainingComponentSelector.ACTION_DECODER,
    "action_decoder": TrainingComponentSelector.ACTION_DECODER,
}


@dataclass(frozen=True)
class TrainabilityReport:
    enabled_objectives: tuple[TrainingObjective, ...]
    trainable_components: tuple[TrainingComponentSelector, ...]
    frozen_components: tuple[TrainingComponentSelector, ...]
    total_parameters: int
    trainable_parameters: int


ComponentResolver = Callable[[nn.Module], list[nn.Module]]


def objective_enabled(training_config: TrainingConfig, objective_name: str) -> bool:
    return training_config.objective_enabled(objective_name)


def objective_weight(training_config: TrainingConfig, objective_name: str) -> float:
    return training_config.objective_weight(objective_name)


def apply_training_component_controls(
    module: nn.Module,
    training_config: TrainingConfig,
) -> TrainabilityReport:
    pipeline = getattr(module, "pipeline", module)
    component_trainable = normalize_component_selectors(training_config.trainable_components)
    component_frozen = normalize_component_selectors(training_config.frozen_components)

    if TrainingComponentSelector.ALL in component_trainable:
        _set_component_requires_grad(
            pipeline,
            selectors=(
                TrainingComponentSelector.VISUAL_TOWER,
                TrainingComponentSelector.POLICY_VARIANT,
                TrainingComponentSelector.ACTION_DECODER,
            ),
            enabled=True,
        )
    else:
        _set_component_requires_grad(
            pipeline,
            selectors=(
                TrainingComponentSelector.VISUAL_TOWER,
                TrainingComponentSelector.POLICY_VARIANT,
                TrainingComponentSelector.ACTION_DECODER,
            ),
            enabled=False,
        )
        _set_component_requires_grad(pipeline, selectors=component_trainable, enabled=True)
    if component_frozen:
        _set_component_requires_grad(pipeline, selectors=component_frozen, enabled=False)
    proprio_context_encoder_auto_enabled = _enable_proprio_context_encoder_when_used(
        pipeline,
        component_frozen=component_frozen,
    )
    generalist_mode_context_encoder_auto_enabled = _enable_generalist_mode_context_encoder_when_used(
        pipeline,
        component_frozen=component_frozen,
    )
    reported_trainable_components = component_trainable
    if (
        proprio_context_encoder_auto_enabled
        and TrainingComponentSelector.ALL not in reported_trainable_components
        and TrainingComponentSelector.VISUAL_TOWER_PROPRIO_CONTEXT_ENCODER not in reported_trainable_components
    ):
        reported_trainable_components = (
            *reported_trainable_components,
            TrainingComponentSelector.VISUAL_TOWER_PROPRIO_CONTEXT_ENCODER,
        )
    if (
        generalist_mode_context_encoder_auto_enabled
        and TrainingComponentSelector.ALL not in reported_trainable_components
        and TrainingComponentSelector.VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER not in reported_trainable_components
    ):
        reported_trainable_components = (
            *reported_trainable_components,
            TrainingComponentSelector.VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER,
        )

    total_parameters = sum(parameter.numel() for parameter in pipeline.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in pipeline.parameters() if parameter.requires_grad)
    return TrainabilityReport(
        enabled_objectives=normalize_enabled_objectives(training_config.enabled_objectives),
        trainable_components=reported_trainable_components,
        frozen_components=component_frozen,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )


def normalize_component_selectors(
    values: tuple[TrainingComponentSelector | str, ...] | list[TrainingComponentSelector | str],
) -> tuple[TrainingComponentSelector, ...]:
    normalized: list[TrainingComponentSelector] = []
    for value in values:
        if isinstance(value, TrainingComponentSelector):
            resolved = value
        else:
            try:
                resolved = COMPONENT_ALIASES[value]
            except KeyError as exc:
                supported = ", ".join(sorted(COMPONENT_ALIASES))
                raise ValueError(
                    f"Unsupported training component selector {value!r}. Supported values: {supported}."
                ) from exc
        if resolved not in normalized:
            normalized.append(resolved)
    return tuple(normalized)


def _set_component_requires_grad(
    pipeline: nn.Module,
    *,
    selectors: tuple[TrainingComponentSelector, ...],
    enabled: bool,
) -> None:
    visited_modules: set[int] = set()
    for selector in selectors:
        for target_module in _resolve_component_modules(pipeline, selector):
            module_id = id(target_module)
            if module_id in visited_modules:
                continue
            visited_modules.add(module_id)
            for parameter in target_module.parameters():
                parameter.requires_grad = enabled


def _resolve_component_modules(pipeline: nn.Module, selector: TrainingComponentSelector) -> list[nn.Module]:
    def _resolve_proprio_context_encoder(module: nn.Module) -> list[nn.Module]:
        encoders = [
            encoder
            for encoder in (
                getattr(module.visual_tower.core, "proprio_context_encoder", None),
                getattr(module.visual_tower.core, "proprio_hidden_context_encoder", None),
            )
            if encoder is not None
        ]
        if not encoders:
            raise ValueError(
                "Training component selector `visual_tower.proprio_context_encoder` requires "
                "`pipeline.visual_tower.core.proprio_context_encoder` or "
                "`pipeline.visual_tower.core.proprio_hidden_context_encoder`."
            )
        return encoders

    def _resolve_generalist_mode_context_encoder(module: nn.Module) -> list[nn.Module]:
        encoder = getattr(module.visual_tower.core, "generalist_mode_context_encoder", None)
        if encoder is None:
            raise ValueError(
                "Training component selector `visual_tower.generalist_mode_context_encoder` requires "
                "`pipeline.visual_tower.core.generalist_mode_context_encoder`."
            )
        return [encoder]

    def _resolve_policy_action_expert(module: nn.Module) -> list[nn.Module]:
        action_expert_modules = module.module_topology().action_expert_modules
        if not action_expert_modules:
            raise ValueError(
                "Training component selector `policy_variant.action_expert` is not "
                "provided by the active policy variant."
            )
        return list(action_expert_modules)

    def _resolve_visual_tower_runtime_backbone(module: nn.Module) -> list[nn.Module]:
        return list(module.module_topology().visual_runtime_modules)

    resolvers: dict[TrainingComponentSelector, ComponentResolver] = {
        TrainingComponentSelector.VISUAL_TOWER: lambda module: [module.visual_tower],
        TrainingComponentSelector.VISUAL_TOWER_FRONTEND: lambda module: [module.visual_tower.frontend],
        TrainingComponentSelector.VISUAL_TOWER_CORE: lambda module: [module.visual_tower.core],
        TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE: _resolve_visual_tower_runtime_backbone,
        TrainingComponentSelector.VISUAL_TOWER_PROPRIO_CONTEXT_ENCODER: _resolve_proprio_context_encoder,
        TrainingComponentSelector.VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER: _resolve_generalist_mode_context_encoder,
        TrainingComponentSelector.POLICY_VARIANT: lambda module: [module.policy_variant],
        TrainingComponentSelector.POLICY_VARIANT_ACTION_EXPERT: _resolve_policy_action_expert,
        TrainingComponentSelector.ACTION_DECODER: lambda module: [module.action_decoder],
    }
    if selector == TrainingComponentSelector.ALL:
        return (
            _resolve_component_modules(pipeline, TrainingComponentSelector.VISUAL_TOWER)
            + _resolve_component_modules(pipeline, TrainingComponentSelector.POLICY_VARIANT)
            + _resolve_component_modules(pipeline, TrainingComponentSelector.ACTION_DECODER)
        )
    try:
        resolver = resolvers[selector]
    except KeyError as exc:
        raise ValueError(f"Unsupported component selector {selector!r}.") from exc
    return resolver(pipeline)


def _enable_proprio_context_encoder_when_used(
    pipeline: nn.Module,
    *,
    component_frozen: tuple[TrainingComponentSelector, ...],
) -> bool:
    """Keep zero-init proprio context trainable unless explicitly disabled.

    The proprio encoder is owned by the shared visual core but semantically
    belongs to the proprio-conditioning adapter. If a run trains only a policy
    module while freezing the main backbone, leaving this zero-init adapter
    frozen makes proprio conditioning a permanent zero path. Inspect the
    assembled core capability so built-in and extension policies behave alike.
    """

    if any(
        selector in component_frozen
        for selector in (
            TrainingComponentSelector.ALL,
            TrainingComponentSelector.VISUAL_TOWER,
            TrainingComponentSelector.VISUAL_TOWER_CORE,
            TrainingComponentSelector.VISUAL_TOWER_PROPRIO_CONTEXT_ENCODER,
        )
    ):
        return False
    core = getattr(getattr(pipeline, "visual_tower", None), "core", None)
    encoders = tuple(
        encoder
        for encoder in (
            getattr(core, "proprio_context_encoder", None),
            getattr(core, "proprio_hidden_context_encoder", None),
        )
        if isinstance(encoder, nn.Module)
    )
    if not encoders:
        return False
    for encoder in encoders:
        for parameter in encoder.parameters():
            parameter.requires_grad = True
    return True


def _enable_generalist_mode_context_encoder_when_used(
    pipeline: nn.Module,
    *,
    component_frozen: tuple[TrainingComponentSelector, ...],
) -> bool:
    """Keep GJD mode control tokens trainable when the main backbone is frozen."""

    if any(
        selector in component_frozen
        for selector in (
            TrainingComponentSelector.ALL,
            TrainingComponentSelector.VISUAL_TOWER,
            TrainingComponentSelector.VISUAL_TOWER_CORE,
            TrainingComponentSelector.VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER,
        )
    ):
        return False
    core = getattr(getattr(pipeline, "visual_tower", None), "core", None)
    encoder = getattr(core, "generalist_mode_context_encoder", None)
    if not isinstance(encoder, nn.Module):
        return False
    for parameter in encoder.parameters():
        parameter.requires_grad = True
    return True
