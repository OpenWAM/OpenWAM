from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
    "decoder": TrainingComponentSelector.VISUAL_TOWER_DECODER,
    "visual_tower.decoder": TrainingComponentSelector.VISUAL_TOWER_DECODER,
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

    total_parameters = sum(parameter.numel() for parameter in pipeline.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in pipeline.parameters() if parameter.requires_grad)
    return TrainabilityReport(
        enabled_objectives=normalize_enabled_objectives(training_config.enabled_objectives),
        trainable_components=component_trainable,
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
    def _resolve_policy_action_expert(module: nn.Module) -> list[nn.Module]:
        action_expert = getattr(module.policy_variant, "action_expert", None)
        if action_expert is None:
            raise ValueError(
                "Training component selector `policy_variant.action_expert` requires "
                "`pipeline.policy_variant.action_expert`."
            )
        return [action_expert]

    resolvers: dict[TrainingComponentSelector, ComponentResolver] = {
        TrainingComponentSelector.VISUAL_TOWER: lambda module: [module.visual_tower],
        TrainingComponentSelector.VISUAL_TOWER_FRONTEND: lambda module: [module.visual_tower.frontend],
        TrainingComponentSelector.VISUAL_TOWER_CORE: lambda module: [module.visual_tower.core],
        TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE: lambda module: [module.visual_tower.core],
        TrainingComponentSelector.VISUAL_TOWER_DECODER: lambda module: [module.visual_tower.decoder],
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
