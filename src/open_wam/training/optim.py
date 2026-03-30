from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from open_wam.configs import OptimizerName, SchedulerName, TrainingConfig


def warmup_constant_lambda(step: int, *, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step >= warmup_steps:
        return 1.0
    return float(step + 1) / float(max(1, warmup_steps))


def collect_trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def build_optimizer(
    module: nn.Module,
    training_config: TrainingConfig,
    *,
    parameters: Iterable[nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    resolved_parameters = list(parameters) if parameters is not None else collect_trainable_parameters(module)
    if not resolved_parameters:
        raise ValueError("No trainable parameters were found when building the optimizer.")
    if training_config.optimizer_name != OptimizerName.ADAMW:
        raise ValueError(f"Unsupported optimizer {training_config.optimizer_name!r}.")
    return torch.optim.AdamW(
        resolved_parameters,
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay,
        foreach=False,
        fused=False,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_name = training_config.scheduler_name
    if scheduler_name == SchedulerName.CONSTANT_WITH_WARMUP:
        scheduler_name = SchedulerName.WARMUP_CONSTANT
    if scheduler_name == SchedulerName.CONSTANT and training_config.warmup_steps > 0:
        scheduler_name = SchedulerName.WARMUP_CONSTANT
    if scheduler_name == SchedulerName.CONSTANT:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    if scheduler_name == SchedulerName.WARMUP_CONSTANT:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=training_config.warmup_steps),
        )
    raise ValueError(f"Unsupported scheduler {training_config.scheduler_name!r}.")
