from __future__ import annotations

import math

from collections.abc import Iterable

import torch
from torch import nn

from open_wam.configs import OptimizerName, SchedulerName, TrainingConfig


def _is_floating_dtype(dtype: torch.dtype | None) -> bool:
    if dtype is None:
        return False
    return torch.empty((), dtype=dtype).is_floating_point()


def _optimizer_state_target_dtype(parameter: object) -> torch.dtype | None:
    grad = getattr(parameter, "grad", None)
    grad_dtype = getattr(grad, "dtype", None)
    if _is_floating_dtype(grad_dtype):
        return grad_dtype
    parameter_dtype = getattr(parameter, "dtype", None)
    if _is_floating_dtype(parameter_dtype):
        return parameter_dtype
    return None


def _normalize_optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        if not isinstance(state, dict):
            continue
        state_dtype = _optimizer_state_target_dtype(parameter)
        if state_dtype is None:
            continue
        for key, value in list(state.items()):
            if key == "step":
                continue
            if torch.is_tensor(value) and torch.is_floating_point(value) and value.dtype != state_dtype:
                state[key] = value.to(dtype=state_dtype)


def warmup_constant_lambda(step: int, *, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step >= warmup_steps:
        return 1.0
    return float(step + 1) / float(max(1, warmup_steps))


def warmup_cosine_lambda(
    step: int, *, warmup_steps: int, total_steps: int, min_ratio: float = 0.0
) -> float:
    """Linear warmup followed by cosine decay to an optional floor.

    The schedule retains explicit
    step indexing, clamping, and minimum-ratio behavior.
    """

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    span = max(1, int(total_steps) - int(warmup_steps))
    progress = min(1.0, max(0.0, (int(step) - int(warmup_steps)) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio + (1.0 - min_ratio) * cosine)


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
    if scheduler_name == SchedulerName.WARMUP_COSINE:
        total = int(getattr(training_config, "num_steps", 0) or 0)
        if total <= 0:
            raise ValueError(
                "`warmup_cosine` needs `training.num_steps` to know the decay horizon."
            )
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_lambda(
                step,
                warmup_steps=training_config.warmup_steps,
                total_steps=total,
            ),
        )
    raise ValueError(f"Unsupported scheduler {training_config.scheduler_name!r}.")
