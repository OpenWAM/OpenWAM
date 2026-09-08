"""Regression coverage for warmup cosine scheduler."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from open_wam.configs import SchedulerName, TrainingConfig
from open_wam.configs.training import parse_training_config
from open_wam.training.optim import build_scheduler, warmup_cosine_lambda


def _reference_formula(
    step: int, *, warmup_steps: int, total_steps: int, min_ratio: float = 0.0
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    span = max(1, int(total_steps) - int(warmup_steps))
    progress = min(1.0, max(0.0, (int(step) - int(warmup_steps)) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio + (1.0 - min_ratio) * cosine)


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 0.01},
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 0.003},
        ],
        lr=0.01,
    )


def _advance(optimizer, scheduler) -> None:
    optimizer.step()
    scheduler.step()


def test_schedule_name_parses_through_typed_config() -> None:
    config = parse_training_config(
        {"scheduler_name": "warmup_cosine", "warmup_steps": 4, "num_steps": 12}
    )
    assert config.scheduler_name is SchedulerName.WARMUP_COSINE
    assert SchedulerName("warmup_cosine") is config.scheduler_name


@pytest.mark.parametrize(
    "step,warmup,total,minimum,expected",
    [
        (0, 4, 12, 0.0, 0.25),
        (1, 4, 12, 0.0, 0.5),
        (3, 4, 12, 0.0, 1.0),
        (4, 4, 12, 0.0, 1.0),
        (8, 4, 12, 0.0, 0.5),
        (12, 4, 12, 0.0, 0.0),
        (20, 4, 12, 0.0, 0.0),
        (0, 0, 12, 0.0, 1.0),
        (6, 0, 12, 0.0, 0.5),
        (12, 0, 12, 0.0, 0.0),
        (8, 4, 12, 0.2, 0.6),
        (12, 4, 12, 0.2, 0.2),
        (20, 4, 12, 0.2, 0.2),
    ],
)
def test_warmup_cosine_boundaries(step, warmup, total, minimum, expected) -> None:
    assert warmup_cosine_lambda(
        step, warmup_steps=warmup, total_steps=total, min_ratio=minimum
    ) == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parametrize("warmup", [-2, 0, 1, 4, 12, 20])
@pytest.mark.parametrize("minimum", [0.0, 0.2])
def test_entire_schedule_matches_reference_formula(warmup, minimum) -> None:
    for step in range(-1, 31):
        kwargs = dict(warmup_steps=warmup, total_steps=12, min_ratio=minimum)
        assert warmup_cosine_lambda(step, **kwargs) == _reference_formula(
            step, **kwargs
        )


@pytest.mark.parametrize("horizon", [None, 0, -1])
def test_warmup_cosine_still_requires_positive_decay_horizon(horizon) -> None:
    config = TrainingConfig(scheduler_name="warmup_cosine", num_steps=horizon)
    with pytest.raises(ValueError, match="training.num_steps"):
        build_scheduler(_optimizer(), config)


@pytest.mark.parametrize("warmup", [0, 4])
@pytest.mark.parametrize("completed_updates", [0, 1, 3, 4, 5, 11, 12, 15])
def test_saved_scheduler_state_continues_reference_lr(
    warmup, completed_updates
) -> None:
    config = TrainingConfig(
        scheduler_name="warmup_cosine", warmup_steps=warmup, num_steps=12
    )
    previous_optimizer = _optimizer()
    previous_scheduler = torch.optim.lr_scheduler.LambdaLR(
        previous_optimizer,
        lr_lambda=lambda step: _reference_formula(
            step, warmup_steps=warmup, total_steps=12
        ),
    )
    uninterrupted_optimizer = _optimizer()
    uninterrupted_scheduler = build_scheduler(uninterrupted_optimizer, config)
    for _ in range(completed_updates):
        _advance(previous_optimizer, previous_scheduler)
        _advance(uninterrupted_optimizer, uninterrupted_scheduler)
        assert uninterrupted_scheduler.get_last_lr() == previous_scheduler.get_last_lr()

    # Match runtime resume ordering: construct both, then restore optimizer and
    # scheduler state. Loading scheduler state alone would not restore group LR.
    resumed_optimizer = _optimizer()
    resumed_scheduler = build_scheduler(resumed_optimizer, config)
    resumed_optimizer.load_state_dict(copy.deepcopy(previous_optimizer.state_dict()))
    resumed_scheduler.load_state_dict(copy.deepcopy(previous_scheduler.state_dict()))
    for _ in range(19):
        assert resumed_scheduler.state_dict() == previous_scheduler.state_dict()
        assert uninterrupted_scheduler.state_dict() == previous_scheduler.state_dict()
        assert [group["lr"] for group in resumed_optimizer.param_groups] == (
            previous_scheduler.get_last_lr()
        )
        assert resumed_scheduler.get_last_lr() == uninterrupted_scheduler.get_last_lr()
        _advance(previous_optimizer, previous_scheduler)
        _advance(uninterrupted_optimizer, uninterrupted_scheduler)
        _advance(resumed_optimizer, resumed_scheduler)


@pytest.mark.parametrize(
    "name,warmup",
    [("constant", 0), ("constant", 4), ("warmup_constant", 4), ("constant_with_warmup", 4)],
)
def test_existing_constant_schedules_are_unchanged(name, warmup) -> None:
    optimizer = _optimizer()
    scheduler = build_scheduler(
        optimizer, TrainingConfig(scheduler_name=name, warmup_steps=warmup)
    )
    for step in range(10):
        scale = 1.0 if warmup <= 0 or step >= warmup else (step + 1) / warmup
        assert scheduler.get_last_lr() == [0.01 * scale, 0.003 * scale]
        _advance(optimizer, scheduler)
