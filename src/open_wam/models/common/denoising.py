"""Derived inference stages; conditioning remains owned by shared semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar

from open_wam.configs import CurrentBlockCoupling
from open_wam.configs.enums import PolicyOutputModality
from open_wam.models.common.dynamics_objectives import DynamicsRolloutPlan

Sample = TypeVar("Sample")
Timestep = TypeVar("Timestep")
Prediction = TypeVar("Prediction")


@dataclass(frozen=True)
class DenoisingStage:
    """Updated streams and completed streams visible as clean conditioning."""

    updates: tuple[PolicyOutputModality, ...]
    clean_conditions: tuple[PolicyOutputModality, ...] = ()


def denoise(
    sample: Sample,
    timesteps: Iterable[Timestep],
    *,
    predict: Callable[[Sample, Timestep], Prediction],
    update: Callable[[Prediction, Timestep, Sample], Sample],
) -> Sample:
    """Advance prepared streams in scheduler order without owning their history.

    The architecture predicts; the resolved schedule updates. Conditioning,
    guidance branches and clean cache writes are explicit caller operations.
    A sample may contain multiple streams advanced together by one schedule.
    """
    for timestep in timesteps:
        sample = update(predict(sample, timestep), timestep, sample)
    return sample


def resolve_denoising_stages(
    *,
    coupling: CurrentBlockCoupling,
    dynamics: DynamicsRolloutPlan | None = None,
    requested: frozenset[PolicyOutputModality] | None = None,
    supplied: frozenset[PolicyOutputModality] = frozenset(),
) -> tuple[DenoisingStage, ...]:
    """Return updated streams in their native execution order.

    This derives execution from already-resolved semantics. It neither changes
    attention/conditioning nor decides which requested outputs are supported.
    Independent stages keep video first to preserve native RNG consumption.
    """
    video = (PolicyOutputModality.VIDEO,)
    action = (PolicyOutputModality.ACTION,)
    if dynamics is not None and dynamics.semantics.is_conditional:
        stages = (
            DenoisingStage(video if dynamics.semantics.video_loss_active else action),
        )
    elif coupling is CurrentBlockCoupling.VIDEO_THEN_ACTION:
        stages = (DenoisingStage(video), DenoisingStage(action, video))
    elif coupling is CurrentBlockCoupling.ACTION_THEN_VIDEO:
        stages = (DenoisingStage(action), DenoisingStage(video, action))
    elif coupling is CurrentBlockCoupling.DECOUPLED_SAME_STEP:
        stages = (DenoisingStage(video), DenoisingStage(action))
    elif coupling in {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }:
        stages = (DenoisingStage(video + action),)
    else:
        raise ValueError(f"Unsupported denoising coupling: {coupling!r}.")
    if requested is not None and not requested:
        raise ValueError("A denoising request must select at least one output.")
    required = set(
        requested
        if requested is not None
        else (modality for stage in stages for modality in stage.updates)
    )
    result = []
    for stage in reversed(stages):
        updates = set(stage.updates) & required - supplied
        if not updates:
            continue
        if len(stage.updates) > 1:
            if coupling is CurrentBlockCoupling.JOINT:
                updates.update(stage.updates)
            elif (
                coupling is CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION
                and action[0] in updates
            ):
                updates.update(video)
            elif (
                coupling is CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO
                and video[0] in updates
            ):
                updates.update(action)
        if updates & supplied:
            raise ValueError(
                "Supplied fixed modalities cannot replace live denoising dependencies."
            )
        required.update(stage.clean_conditions)
        result.append(
            DenoisingStage(
                tuple(modality for modality in stage.updates if modality in updates),
                stage.clean_conditions,
            )
        )
    return tuple(reversed(result))


def denoise_stages(
    sample: Sample,
    stages: tuple[DenoisingStage, ...],
    *,
    timesteps: Callable[[DenoisingStage], Iterable[Timestep]],
    prepare: Callable[[DenoisingStage, Sample], Sample],
    predict: Callable[[DenoisingStage, Sample, Timestep], Prediction],
    update: Callable[[DenoisingStage, Prediction, Timestep, Sample], Sample],
) -> Sample:
    """One architecture-independent stage lifecycle, including cache scope setup."""
    for stage in stages:
        sample = prepare(stage, sample)
        sample = denoise(
            sample,
            timesteps(stage),
            predict=lambda value, time: predict(stage, value, time),
            update=lambda prediction, time, value: update(
                stage, prediction, time, value
            ),
        )
    return sample


def independently_generated_modalities(
    coupling: CurrentBlockCoupling,
) -> frozenset[PolicyOutputModality]:
    """Outputs that do not need another newly generated modality.

    A coupled stage cannot omit a stream. In ordered conditioning only the
    first stage is independent; decoupled stages are independent of each other.
    Architectures may advertise a subset according to their implementation.
    """
    return frozenset(
        stage.updates[0]
        for stage in resolve_denoising_stages(coupling=coupling)
        if len(stage.updates) == 1 and not stage.clean_conditions
    )
