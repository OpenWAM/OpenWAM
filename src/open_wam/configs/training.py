from __future__ import annotations

from dataclasses import dataclass

from .enums import (
    OptimizerName,
    SampleLossWeightMode,
    SchedulerName,
    TrainingComponentSelector,
    TrainingObjective,
    coerce_fields,
)


OBJECTIVE_ALIASES = {
    "action": TrainingObjective.ACTION,
    "latent": TrainingObjective.LATENT,
    "video": TrainingObjective.LATENT,
}


def normalize_enabled_objectives(
    values: tuple[TrainingObjective | str, ...] | list[TrainingObjective | str],
) -> tuple[TrainingObjective, ...]:
    normalized: list[TrainingObjective] = []
    for value in values:
        try:
            resolved = OBJECTIVE_ALIASES[value]
        except KeyError as exc:
            supported = ", ".join(sorted(OBJECTIVE_ALIASES))
            raise ValueError(f"Unsupported training objective {value!r}. Supported values: {supported}.") from exc
        if resolved not in normalized:
            normalized.append(resolved)
    if not normalized:
        raise ValueError("At least one training objective must be enabled.")
    return tuple(normalized)


@dataclass(frozen=True)
class TrainingConfig:
    """Training-layer config shared by all policy variants and runtimes."""

    # Diffusion noise schedule knobs
    video_num_train_timesteps: int = 1000
    action_num_train_timesteps: int = 1000
    video_sigma_shift: float = 5.0
    action_sigma_shift: float = 1.0
    use_teacher_forcing: bool = False
    chunk_size: int = 2
    window_size: int = 8

    # Optimization knobs
    optimizer_name: OptimizerName = OptimizerName.ADAMW
    scheduler_name: SchedulerName = SchedulerName.CONSTANT
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = None
    num_steps: int | None = None
    text_condition_dropout_prob: float = 0.0

    # Objective-selection knobs
    enabled_objectives: tuple[TrainingObjective, ...] = (
        TrainingObjective.ACTION,
        TrainingObjective.LATENT,
    )
    latent_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    sample_loss_weight_mode: SampleLossWeightMode = SampleLossWeightMode.NONE
    sample_loss_weight_reference_steps: float | None = None
    sample_loss_weight_min: float | None = None
    sample_loss_weight_max: float | None = None

    # Trainability knobs
    trainable_components: tuple[TrainingComponentSelector, ...] = (TrainingComponentSelector.ALL,)
    frozen_components: tuple[TrainingComponentSelector, ...] = ()

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "optimizer_name": OptimizerName,
                "scheduler_name": SchedulerName,
                "sample_loss_weight_mode": SampleLossWeightMode,
            },
            enum_tuple_fields={
                "trainable_components": TrainingComponentSelector,
                "frozen_components": TrainingComponentSelector,
            },
            transforms={
                "enabled_objectives": normalize_enabled_objectives,
            },
        )
        if self.sample_loss_weight_reference_steps is not None and self.sample_loss_weight_reference_steps <= 0:
            raise ValueError("`sample_loss_weight_reference_steps` must be positive when set.")
        if self.sample_loss_weight_min is not None and self.sample_loss_weight_min < 0:
            raise ValueError("`sample_loss_weight_min` must be non-negative when set.")
        if self.sample_loss_weight_max is not None and self.sample_loss_weight_max <= 0:
            raise ValueError("`sample_loss_weight_max` must be positive when set.")
        if (
            self.sample_loss_weight_min is not None
            and self.sample_loss_weight_max is not None
            and self.sample_loss_weight_min > self.sample_loss_weight_max
        ):
            raise ValueError("`sample_loss_weight_min` cannot exceed `sample_loss_weight_max`.")

    def objective_enabled(self, objective_name: TrainingObjective | str) -> bool:
        resolved_name = OBJECTIVE_ALIASES.get(objective_name, objective_name)
        return resolved_name in normalize_enabled_objectives(self.enabled_objectives)

    def objective_weight(self, objective_name: TrainingObjective | str) -> float:
        resolved_name = OBJECTIVE_ALIASES.get(objective_name, objective_name)
        if not self.objective_enabled(resolved_name):
            return 0.0
        if resolved_name == TrainingObjective.LATENT:
            return float(self.latent_loss_weight)
        if resolved_name == TrainingObjective.ACTION:
            return float(self.action_loss_weight)
        raise ValueError(f"Unsupported objective {objective_name!r}.")
