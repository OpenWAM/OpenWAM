from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .coercion import coerce_enum
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


def parse_training_config(raw_value: Mapping[str, Any] | None) -> TrainingConfig:
    """Parse the shared optimization and objective section."""

    raw = raw_value or {}
    return TrainingConfig(
        video_num_train_timesteps=raw.get("video_num_train_timesteps", 1000),
        action_num_train_timesteps=raw.get("action_num_train_timesteps", 1000),
        video_sigma_shift=raw.get("video_sigma_shift", 5.0),
        action_sigma_shift=raw.get("action_sigma_shift", 1.0),
        use_teacher_forcing=raw.get("use_teacher_forcing", False),
        chunk_size=raw.get("chunk_size", 2),
        window_size=raw.get("window_size", 8),
        optimizer_name=coerce_enum(
            OptimizerName,
            raw.get("optimizer_name", "adamw"),
        ),
        scheduler_name=coerce_enum(
            SchedulerName,
            raw.get("scheduler_name", "constant"),
        ),
        learning_rate=raw.get("learning_rate", 1e-4),
        beta1=raw.get("beta1", 0.9),
        beta2=raw.get("beta2", 0.999),
        weight_decay=raw.get("weight_decay", 0.0),
        warmup_steps=raw.get("warmup_steps", 0),
        gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 1),
        max_grad_norm=raw.get("max_grad_norm"),
        num_steps=raw.get("num_steps"),
        text_condition_dropout_prob=raw.get("text_condition_dropout_prob", 0.0),
        enabled_objectives=tuple(raw.get("enabled_objectives", ("action", "latent"))),
        latent_loss_weight=raw.get("latent_loss_weight", 1.0),
        action_loss_weight=raw.get("action_loss_weight", 1.0),
        sample_loss_weight_mode=coerce_enum(
            SampleLossWeightMode,
            raw.get("sample_loss_weight_mode", "none"),
        ),
        sample_loss_weight_reference_steps=raw.get(
            "sample_loss_weight_reference_steps"
        ),
        sample_loss_weight_min=raw.get("sample_loss_weight_min"),
        sample_loss_weight_max=raw.get("sample_loss_weight_max"),
        trainable_components=tuple(raw.get("trainable_components", ("all",))),
        frozen_components=tuple(raw.get("frozen_components", ())),
    )
