from __future__ import annotations

from dataclasses import dataclass, field

from .enums import AuxiliaryValidationSource, DataSplit, JointDenoiseTrainingMode, coerce_fields


@dataclass(frozen=True)
class AuxiliaryValidationTaskConfig:
    """One optional validation probe run alongside the primary validation set."""

    name: str
    mode_override: JointDenoiseTrainingMode | None = None
    dataset_split: DataSplit = DataSplit.VAL
    source: AuxiliaryValidationSource = AuxiliaryValidationSource.DATASET
    max_batches: int | None = 16
    report_prefix: str | None = None
    drop_text_conditioning: bool | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={"dataset_split": DataSplit, "source": AuxiliaryValidationSource},
            optional_enum_fields={"mode_override": JointDenoiseTrainingMode},
        )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("`validation.auxiliary_tasks[].name` must be non-empty.")
        if self.report_prefix is not None and (not isinstance(self.report_prefix, str) or not self.report_prefix):
            raise ValueError("`validation.auxiliary_tasks[].report_prefix` must be non-empty when set.")
        if self.max_batches is not None:
            if isinstance(self.max_batches, bool):
                raise ValueError("`validation.auxiliary_tasks[].max_batches` must be non-negative or null.")
            max_batches = int(self.max_batches)
            if max_batches < 0:
                raise ValueError("`validation.auxiliary_tasks[].max_batches` must be non-negative or null.")
            object.__setattr__(self, "max_batches", max_batches)
        if not isinstance(self.enabled, bool):
            raise ValueError("`validation.auxiliary_tasks[].enabled` must be boolean.")
        if self.drop_text_conditioning is not None and not isinstance(self.drop_text_conditioning, bool):
            raise ValueError("`validation.auxiliary_tasks[].drop_text_conditioning` must be boolean or null.")

    @property
    def phase(self) -> str:
        return self.report_prefix or self.name

    @property
    def should_drop_text(self) -> bool:
        if self.drop_text_conditioning is not None:
            return bool(self.drop_text_conditioning)
        return self.mode_override in {
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
            JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
        }


@dataclass(frozen=True)
class ValidationConfig:
    """Validation configuration independent from training loop mechanics."""

    auxiliary_tasks: tuple[AuxiliaryValidationTaskConfig, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        tasks = tuple(
            task if isinstance(task, AuxiliaryValidationTaskConfig) else AuxiliaryValidationTaskConfig(**task)
            for task in self.auxiliary_tasks
        )
        names = [task.name for task in tasks]
        if len(names) != len(set(names)):
            raise ValueError("`validation.auxiliary_tasks` entries must have unique names.")
        phases = [task.phase for task in tasks if task.enabled and task.max_batches != 0]
        if len(phases) != len(set(phases)):
            raise ValueError("Enabled `validation.auxiliary_tasks` entries must have unique report prefixes.")
        object.__setattr__(self, "auxiliary_tasks", tasks)
