from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .coercion import coerce_enum, coerce_optional_enum
from .enums import (
    AuxiliaryValidationSource,
    DataSplit,
    DynamicsObjective,
    coerce_fields,
)


@dataclass(frozen=True)
class AuxiliaryValidationTaskConfig:
    """One optional validation probe run alongside the primary validation set."""

    name: str
    mode_override: DynamicsObjective | None = None
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
            optional_enum_fields={"mode_override": DynamicsObjective},
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
        if (
            self.mode_override is not None
            and self.mode_override.is_conditional
            and self.drop_text_conditioning is not None
        ):
            raise ValueError(
                "Conditional FDM/IDM validation always removes task text; "
                "do not set `validation.auxiliary_tasks[].drop_text_conditioning`."
            )

    @property
    def phase(self) -> str:
        return self.report_prefix or self.name

    @property
    def should_drop_text(self) -> bool:
        if self.mode_override is not None and self.mode_override.is_conditional:
            return True
        return bool(self.drop_text_conditioning)


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


def parse_validation_config(raw_value: Mapping[str, Any] | None) -> ValidationConfig:
    """Parse optional validation probes independently from loop mechanics."""

    raw = raw_value or {}
    tasks_raw = raw.get("auxiliary_tasks", ())
    if tasks_raw is None:
        tasks_raw = ()
    if not isinstance(tasks_raw, (list, tuple)):
        raise ValueError("Expected `validation.auxiliary_tasks` to be a list.")
    tasks: list[AuxiliaryValidationTaskConfig] = []
    for item in tasks_raw:
        if not isinstance(item, dict):
            raise ValueError(
                "Expected each `validation.auxiliary_tasks` entry to be a mapping."
            )
        if "name" not in item:
            raise ValueError(
                "Expected each `validation.auxiliary_tasks` entry to include `name`."
            )
        tasks.append(
            AuxiliaryValidationTaskConfig(
                name=item["name"],
                mode_override=coerce_optional_enum(
                    DynamicsObjective,
                    item.get("mode_override"),
                ),
                dataset_split=coerce_enum(
                    DataSplit,
                    item.get("dataset_split", DataSplit.VAL),
                ),
                source=coerce_enum(
                    AuxiliaryValidationSource,
                    item.get("source", AuxiliaryValidationSource.DATASET),
                ),
                max_batches=item.get("max_batches", 16),
                report_prefix=item.get("report_prefix"),
                drop_text_conditioning=item.get("drop_text_conditioning"),
                enabled=item.get("enabled", True),
            )
        )
    return ValidationConfig(auxiliary_tasks=tuple(tasks))
