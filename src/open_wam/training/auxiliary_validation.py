from __future__ import annotations

from dataclasses import dataclass, replace
import inspect

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from open_wam.configs import AuxiliaryValidationTaskConfig, ExperimentConfig
from open_wam.configs.enums import AuxiliaryValidationSource, DataSplit
from open_wam.contracts import (
    GENERALIST_TRAINING_BUCKET_METADATA_KEY,
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
)


@dataclass(frozen=True)
class AuxiliaryValidationRun:
    """Runtime-ready auxiliary validation task."""

    config: AuxiliaryValidationTaskConfig
    loader: DataLoader
    resolved_source: str


class AuxiliaryValidationDataset(Dataset):
    """Apply validation-only metadata overrides without changing source datasets."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        task: AuxiliaryValidationTaskConfig,
    ) -> None:
        self.dataset = dataset
        self.task = task

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        metadata = dict(getattr(sample, "metadata", {}) or {})
        if self.task.mode_override is not None:
            metadata[GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY] = self.task.mode_override.value
            metadata[GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY] = self.task.should_drop_text
            metadata.setdefault(GENERALIST_TRAINING_SOURCE_METADATA_KEY, "auxiliary_validation")
            metadata.setdefault(GENERALIST_TRAINING_BUCKET_METADATA_KEY, self.task.name)
            metadata["generalist_validation_task"] = self.task.name
            metadata["generalist_validation_phase"] = self.task.phase
            metadata["generalist_validation_requested_source"] = self.task.source.value
        updates = {"metadata": metadata}
        if self.task.should_drop_text:
            if hasattr(sample, "task_text"):
                updates["task_text"] = None
            if hasattr(sample, "text_context"):
                text_context = getattr(sample, "text_context")
                negative_text_context = getattr(sample, "negative_text_context", None)
                if negative_text_context is not None:
                    updates["text_context"] = negative_text_context.clone()
                elif text_context is not None:
                    updates["text_context"] = torch.zeros_like(text_context)
        return replace(sample, **updates)


def build_auxiliary_validation_runs(
    config: ExperimentConfig,
    strategy,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[AuxiliaryValidationRun, ...]:
    runs: list[AuxiliaryValidationRun] = []
    seen_phases: set[str] = set()
    for task in config.validation.auxiliary_tasks:
        if not task.enabled or task.max_batches == 0:
            continue
        if task.phase in seen_phases:
            raise ValueError(f"Duplicate auxiliary validation report prefix {task.phase!r}.")
        seen_phases.add(task.phase)
        source_loader = train_loader if task.dataset_split == DataSplit.TRAIN else val_loader
        source_dataset, resolved_source = _resolve_auxiliary_validation_source(source_loader.dataset, task=task)
        dataset = AuxiliaryValidationDataset(source_dataset, task=task)
        sampler = (
            DistributedSampler(dataset, shuffle=False, num_replicas=strategy.world_size, rank=strategy.rank)
            if strategy.distributed
            else None
        )
        runs.append(
            AuxiliaryValidationRun(
                config=task,
                loader=DataLoader(
                    dataset,
                    batch_size=source_loader.batch_size,
                    shuffle=False,
                    num_workers=source_loader.num_workers,
                    sampler=sampler,
                    collate_fn=source_loader.collate_fn,
                    pin_memory=source_loader.pin_memory,
                ),
                resolved_source=resolved_source,
            )
        )
    return tuple(runs)


def _resolve_auxiliary_validation_source(
    dataset: Dataset,
    *,
    task: AuxiliaryValidationTaskConfig,
) -> tuple[Dataset, str]:
    if task.source == AuxiliaryValidationSource.DATASET:
        return dataset, AuxiliaryValidationSource.DATASET.value
    if task.source == AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS_IF_AVAILABLE:
        return _resolve_named_auxiliary_validation_source(
            dataset,
            task=task,
            source=AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS,
            fallback=(dataset, AuxiliaryValidationSource.DATASET.value),
        )
    return _resolve_named_auxiliary_validation_source(dataset, task=task, source=task.source)


def _resolve_named_auxiliary_validation_source(
    dataset: Dataset,
    *,
    task: AuxiliaryValidationTaskConfig,
    source: AuxiliaryValidationSource,
    fallback: tuple[Dataset, str] | None = None,
) -> tuple[Dataset, str]:
    build_source_view = getattr(dataset, "build_source_view", None)
    if callable(build_source_view):
        source_view_kwargs = {
            "source": source.value,
            "mode": task.mode_override.value if task.mode_override is not None else "joint",
            "bucket_name": task.name,
            "drop_text": task.should_drop_text,
        }
        try:
            source_view_parameters = inspect.signature(build_source_view).parameters
        except (TypeError, ValueError):
            source_view_parameters = {}
        if "spread_indices" in source_view_parameters:
            source_view_kwargs["spread_indices"] = True
        view = build_source_view(**source_view_kwargs)
        if isinstance(view, Dataset):
            return view, source.value
    attribute_by_source = {
        AuxiliaryValidationSource.REAL_DEMO: "real_dataset",
        AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS: "counterfactual_dataset",
    }
    attribute = attribute_by_source.get(source)
    if attribute is not None and hasattr(dataset, attribute):
        resolved = getattr(dataset, attribute)
        if isinstance(resolved, Dataset):
            return resolved, source.value
    if fallback is not None:
        return fallback
    raise ValueError(
        f"Auxiliary validation task {task.name!r} requested source {task.source.value!r}, "
        f"but the selected {task.dataset_split.value!r} dataset does not expose that source."
    )


def _auxiliary_validation_summary_metrics(
    *,
    task: AuxiliaryValidationTaskConfig,
    metrics: dict[str, float],
    batch_count: float,
) -> dict[str, float]:
    summary: dict[str, float] = {"count": float(batch_count)}
    for namespace in ("joint_denoise", "mot_generalist"):
        action_active_key = f"{namespace}/action_loss_active"
        latent_active_key = f"{namespace}/latent_loss_active"
        if action_active_key in metrics:
            summary["action_loss_active"] = metrics[action_active_key]
        if latent_active_key in metrics:
            summary["latent_loss_active"] = metrics[latent_active_key]
        if task.mode_override is None:
            continue
        mode = task.mode_override.value
        mode_count_key = f"{namespace}/{mode}/count"
        if mode_count_key in metrics:
            summary["mode_fraction"] = metrics[mode_count_key]
    return summary
