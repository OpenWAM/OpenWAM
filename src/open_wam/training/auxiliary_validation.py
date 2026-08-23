from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from open_wam.configs import AuxiliaryValidationTaskConfig, ExperimentConfig
from open_wam.configs.enums import (
    AuxiliaryValidationSource,
    DataSplit,
    DynamicsObjective,
)
from open_wam.configs.policy_video_action import resolve_fixed_conditioning_mode
from open_wam.contracts import (
    DYNAMICS_ROUTING_BUCKET_METADATA_KEY,
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY,
    DYNAMICS_ROUTING_MODE_METADATA_KEY,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
)
from open_wam.data import DynamicsSourceViewProvider


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
            metadata[DYNAMICS_ROUTING_MODE_METADATA_KEY] = self.task.mode_override.value
            metadata[DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY] = self.task.should_drop_text
            metadata.setdefault(DYNAMICS_ROUTING_SOURCE_METADATA_KEY, "auxiliary_validation")
            metadata.setdefault(DYNAMICS_ROUTING_BUCKET_METADATA_KEY, self.task.name)
            metadata["generalist_validation_task"] = self.task.name
            metadata["generalist_validation_phase"] = self.task.phase
            metadata["generalist_validation_requested_source"] = self.task.source.value
        updates = {"metadata": metadata}
        if self.task.should_drop_text:
            if hasattr(sample, "task_text"):
                updates["task_text"] = None
            if hasattr(sample, "text_context"):
                text_context = sample.text_context
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
    fixed_mode = resolve_fixed_conditioning_mode(config.policy_variant)
    for task in config.validation.auxiliary_tasks:
        if not task.enabled or task.max_batches == 0:
            continue
        if (
            fixed_mode is not None
            and task.mode_override is not None
            and task.mode_override != fixed_mode
        ):
            raise ValueError(
                f"Auxiliary validation task {task.name!r} requests mode "
                f"{task.mode_override.value!r}, but policy program "
                f"{config.policy_variant.program.value!r} fixes mode "
                f"{fixed_mode.value!r}."
            )
        effective_task = (
            replace(task, mode_override=fixed_mode)
            if fixed_mode is not None and task.mode_override is None
            else task
        )
        if effective_task.phase in seen_phases:
            raise ValueError(
                f"Duplicate auxiliary validation report prefix {effective_task.phase!r}."
            )
        seen_phases.add(effective_task.phase)
        source_loader = (
            train_loader
            if effective_task.dataset_split == DataSplit.TRAIN
            else val_loader
        )
        source_dataset, resolved_source = _resolve_auxiliary_validation_source(
            source_loader.dataset,
            task=effective_task,
        )
        dataset = AuxiliaryValidationDataset(source_dataset, task=effective_task)
        sampler = (
            DistributedSampler(dataset, shuffle=False, num_replicas=strategy.world_size, rank=strategy.rank)
            if strategy.distributed
            else None
        )
        runs.append(
            AuxiliaryValidationRun(
                config=effective_task,
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
    conditional_mode = (
        task.mode_override is not None and task.mode_override.is_conditional
    )
    if task.source == AuxiliaryValidationSource.DATASET:
        return dataset, AuxiliaryValidationSource.DATASET.value
    if task.source == AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS_IF_AVAILABLE:
        if isinstance(dataset, DynamicsSourceViewProvider):
            source = (
                AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS
                if dataset.has_route(
                    source=AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS.value,
                    mode=(
                        task.mode_override.value
                        if task.mode_override is not None
                        else DynamicsObjective.JOINT.value
                    ),
                )
                else AuxiliaryValidationSource.REAL_DEMO
            )
            return _resolve_named_auxiliary_validation_source(
                dataset,
                task=task,
                source=source,
            )
        if conditional_mode:
            raise ValueError(
                f"Conditional auxiliary validation task {task.name!r} requires "
                "a dynamics-routed dataset; no source-view provider is available."
            )
        return dataset, AuxiliaryValidationSource.DATASET.value
    return _resolve_named_auxiliary_validation_source(dataset, task=task, source=task.source)


def _resolve_named_auxiliary_validation_source(
    dataset: Dataset,
    *,
    task: AuxiliaryValidationTaskConfig,
    source: AuxiliaryValidationSource,
) -> tuple[Dataset, str]:
    mode = (
        task.mode_override.value
        if task.mode_override is not None
        else DynamicsObjective.JOINT.value
    )
    if isinstance(dataset, DynamicsSourceViewProvider) and not dataset.has_route(
        source=source.value,
        mode=mode,
    ):
        raise ValueError(
            f"Auxiliary validation task {task.name!r} requested source {source.value!r}, "
            f"but that source is not available in the selected {task.dataset_split.value!r} dataset."
        )
    if isinstance(dataset, DynamicsSourceViewProvider):
        view = dataset.build_source_view(
            source=source.value,
            mode=mode,
            bucket_name=task.name,
            spread_indices=True,
        )
        if isinstance(view, Dataset):
            return view, source.value
    raise ValueError(
        f"Auxiliary validation task {task.name!r} requested source {task.source.value!r}, "
        f"but the selected {task.dataset_split.value!r} dataset does not expose that source."
    )


def _auxiliary_validation_summary_metrics(
    *,
    task: AuxiliaryValidationTaskConfig,
    metrics: dict[str, float],
    batch_count: float,
    dynamics_metric_namespace: str | None,
) -> dict[str, float]:
    summary: dict[str, float] = {"count": float(batch_count)}
    if dynamics_metric_namespace is None:
        return summary
    action_active_key = f"{dynamics_metric_namespace}/action_loss_active"
    latent_active_key = f"{dynamics_metric_namespace}/latent_loss_active"
    if action_active_key in metrics:
        summary["action_loss_active"] = metrics[action_active_key]
    if latent_active_key in metrics:
        summary["latent_loss_active"] = metrics[latent_active_key]
    if task.mode_override is not None:
        mode_count_key = (
            f"{dynamics_metric_namespace}/{task.mode_override.value}/count"
        )
        if mode_count_key in metrics:
            summary["mode_fraction"] = metrics[mode_count_key]
    return summary
