from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from open_wam.configs import BatchAdapterName, SampleLossWeightMode, TrainingConfig
from open_wam.data import (
    LatentWAMBatch,
    WAMBatch,
    move_latent_wam_batch_to_device,
    move_wam_batch_to_device,
)
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.pipelines import VariantPipeline, VariantPipelineTrainOutput


@dataclass
class PreparedTrainInput:
    policy_batch: PolicyTrainBatch
    views: dict[str, torch.Tensor] | None = None
    video_latents: torch.Tensor | None = None
    canonical_video: torch.Tensor | None = None
    text_context: torch.Tensor | None = None
    negative_text_context: torch.Tensor | None = None


@dataclass
class TrainStepResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    output: VariantPipelineTrainOutput


class BatchAdapter(Protocol):
    def move_to_device(self, batch, device: torch.device): ...
    def prepare(self, batch) -> PreparedTrainInput: ...


def build_policy_train_batch(
    *,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    state: torch.Tensor | None,
    state_mask: torch.Tensor | None,
    task_text: tuple[str | None, ...] | None,
    metadata: tuple[dict[str, object], ...],
    video_latents: torch.Tensor | None = None,
) -> PolicyTrainBatch:
    extra = {
        "task_text": task_text,
        "metadata": metadata,
        "state_mask": state_mask,
    }
    if video_latents is not None:
        extra["video_latents"] = video_latents
    return PolicyTrainBatch(
        actions=actions,
        action_mask=action_mask,
        state=state,
        extra=extra,
    )


class ViewBatchAdapter:
    """Prepare RGB-window batches for the shared pipeline."""

    def move_to_device(self, batch: WAMBatch, device: torch.device) -> WAMBatch:
        return move_wam_batch_to_device(batch, device)

    def prepare(self, batch: WAMBatch) -> PreparedTrainInput:
        # Raw RGB batches do not carry latent tensors by contract. Keep the
        # policy-batch extras additive for the rare cases where a view batch
        # subtype chooses to include them.
        video_latents = getattr(batch, "video_latents", None)
        return PreparedTrainInput(
            views=batch.views,
            policy_batch=build_policy_train_batch(
                actions=batch.actions,
                action_mask=batch.action_mask,
                state=batch.state,
                state_mask=batch.state_mask,
                task_text=batch.task_text,
                metadata=batch.metadata,
                video_latents=video_latents,
            ),
        )


class LatentBatchAdapter:
    """Prepare latent-first batches for the shared pipeline."""

    def move_to_device(self, batch: LatentWAMBatch, device: torch.device) -> LatentWAMBatch:
        return move_latent_wam_batch_to_device(batch, device)

    def prepare(self, batch: LatentWAMBatch) -> PreparedTrainInput:
        return PreparedTrainInput(
            video_latents=batch.video_latents,
            canonical_video=batch.canonical_video,
            text_context=batch.text_context,
            negative_text_context=batch.negative_text_context,
            policy_batch=build_policy_train_batch(
                actions=batch.actions,
                action_mask=batch.action_mask,
                state=batch.state,
                state_mask=batch.state_mask,
                task_text=batch.task_text,
                metadata=batch.metadata,
            ),
        )


def build_batch_adapter(name: BatchAdapterName | str) -> BatchAdapter:
    if name == BatchAdapterName.VIEWS:
        return ViewBatchAdapter()
    if name == BatchAdapterName.LATENTS:
        return LatentBatchAdapter()
    raise ValueError(f"Unsupported batch_adapter {name!r}.")


class PipelineTrainStepExecutor:
    """Run one shared-pipeline train forward pass from prepared runtime batches."""

    def __init__(
        self,
        *,
        pipeline: VariantPipeline,
        batch_adapter: BatchAdapter,
        training_config: TrainingConfig,
    ) -> None:
        self.pipeline = pipeline
        self.batch_adapter = batch_adapter
        self.training_config = training_config

    def forward_train(self, batch) -> TrainStepResult:
        prepared = self.batch_adapter.prepare(batch)
        prepared = self._apply_text_condition_dropout(prepared)
        if prepared.views is not None:
            output = self.pipeline(
                views=prepared.views,
                batch=prepared.policy_batch,
            )
        else:
            assert prepared.video_latents is not None
            output = self.pipeline(
                video_latents=prepared.video_latents,
                batch=prepared.policy_batch,
                canonical_video=prepared.canonical_video,
                text_context=prepared.text_context,
                negative_text_context=prepared.negative_text_context,
            )
        sample_loss_weight = resolve_sample_loss_weight(
            training_config=self.training_config,
            batch=prepared.policy_batch,
        )
        loss = output.decoder_output.loss * sample_loss_weight
        metrics = {
            "loss": loss.detach(),
            **{name: value.detach() for name, value in output.decoder_output.metrics.items()},
        }
        if self.training_config.sample_loss_weight_mode != SampleLossWeightMode.NONE:
            metrics["unweighted_loss"] = output.decoder_output.loss.detach()
            metrics["sample_loss_weight"] = sample_loss_weight.detach()
        return TrainStepResult(loss=loss, metrics=metrics, output=output)

    def _apply_text_condition_dropout(self, prepared: PreparedTrainInput) -> PreparedTrainInput:
        prob = float(self.training_config.text_condition_dropout_prob)
        if prob <= 0.0 or prepared.text_context is None or not self.pipeline.training:
            return prepared
        batch_size = prepared.text_context.shape[0]
        drop_mask = torch.rand(batch_size, device=prepared.text_context.device) < prob
        if not bool(drop_mask.any()):
            return prepared
        text_context = prepared.text_context.clone()
        if prepared.negative_text_context is not None:
            text_context[drop_mask] = prepared.negative_text_context[drop_mask]
        else:
            text_context[drop_mask] = 0.0
        return PreparedTrainInput(
            policy_batch=prepared.policy_batch,
            views=prepared.views,
            video_latents=prepared.video_latents,
            canonical_video=prepared.canonical_video,
            text_context=text_context,
            negative_text_context=prepared.negative_text_context,
        )


def resolve_sample_loss_weight(
    *,
    training_config: TrainingConfig,
    batch: PolicyTrainBatch,
) -> torch.Tensor:
    mode = training_config.sample_loss_weight_mode
    if mode == SampleLossWeightMode.NONE:
        return torch.ones((), dtype=torch.float32, device=batch.actions.device)

    valid_action_steps = _per_sample_valid_action_steps(batch)
    if valid_action_steps.shape[0] != 1:
        raise ValueError(
            "sample_loss_weight_mode currently requires train_batch_size=1 because decoder_output.loss is "
            "already reduced to a scalar before runtime weighting. Use gradient_accumulation_steps for larger "
            "effective batches or disable sample_loss_weight_mode."
        )
    reference_steps = training_config.sample_loss_weight_reference_steps
    if reference_steps is None:
        reference_steps = _metadata_mean_float(
            batch.extra.get("metadata"),
            "dataset_mean_valid_action_steps",
        )
    if reference_steps is None:
        reference_steps = float(valid_action_steps.detach().mean().clamp_min(1.0).item())
    reference = torch.tensor(
        float(reference_steps),
        dtype=torch.float32,
        device=valid_action_steps.device,
    ).clamp_min(1.0)
    normalized = valid_action_steps / reference
    if mode == SampleLossWeightMode.VALID_ACTION_STEPS:
        weights = normalized
    elif mode == SampleLossWeightMode.SQRT_VALID_ACTION_STEPS:
        weights = torch.sqrt(normalized.clamp_min(0.0))
    else:
        raise ValueError(f"Unsupported sample_loss_weight_mode {mode!r}.")

    if training_config.sample_loss_weight_min is not None:
        weights = weights.clamp_min(float(training_config.sample_loss_weight_min))
    if training_config.sample_loss_weight_max is not None:
        weights = weights.clamp_max(float(training_config.sample_loss_weight_max))
    return weights.mean()


def _per_sample_valid_action_steps(batch: PolicyTrainBatch) -> torch.Tensor:
    if batch.action_mask is None:
        return torch.full(
            (batch.actions.shape[0],),
            fill_value=float(batch.actions.shape[1]),
            dtype=torch.float32,
            device=batch.actions.device,
        )
    if batch.action_mask.ndim < 3:
        raise ValueError(
            "Expected action_mask to have shape [batch, time, dim] when sample loss weighting is enabled, "
            f"got {tuple(batch.action_mask.shape)}."
        )
    valid_step_mask = batch.action_mask.float().sum(dim=-1) > 0
    return valid_step_mask.float().sum(dim=-1).clamp_min(1.0)


def _metadata_mean_float(metadata: object, key: str) -> float | None:
    if not isinstance(metadata, (tuple, list)):
        return None
    values: list[float] = []
    for item in metadata:
        if not isinstance(item, Mapping):
            continue
        value = item.get(key)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))
