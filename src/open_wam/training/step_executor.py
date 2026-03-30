from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from open_wam.configs import BatchAdapterName, TrainingConfig
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
) -> PolicyTrainBatch:
    return PolicyTrainBatch(
        actions=actions,
        action_mask=action_mask,
        state=state,
        extra={
            "task_text": task_text,
            "metadata": metadata,
            "state_mask": state_mask,
        },
    )


class ViewBatchAdapter:
    """Prepare RGB-window batches for the shared pipeline."""

    def move_to_device(self, batch: WAMBatch, device: torch.device) -> WAMBatch:
        return move_wam_batch_to_device(batch, device)

    def prepare(self, batch: WAMBatch) -> PreparedTrainInput:
        return PreparedTrainInput(
            views=batch.views,
            policy_batch=build_policy_train_batch(
                actions=batch.actions,
                action_mask=batch.action_mask,
                state=batch.state,
                state_mask=batch.state_mask,
                task_text=batch.task_text,
                metadata=batch.metadata,
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
        metrics = {
            "loss": output.decoder_output.loss.detach(),
            **{name: value.detach() for name, value in output.decoder_output.metrics.items()},
        }
        return TrainStepResult(loss=output.decoder_output.loss, metrics=metrics, output=output)

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
