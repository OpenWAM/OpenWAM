"""Batch-preserving frontend/decoder bridge for variable-length policy execution.

Only lightweight preparation and decoding are per-sample. The policy receives
all prepared samples in one call and owns padding/packing of its heavy layers.
Trimming here prevents collation padding from entering conditioning or losses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from open_wam.configs.enums import BatchingMode
from open_wam.data.latent_contracts import LATENT_SAMPLE_TENSOR_AXES
from open_wam.models.action_decoders import ActionDecoderTrainOutput
from open_wam.models.policy_variants import PolicyTrainBatch, PolicyTrainOutput

if TYPE_CHECKING:
    from .variant_pipeline import VariantPipeline, VariantPipelineTrainOutput


@dataclass(frozen=True)
class LatentTrainSampleInput:
    video_latents: torch.Tensor
    batch: PolicyTrainBatch
    canonical_video: torch.Tensor | None
    text_context: torch.Tensor | None
    negative_text_context: torch.Tensor | None


def split_latent_train_batch(
    video_latents: torch.Tensor,
    batch: PolicyTrainBatch,
    *,
    canonical_video: torch.Tensor | None = None,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
) -> tuple[LatentTrainSampleInput, ...]:
    """Restore each original tensor and metadata item from a collated batch."""

    if video_latents.ndim != 5 or batch.actions.ndim != 3:
        raise ValueError(
            "Variable batches require video [B,C,T,H,W] and actions [B,T,D]."
        )
    size = int(video_latents.shape[0])
    lengths = batch.extra.get("sequence_lengths")
    tensor_lengths = batch.extra.get("tensor_lengths")
    metadata = batch.extra.get("metadata")
    if (
        size <= 0
        or batch.actions.shape[0] != size
        or not isinstance(lengths, (tuple, list))
        or len(lengths) != size
        or not isinstance(tensor_lengths, Mapping)
        or not isinstance(metadata, (tuple, list))
        or len(metadata) != size
    ):
        raise ValueError(
            "Variable batches require lengths and metadata for every sample."
        )
    for length in lengths:
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or not 0 < length <= video_latents.shape[2]
        ):
            raise ValueError(
                "sequence_lengths must contain positive original latent lengths."
            )

    def slice_tensor(name: str, value: torch.Tensor | None, index: int):
        if value is None:
            return None
        axis = LATENT_SAMPLE_TENSOR_AXES[name] + 1
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim <= axis
            or value.shape[0] != size
        ):
            raise ValueError(f"Invalid batched tensor {name!r}.")
        field_lengths = tensor_lengths.get(name)
        if not isinstance(field_lengths, (tuple, list)) or len(field_lengths) != size:
            raise ValueError(f"Missing original tensor lengths for {name!r}.")
        length = field_lengths[index]
        if length is None:
            return None
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or not 0 <= length <= value.shape[axis]
        ):
            raise ValueError(f"Invalid original tensor length for {name!r}.")
        slices = [slice(None)] * value.ndim
        slices[0] = slice(index, index + 1)
        slices[axis] = slice(0, length)
        return value[tuple(slices)]

    result = []
    for index, length in enumerate(lengths):
        if not isinstance(metadata[index], Mapping):
            raise TypeError("Every variable-batch metadata item must be a mapping.")
        video = slice_tensor("video_latents", video_latents, index)
        if video.shape[2] != length:
            raise ValueError(
                "sequence_lengths and video_latents tensor_lengths disagree."
            )
        extra = {
            key: value
            for key, value in batch.extra.items()
            if key
            not in {
                "batching_mode",
                "sequence_lengths",
                "tensor_lengths",
                "metadata",
                "task_text",
                "video_latents",
            }
        }
        for name in LATENT_SAMPLE_TENSOR_AXES.keys() & extra.keys():
            extra[name] = slice_tensor(name, extra[name], index)
        extra["metadata"] = (dict(metadata[index]),)
        extra["batching_video_capacity"] = int(video_latents.shape[2])
        for key in ("task_text", "source_task_text"):
            task_text = batch.extra.get(key)
            if task_text is not None:
                if len(task_text) != size:
                    raise ValueError(f"{key} must have one item per sequence.")
                extra[key] = (task_text[index],)
        if "video_latents" in batch.extra:
            extra["video_latents"] = video
        sample_batch = replace(
            batch,
            actions=slice_tensor("actions", batch.actions, index),
            action_mask=slice_tensor("action_mask", batch.action_mask, index),
            state=slice_tensor("state", batch.state, index),
            source_text_context=slice_tensor(
                "text_context", batch.source_text_context, index
            ),
            extra=extra,
        )
        result.append(
            LatentTrainSampleInput(
                video_latents=video,
                batch=sample_batch,
                canonical_video=slice_tensor("canonical_video", canonical_video, index),
                text_context=slice_tensor("text_context", text_context, index),
                negative_text_context=slice_tensor(
                    "negative_text_context", negative_text_context, index
                ),
            )
        )
    return tuple(result)


def _mean_metrics(
    items: tuple[dict[str, torch.Tensor], ...],
) -> dict[str, torch.Tensor]:
    keys = set(items[0])
    if any(set(item) != keys for item in items[1:]):
        raise ValueError(
            "All samples in a variable batch must expose the same metric contract."
        )
    return {
        key: torch.stack([item[key] for item in items]).mean(dim=0)
        for key in sorted(keys)
    }


def forward_variable_latent_batch(
    pipeline: VariantPipeline,
    video_latents: torch.Tensor,
    batch: PolicyTrainBatch,
    *,
    batching_mode: BatchingMode,
    canonical_video: torch.Tensor | None = None,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
) -> VariantPipelineTrainOutput:
    """Prepare separate sequences, execute heavy layers once, average sample losses."""

    from .variant_pipeline import VariantPipelineTrainOutput

    samples = split_latent_train_batch(
        video_latents,
        batch,
        canonical_video=canonical_video,
        text_context=text_context,
        negative_text_context=negative_text_context,
    )
    visual_outputs = tuple(
        pipeline.prepare_visual_outputs_from_latents(
            sample.video_latents,
            task_text=sample.batch.extra.get("task_text"),
            text_context=sample.text_context,
            negative_text_context=sample.negative_text_context,
            canonical_video=sample.canonical_video,
        )
        for sample in samples
    )
    prepared_inputs = tuple(
        pipeline.policy_variant.prepare_train_inputs(visual, sample.batch)
        for visual, sample in zip(visual_outputs, samples, strict=True)
    )
    execution = pipeline.policy_variant.forward_train_batch(
        visual_tower=pipeline.visual_tower,
        visual_outputs=visual_outputs,
        prepared_inputs=prepared_inputs,
        batching_mode=batching_mode,
    )
    if isinstance(execution, PolicyTrainOutput):
        if execution.policy_features.shape[0] != len(samples):
            raise ValueError("Native batch output must preserve the original sample count.")
        return VariantPipelineTrainOutput(
            visual_outputs=None,
            policy_output=execution,
            decoder_output=pipeline.resolve_train_decoder_output(execution, batch),
        )
    policies = tuple(execution)
    if len(policies) != len(samples):
        raise ValueError(
            "Policy batch execution must return one output per original sequence."
        )
    sample_outputs = tuple(
        VariantPipelineTrainOutput(
            visual_outputs=visual,
            policy_output=policy,
            decoder_output=pipeline.resolve_train_decoder_output(
                policy, prepared.batch
            ),
        )
        for visual, policy, prepared in zip(
            visual_outputs, policies, prepared_inputs, strict=True
        )
    )
    action_capacity = int(batch.actions.shape[1])
    predictions = []
    for output in sample_outputs:
        prediction = output.decoder_output.action_pred
        if prediction.shape[0] != 1 or prediction.shape[1] > action_capacity:
            raise ValueError(
                "Decoder prediction does not fit the original collated action shape."
            )
        predictions.append(
            F.pad(prediction, (0, 0, 0, action_capacity - prediction.shape[1]))
        )
    decoder = ActionDecoderTrainOutput(
        action_pred=torch.cat(predictions, dim=0),
        loss=torch.stack(
            [output.decoder_output.loss for output in sample_outputs]
        ).mean(),
        metrics=_mean_metrics(
            tuple(output.decoder_output.metrics for output in sample_outputs)
        ),
        aux={"batching_mode": batching_mode.value, "sample_count": len(samples)},
    )
    policy = PolicyTrainOutput(
        policy_features=policies[0].policy_features.new_zeros(
            len(samples), 0, policies[0].policy_features.shape[-1]
        ),
        metrics=_mean_metrics(tuple(output.metrics for output in policies)),
        aux={
            "batching_mode": batching_mode.value,
            "sample_count": len(samples),
            "sequence_lengths": tuple(batch.extra["sequence_lengths"]),
        },
    )
    return VariantPipelineTrainOutput(
        visual_outputs=None,
        policy_output=policy,
        decoder_output=decoder,
        sample_outputs=sample_outputs,
    )
