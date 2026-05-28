from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LatentWAMSample:
    """One latent-first dataset sample before collation."""

    video_latents: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor | None = None
    state: torch.Tensor | None = None
    state_mask: torch.Tensor | None = None
    task_text: str | None = None
    text_context: torch.Tensor | None = None
    negative_text_context: torch.Tensor | None = None
    canonical_video: torch.Tensor | None = None
    condition_latents: torch.Tensor | None = None
    proprio_context_state: torch.Tensor | None = None
    proprio_context_state_mask: torch.Tensor | None = None
    proprio_context_frames: torch.Tensor | None = None
    proprio_context_frames_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatentWAMBatch:
    """Uniform latent-first batch contract returned by latent dataset adapters."""

    video_latents: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor | None = None
    state: torch.Tensor | None = None
    state_mask: torch.Tensor | None = None
    task_text: tuple[str | None, ...] | None = None
    text_context: torch.Tensor | None = None
    negative_text_context: torch.Tensor | None = None
    canonical_video: torch.Tensor | None = None
    condition_latents: torch.Tensor | None = None
    proprio_context_state: torch.Tensor | None = None
    proprio_context_state_mask: torch.Tensor | None = None
    proprio_context_frames: torch.Tensor | None = None
    proprio_context_frames_mask: torch.Tensor | None = None
    metadata: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def collate_latent_wam_samples(samples: list[LatentWAMSample]) -> LatentWAMBatch:
    """Collate latent-first samples into one batch."""

    if not samples:
        raise ValueError("Cannot collate an empty latent WAM sample list.")

    video_latents = torch.stack([sample.video_latents for sample in samples], dim=0)
    actions = torch.stack([sample.actions for sample in samples], dim=0)
    action_mask = _stack_optional_tensor(samples, "action_mask")
    state = _stack_optional_tensor(samples, "state")
    state_mask = _stack_optional_tensor(samples, "state_mask")
    text_context = _stack_optional_tensor(samples, "text_context")
    negative_text_context = _stack_optional_tensor(samples, "negative_text_context")
    canonical_video = _stack_optional_tensor(samples, "canonical_video")
    condition_latents = _stack_optional_tensor(samples, "condition_latents")
    proprio_context_state = _stack_optional_tensor(samples, "proprio_context_state")
    proprio_context_state_mask = _stack_optional_tensor(samples, "proprio_context_state_mask")
    proprio_context_frames = _stack_optional_tensor(samples, "proprio_context_frames")
    proprio_context_frames_mask = _stack_optional_tensor(samples, "proprio_context_frames_mask")
    task_text = tuple(sample.task_text for sample in samples)
    metadata = tuple(_metadata_with_action_stats(sample) for sample in samples)
    return LatentWAMBatch(
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        state=state,
        state_mask=state_mask,
        task_text=task_text,
        text_context=text_context,
        negative_text_context=negative_text_context,
        canonical_video=canonical_video,
        condition_latents=condition_latents,
        proprio_context_state=proprio_context_state,
        proprio_context_state_mask=proprio_context_state_mask,
        proprio_context_frames=proprio_context_frames,
        proprio_context_frames_mask=proprio_context_frames_mask,
        metadata=metadata,
    )


def move_latent_wam_batch_to_device(
    batch: LatentWAMBatch,
    device: torch.device | str,
) -> LatentWAMBatch:
    """Move all tensor fields in a latent batch to the target device."""

    return LatentWAMBatch(
        video_latents=batch.video_latents.to(device),
        actions=batch.actions.to(device),
        action_mask=batch.action_mask.to(device) if batch.action_mask is not None else None,
        state=batch.state.to(device) if batch.state is not None else None,
        state_mask=batch.state_mask.to(device) if batch.state_mask is not None else None,
        task_text=batch.task_text,
        text_context=batch.text_context.to(device) if batch.text_context is not None else None,
        negative_text_context=(
            batch.negative_text_context.to(device) if batch.negative_text_context is not None else None
        ),
        canonical_video=batch.canonical_video.to(device) if batch.canonical_video is not None else None,
        condition_latents=batch.condition_latents.to(device) if batch.condition_latents is not None else None,
        proprio_context_state=(
            batch.proprio_context_state.to(device) if batch.proprio_context_state is not None else None
        ),
        proprio_context_state_mask=(
            batch.proprio_context_state_mask.to(device) if batch.proprio_context_state_mask is not None else None
        ),
        proprio_context_frames=(
            batch.proprio_context_frames.to(device) if batch.proprio_context_frames is not None else None
        ),
        proprio_context_frames_mask=(
            batch.proprio_context_frames_mask.to(device) if batch.proprio_context_frames_mask is not None else None
        ),
        metadata=batch.metadata,
    )


def _stack_optional_tensor(samples: list[LatentWAMSample], field_name: str) -> torch.Tensor | None:
    values = [getattr(sample, field_name) for sample in samples]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Inconsistent optional field '{field_name}' across the latent batch.")
    return torch.stack(values, dim=0)  # type: ignore[arg-type]


def _metadata_with_action_stats(sample: LatentWAMSample) -> dict[str, Any]:
    metadata = dict(sample.metadata)
    action_mask = sample.action_mask
    if action_mask is None:
        valid_steps = int(sample.actions.shape[0])
        valid_values = int(sample.actions.numel())
    else:
        reduced = action_mask.float().sum(dim=-1)
        valid_steps = int((reduced > 0).sum().item())
        valid_values = int(action_mask.float().sum().item())
    metadata.setdefault("valid_action_steps", valid_steps)
    metadata.setdefault("valid_action_values", valid_values)
    return metadata
