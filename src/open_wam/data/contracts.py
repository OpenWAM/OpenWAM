from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class WAMSample:
    """One dataset sample before collation.

    Attributes:
        views:
            Raw RGB videos per camera view. Each tensor is `[T, H, W, 3]`.
        actions:
            Action targets aligned to the sample anchor, `[H_action, D_action]`.
            The exact representation is dataset-configured: it may be the raw
            dataset action, a reference-relative EEF pose target, or another
            transformed control target exposed by the data layer.
        action_mask:
            Valid action dimensions for padded schemas, same shape as `actions`.
        state:
            Optional state history aligned to the current anchor, `[H_state, D_state]`.
        state_mask:
            Valid state dimensions for padded schemas, same shape as `state`.
        task_text:
            Natural-language task instruction for this window.
        metadata:
            Per-sample bookkeeping kept outside the protected backbone.
    """

    views: dict[str, torch.Tensor]
    actions: torch.Tensor
    action_mask: torch.Tensor | None = None
    state: torch.Tensor | None = None
    state_mask: torch.Tensor | None = None
    task_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WAMBatch:
    """Uniform batch contract returned by all dataset adapters.

    This is the common data-layer artifact that future head variants should
    consume through the Lightning module and pipeline. The batch is intentionally
    independent from any specific action-head placement strategy.
    """

    views: dict[str, torch.Tensor]
    actions: torch.Tensor
    action_mask: torch.Tensor | None = None
    state: torch.Tensor | None = None
    state_mask: torch.Tensor | None = None
    task_text: tuple[str | None, ...] | None = None
    metadata: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def collate_wam_samples(samples: list[WAMSample]) -> WAMBatch:
    """Collate uniform samples into one batch.

    View tensors remain grouped by camera name so the canonicalizer can keep
    dataset-specific layout decisions outside the model code.
    """

    if not samples:
        raise ValueError("Cannot collate an empty WAM sample list.")

    view_names = tuple(samples[0].views.keys())
    views = {
        view_name: torch.stack([sample.views[view_name] for sample in samples], dim=0)
        for view_name in view_names
    }
    actions = torch.stack([sample.actions for sample in samples], dim=0)
    action_mask = _stack_optional_tensor(samples, "action_mask")
    state = _stack_optional_tensor(samples, "state")
    state_mask = _stack_optional_tensor(samples, "state_mask")
    task_text = tuple(sample.task_text for sample in samples)
    metadata = tuple(sample.metadata for sample in samples)
    return WAMBatch(
        views=views,
        actions=actions,
        action_mask=action_mask,
        state=state,
        state_mask=state_mask,
        task_text=task_text,
        metadata=metadata,
    )


def move_wam_batch_to_device(batch: WAMBatch, device: torch.device | str) -> WAMBatch:
    """Move all tensor fields in a uniform batch to the target device."""

    return WAMBatch(
        views={name: value.to(device) for name, value in batch.views.items()},
        actions=batch.actions.to(device),
        action_mask=batch.action_mask.to(device) if batch.action_mask is not None else None,
        state=batch.state.to(device) if batch.state is not None else None,
        state_mask=batch.state_mask.to(device) if batch.state_mask is not None else None,
        task_text=batch.task_text,
        metadata=batch.metadata,
    )


def _stack_optional_tensor(samples: list[WAMSample], field_name: str) -> torch.Tensor | None:
    values = [getattr(sample, field_name) for sample in samples]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Inconsistent optional field '{field_name}' across the batch.")
    return torch.stack(values, dim=0)  # type: ignore[arg-type]
