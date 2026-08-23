"""Canonical proprio-context selection for policy runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch

from .chunked_attention_visibility import (
    align_frame_context_to_previous_chunk_boundary,
)


class ProprioContextGranularity(StrEnum):
    """Sampling granularity of a hidden-state proprio sequence."""

    FRAME = "frame"
    CHUNK = "chunk"


@dataclass(frozen=True, slots=True)
class HiddenProprioContext:
    """Validated hidden-state proprio values and their source granularity."""

    values: torch.Tensor
    granularity: ProprioContextGranularity

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError(
                "Hidden proprio context values must be a tensor, "
                f"got {type(self.values).__name__}."
            )
        object.__setattr__(
            self,
            "granularity",
            ProprioContextGranularity(self.granularity),
        )


def select_latest_proprio_state(
    state: torch.Tensor | None,
) -> torch.Tensor | None:
    """Select the latest state from either anchor or history layout."""

    if state is None or state.ndim == 2:
        return state
    if state.ndim == 3:
        return state[:, -1, :]
    raise ValueError(
        "Proprio context expects [B, state_dim] or [B, T, state_dim], "
        f"got {tuple(state.shape)}."
    )


def resolve_hidden_proprio_context(
    extra: Mapping[str, Any],
    *,
    require_frame_aligned: bool,
    label: str,
) -> HiddenProprioContext:
    """Resolve and mask the canonical frame- or chunk-level proprio payload."""

    value = _optional_tensor(extra, "proprio_context_frames", label=label)
    mask = _optional_tensor(extra, "proprio_context_frames_mask", label=label)
    granularity = ProprioContextGranularity.FRAME
    if value is None:
        value = _optional_tensor(extra, "proprio_context_state", label=label)
        mask = _optional_tensor(extra, "proprio_context_state_mask", label=label)
        granularity = ProprioContextGranularity.CHUNK
        if require_frame_aligned and value is not None:
            raise ValueError(
                "The selected sequence contract requires frame-level "
                "`proprio_context_frames`; chunk-level `proprio_context_state` "
                "cannot be safely aligned to model-visible frame and chunk boundaries."
            )
    if value is None:
        raise ValueError(
            "proprio_context_mode=per_chunk_additive requires frame- or "
            f"chunk-level proprio context for {label}."
        )
    if value.ndim != 3:
        raise ValueError(
            "Hidden proprio context expects shape [B, frames_or_chunks, state_dim], "
            f"got {tuple(value.shape)} for {label}."
        )
    if mask is not None:
        if tuple(mask.shape) != tuple(value.shape):
            raise ValueError(
                "Hidden proprio context mask must match the context tensor shape, "
                f"got mask={tuple(mask.shape)}, state={tuple(value.shape)} for {label}."
            )
        value = value * mask.to(device=value.device, dtype=value.dtype)
    return HiddenProprioContext(values=value, granularity=granularity)


def prepend_hidden_proprio_context(
    context: HiddenProprioContext,
    *,
    prefix_state: torch.Tensor | None,
    target_frame_count: int,
    label: str,
) -> HiddenProprioContext:
    """Prepend the state paired with an external video-condition frame.

    Frame-aligned payloads are trimmed to the target sequence before the
    prefix is added. Chunk-aligned payloads already represent target chunks,
    so their complete sequence is preserved.
    """

    selected_prefix = select_latest_proprio_state(prefix_state)
    if selected_prefix is None:
        raise ValueError(f"{label} requires state for the condition frame.")
    values = context.values
    if values.ndim != 3:
        raise ValueError(
            f"{label} expects hidden proprio shape [B, frames_or_chunks, state_dim], "
            f"got {tuple(values.shape)}."
        )
    if selected_prefix.ndim != 2:
        raise ValueError(
            f"{label} expects prefix state shape [B, state_dim], "
            f"got {tuple(selected_prefix.shape)}."
        )
    if tuple(selected_prefix.shape) != (
        int(values.shape[0]),
        int(values.shape[2]),
    ):
        raise ValueError(
            f"{label} prefix state must match hidden proprio batch/state dimensions, "
            f"got prefix={tuple(selected_prefix.shape)}, context={tuple(values.shape)}."
        )

    target_frames = int(target_frame_count)
    if target_frames < 0:
        raise ValueError(
            f"{label} requires target_frame_count >= 0, got {target_frames}."
        )
    if context.granularity is ProprioContextGranularity.FRAME:
        if int(values.shape[1]) < target_frames:
            raise ValueError(
                f"{label} expects at least one state per target frame, "
                f"got context={tuple(values.shape)}, target_frames={target_frames}."
            )
        target_values = values[:, :target_frames, :]
    else:
        target_values = values

    prefixed_values = torch.cat(
        [
            selected_prefix[:, None, :].to(
                device=values.device,
                dtype=values.dtype,
            ),
            target_values,
        ],
        dim=1,
    )
    return HiddenProprioContext(
        values=prefixed_values,
        granularity=context.granularity,
    )


def project_hidden_proprio_context_to_frames(
    context: HiddenProprioContext,
    *,
    num_frames: int,
    chunk_size: int,
    chunk_origin_frame: int = 0,
    prefix_frames: int = 0,
) -> torch.Tensor:
    """Project frame- or chunk-sampled state onto model-visible frames.

    Frame-sampled state is anchored at the immediately preceding chunk
    boundary. Chunk-sampled state is already boundary-aligned by the data
    adapter and is expanded over the matching target chunk. A separately
    prepended condition frame remains its own prefix entry.
    """

    values = context.values
    if values.ndim != 3:
        raise ValueError(
            "Hidden proprio context expects shape "
            "[B, frames_or_chunks, state_dim], "
            f"got {tuple(values.shape)}."
        )
    frame_count = int(num_frames)
    if frame_count < 0:
        raise ValueError(
            f"Hidden proprio projection requires num_frames >= 0, got {frame_count}."
        )
    prefix_count = int(prefix_frames)
    if prefix_count < 0 or prefix_count > frame_count:
        raise ValueError(
            "Hidden proprio projection requires 0 <= prefix_frames <= num_frames, "
            f"got prefix_frames={prefix_count}, num_frames={frame_count}."
        )
    state_count = int(values.shape[1])
    if state_count <= 0 and frame_count > 0:
        raise ValueError(
            "Hidden proprio projection requires at least one source state."
        )
    if frame_count == 0:
        return values[:, :0, :]

    resolved_chunk_size = max(1, int(chunk_size))
    chunk_origin = int(chunk_origin_frame)
    target_frame_count = frame_count - prefix_count
    target_frame_ids = torch.arange(
        target_frame_count,
        device=values.device,
        dtype=torch.long,
    )

    if context.granularity is ProprioContextGranularity.FRAME:
        required_states = prefix_count + target_frame_count
        if state_count < required_states:
            raise ValueError(
                "Frame-level hidden proprio context is too short for the model "
                f"sequence: state_count={state_count}, required_states={required_states}."
            )
        if prefix_count == 0:
            return align_frame_context_to_previous_chunk_boundary(
                values,
                num_frames=frame_count,
                chunk_origin_frame=chunk_origin,
                chunk_size=resolved_chunk_size,
            )
        target_boundary_ids = (
            torch.div(
                target_frame_ids - chunk_origin,
                resolved_chunk_size,
                rounding_mode="floor",
            )
            * resolved_chunk_size
            + chunk_origin
            - 1
            + prefix_count
        ).clamp(min=0, max=state_count - 1)
        target_context = values.index_select(dim=1, index=target_boundary_ids)
    else:
        target_chunk_ids = torch.div(
            (target_frame_ids - chunk_origin).clamp_min(0),
            resolved_chunk_size,
            rounding_mode="floor",
        )
        target_source_ids = target_chunk_ids + prefix_count
        required_states = (
            prefix_count
            if target_source_ids.numel() == 0
            else int(target_source_ids.max().item()) + 1
        )
        if state_count < required_states:
            raise ValueError(
                "Chunk-level hidden proprio context is too short for the model "
                f"sequence: state_count={state_count}, required_states={required_states}."
            )
        target_context = values.index_select(dim=1, index=target_source_ids)

    if prefix_count == 0:
        return target_context
    return torch.cat([values[:, :prefix_count, :], target_context], dim=1)


def _optional_tensor(
    values: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> torch.Tensor | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Hidden proprio `{key}` must be a tensor for {label}, "
            f"got {type(value).__name__}."
        )
    return value


__all__ = [
    "HiddenProprioContext",
    "ProprioContextGranularity",
    "prepend_hidden_proprio_context",
    "project_hidden_proprio_context_to_frames",
    "resolve_hidden_proprio_context",
    "select_latest_proprio_state",
]
