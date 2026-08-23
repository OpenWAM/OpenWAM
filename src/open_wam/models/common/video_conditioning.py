"""Canonical video-condition tensor selection for policy runtimes."""

from __future__ import annotations

import torch


def resolve_video_condition_latents(
    video_latents: torch.Tensor,
    condition_latents: object,
    *,
    enabled: bool,
    required: bool,
    label: str = "Video conditioning",
) -> torch.Tensor | None:
    """Validate optional condition latents against the canonical video layout."""

    _validate_video_latents(video_latents)
    if not enabled:
        return None
    if condition_latents is None:
        if required:
            raise ValueError(
                f"{label} requires `condition_latents`, but the latent batch did not provide them."
            )
        return None
    if not isinstance(condition_latents, torch.Tensor):
        raise TypeError(
            f"{label} `condition_latents` must be a tensor, "
            f"got {type(condition_latents).__name__}."
        )
    _validate_condition_latents(video_latents, condition_latents, label=label)
    return condition_latents.to(
        device=video_latents.device,
        dtype=video_latents.dtype,
    )


def select_first_frame_condition_latents(
    video_latents: torch.Tensor,
    *,
    condition_latents: torch.Tensor | None = None,
    label: str,
) -> tuple[torch.Tensor, str]:
    """Select one clean condition frame from canonical latent tensors."""

    _validate_video_latents(video_latents)
    if condition_latents is None:
        return video_latents[:, :, :1], "video_latents"
    _validate_condition_latents(video_latents, condition_latents, label=label)
    return (
        condition_latents[:, :, :1].to(
            device=video_latents.device,
            dtype=video_latents.dtype,
        ),
        "condition_latents",
    )


def resolve_full_window_condition_latents(
    video_latents: torch.Tensor,
    condition_latents: torch.Tensor | None,
    *,
    label: str,
) -> tuple[torch.Tensor | None, str]:
    """Resolve an optional clean condition tensor matching a full video window."""

    _validate_video_latents(video_latents)
    if condition_latents is None:
        return None, "video_latents"
    _validate_condition_latents(video_latents, condition_latents, label=label)
    if tuple(condition_latents.shape) != tuple(video_latents.shape):
        raise ValueError(
            f"{label} condition_latents must match video_latents exactly for "
            "full-window conditioning, "
            f"got condition={tuple(condition_latents.shape)}, "
            f"video={tuple(video_latents.shape)}."
        )
    return (
        condition_latents.to(
            device=video_latents.device,
            dtype=video_latents.dtype,
        ),
        "condition_latents",
    )


def build_repeated_first_frame_condition(
    video_latents: torch.Tensor,
    *,
    target_frames: int,
) -> torch.Tensor:
    """Repeat the anchor frame across a clean current-frame condition chunk."""

    first_frame_latents, _ = select_first_frame_condition_latents(
        video_latents,
        label="Current-frame action chunks",
    )
    target_frames = int(target_frames)
    if target_frames <= 0:
        raise ValueError(
            "Current-frame action chunks require positive target_frames, "
            f"got {target_frames}."
        )
    return first_frame_latents.repeat(1, 1, target_frames, 1, 1)


def _validate_video_latents(video_latents: torch.Tensor) -> None:
    if video_latents.ndim != 5:
        raise ValueError(
            "Expected video latents shaped [B, C, T, H, W], "
            f"got {tuple(video_latents.shape)}."
        )


def _validate_condition_latents(
    video_latents: torch.Tensor,
    condition_latents: torch.Tensor,
    *,
    label: str,
) -> None:
    if condition_latents.ndim != 5:
        raise ValueError(
            f"{label} condition_latents must have shape `[B, C, T, H, W]`, "
            f"got {tuple(condition_latents.shape)}."
        )
    if tuple(condition_latents.shape[:2]) != tuple(video_latents.shape[:2]):
        raise ValueError(
            f"{label} condition_latents batch/channel dimensions must match video_latents, "
            f"got condition={tuple(condition_latents.shape)}, "
            f"video={tuple(video_latents.shape)}."
        )
    if int(condition_latents.shape[2]) < 1:
        raise ValueError(
            f"{label} condition_latents must contain at least one latent frame."
        )
    if tuple(condition_latents.shape[-2:]) != tuple(video_latents.shape[-2:]):
        raise ValueError(
            f"{label} condition_latents spatial shape must match video_latents, "
            f"got condition={tuple(condition_latents.shape)}, "
            f"video={tuple(video_latents.shape)}."
        )


__all__ = [
    "build_repeated_first_frame_condition",
    "resolve_full_window_condition_latents",
    "resolve_video_condition_latents",
    "select_first_frame_condition_latents",
]
