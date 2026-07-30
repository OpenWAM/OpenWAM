from __future__ import annotations

import torch


def select_first_frame_condition_latents(
    video_latents: torch.Tensor,
    *,
    condition_latents: torch.Tensor | None = None,
    label: str,
) -> tuple[torch.Tensor, str]:
    """Select one clean condition frame for a parallel-stream runtime."""

    if video_latents.ndim != 5:
        raise ValueError(
            f"Expected video latents shaped [B, C, F, H, W], got {tuple(video_latents.shape)}."
        )
    if condition_latents is None:
        return video_latents[:, :, :1], "video_latents"
    if condition_latents.ndim != 5:
        raise ValueError(
            f"{label} condition_latents must have shape `[B, C, T, H, W]`, "
            f"got {tuple(condition_latents.shape)}."
        )
    expected_prefix = (video_latents.shape[0], video_latents.shape[1])
    if tuple(condition_latents.shape[:2]) != expected_prefix:
        raise ValueError(
            f"{label} condition_latents batch/channel dimensions must match video_latents, "
            f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
        )
    if condition_latents.shape[2] < 1:
        raise ValueError(
            f"{label} condition_latents must contain at least one latent frame."
        )
    if tuple(condition_latents.shape[-2:]) != tuple(video_latents.shape[-2:]):
        raise ValueError(
            f"{label} condition_latents spatial shape must match video_latents, "
            f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
        )
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

    if condition_latents is None:
        return None, "video_latents"
    if condition_latents.ndim != 5:
        raise ValueError(
            f"{label} condition_latents must have shape `[B, C, T, H, W]`, "
            f"got {tuple(condition_latents.shape)}."
        )
    if tuple(condition_latents.shape) != tuple(video_latents.shape):
        raise ValueError(
            f"{label} condition_latents must match video_latents exactly for full-window conditioning, "
            f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
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


__all__ = [
    "build_repeated_first_frame_condition",
    "resolve_full_window_condition_latents",
    "select_first_frame_condition_latents",
]
