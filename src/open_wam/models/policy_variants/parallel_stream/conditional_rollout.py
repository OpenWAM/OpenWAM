from __future__ import annotations

import torch

from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.contracts import (
    DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
)
from open_wam.models.common.dynamics_contracts import DynamicsRolloutGeometry

from .runtime_semantics import (
    prefix_visibility_mode_for_history_visibility,
    prefix_visibility_mode_for_policy,
)

__all__ = [
    "dynamics_rollout_prefix_visibility_mode",
    "select_dynamics_warmup_history_suffix",
    "slice_dynamics_conditioning_chunk",
    "uses_dynamics_mode_text_token",
]


def dynamics_rollout_prefix_visibility_mode(
    geometry: DynamicsRolloutGeometry,
    policy_config: ParallelStreamPolicyConfig,
) -> str:
    """Adapt shared history semantics to the parallel cache-prefix contract."""

    if (
        geometry.conditional_history_policy
        == DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY
    ):
        return prefix_visibility_mode_for_history_visibility(
            geometry.history_stream_visibility
        )
    if geometry.conditional_history_policy is not None:
        raise ValueError(
            "Parallel Stream does not implement conditional history policy "
            f"{geometry.conditional_history_policy!r}."
        )
    return prefix_visibility_mode_for_policy(policy_config)


def select_dynamics_warmup_history_suffix(
    *,
    video_latents: torch.Tensor,
    action_latents: torch.Tensor,
    frame_start: int,
    geometry: DynamicsRolloutGeometry,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Keep only the rollout-local history chunk for conditional warmup."""

    if geometry.conditional_history_policy is None:
        return video_latents, action_latents, int(frame_start), 0
    available_frames = min(
        int(video_latents.shape[2]),
        int(action_latents.shape[2]),
    )
    retained_frames = min(int(geometry.frame_chunk_size), available_frames)
    if retained_frames <= 0:
        return (
            video_latents[:, :, :0],
            action_latents[:, :, :0],
            int(frame_start),
            0,
        )
    dropped_frames = available_frames - retained_frames
    return (
        video_latents[:, :, -retained_frames:].contiguous(),
        action_latents[:, :, -retained_frames:].contiguous(),
        int(frame_start) + int(dropped_frames),
        int(dropped_frames),
    )


def slice_dynamics_conditioning_chunk(
    value: torch.Tensor | None,
    *,
    target_frames: int,
    source: str,
) -> torch.Tensor | None:
    """Select a leading conditioning chunk or reject insufficient history."""

    if value is None:
        return None
    observed_frames = int(value.shape[2])
    if observed_frames == target_frames:
        return value
    if observed_frames < target_frames:
        raise ValueError(
            f"{source} provides {observed_frames} frames but conditional dynamics "
            f"rollout needs {target_frames}."
        )
    return value[:, :, :target_frames].contiguous()


def uses_dynamics_mode_text_token(
    policy_config: ParallelStreamPolicyConfig,
) -> bool:
    """Return whether rollout text receives the learned dynamics mode token."""

    return policy_config.generalist_mode_text_token
