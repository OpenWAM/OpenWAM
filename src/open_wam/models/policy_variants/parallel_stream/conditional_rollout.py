from __future__ import annotations

import torch

from open_wam.configs.enums import (
    JointDenoiseTrainingMode,
    ParallelHistoryStreamVisibility,
)
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.models.common.joint_conditioning import (
    generalist_joint_conditioning_chunk_size,
    generalist_joint_conditioning_window_size,
    is_conditional_joint_conditioning_mode,
)

from .runtime_semantics import (
    prefix_visibility_mode_for_policy,
    resolve_parallel_history_stream_visibility,
)

__all__ = [
    "generalist_conditioning_chunk_size",
    "generalist_conditioning_history_stream_visibility",
    "generalist_conditioning_prefix_visibility_mode",
    "generalist_conditioning_window_size",
    "is_conditional_joint_denoise_mode",
    "resolve_action_conditioning_mode",
    "select_conditional_warmup_history_suffix",
    "slice_conditioning_chunk",
    "uses_generalist_mode_text_token",
]


def resolve_action_conditioning_mode(
    action_conditioning_mode: JointDenoiseTrainingMode | str,
) -> JointDenoiseTrainingMode:
    """Map rollout-facing labels to the shared GJD training-mode enum."""

    raw_value = str(
        getattr(action_conditioning_mode, "value", action_conditioning_mode)
    )
    direct_values = {mode.value: mode for mode in JointDenoiseTrainingMode}
    if raw_value in direct_values:
        return direct_values[raw_value]
    aliases = {
        "joint": JointDenoiseTrainingMode.JOINT,
        "vanilla_joint_rollout": JointDenoiseTrainingMode.JOINT,
        "clean_action_feedback": JointDenoiseTrainingMode.JOINT,
        "fdm": JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        "forced_action_joint_fdm": (
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
        ),
        "idm": JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
    }
    try:
        return aliases[raw_value]
    except KeyError as exc:
        supported = ", ".join(sorted(set(direct_values) | set(aliases)))
        raise ValueError(
            f"Unsupported joint-denoise rollout mode {raw_value!r}. "
            f"Supported modes: {supported}."
        ) from exc


def is_conditional_joint_denoise_mode(
    mode: JointDenoiseTrainingMode | str,
) -> bool:
    """Return whether a rollout is conditional FDM or IDM rather than joint."""

    return is_conditional_joint_conditioning_mode(
        mode,
        joint_mode=JointDenoiseTrainingMode.JOINT,
        action_conditioned_video_mode=(
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
        ),
        video_conditioned_action_mode=(
            JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION
        ),
    )


def generalist_conditioning_window_size(
    mode: JointDenoiseTrainingMode | str,
    *,
    fallback_window_size: int,
) -> int:
    """Resolve the local exact-attention window for one rollout mode."""

    return generalist_joint_conditioning_window_size(
        mode,
        joint_mode=JointDenoiseTrainingMode.JOINT,
        action_conditioned_video_mode=(
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
        ),
        video_conditioned_action_mode=(
            JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION
        ),
        fallback_window_size=fallback_window_size,
    )


def generalist_conditioning_chunk_size(
    mode: JointDenoiseTrainingMode | str,
    *,
    fallback_chunk_size: int,
) -> int:
    """Resolve the exact-runtime chunk size for one rollout mode."""

    return generalist_joint_conditioning_chunk_size(
        mode,
        joint_mode=JointDenoiseTrainingMode.JOINT,
        action_conditioned_video_mode=(
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
        ),
        video_conditioned_action_mode=(
            JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION
        ),
        fallback_chunk_size=fallback_chunk_size,
    )


def generalist_conditioning_history_stream_visibility(
    mode: JointDenoiseTrainingMode | str,
    policy_config: ParallelStreamPolicyConfig,
) -> ParallelHistoryStreamVisibility:
    """Restrict conditional rollouts to the latest clean video history."""

    if is_conditional_joint_denoise_mode(mode):
        return ParallelHistoryStreamVisibility.VIDEO_ONLY
    return resolve_parallel_history_stream_visibility(policy_config)


def generalist_conditioning_prefix_visibility_mode(
    mode: JointDenoiseTrainingMode | str,
    policy_config: ParallelStreamPolicyConfig,
) -> str:
    """Map conditional history semantics to the cache-prefix contract."""

    if is_conditional_joint_denoise_mode(mode):
        return "video_history_only"
    return prefix_visibility_mode_for_policy(policy_config)


def select_conditional_warmup_history_suffix(
    *,
    video_latents: torch.Tensor,
    action_latents: torch.Tensor,
    frame_start: int,
    frame_chunk_size: int,
    mode: JointDenoiseTrainingMode | str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Keep only the rollout-local history chunk for conditional warmup."""

    if not is_conditional_joint_denoise_mode(mode):
        return video_latents, action_latents, int(frame_start), 0
    available_frames = min(
        int(video_latents.shape[2]),
        int(action_latents.shape[2]),
    )
    retained_frames = min(max(1, int(frame_chunk_size)), available_frames)
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


def slice_conditioning_chunk(
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
            f"{source} provides {observed_frames} frames but conditional GJD "
            f"rollout needs {target_frames}."
        )
    return value[:, :, :target_frames].contiguous()


def uses_generalist_mode_text_token(
    policy_config: ParallelStreamPolicyConfig,
) -> bool:
    """Return whether rollout text receives the learned GJD mode token."""

    return bool(getattr(policy_config, "generalist_mode_text_token", False))
