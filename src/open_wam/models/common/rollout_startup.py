from __future__ import annotations

from dataclasses import dataclass

import torch


STRICT_STARTUP_DEPRECATION_MESSAGE = (
    "Exact realtime startup with generation_frame_start < 1 is deprecated because generated or "
    "synthetic frame-0 actions can be recorded as valid action context. Use the strict frame-0 "
    "condition -> frames 1..4 execution contract instead."
)


@dataclass(frozen=True)
class StrictStartupPlan:
    """Shared rollout-startup geometry for video-prefix/action-generation chunks."""

    step_index: int
    current_start_frame: int
    frame_chunk_size: int
    action_tokens_per_frame: int
    action_horizon: int
    video_prefix_frames: int
    generation_frame_start: int
    action_prefix_tokens: int
    current_action_sequence_tokens: int

    @property
    def is_startup(self) -> bool:
        return self.video_prefix_frames > 0

    def chunk_origin_frame(self, history_frames: int) -> int:
        return int(history_frames) + int(self.video_prefix_frames)


def resolve_strict_startup_plan(
    *,
    step_index: int,
    current_start_frame: int,
    frame_chunk_size: int,
    action_tokens_per_frame: int,
    action_horizon: int,
) -> StrictStartupPlan:
    """Resolve strict startup geometry without changing rollout semantics.

    Startup is exactly the existing contract: only the first call at frame 0
    receives a one-frame video prefix, and action generation begins at frame 1.
    Later calls use no prefix and keep their cursor-provided start frame.
    """

    step_index = int(step_index)
    current_start_frame = int(current_start_frame)
    frame_chunk_size = int(frame_chunk_size)
    action_tokens_per_frame = int(action_tokens_per_frame)
    action_horizon = int(action_horizon)
    if step_index < 0:
        raise ValueError(f"Expected step_index >= 0, got {step_index}.")
    if current_start_frame < 0:
        raise ValueError(f"Expected current_start_frame >= 0, got {current_start_frame}.")
    if frame_chunk_size <= 0:
        raise ValueError(f"Expected frame_chunk_size > 0, got {frame_chunk_size}.")
    if action_tokens_per_frame <= 0:
        raise ValueError(f"Expected action_tokens_per_frame > 0, got {action_tokens_per_frame}.")
    if action_horizon <= 0:
        raise ValueError(f"Expected action_horizon > 0, got {action_horizon}.")

    video_prefix_frames = 1 if step_index == 0 and current_start_frame == 0 else 0
    generation_frame_start = current_start_frame + video_prefix_frames
    action_prefix_tokens = video_prefix_frames * action_tokens_per_frame
    current_action_sequence_tokens = action_prefix_tokens + action_horizon
    return StrictStartupPlan(
        step_index=step_index,
        current_start_frame=current_start_frame,
        frame_chunk_size=frame_chunk_size,
        action_tokens_per_frame=action_tokens_per_frame,
        action_horizon=action_horizon,
        video_prefix_frames=video_prefix_frames,
        generation_frame_start=generation_frame_start,
        action_prefix_tokens=action_prefix_tokens,
        current_action_sequence_tokens=current_action_sequence_tokens,
    )


def build_strict_action_context_mask(
    *,
    batch_size: int,
    history_action_tokens: int,
    current_action_sequence_tokens: int,
    invalid_current_prefix_tokens: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an action-context validity mask for packed rollout attention.

    History actions stay valid. The current chunk's startup action prefix is
    materialized only to keep token alignment with the observed video prefix,
    so those tokens are hidden from attention.
    """

    batch_size = int(batch_size)
    history_action_tokens = int(history_action_tokens)
    current_action_sequence_tokens = int(current_action_sequence_tokens)
    invalid_current_prefix_tokens = int(invalid_current_prefix_tokens)
    if batch_size <= 0:
        raise ValueError(f"Expected batch_size > 0, got {batch_size}.")
    if history_action_tokens < 0:
        raise ValueError(f"Expected history_action_tokens >= 0, got {history_action_tokens}.")
    if current_action_sequence_tokens <= 0:
        raise ValueError(
            f"Expected current_action_sequence_tokens > 0, got {current_action_sequence_tokens}."
        )
    if invalid_current_prefix_tokens < 0:
        raise ValueError(
            f"Expected invalid_current_prefix_tokens >= 0, got {invalid_current_prefix_tokens}."
        )
    if invalid_current_prefix_tokens > current_action_sequence_tokens:
        raise ValueError(
            "Invalid startup action prefix cannot exceed the current action sequence, "
            f"got invalid_current_prefix_tokens={invalid_current_prefix_tokens}, "
            f"current_action_sequence_tokens={current_action_sequence_tokens}."
        )

    mask = torch.ones(
        batch_size,
        history_action_tokens + current_action_sequence_tokens,
        1,
        device=device,
        dtype=dtype,
    )
    if invalid_current_prefix_tokens > 0:
        start = history_action_tokens
        mask[:, start : start + invalid_current_prefix_tokens] = 0.0
    return mask


def require_strict_startup_generation_frame(generation_frame_start: int) -> None:
    if int(generation_frame_start) < 1:
        raise ValueError(STRICT_STARTUP_DEPRECATION_MESSAGE)


def strict_startup_conditioning_frame_index(generation_frame_start: int) -> int:
    require_strict_startup_generation_frame(generation_frame_start)
    return int(generation_frame_start) - 1
