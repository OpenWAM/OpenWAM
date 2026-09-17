"""Architecture-independent preparation and publication of video/action chunks."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from open_wam.models.common.dynamics_objectives import DynamicsRolloutPlan
from open_wam.models.common.temporal_windows import resolve_interleaved_history_frames
from open_wam.models.common.video_action_state import VideoActionRolloutState
from open_wam.models.policy_variants.contracts import (
    PolicyGeneratedVideo,
    PolicyInferState,
)


def select_action_chunk(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    action_dim: int,
    frame_count: int,
    actions_per_frame: int,
    label: str,
) -> torch.Tensor | None:
    """Validate model-space controls before selecting a complete future chunk."""
    if value is None:
        return None
    if value.ndim != 3 or value.shape[0] != batch_size or value.shape[-1] != action_dim:
        raise ValueError(
            f"Dynamics rollout {label!r} must have shape [B, H, D] with "
            f"batch={batch_size}, action_dim={action_dim}; got {tuple(value.shape)}."
        )
    horizon = frame_count * actions_per_frame
    if value.shape[1] < horizon or value.shape[1] % actions_per_frame:
        raise ValueError(
            f"Dynamics rollout {label!r} must be frame-aligned to "
            f"actions_per_frame={actions_per_frame} and contain at least {horizon} "
            f"action steps; got {value.shape[1]}."
        )
    return value[:, :horizon].contiguous()


@dataclass(frozen=True)
class VideoActionSequence:
    """Aligned model-space streams; token packing belongs to the model adapter."""

    video: torch.Tensor
    action: torch.Tensor
    action_mask: torch.Tensor
    proprio: torch.Tensor | None
    frame_start: int
    history_frames: int
    startup_frames: int
    chunk_frames: int
    actions_per_frame: int
    max_history_frames: int
    forced_action: torch.Tensor | None
    commit_action: torch.Tensor | None

    @property
    def generation_start(self) -> int:
        return self.frame_start + self.history_frames

    @property
    def history_action_tokens(self) -> int:
        return self.history_frames * self.actions_per_frame

    @property
    def retention_frames(self) -> int:
        return self.max_history_frames + self.chunk_frames

    def publish(
        self,
        state: PolicyInferState,
        *,
        video: torch.Tensor,
        action: torch.Tensor,
        retain_future_video: bool,
        retain_future_action: bool,
    ) -> PolicyInferState:
        """Publish new semantic history without modifying the input snapshot."""
        runtime = state.variant_state or VideoActionRolloutState()
        observed_frame_end = (
            self.generation_start
            if runtime.past_clean_latents is None
            else state.observed_frame_end
        )
        video_end = self.history_frames + (
            self.chunk_frames if retain_future_video else 0
        )
        committed = action.clone()
        mask = self.action_mask.clone()
        if self.commit_action is not None:
            committed[:, self.history_action_tokens :] = self.commit_action
        if not retain_future_action and self.forced_action is None:
            # The startup placeholder and ungenerated controls are never history.
            start = (self.history_frames - self.startup_frames) * self.actions_per_frame
            committed[:, start:] = 0
            mask[:, start:] = 0
        retention = self.retention_frames
        runtime = replace(
            runtime,
            past_clean_latents=video[:, :, :video_end][:, :, -retention:].detach(),
            past_clean_actions=committed[
                :, -retention * self.actions_per_frame :
            ].detach(),
            past_clean_action_mask=mask[
                :, -retention * self.actions_per_frame :
            ].detach(),
            past_hidden_proprio_states=(
                None
                if self.proprio is None
                else self.proprio[:, :video_end][:, -retention:].detach()
            ),
            pending_predicted_video_frames=self.chunk_frames
            if retain_future_video
            else 0,
            chunk_advance_frames=self.chunk_frames,
        )
        return replace(
            state,
            observed_frame_end=observed_frame_end,
            cursor=replace(
                state.cursor,
                current_start_frame=self.generation_start + self.chunk_frames,
                block_index=state.step_index + 1,
                chunk_size=self.chunk_frames,
            ),
            variant_state=runtime,
            revision=state.revision + 1,
        )


def prepare_video_action_sequence(
    *,
    observation: torch.Tensor,
    state: PolicyInferState,
    chunk_frames: int,
    actions_per_frame: int,
    action_dim: int,
    window_size: int,
    dynamics: DynamicsRolloutPlan,
    supplied_video: PolicyGeneratedVideo | None = None,
) -> VideoActionSequence:
    """Resolve t0, history bounds, action validity and future inputs once.

    The observed startup frame has no reaching action. Conditional objectives
    retain one boundary frame, regardless of the planning history in the session.
    No random draws or backbone operations occur here.
    """
    if observation.ndim != 5 or observation.shape[2] < 1:
        raise ValueError("Observation must be nonempty [B, C, T, H, W].")
    if min(chunk_frames, actions_per_frame, action_dim) <= 0:
        raise ValueError("Video/action geometry must be positive.")
    dynamics.require_generation_inputs()
    runtime = state.variant_state or VideoActionRolloutState()
    runtime.require_complete_video_history()
    batch, channels, _, height, width = observation.shape
    video = runtime.past_clean_latents
    actions = runtime.past_clean_actions
    mask = runtime.past_clean_action_mask
    proprio = runtime.past_hidden_proprio_states
    latest_proprio = runtime.hidden_proprio_state
    startup = int(video is None)
    generation_start = state.cursor.current_start_frame + startup
    keep = (
        1
        if dynamics.semantics.is_conditional
        else resolve_interleaved_history_frames(
            window_size=window_size,
            frame_chunk_size=chunk_frames,
        )
    )
    if startup:
        video = observation[:, :, -1:]
        actions = observation.new_zeros(batch, actions_per_frame, action_dim)
        mask = observation.new_zeros(batch, actions_per_frame, 1)
        proprio = (
            None if latest_proprio is None else latest_proprio[:, None].to(observation)
        )
    else:
        if video.ndim != 5 or (video.shape[0], video.shape[1], *video.shape[-2:]) != (
            batch,
            channels,
            height,
            width,
        ):
            raise ValueError(
                "Observed video geometry must match recurrent video history."
            )
        if actions is None or actions.shape != (
            batch,
            video.shape[2] * actions_per_frame,
            action_dim,
        ):
            raise ValueError(
                "Recurrent actions must be frame-aligned with video history."
            )
        video, actions = (
            video[:, :, -keep:].to(observation),
            actions[:, -keep * actions_per_frame :].to(observation),
        )
        if mask is None:
            mask = actions.new_ones((*actions.shape[:2], 1))
        else:
            mask = mask[:, -actions.shape[1] :].to(observation)
        if proprio is not None:
            proprio = proprio[:, -video.shape[2] :].to(observation)
    history_frames = video.shape[2]
    if latest_proprio is not None:
        if proprio is None or proprio.shape[:2] != (batch, history_frames):
            raise ValueError("Per-chunk proprio requires aligned observation history.")
        proprio = torch.cat(
            [
                proprio,
                latest_proprio[:, None].to(observation).expand(-1, chunk_frames, -1),
            ],
            dim=1,
        )
    forced = select_action_chunk(
        dynamics.clean_action,
        batch_size=batch,
        action_dim=action_dim,
        frame_count=chunk_frames,
        actions_per_frame=actions_per_frame,
        label="clean_action",
    )
    committed = select_action_chunk(
        dynamics.history_action,
        batch_size=batch,
        action_dim=action_dim,
        frame_count=chunk_frames,
        actions_per_frame=actions_per_frame,
        label="history_action",
    )
    expected = (batch, channels, chunk_frames, height, width)
    future_video = observation.new_zeros(expected)
    if supplied_video is not None:
        if (
            supplied_video.frame_start != generation_start
            or supplied_video.latents.shape != expected
        ):
            raise ValueError(
                "Supplied video must match the consumer's generated frame span and latent geometry."
            )
        future_video = supplied_video.latents.to(observation)
    if dynamics.clean_video is not None:
        future_video = dynamics.clean_video[:, :, :chunk_frames].to(observation)
        if future_video.shape != expected:
            raise ValueError(
                "Conditional video must match the generated chunk geometry."
            )
    future_action = (
        observation.new_zeros(batch, chunk_frames * actions_per_frame, action_dim)
        if forced is None
        else forced.to(observation)
    )
    return VideoActionSequence(
        video=torch.cat([video, future_video], dim=2),
        action=torch.cat([actions, future_action], dim=1),
        action_mask=torch.cat(
            [mask, future_action.new_ones((*future_action.shape[:2], 1))], dim=1
        ),
        proprio=proprio,
        frame_start=generation_start - history_frames,
        history_frames=history_frames,
        startup_frames=startup,
        chunk_frames=chunk_frames,
        actions_per_frame=actions_per_frame,
        max_history_frames=keep,
        forced_action=None if forced is None else forced.to(observation),
        commit_action=None if committed is None else committed.to(observation),
    )
