"""Reconcile model-space speculative history with executed observations."""

from __future__ import annotations

from dataclasses import replace

import torch

from open_wam.models.policy_variants.contracts import (
    PolicyInferState,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
    PolicyTemporalSpan,
)
from open_wam.models.common.video_action_state import VideoActionRolloutState
from open_wam.models.common.temporal_windows import resolve_interleaved_cache_frames


def reconcile_video_action_observed_history(
    *,
    policy_state: PolicyInferState | None,
    history: PolicyObservedHistory,
    action_tokens_per_frame: int,
    action_dim: int,
) -> PolicyObservedHistoryOutput:
    """Replace a speculative interval with real video/action/proprio history."""
    span = None
    if history.execution_commit is not None:
        if policy_state is None:
            raise ValueError("An execution commit requires an existing policy state.")
        span = _validate_temporal_commit(
            policy_state=policy_state,
            history=history,
        )
    return commit_video_action_observed_history(
        policy_state=policy_state,
        history=history,
        observed_span=span,
        action_tokens_per_frame=action_tokens_per_frame,
        action_dim=action_dim,
    )


def commit_video_action_observed_history(
    *,
    policy_state: PolicyInferState | None,
    history: PolicyObservedHistory,
    observed_span: PolicyTemporalSpan | None,
    action_tokens_per_frame: int,
    action_dim: int,
) -> PolicyObservedHistoryOutput:
    """Commit canonical observations, whether appending or replacing predictions.

    Adapters resolve the observed span; this function alone owns stream slicing,
    bounded retention and cursor publication. Without a span, replace the latest
    predicted chunk without moving the caller's cursor.
    """

    runtime_state = (
        policy_state.variant_state
        if policy_state is not None
        and isinstance(policy_state.variant_state, VideoActionRolloutState)
        else None
    )
    if runtime_state is None:
        return PolicyObservedHistoryOutput(
            next_state=policy_state,
            debug={"warmup_skipped": True, "reason": "no_video_action_runtime_state"},
        )

    # Validate and construct the replacement before publishing any state change.

    temporal_geometry = policy_state.temporal_geometry
    if temporal_geometry is None:
        raise RuntimeError(
            "Video/action observed-history reconciliation requires resolved "
            "session temporal geometry."
        )

    real_latents = history.video_latents
    if real_latents.ndim != 5 or int(real_latents.shape[2]) <= 0:
        raise ValueError(
            "Observed rollout video latents must be [B, C, T, H, W] with T > 0, "
            f"got {tuple(real_latents.shape)}."
        )
    runtime_device = real_latents.device
    runtime_dtype = real_latents.dtype
    replacement_frames = (
        max(0, int(policy_state.cursor.current_start_frame) - observed_span.start_frame)
        if observed_span is not None
        else 0
    )
    frame_chunk_size = (
        replacement_frames
        if replacement_frames > 0 and runtime_state.past_clean_latents is not None
        else int(temporal_geometry.frame_chunk_size)
    )
    inference_window_size = int(temporal_geometry.attention_window_size)
    history_window_frames = max(
        int(real_latents.shape[2]),
        resolve_interleaved_cache_frames(
            window_size=inference_window_size,
            frame_chunk_size=frame_chunk_size,
        ),
    )
    action_tokens_per_frame = int(action_tokens_per_frame)
    if action_tokens_per_frame <= 0:
        raise ValueError(
            "Video/action observed-history action_tokens_per_frame must be positive, "
            f"got {action_tokens_per_frame}."
        )
    committed_span = observed_span
    if committed_span is not None:
        if int(real_latents.shape[2]) != committed_span.frame_count:
            raise ValueError(
                "Observed-history video must contain exactly the committed model frames: "
                f"observed={real_latents.shape[2]}, committed={committed_span.frame_count}."
            )
        if history.action_history is not None and (
            history.action_history.ndim != 3
            or history.action_history.shape[1]
            != committed_span.frame_count * action_tokens_per_frame
        ):
            raise ValueError(
                "Observed-history actions must contain exactly the committed model-frame groups: "
                f"shape={tuple(history.action_history.shape)}, "
                f"expected_tokens={committed_span.frame_count * action_tokens_per_frame}."
            )
    speculative_action_tokens = (
        max(
            runtime_state.chunk_advance_frames,
            runtime_state.pending_predicted_video_frames,
        )
        * action_tokens_per_frame
        if observed_span is None
        else replacement_frames * action_tokens_per_frame
    )

    past_latents = runtime_state.past_clean_latents
    past_hidden_proprio = runtime_state.past_hidden_proprio_states
    pending_pred_latent_frames = int(runtime_state.pending_predicted_video_frames)
    dropped_pred_latent_frames = 0
    if past_latents is None:
        base_latents = None
    else:
        past_latents = past_latents.to(
            device=runtime_device,
            dtype=runtime_dtype,
        )
        if observed_span is not None:
            # An action-only prediction advances the cursor without appending
            # video. Locate observations against the stored span, not that cursor.
            video_end = int(policy_state.cursor.current_start_frame) - max(
                0, runtime_state.chunk_advance_frames - pending_pred_latent_frames
            )
            start = observed_span.start_frame
            if start > video_end:
                raise ValueError("Observation commit leaves a gap in video history.")
            retained = max(0, start - (video_end - int(past_latents.shape[2])))
            dropped_pred_latent_frames = int(past_latents.shape[2]) - retained
        else:
            dropped_pred_latent_frames = min(
                max(0, pending_pred_latent_frames), int(past_latents.shape[2])
            )
        if dropped_pred_latent_frames <= 0:
            base_latents = past_latents
        else:
            base_latents = past_latents[:, :, :-dropped_pred_latent_frames]
    if base_latents is None or int(base_latents.shape[2]) == 0:
        combined_latents = real_latents
    else:
        combined_latents = torch.cat([base_latents, real_latents], dim=2)
    runtime_state = replace(
        runtime_state,
        past_clean_latents=combined_latents[:, :, -history_window_frames:].detach(),
        pending_predicted_video_frames=0,
        chunk_advance_frames=0,
    )

    runtime_state, appended_hidden_proprio_frames = _reconcile_hidden_proprio_history(
        runtime_state=runtime_state,
        proprio_history=history.proprio_history,
        past_hidden_proprio=past_hidden_proprio,
        real_latent_frames=int(real_latents.shape[2]),
        dropped_pred_latent_frames=dropped_pred_latent_frames,
        history_window_frames=history_window_frames,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
    )
    runtime_state, appended_action_tokens, dropped_pred_action_tokens = (
        _reconcile_action_history(
            runtime_state=runtime_state,
            action_history=history.action_history,
            action_mask=history.action_mask,
            action_dim=int(action_dim),
            action_tokens_per_frame=action_tokens_per_frame,
            speculative_action_tokens=speculative_action_tokens,
            observed_frames=int(real_latents.shape[2]),
            history_window_frames=history_window_frames,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
        )
    )
    return PolicyObservedHistoryOutput(
        next_state=replace(
            policy_state,
            variant_state=runtime_state,
            cursor=policy_state.cursor
            if committed_span is None
            else replace(
                policy_state.cursor,
                current_start_frame=committed_span.end_frame,
            ),
            revision=policy_state.revision + 1,
            observed_frame_end=(
                policy_state.cursor.current_start_frame
                if committed_span is None
                else committed_span.end_frame
            ),
        ),
        applied=True,
        debug={
            "warmup_skipped": False,
            "real_obs_frames": int(history.observation_frame_count),
            "real_latent_frames": int(real_latents.shape[2]),
            "past_clean_latent_frames_after": int(
                runtime_state.past_clean_latents.shape[2]
            ),
            "past_clean_action_frames_after": (
                0
                if runtime_state.past_clean_actions is None
                else int(
                    runtime_state.past_clean_actions.shape[1] // action_tokens_per_frame
                )
            ),
            "appended_action_tokens": int(appended_action_tokens),
            "appended_hidden_proprio_frames": int(appended_hidden_proprio_frames),
            "pending_pred_latent_frames_before": int(pending_pred_latent_frames),
            "dropped_pred_latent_frames": int(dropped_pred_latent_frames),
            "dropped_pred_action_tokens": int(dropped_pred_action_tokens),
            "speculative_action_tokens": int(speculative_action_tokens),
            "inference_window_size": int(inference_window_size),
            "history_window_frames": int(history_window_frames),
            "committed_frame_start": (
                None if committed_span is None else int(committed_span.start_frame)
            ),
            "committed_frame_count": (
                None if committed_span is None else int(committed_span.frame_count)
            ),
            "committed_frame_end": (
                None if committed_span is None else int(committed_span.end_frame)
            ),
        },
    )


def _validate_temporal_commit(
    *,
    policy_state: PolicyInferState,
    history: PolicyObservedHistory,
) -> PolicyTemporalSpan | None:
    """Validate one observed interval before mutating recurrent state."""

    commit = history.execution_commit
    if commit is None:
        return None
    speculative_span = commit.speculative_span
    executed_span = commit.executed_span
    if int(policy_state.cursor.current_start_frame) != int(speculative_span.end_frame):
        raise ValueError(
            "Observed-history execution does not match the policy's speculative "
            "cursor: "
            f"cursor={policy_state.cursor.current_start_frame}, "
            f"speculative_end={speculative_span.end_frame}."
        )
    return executed_span


def _reconcile_hidden_proprio_history(
    *,
    runtime_state: VideoActionRolloutState,
    proprio_history: torch.Tensor | None,
    past_hidden_proprio: torch.Tensor | None,
    real_latent_frames: int,
    dropped_pred_latent_frames: int,
    history_window_frames: int,
    runtime_device: torch.device,
    runtime_dtype: torch.dtype,
) -> tuple[VideoActionRolloutState, int]:
    if past_hidden_proprio is None:
        return runtime_state, 0
    if proprio_history is None:
        # Keep estimated states only where the executed prefix already has
        # aligned rows. Missing observations must not retain an unexecuted tail.
        retained_frames = int(past_hidden_proprio.shape[1]) - dropped_pred_latent_frames
        end = retained_frames + real_latent_frames
        if retained_frames < 0 or end > int(past_hidden_proprio.shape[1]):
            raise ValueError(
                "Observed video extends beyond the available proprio history; "
                "provide aligned proprio history for this commit."
            )
        return replace(
            runtime_state,
            past_hidden_proprio_states=past_hidden_proprio[:, :end][
                :, -history_window_frames:
            ].detach(),
        ), 0
    raw_state = proprio_history.to(
        device=runtime_device,
        dtype=runtime_dtype,
    )
    if raw_state.ndim == 2:
        raw_state = raw_state.unsqueeze(0)
    if raw_state.ndim != 3:
        raise ValueError(
            "Observed rollout proprio history must be [T, D] or [B, T, D], "
            f"got {tuple(raw_state.shape)}."
        )
    if int(raw_state.shape[1]) <= 0:
        return runtime_state, 0
    if int(raw_state.shape[1]) >= real_latent_frames:
        latent_state = raw_state[:, -real_latent_frames:, :]
    else:
        pad_count = real_latent_frames - int(raw_state.shape[1])
        latent_state = torch.cat(
            [
                raw_state,
                raw_state[:, -1:, :].expand(-1, pad_count, -1),
            ],
            dim=1,
        )
    past_hidden_proprio = past_hidden_proprio.to(
        device=runtime_device,
        dtype=runtime_dtype,
    )
    base_hidden = (
        past_hidden_proprio[:, :-dropped_pred_latent_frames]
        if dropped_pred_latent_frames > 0
        else past_hidden_proprio
    )
    if int(base_hidden.shape[1]) == 0:
        combined_hidden = latent_state
    else:
        combined_hidden = torch.cat([base_hidden, latent_state], dim=1)
    return replace(
        runtime_state,
        past_hidden_proprio_states=combined_hidden[:, -history_window_frames:].detach(),
    ), int(latent_state.shape[1])


def _reconcile_action_history(
    *,
    runtime_state: VideoActionRolloutState,
    action_history: torch.Tensor | None,
    action_mask: torch.Tensor | None,
    action_dim: int,
    action_tokens_per_frame: int,
    speculative_action_tokens: int,
    observed_frames: int,
    history_window_frames: int,
    runtime_device: torch.device,
    runtime_dtype: torch.dtype,
) -> tuple[VideoActionRolloutState, int, int]:
    if action_history is None:
        if action_mask is not None:
            raise ValueError("Observed action_mask requires action_history.")
        # No measured controls were supplied. Retain the corresponding planned
        # actions, but discard every speculative action beyond the observed end.
        future_tokens = max(
            0, speculative_action_tokens - observed_frames * action_tokens_per_frame
        )
        if future_tokens and runtime_state.past_clean_actions is not None:
            runtime_state = replace(
                runtime_state,
                past_clean_actions=runtime_state.past_clean_actions[:, :-future_tokens],
                past_clean_action_mask=None
                if runtime_state.past_clean_action_mask is None
                else runtime_state.past_clean_action_mask[:, :-future_tokens],
            )
        return runtime_state, 0, 0
    if action_history.ndim != 3 or int(action_history.shape[-1]) != action_dim:
        raise ValueError(
            "Observed rollout action history must be [B, T_action, D_action] "
            "with the policy action dimension, "
            f"got {tuple(action_history.shape)}, action_dim={action_dim}."
        )
    warm_actions = action_history[:, : observed_frames * action_tokens_per_frame].to(
        device=runtime_device,
        dtype=runtime_dtype,
    )
    if action_mask is not None:
        if action_mask.shape != (*action_history.shape[:2], 1) or not bool(
            ((action_mask == 0) | (action_mask == 1)).all()
        ):
            raise ValueError("Observed action_mask must be binary [B, T_action, 1].")
        observed_mask = action_mask[:, : warm_actions.shape[1]].to(warm_actions)
        warm_actions = warm_actions.masked_fill(observed_mask == 0, 0)
    else:
        observed_mask = warm_actions.new_ones((*warm_actions.shape[:2], 1))
    if int(warm_actions.shape[1]) <= 0:
        return runtime_state, 0, 0
    past_actions = runtime_state.past_clean_actions
    if past_actions is None:
        base_actions = None
        base_mask = None
        dropped_pred_action_tokens = 0
    else:
        if int(past_actions.shape[-1]) != action_dim:
            raise ValueError(
                "Observed rollout action dimension must match packed history, "
                f"got observed={action_dim}, history={past_actions.shape[-1]}."
            )
        past_actions = past_actions.to(
            device=runtime_device,
            dtype=runtime_dtype,
        )
        dropped_pred_action_tokens = min(
            speculative_action_tokens,
            int(past_actions.shape[1]),
        )
        base_actions = (
            past_actions[:, :-dropped_pred_action_tokens]
            if dropped_pred_action_tokens > 0
            else past_actions
        )
        past_mask = runtime_state.past_clean_action_mask
        base_mask = (
            None
            if past_mask is None
            else (
                past_mask[:, :-dropped_pred_action_tokens]
                if dropped_pred_action_tokens > 0
                else past_mask
            )
        )
    if base_actions is None or int(base_actions.shape[1]) == 0:
        combined_actions = warm_actions
    else:
        combined_actions = torch.cat([base_actions, warm_actions], dim=1)
    max_action_history_tokens = history_window_frames * action_tokens_per_frame
    if base_actions is not None and base_actions.shape[1]:
        if base_mask is None:
            base_mask = base_actions.new_ones((*base_actions.shape[:2], 1))
        observed_mask = torch.cat([base_mask.to(observed_mask), observed_mask], dim=1)
    return (
        replace(
            runtime_state,
            past_clean_actions=combined_actions[
                :, -max_action_history_tokens:
            ].detach(),
            past_clean_action_mask=observed_mask[
                :, -max_action_history_tokens:
            ].detach(),
        ),
        int(warm_actions.shape[1]),
        int(dropped_pred_action_tokens),
    )
