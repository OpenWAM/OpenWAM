"""Reconcile speculative packed DualExpert history with executed observations."""

from __future__ import annotations

import torch

from ..contracts import (
    PolicyInferState,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
    PolicyTemporalSpan,
)
from .contracts import DualExpertRuntimeState
from .rollout_geometry import (
    resolve_dual_expert_rollout_cache_window_frames,
)


def reconcile_dual_expert_observed_history(
    *,
    policy_state: PolicyInferState | None,
    history: PolicyObservedHistory,
    action_horizon: int,
    action_tokens_per_frame: int,
    action_dim: int,
) -> PolicyObservedHistoryOutput:
    """Replace the last speculative chunk with real video/action/proprio history."""

    runtime_state = (
        policy_state.variant_state
        if policy_state is not None
        and isinstance(policy_state.variant_state, DualExpertRuntimeState)
        else None
    )
    if runtime_state is None:
        return PolicyObservedHistoryOutput(
            next_state=policy_state,
            debug={"warmup_skipped": True, "reason": "no_dual_expert_runtime_state"},
        )

    temporal_geometry = policy_state.temporal_geometry
    if temporal_geometry is None:
        raise RuntimeError(
            "DualExpert observed-history reconciliation requires resolved "
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
    frame_chunk_size = (
        int(history.execution_commit.speculative_span.frame_count)
        if history.execution_commit is not None
        else int(temporal_geometry.frame_chunk_size)
    )
    inference_window_size = int(temporal_geometry.attention_window_size)
    history_window_frames = max(
        int(real_latents.shape[2]),
        resolve_dual_expert_rollout_cache_window_frames(
            window_size=inference_window_size,
            frame_chunk_size=frame_chunk_size,
        ),
    )
    action_tokens_per_frame = int(action_tokens_per_frame)
    if action_tokens_per_frame <= 0:
        raise ValueError(
            "DualExpert observed-history action_tokens_per_frame must be positive, "
            f"got {action_tokens_per_frame}."
        )
    committed_span = _validate_temporal_commit(
        policy_state=policy_state,
        runtime_state=runtime_state,
        history=history,
        real_latent_frames=int(real_latents.shape[2]),
        action_tokens_per_frame=action_tokens_per_frame,
    )
    speculative_action_tokens = (
        int(action_horizon)
        if history.execution_commit is None
        else int(history.execution_commit.speculative_span.frame_count)
        * action_tokens_per_frame
    )

    past_latents = runtime_state.past_clean_latents
    past_hidden_proprio = runtime_state.past_hidden_proprio_states
    pending_pred_latent_frames = int(
        runtime_state.pending_predicted_video_frames
    )
    dropped_pred_latent_frames = 0
    if past_latents is None:
        base_latents = None
    else:
        past_latents = past_latents.to(
            device=runtime_device,
            dtype=runtime_dtype,
        )
        dropped_pred_latent_frames = min(
            max(0, pending_pred_latent_frames),
            int(past_latents.shape[2]),
        )
        if dropped_pred_latent_frames <= 0:
            base_latents = past_latents
        elif (
            int(policy_state.step_index) <= 1
            and dropped_pred_latent_frames >= int(past_latents.shape[2])
        ):
            # Keep the bootstrap observation when the first rewind otherwise
            # removes the entire startup prediction.
            base_latents = past_latents[:, :, :1]
            dropped_pred_latent_frames = max(
                0,
                int(past_latents.shape[2]) - 1,
            )
        else:
            base_latents = past_latents[:, :, :-dropped_pred_latent_frames]
    if base_latents is None or int(base_latents.shape[2]) == 0:
        combined_latents = real_latents
    else:
        combined_latents = torch.cat([base_latents, real_latents], dim=2)
    runtime_state.past_clean_latents = combined_latents[
        :, :, -history_window_frames:
    ].detach()
    runtime_state.pending_predicted_video_frames = 0

    appended_hidden_proprio_frames = _reconcile_hidden_proprio_history(
        runtime_state=runtime_state,
        proprio_history=history.proprio_history,
        past_hidden_proprio=past_hidden_proprio,
        real_latent_frames=int(real_latents.shape[2]),
        dropped_pred_latent_frames=dropped_pred_latent_frames,
        history_window_frames=history_window_frames,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
    )
    appended_action_tokens, dropped_pred_action_tokens = (
        _reconcile_action_history(
            runtime_state=runtime_state,
            action_history=history.action_history,
            action_horizon=int(action_horizon),
            action_dim=int(action_dim),
            action_tokens_per_frame=action_tokens_per_frame,
            speculative_action_tokens=speculative_action_tokens,
            history_window_frames=history_window_frames,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
        )
    )
    _apply_temporal_commit(
        policy_state=policy_state,
        runtime_state=runtime_state,
        committed_span=committed_span,
    )
    return PolicyObservedHistoryOutput(
        next_state=policy_state,
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
                    runtime_state.past_clean_actions.shape[1]
                    // action_tokens_per_frame
                )
            ),
            "appended_action_tokens": int(appended_action_tokens),
            "appended_hidden_proprio_frames": int(
                appended_hidden_proprio_frames
            ),
            "pending_pred_latent_frames_before": int(
                pending_pred_latent_frames
            ),
            "dropped_pred_latent_frames": int(
                dropped_pred_latent_frames
            ),
            "dropped_pred_action_tokens": int(
                dropped_pred_action_tokens
            ),
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
    runtime_state: DualExpertRuntimeState,
    history: PolicyObservedHistory,
    real_latent_frames: int,
    action_tokens_per_frame: int,
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
    if int(runtime_state.chunk_advance_frames) != int(
        speculative_span.frame_count
    ):
        raise ValueError(
            "Observed-history execution does not match the policy's pending "
            "chunk geometry: "
            f"pending={runtime_state.chunk_advance_frames}, "
            f"speculative={speculative_span.frame_count}."
        )
    if int(real_latent_frames) != int(executed_span.frame_count):
        raise ValueError(
            "Observed-history video must contain exactly the committed model "
            "frames: "
            f"observed={real_latent_frames}, committed={executed_span.frame_count}."
        )
    if history.action_history is not None and history.action_history.ndim == 3:
        expected_action_tokens = int(executed_span.frame_count) * int(
            action_tokens_per_frame
        )
        if int(history.action_history.shape[1]) != expected_action_tokens:
            raise ValueError(
                "Observed-history actions must contain exactly the committed "
                "model-frame groups: "
                f"observed={history.action_history.shape[1]}, "
                f"expected={expected_action_tokens}."
            )
    return executed_span


def _apply_temporal_commit(
    *,
    policy_state: PolicyInferState,
    runtime_state: DualExpertRuntimeState,
    committed_span: PolicyTemporalSpan | None,
) -> None:
    if committed_span is None:
        return
    policy_state.cursor.current_start_frame = int(committed_span.end_frame)
    runtime_state.next_condition_frame_start = int(committed_span.end_frame)
    runtime_state.chunk_advance_frames = 0


def _reconcile_hidden_proprio_history(
    *,
    runtime_state: DualExpertRuntimeState,
    proprio_history: torch.Tensor | None,
    past_hidden_proprio: torch.Tensor | None,
    real_latent_frames: int,
    dropped_pred_latent_frames: int,
    history_window_frames: int,
    runtime_device: torch.device,
    runtime_dtype: torch.dtype,
) -> int:
    if proprio_history is None or past_hidden_proprio is None:
        return 0
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
        return 0
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
    runtime_state.past_hidden_proprio_states = combined_hidden[
        :, -history_window_frames:
    ].detach()
    return int(latent_state.shape[1])


def _reconcile_action_history(
    *,
    runtime_state: DualExpertRuntimeState,
    action_history: torch.Tensor | None,
    action_horizon: int,
    action_dim: int,
    action_tokens_per_frame: int,
    speculative_action_tokens: int,
    history_window_frames: int,
    runtime_device: torch.device,
    runtime_dtype: torch.dtype,
) -> tuple[int, int]:
    if action_history is None:
        return 0, 0
    if action_history.ndim != 3 or int(action_history.shape[-1]) != action_dim:
        raise ValueError(
            "Observed rollout action history must be [B, T_action, D_action] "
            "with the policy action dimension, "
            f"got {tuple(action_history.shape)}, action_dim={action_dim}."
        )
    warm_actions = action_history[:, :action_horizon].to(
        device=runtime_device,
        dtype=runtime_dtype,
    )
    if int(warm_actions.shape[1]) <= 0:
        return 0, 0
    past_actions = runtime_state.past_clean_actions
    if past_actions is None:
        base_actions = None
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
    if base_actions is None or int(base_actions.shape[1]) == 0:
        combined_actions = warm_actions
    else:
        combined_actions = torch.cat([base_actions, warm_actions], dim=1)
    max_action_history_tokens = history_window_frames * action_tokens_per_frame
    runtime_state.past_clean_actions = combined_actions[
        :, -max_action_history_tokens:
    ].detach()
    return int(warm_actions.shape[1]), int(dropped_pred_action_tokens)
