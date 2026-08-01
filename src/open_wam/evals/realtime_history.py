"""Fallback quarantine and observation-history contracts for realtime rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from open_wam.configs.enums import (
    FallbackHistoryDecision,
    FallbackHistoryPolicy,
    GripperRepresentation,
)

if TYPE_CHECKING:
    from open_wam.configs import ExperimentConfig


__all__ = [
    "ActionFallbackHistoryState",
    "FrameFallbackHistoryState",
    "action_advances_model_timeline",
    "append_action_observation",
    "append_frame_history_record",
    "append_observation_window",
    "copy_observation",
    "copy_observation_window",
    "fallback_absolute_tail_start",
    "fallback_policy_freezes_model_timeline",
    "frame_contains_fallback_action",
    "proprio_state_to_numpy",
]


@dataclass
class FrameFallbackHistoryState:
    """Fallback quarantine counters for frame-grouped observation history."""

    policy: FallbackHistoryPolicy
    quarantine_active: bool = False
    clean_frames_since_fallback: int = 0
    hidden_history_frames: int = 0
    hidden_history_raw_observations: int = 0
    hidden_fallback_frames: int = 0
    hidden_fallback_raw_observations: int = 0
    hidden_washout_frames: int = 0
    hidden_washout_raw_observations: int = 0
    fallback_quarantine_count: int = 0
    quarantine_records: list[dict[str, Any]] | None = None


@dataclass
class ActionFallbackHistoryState:
    """Fallback quarantine counters for action-rate observation history."""

    policy: FallbackHistoryPolicy
    quarantine_active: bool = False
    clean_actions_since_fallback: int = 0
    hidden_history_actions: int = 0
    hidden_fallback_actions: int = 0
    hidden_washout_actions: int = 0
    fallback_quarantine_count: int = 0
    quarantine_observations: list[dict[str, np.ndarray]] | None = None


def frame_contains_fallback_action(action_sources: list[str]) -> bool:
    """Return whether any action in a frame came from a fallback source."""

    return any(str(source).startswith("fallback_") for source in action_sources)


def fallback_policy_freezes_model_timeline(policy: FallbackHistoryPolicy) -> bool:
    """Return whether fallback actions should leave model time unchanged."""

    return policy is not FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY


def action_advances_model_timeline(
    action_source: str,
    *,
    fallback_history_policy: FallbackHistoryPolicy,
) -> bool:
    """Resolve whether one executed simulator action advances model history."""

    return not (
        str(action_source).startswith("fallback_")
        and fallback_policy_freezes_model_timeline(fallback_history_policy)
    )


def fallback_absolute_tail_start(config: ExperimentConfig) -> int | None:
    """Resolve the first absolute action channel preserved by hold-state fallback."""

    action_target = getattr(getattr(config, "data", None), "action_target", None)
    gripper_representation = getattr(action_target, "gripper_representation", None)
    if gripper_representation == GripperRepresentation.ACTION_COMMAND:
        return None
    if str(gripper_representation) == GripperRepresentation.ACTION_COMMAND.value:
        return None
    if bool(getattr(action_target, "include_gripper", False)):
        return 6
    return None


def append_frame_history_record(
    *,
    pending_history: list[dict[str, Any]],
    state: FrameFallbackHistoryState,
    absolute_frame_index: int,
    current_obs: dict[str, np.ndarray],
    frame_obs_sequence: list[dict[str, np.ndarray]],
    frame_actions: list[np.ndarray],
    frame_action_sources: list[str],
    frame_chunk_size: int,
    proprio_state: np.ndarray | torch.Tensor | None = None,
) -> FallbackHistoryDecision:
    """Append or quarantine one frame-grouped observed-history record."""

    contains_fallback_action = frame_contains_fallback_action(frame_action_sources)
    history_record = {
        "absolute_frame_index": int(absolute_frame_index),
        "obs": {key: np.array(value, copy=True) for key, value in current_obs.items()},
        "obs_sequence": [
            {key: np.array(value, copy=True) for key, value in obs.items()}
            for obs in frame_obs_sequence
        ],
        "raw_actions": np.stack(frame_actions, axis=0).astype(np.float32),
        "source": _frame_source_label(frame_action_sources),
        "action_sources": [str(source) for source in frame_action_sources],
        "contains_fallback_action": bool(contains_fallback_action),
    }
    if proprio_state is not None:
        history_record["proprio_state"] = proprio_state_to_numpy(proprio_state)
    raw_observation_count = len(frame_obs_sequence)
    policy = state.policy
    if policy is FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY:
        pending_history.append(history_record)
        return FallbackHistoryDecision.INCLUDED

    washout_frames_required = _washout_frames_required(
        policy,
        frame_chunk_size=frame_chunk_size,
    )
    if contains_fallback_action:
        if not state.quarantine_active:
            state.fallback_quarantine_count += 1
        state.quarantine_active = True
        state.clean_frames_since_fallback = 0
        state.quarantine_records = []
        _record_hidden_frame(
            state,
            raw_observation_count=raw_observation_count,
            hidden_kind="fallback",
        )
        return FallbackHistoryDecision.FALLBACK
    if not state.quarantine_active:
        pending_history.append(history_record)
        return FallbackHistoryDecision.INCLUDED

    state.clean_frames_since_fallback += 1
    if state.quarantine_records is None:
        state.quarantine_records = []
    state.quarantine_records.append(history_record)
    _record_hidden_frame(
        state,
        raw_observation_count=raw_observation_count,
        hidden_kind="washout",
    )
    if state.clean_frames_since_fallback >= washout_frames_required:
        pending_history.extend(state.quarantine_records)
        state.quarantine_records = []
        state.quarantine_active = False
        state.clean_frames_since_fallback = 0
    return FallbackHistoryDecision.WASHOUT


def copy_observation(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy one observation record without sharing array storage."""

    return {key: np.array(value, copy=True) for key, value in obs.items()}


def copy_observation_window(
    obs_window: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Copy an observation window without sharing record or array storage."""

    return [copy_observation(obs) for obs in obs_window]


def append_observation_window(
    obs_window: list[dict[str, np.ndarray]],
    obs: dict[str, np.ndarray],
    *,
    max_window_frames: int,
) -> list[dict[str, np.ndarray]]:
    """Append a copied observation and retain only the configured tail."""

    obs_window.append(copy_observation(obs))
    if len(obs_window) > int(max_window_frames):
        del obs_window[: -int(max_window_frames)]
    return obs_window


def append_action_observation(
    *,
    model_obs_window: list[dict[str, np.ndarray]],
    state: ActionFallbackHistoryState,
    current_obs: dict[str, np.ndarray],
    action_source: str,
    clean_actions_required: int,
    max_window_frames: int,
) -> FallbackHistoryDecision:
    """Append or quarantine one action-rate model observation."""

    contains_fallback_action = str(action_source).startswith("fallback_")
    if state.policy is FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY:
        append_observation_window(
            model_obs_window,
            current_obs,
            max_window_frames=max_window_frames,
        )
        return FallbackHistoryDecision.INCLUDED

    if contains_fallback_action:
        if not state.quarantine_active:
            state.fallback_quarantine_count += 1
        state.quarantine_active = True
        state.clean_actions_since_fallback = 0
        state.quarantine_observations = []
        state.hidden_history_actions += 1
        state.hidden_fallback_actions += 1
        return FallbackHistoryDecision.FALLBACK

    if not state.quarantine_active:
        append_observation_window(
            model_obs_window,
            current_obs,
            max_window_frames=max_window_frames,
        )
        return FallbackHistoryDecision.INCLUDED

    state.clean_actions_since_fallback += 1
    if state.quarantine_observations is None:
        state.quarantine_observations = []
    state.quarantine_observations.append(copy_observation(current_obs))
    state.hidden_history_actions += 1
    state.hidden_washout_actions += 1
    if state.clean_actions_since_fallback >= int(clean_actions_required):
        for obs in state.quarantine_observations:
            append_observation_window(
                model_obs_window,
                obs,
                max_window_frames=max_window_frames,
            )
        state.quarantine_observations = []
        state.quarantine_active = False
        state.clean_actions_since_fallback = 0
    return FallbackHistoryDecision.WASHOUT


def proprio_state_to_numpy(
    proprio_state: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Detach one proprio state as a copied float32 NumPy array."""

    if isinstance(proprio_state, torch.Tensor):
        array = proprio_state.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        array = np.asarray(proprio_state, dtype=np.float32)
    return np.asarray(array, dtype=np.float32).reshape(-1).copy()


def _frame_source_label(action_sources: list[str]) -> str:
    if not action_sources:
        return "unknown"
    first_source = str(action_sources[0])
    if all(str(source) == first_source for source in action_sources):
        return first_source
    return "mixed"


def _washout_frames_required(
    policy: FallbackHistoryPolicy,
    *,
    frame_chunk_size: int,
) -> int:
    if policy is FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK:
        return int(frame_chunk_size)
    return 0


def _record_hidden_frame(
    state: FrameFallbackHistoryState,
    *,
    raw_observation_count: int,
    hidden_kind: str,
) -> None:
    state.hidden_history_frames += 1
    state.hidden_history_raw_observations += int(raw_observation_count)
    if hidden_kind == "fallback":
        state.hidden_fallback_frames += 1
        state.hidden_fallback_raw_observations += int(raw_observation_count)
        return
    if hidden_kind == "washout":
        state.hidden_washout_frames += 1
        state.hidden_washout_raw_observations += int(raw_observation_count)
        return
    raise ValueError(f"Unsupported hidden_kind={hidden_kind!r}.")
