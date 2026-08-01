from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from open_wam.configs import (
    FallbackHistoryDecision,
    FallbackHistoryPolicy,
    GripperRepresentation,
)
from open_wam.evals import realtime_history


def _frame_payload() -> tuple[
    dict[str, np.ndarray],
    list[dict[str, np.ndarray]],
    list[np.ndarray],
]:
    current_obs = {"image": np.full((1, 1, 3), 9, dtype=np.uint8)}
    observations = [
        {"image": np.full((1, 1, 3), 2 + index, dtype=np.uint8)}
        for index in range(4)
    ]
    actions = [np.full((7,), float(index), dtype=np.float32) for index in range(4)]
    return current_obs, observations, actions


def test_frame_history_includes_copied_records_and_typed_decision() -> None:
    pending_history: list[dict[str, Any]] = []
    state = realtime_history.FrameFallbackHistoryState(
        policy=FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY,
    )
    current_obs, observations, actions = _frame_payload()
    proprio = torch.tensor([[1.0, 2.0]], dtype=torch.float64)

    decision = realtime_history.append_frame_history_record(
        pending_history=pending_history,
        state=state,
        absolute_frame_index=4,
        current_obs=current_obs,
        frame_obs_sequence=observations,
        frame_actions=actions,
        frame_action_sources=["history_replan"] * 4,
        frame_chunk_size=4,
        proprio_state=proprio,
    )

    assert decision is FallbackHistoryDecision.INCLUDED
    assert decision.value == "included"
    assert len(pending_history) == 1
    record = pending_history[0]
    assert record["source"] == "history_replan"
    np.testing.assert_array_equal(
        record["proprio_state"],
        np.array([1.0, 2.0], dtype=np.float32),
    )
    current_obs["image"][...] = 0
    observations[0]["image"][...] = 0
    actions[0][...] = -1
    np.testing.assert_array_equal(
        record["obs"]["image"],
        np.full((1, 1, 3), 9, dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        record["obs_sequence"][0]["image"],
        np.full((1, 1, 3), 2, dtype=np.uint8),
    )
    np.testing.assert_array_equal(record["raw_actions"][0], np.zeros((7,), dtype=np.float32))


def test_frame_history_quarantines_fallback_until_clean_chunk() -> None:
    pending_history: list[dict[str, Any]] = []
    state = realtime_history.FrameFallbackHistoryState(
        policy=FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
    )
    current_obs, observations, actions = _frame_payload()

    decision = realtime_history.append_frame_history_record(
        pending_history=pending_history,
        state=state,
        absolute_frame_index=4,
        current_obs=current_obs,
        frame_obs_sequence=observations,
        frame_actions=actions,
        frame_action_sources=["fallback_hold_state"] * 4,
        frame_chunk_size=4,
    )
    assert decision is FallbackHistoryDecision.FALLBACK

    decisions = []
    for frame_index in range(5, 9):
        decisions.append(
            realtime_history.append_frame_history_record(
                pending_history=pending_history,
                state=state,
                absolute_frame_index=frame_index,
                current_obs=current_obs,
                frame_obs_sequence=observations,
                frame_actions=actions,
                frame_action_sources=["history_replan"] * 4,
                frame_chunk_size=4,
            )
        )

    assert decisions == [FallbackHistoryDecision.WASHOUT] * 4
    assert [record["absolute_frame_index"] for record in pending_history] == [5, 6, 7, 8]
    assert not state.quarantine_active
    assert state.hidden_fallback_frames == 1
    assert state.hidden_washout_frames == 4


def test_action_history_quarantines_and_releases_copied_observations() -> None:
    model_window = [{"image": np.array([0], dtype=np.uint8)}]
    state = realtime_history.ActionFallbackHistoryState(
        policy=FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
    )

    fallback = realtime_history.append_action_observation(
        model_obs_window=model_window,
        state=state,
        current_obs={"image": np.array([1], dtype=np.uint8)},
        action_source="fallback_hold_state",
        clean_actions_required=2,
        max_window_frames=2,
    )
    first_clean = realtime_history.append_action_observation(
        model_obs_window=model_window,
        state=state,
        current_obs={"image": np.array([2], dtype=np.uint8)},
        action_source="history_replan",
        clean_actions_required=2,
        max_window_frames=2,
    )
    second_clean = realtime_history.append_action_observation(
        model_obs_window=model_window,
        state=state,
        current_obs={"image": np.array([3], dtype=np.uint8)},
        action_source="history_replan",
        clean_actions_required=2,
        max_window_frames=2,
    )

    assert fallback is FallbackHistoryDecision.FALLBACK
    assert first_clean is second_clean is FallbackHistoryDecision.WASHOUT
    assert [int(obs["image"][0]) for obs in model_window] == [2, 3]
    assert not state.quarantine_active


def test_observation_copy_helpers_do_not_share_array_storage() -> None:
    source = {"image": np.array([1], dtype=np.uint8)}
    copied = realtime_history.copy_observation(source)
    window = realtime_history.copy_observation_window([source])
    realtime_history.append_observation_window(
        window,
        {"image": np.array([2], dtype=np.uint8)},
        max_window_frames=1,
    )

    source["image"][0] = 9
    assert int(copied["image"][0]) == 1
    assert [int(obs["image"][0]) for obs in window] == [2]


def test_fallback_timeline_and_absolute_tail_policies_are_explicit() -> None:
    assert realtime_history.action_advances_model_timeline(
        "history_replan",
        fallback_history_policy=FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
    )
    assert not realtime_history.action_advances_model_timeline(
        "fallback_hold_state",
        fallback_history_policy=FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
    )
    assert realtime_history.action_advances_model_timeline(
        "fallback_hold_state",
        fallback_history_policy=FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY,
    )

    action_command = SimpleNamespace(
        data=SimpleNamespace(
            action_target=SimpleNamespace(
                include_gripper=True,
                gripper_representation=GripperRepresentation.ACTION_COMMAND,
            )
        )
    )
    absolute_gripper = SimpleNamespace(
        data=SimpleNamespace(
            action_target=SimpleNamespace(
                include_gripper=True,
                gripper_representation=GripperRepresentation.FIRST_CHANNEL,
            )
        )
    )
    assert realtime_history.fallback_absolute_tail_start(action_command) is None
    assert realtime_history.fallback_absolute_tail_start(absolute_gripper) == 6


def test_proprio_state_to_numpy_flattens_casts_and_copies() -> None:
    source = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    converted = realtime_history.proprio_state_to_numpy(source)

    source[0, 0] = 9.0
    assert converted.dtype == np.float32
    assert converted.shape == (2,)
    np.testing.assert_array_equal(converted, np.array([1.0, 2.0], dtype=np.float32))
