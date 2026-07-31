from __future__ import annotations

import pytest
import torch

from open_wam.models.policy_variants import (
    PolicyInferState,
    PolicyObservedHistory,
)
from open_wam.models.policy_variants.mot.contracts import MoTRuntimeState
from open_wam.models.policy_variants.mot.observed_history import (
    reconcile_mot_observed_history,
)


def _video(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1, 1)


def _reconcile(
    runtime_state: MoTRuntimeState,
    *,
    real_latents: torch.Tensor,
    step_index: int = 2,
    action_history: torch.Tensor | None = None,
    proprio_history: torch.Tensor | None = None,
    observation_frame_count: int = 16,
    inference_window_size: int = 30,
    rollout_frame_chunk_size: int | None = None,
):
    policy_state = PolicyInferState(
        step_index=step_index,
        variant_state=runtime_state,
    )
    return reconcile_mot_observed_history(
        policy_state=policy_state,
        history=PolicyObservedHistory(
            video_latents=real_latents,
            observation_frame_count=observation_frame_count,
            action_history=action_history,
            proprio_history=proprio_history,
            inference_window_size=inference_window_size,
            rollout_frame_chunk_size=rollout_frame_chunk_size,
        ),
        default_inference_window_size=64,
        default_frame_chunk_size=4,
        action_horizon=16,
        action_dim=7,
    )


def test_first_chunk_keeps_bootstrap_and_replaces_speculative_tail() -> None:
    past_actions = torch.arange(16 * 7, dtype=torch.float32).view(1, 16, 7)
    real_actions = torch.full((1, 8, 7), 500.0)
    past_proprio = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    real_proprio = torch.arange(8, dtype=torch.float32).view(4, 2) + 100.0
    runtime_state = MoTRuntimeState(
        past_clean_latents=_video([0.0, 1.0, 2.0, 3.0]),
        past_clean_actions=past_actions,
        past_hidden_proprio_states=past_proprio,
        pending_predicted_video_frames=4,
    )

    output = _reconcile(
        runtime_state,
        real_latents=_video([10.0, 11.0, 12.0, 13.0]),
        step_index=1,
        action_history=real_actions,
        proprio_history=real_proprio,
    )

    assert output.next_state is not None
    assert output.next_state.variant_state is runtime_state
    torch.testing.assert_close(
        runtime_state.past_clean_latents,
        _video([0.0, 10.0, 11.0, 12.0, 13.0]),
    )
    torch.testing.assert_close(
        runtime_state.past_hidden_proprio_states,
        torch.cat([past_proprio[:, :1], real_proprio.unsqueeze(0)], dim=1),
    )
    torch.testing.assert_close(runtime_state.past_clean_actions, real_actions)
    assert runtime_state.pending_predicted_video_frames == 0
    assert output.debug["dropped_pred_latent_frames"] == 3
    assert output.debug["dropped_pred_action_tokens"] == 16
    assert output.debug["appended_action_tokens"] == 8
    assert output.debug["past_clean_action_frames_after"] == 2


def test_later_chunk_replaces_exact_speculative_video_and_action_horizons() -> None:
    past_actions = torch.arange(40 * 7, dtype=torch.float32).view(1, 40, 7)
    real_actions = torch.full((1, 16, 7), -4.0)
    past_proprio = torch.arange(20, dtype=torch.float32).view(1, 10, 2)
    real_proprio = torch.full((1, 4, 2), 9.0)
    runtime_state = MoTRuntimeState(
        past_clean_latents=_video([float(index) for index in range(10)]),
        past_clean_actions=past_actions,
        past_hidden_proprio_states=past_proprio,
        pending_predicted_video_frames=4,
    )

    output = _reconcile(
        runtime_state,
        real_latents=_video([20.0, 21.0, 22.0, 23.0]),
        action_history=real_actions,
        proprio_history=real_proprio,
    )

    torch.testing.assert_close(
        runtime_state.past_clean_latents,
        _video([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0, 21.0, 22.0, 23.0]),
    )
    torch.testing.assert_close(
        runtime_state.past_hidden_proprio_states,
        torch.cat([past_proprio[:, :6], real_proprio], dim=1),
    )
    torch.testing.assert_close(
        runtime_state.past_clean_actions,
        torch.cat([past_actions[:, :24], real_actions], dim=1),
    )
    assert output.debug["dropped_pred_latent_frames"] == 4
    assert output.debug["dropped_pred_action_tokens"] == 16


def test_history_window_trims_video_proprio_and_action_to_shared_frame_capacity() -> None:
    runtime_state = MoTRuntimeState(
        past_clean_latents=_video([float(index) for index in range(10)]),
        past_clean_actions=torch.arange(40 * 7, dtype=torch.float32).view(1, 40, 7),
        past_hidden_proprio_states=torch.arange(20, dtype=torch.float32).view(1, 10, 2),
        pending_predicted_video_frames=0,
    )
    real_latents = _video([20.0, 21.0, 22.0, 23.0])
    real_actions = torch.full((1, 8, 7), -8.0)
    real_proprio = torch.full((4, 2), 7.0)

    output = _reconcile(
        runtime_state,
        real_latents=real_latents,
        action_history=real_actions,
        proprio_history=real_proprio,
        inference_window_size=2,
        rollout_frame_chunk_size=2,
    )

    assert output.debug["history_window_frames"] == 4
    torch.testing.assert_close(runtime_state.past_clean_latents, real_latents)
    torch.testing.assert_close(
        runtime_state.past_hidden_proprio_states,
        real_proprio.unsqueeze(0),
    )
    assert runtime_state.past_clean_actions is not None
    assert runtime_state.past_clean_actions.shape == (1, 16, 7)
    torch.testing.assert_close(runtime_state.past_clean_actions[:, -8:], real_actions)


def test_short_proprio_history_repeats_last_state_per_real_latent() -> None:
    past_proprio = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
    runtime_state = MoTRuntimeState(
        past_clean_latents=_video([0.0, 1.0, 2.0, 3.0, 4.0]),
        past_hidden_proprio_states=past_proprio,
        pending_predicted_video_frames=2,
    )
    real_proprio = torch.tensor([[100.0, 101.0], [200.0, 201.0]])

    _reconcile(
        runtime_state,
        real_latents=_video([10.0, 11.0, 12.0, 13.0]),
        proprio_history=real_proprio,
    )

    expected_real = torch.tensor(
        [[[100.0, 101.0], [200.0, 201.0], [200.0, 201.0], [200.0, 201.0]]]
    )
    torch.testing.assert_close(
        runtime_state.past_hidden_proprio_states,
        torch.cat([past_proprio[:, :3], expected_real], dim=1),
    )


def test_absent_proprio_cache_is_not_initialized_by_reconciliation() -> None:
    runtime_state = MoTRuntimeState(
        past_clean_latents=_video([0.0]),
        past_hidden_proprio_states=None,
    )

    output = _reconcile(
        runtime_state,
        real_latents=_video([1.0]),
        proprio_history=torch.ones(1, 2),
    )

    assert runtime_state.past_hidden_proprio_states is None
    assert output.debug["appended_hidden_proprio_frames"] == 0


def test_non_mot_state_is_a_noop() -> None:
    policy_state = PolicyInferState(step_index=3, variant_state={"owned": "elsewhere"})
    output = reconcile_mot_observed_history(
        policy_state=policy_state,
        history=PolicyObservedHistory(
            video_latents=_video([1.0]),
            observation_frame_count=4,
        ),
        default_inference_window_size=30,
        default_frame_chunk_size=4,
        action_horizon=16,
        action_dim=7,
    )

    assert output.next_state is policy_state
    assert output.debug == {
        "warmup_skipped": True,
        "reason": "no_mot_runtime_state",
    }


@pytest.mark.parametrize(
    "action_history",
    (
        torch.zeros(16, 7),
        torch.zeros(1, 16, 6),
        torch.tensor(1.0),
    ),
)
def test_action_history_requires_policy_action_schema(
    action_history: torch.Tensor,
) -> None:
    runtime_state = MoTRuntimeState(past_clean_latents=_video([0.0]))

    with pytest.raises(ValueError, match="policy action dimension"):
        _reconcile(
            runtime_state,
            real_latents=_video([1.0]),
            action_history=action_history,
        )


def test_video_history_requires_nonempty_bcthw_latents() -> None:
    runtime_state = MoTRuntimeState()

    with pytest.raises(ValueError, match=r"\[B, C, T, H, W\]"):
        _reconcile(
            runtime_state,
            real_latents=torch.zeros(1, 1, 0, 1, 1),
        )
