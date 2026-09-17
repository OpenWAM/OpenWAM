"""Shared sequence preparation must not depend on model token packing."""

import pytest
import torch

from open_wam.configs import DynamicsObjective, VideoActionProgram
from open_wam.models.common.dynamics_objectives import resolve_dynamics_rollout_plan
from open_wam.models.common.rollout import RolloutCursor
from open_wam.models.common.video_action_layout import prepare_video_action_sequence
from open_wam.models.common.video_action_state import VideoActionRolloutState
from open_wam.models.policy_variants import DynamicsRolloutRequest, PolicyInferState


def sequence(*, request=None, state=None, chunk=2, observation=None):
    return prepare_video_action_sequence(
        observation=torch.full((1, 4, 1, 2, 2), 7.0)
        if observation is None
        else observation,
        state=PolicyInferState() if state is None else state,
        chunk_frames=chunk,
        actions_per_frame=2,
        action_dim=3,
        window_size=5,
        dynamics=resolve_dynamics_rollout_plan(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
            request=request,
        ),
    )


def test_shared_window_aligns_all_history_streams():
    video = torch.arange(80.0).reshape(1, 4, 5, 2, 2)
    action = torch.arange(30.0).reshape(1, 10, 3)
    proprio = torch.arange(15.0).reshape(1, 5, 3)
    current = torch.tensor([[101.0, 102.0, 103.0]])
    state = PolicyInferState(
        cursor=RolloutCursor(current_start_frame=5),
        variant_state=VideoActionRolloutState(
            past_clean_latents=video,
            past_clean_actions=action,
            past_hidden_proprio_states=proprio,
            hidden_proprio_state=current,
        ),
    )
    result = sequence(state=state)
    assert result.history_frames == 4
    assert result.generation_start == 5
    torch.testing.assert_close(result.video[:, :, :4], video[:, :, -4:])
    torch.testing.assert_close(result.action[:, :8], action[:, -8:])
    torch.testing.assert_close(
        result.proprio,
        torch.cat([proprio[:, -4:], current[:, None].expand(-1, 2, -1)], dim=1),
    )


@pytest.mark.parametrize("chunk", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "objective",
    [
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
    ],
)
def test_conditional_inputs_preserve_t0_and_select_exact_future_span(chunk, objective):
    actions = torch.arange(24.0).reshape(1, 8, 3)
    video = torch.arange(64.0).reshape(1, 4, 4, 2, 2)
    request = DynamicsRolloutRequest(
        objective=objective,
        clean_action=actions
        if objective is DynamicsObjective.ACTION_CONDITIONED_VIDEO
        else None,
        clean_video=video
        if objective is DynamicsObjective.VIDEO_CONDITIONED_ACTION
        else None,
        history_action=actions,
    )
    result = sequence(request=request, chunk=chunk)
    assert (
        result.history_frames == result.startup_frames == result.generation_start == 1
    )
    assert result.video.shape[2] == chunk + 1
    assert not result.action_mask[:, :2].any()
    assert result.action_mask[:, 2:].all()
    torch.testing.assert_close(result.video[:, :, :1], torch.full((1, 4, 1, 2, 2), 7.0))
    torch.testing.assert_close(result.commit_action, actions[:, : 2 * chunk])
    if objective is DynamicsObjective.VIDEO_CONDITIONED_ACTION:
        torch.testing.assert_close(result.video[:, :, 1:], video[:, :, :chunk])
    else:
        torch.testing.assert_close(result.forced_action, actions[:, : 2 * chunk])


def test_shared_preparation_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="frame-aligned"):
        sequence(
            request=DynamicsRolloutRequest(
                objective=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
                clean_action=torch.zeros(1, 5, 3),
            )
        )
    state = PolicyInferState(
        variant_state=VideoActionRolloutState(
            past_clean_latents=torch.zeros(1, 4, 5, 2, 2),
            past_clean_actions=torch.zeros(1, 8, 3),
        )
    )
    with pytest.raises(ValueError, match="frame-aligned"):
        sequence(state=state)
    with pytest.raises(ValueError, match="Observation must be nonempty"):
        sequence(observation=torch.zeros(1, 4, 0, 2, 2))


def test_publish_is_immutable_and_ungenerated_actions_are_invalid():
    state = PolicyInferState()
    prepared = sequence()
    next_state = prepared.publish(
        state,
        video=prepared.video,
        action=prepared.action,
        retain_future_video=True,
        retain_future_action=False,
    )
    assert state.variant_state is None
    assert state.cursor.current_start_frame == 0
    assert next_state.cursor.current_start_frame == 3
    assert not next_state.variant_state.past_clean_action_mask.any()
    assert next_state.cursor.block_index == 1
