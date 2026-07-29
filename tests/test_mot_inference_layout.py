from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from open_wam.configs import MoTGeneralistTrainingMode
from open_wam.models.policy_variants.mot import (
    MoTPackedHistory,
    MoTPackedInferenceLayout,
)
from open_wam.models.policy_variants.mot.contracts import MoTRuntimeState


def _layout(*, startup_prefix: bool = False) -> MoTPackedInferenceLayout:
    video_prefix_frames = 1 if startup_prefix else 0
    action_prefix_tokens = 2 if startup_prefix else 0
    return MoTPackedInferenceLayout(
        batch_size=1,
        action_dim=3,
        configured_action_horizon=4,
        frame_chunk_size=2,
        action_tokens_per_frame=2,
        current_video_prefix_frames=video_prefix_frames,
        current_video_sequence_frames=video_prefix_frames + 2,
        current_action_prefix_tokens=action_prefix_tokens,
        current_action_sequence_tokens=action_prefix_tokens + 4,
        video_channels=4,
        video_height=2,
        video_width=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_packed_history_window_aligns_video_action_and_hidden_proprio_tails() -> None:
    layout = _layout()
    video_history = torch.arange(1 * 4 * 5 * 2 * 2, dtype=torch.float32).reshape(
        1,
        4,
        5,
        2,
        2,
    )
    action_history = torch.arange(1 * 8 * 3, dtype=torch.float32).reshape(1, 8, 3)
    hidden_history = torch.arange(1 * 5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    current_hidden = torch.tensor([[101.0, 102.0, 103.0]])
    runtime_state = MoTRuntimeState(
        past_clean_latents=video_history,
        past_clean_actions=action_history,
        past_hidden_proprio_states=hidden_history,
    )

    history = MoTPackedHistory.from_runtime_state(
        runtime_state,
        layout=layout,
        current_video_latents=torch.zeros(1, 4, 2, 2, 2),
    ).select_window(
        layout=layout,
        history_window_frames=5,
        hidden_proprio_state=current_hidden,
        require_hidden_proprio_history=True,
    )

    assert history.frames == 3
    assert history.max_frames == 3
    assert history.action_tokens == 6
    torch.testing.assert_close(history.video_latents, video_history[:, :, -3:])
    torch.testing.assert_close(history.action_latents, action_history[:, -6:])
    torch.testing.assert_close(
        history.hidden_proprio_sequence,
        torch.cat(
            [
                hidden_history[:, -3:],
                current_hidden[:, None, :].expand(-1, 2, -1),
            ],
            dim=1,
        ),
    )


def test_conditional_rollout_inputs_strip_startup_action_prefix_and_prepend_video_anchor() -> None:
    layout = _layout(startup_prefix=True)
    forced_action_sequence = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    commit_actions = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    video_condition = torch.arange(1 * 4 * 2 * 2 * 2, dtype=torch.float32).reshape(
        1,
        4,
        2,
        2,
        2,
    )
    current_clean_video = torch.full((1, 4, 3, 2, 2), 7.0)

    inputs = layout.resolve_conditional_rollout_inputs(
        {
            "mot_forced_action_latents": forced_action_sequence,
            "mot_commit_action_latents": commit_actions,
            "mot_video_condition_latents": video_condition,
        },
        generalist_rollout_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        current_clean_video=current_clean_video,
    )

    torch.testing.assert_close(
        inputs.forced_action_latents,
        forced_action_sequence[:, 2:],
    )
    torch.testing.assert_close(inputs.commit_action_latents, commit_actions)
    torch.testing.assert_close(
        inputs.video_condition_latents,
        torch.cat([current_clean_video[:, :, :1], video_condition], dim=2),
    )


def test_packed_inference_layout_rejects_incoherent_video_geometry() -> None:
    with pytest.raises(ValueError, match="prefix plus one chunk"):
        replace(_layout(startup_prefix=True), current_video_sequence_frames=2)
