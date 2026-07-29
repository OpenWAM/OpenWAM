from __future__ import annotations

import pytest
import torch

from open_wam.data import (
    GENERALIST_CONDITIONAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE,
    GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
    GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON,
    project_real_conditional_sample_to_target_only,
)
from open_wam.data.latent_contracts import LatentWAMSample


def _sample(
    *,
    frames: int = 4,
    action_steps_per_frame: int = 2,
    metadata: dict | None = None,
    condition_latents: bool = True,
) -> LatentWAMSample:
    action_steps = frames * action_steps_per_frame
    return LatentWAMSample(
        video_latents=torch.arange(
            2 * frames * 2 * 2,
            dtype=torch.float32,
        ).reshape(2, frames, 2, 2),
        actions=torch.arange(
            action_steps * 3,
            dtype=torch.float32,
        ).reshape(action_steps, 3),
        action_mask=torch.ones(action_steps, 3),
        state=torch.full((2, 3), -1.0),
        state_mask=torch.ones(2, 3),
        canonical_video=torch.arange(
            frames * 3 * 2 * 2,
            dtype=torch.float32,
        ).reshape(frames, 3, 2, 2),
        condition_latents=(
            torch.arange(
                100,
                100 + 2 * frames * 2 * 2,
                dtype=torch.float32,
            ).reshape(2, frames, 2, 2)
            if condition_latents
            else None
        ),
        proprio_context_state=torch.arange(
            frames * 3,
            dtype=torch.float32,
        ).reshape(frames, 3),
        proprio_context_state_mask=torch.ones(frames, 3),
        proprio_context_frames=torch.arange(
            200,
            200 + frames * 3,
            dtype=torch.float32,
        ).reshape(frames, 3),
        proprio_context_frames_mask=torch.ones(frames, 3),
        metadata=dict(metadata or {}),
    )


def test_target_only_projection_uses_previous_boundary_as_t0() -> None:
    sample = _sample(
        metadata={
            "history_frames": 3,
            "observation_frame_indices": [10, 14, 18, 22],
            "segment_valid_latent_frames": 4,
            "context_prefix_frames_requested": 3,
            "subwindow_action_start": 8,
        }
    )

    projected = project_real_conditional_sample_to_target_only(sample)

    assert projected.video_latents.shape == (2, 2, 2, 2)
    assert torch.equal(projected.video_latents, sample.video_latents[:, 2:])
    assert projected.condition_latents is not None
    assert torch.equal(projected.condition_latents, sample.condition_latents[:, 2:])
    assert projected.canonical_video is not None
    assert torch.equal(projected.canonical_video, sample.canonical_video[2:])
    assert torch.equal(projected.actions[:2], torch.zeros(2, 3))
    assert torch.equal(projected.actions[2:], sample.actions[4:6])
    assert projected.action_mask is not None
    assert torch.equal(projected.action_mask[:2], torch.zeros(2, 3))
    assert torch.equal(projected.action_mask[2:], torch.ones(2, 3))
    assert projected.state is not None
    assert projected.proprio_context_frames is not None
    assert torch.equal(projected.state, sample.proprio_context_frames[:2])
    assert torch.equal(
        projected.proprio_context_frames,
        sample.proprio_context_frames[2:],
    )
    assert projected.metadata["generalist_conditional_contract"] == (
        GENERALIST_CONDITIONAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE
    )
    assert projected.metadata["generalist_gjd_chunk_contract"] == (
        GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON
    )
    assert projected.metadata["generalist_conditional_history_policy"] == (
        GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY
    )
    assert projected.metadata["generalist_conditional_boundary_source"] == (
        "history_frames"
    )
    assert projected.metadata["observation_frame_indices"] == [18, 22]
    assert projected.metadata["loss_frame_start"] == 1
    assert projected.metadata["loss_frame_end"] == 2
    assert projected.metadata["subwindow_action_start"] == 12
    assert projected.metadata["state_anchor_source_frame"] == 1


def test_target_only_projection_without_condition_anchors_state_on_t0() -> None:
    sample = _sample(
        condition_latents=False,
        metadata={"current_start_frame_in_sample": 3},
    )

    projected = project_real_conditional_sample_to_target_only(sample)

    assert projected.state is not None
    assert projected.proprio_context_frames is not None
    assert torch.equal(projected.state, sample.proprio_context_frames[1:3])
    assert projected.metadata["state_anchor_source_frame"] == 2
    assert projected.metadata["state_anchor_frame_in_sample"] == 0


def test_target_only_projection_defaults_to_first_frame_boundary() -> None:
    sample = _sample(metadata={})

    projected = project_real_conditional_sample_to_target_only(sample)

    assert projected.video_latents.shape == sample.video_latents.shape
    assert projected.metadata["generalist_conditional_boundary_source"] == (
        "default_first_frame"
    )
    assert projected.metadata["generalist_conditional_source_t0_frame_in_sample"] == 0
    assert projected.action_mask is not None
    assert torch.equal(projected.action_mask[:2], torch.zeros(2, 3))
    assert torch.equal(projected.action_mask[2:], torch.ones(6, 3))


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (
            _sample(frames=1),
            "requires at least two latent frames",
        ),
        (
            LatentWAMSample(
                video_latents=torch.zeros(2, 3, 2, 2),
                actions=torch.zeros(5, 3),
            ),
            "requires frame-aligned actions",
        ),
        (
            _sample(metadata={"history_frames": 4}),
            "requires at least one future frame",
        ),
    ],
)
def test_target_only_projection_rejects_invalid_layouts(
    sample: LatentWAMSample,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        project_real_conditional_sample_to_target_only(sample)
