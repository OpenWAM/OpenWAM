from __future__ import annotations

import pytest
import torch

from open_wam.configs import (
    DynamicsObjective,
    HistoryStreamVisibility,
    ParallelStreamPolicyConfig,
    VideoActionProgram,
)
from open_wam.models.common.dynamics_objectives import (
    dynamics_objective_attention_window_size,
    dynamics_objective_rollout_chunk_size,
    is_conditional_dynamics_objective,
    resolve_dynamics_objective,
    resolve_dynamics_rollout_geometry,
)
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.conditional_rollout import (
    dynamics_rollout_prefix_visibility_mode,
    select_dynamics_warmup_history_suffix,
    slice_dynamics_conditioning_chunk,
    uses_dynamics_mode_text_token,
)


def _policy_config(**overrides: object) -> ParallelStreamPolicyConfig:
    overrides.setdefault("program", VideoActionProgram.VIDEO_THEN_ACTION)
    return ParallelStreamPolicyConfig(hidden_size=32, **overrides)


def _rollout_geometry(
    mode: DynamicsObjective,
    config: ParallelStreamPolicyConfig,
    *,
    frame_chunk_size: int = 4,
):
    return resolve_dynamics_rollout_geometry(
        mode,
        fallback_frame_chunk_size=frame_chunk_size,
        fallback_attention_window_size=30,
        fallback_history_stream_visibility=config.history_stream_visibility,
    )


def test_reference_runtime_conditional_rollout_names_alias_canonical_contract() -> None:
    aliases = {
        "_rollout_chunk_size_for_generalist_conditioning": (
            dynamics_objective_rollout_chunk_size
        ),
        "_generalist_mode_for_action_conditioning": (resolve_dynamics_objective),
        "_is_conditional_joint_denoise_mode": is_conditional_dynamics_objective,
        "_prefix_visibility_mode_for_generalist_conditioning": (
            dynamics_rollout_prefix_visibility_mode
        ),
        "_select_conditional_warmup_history_suffix": (
            select_dynamics_warmup_history_suffix
        ),
        "_slice_conditioning_chunk": slice_dynamics_conditioning_chunk,
        "_uses_generalist_mode_text_token": uses_dynamics_mode_text_token,
        "_window_size_for_generalist_conditioning": (
            dynamics_objective_attention_window_size
        ),
    }

    for legacy_name, canonical in aliases.items():
        assert getattr(reference_runtime, legacy_name) is canonical


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("joint", DynamicsObjective.JOINT),
        ("action_conditioned_video", DynamicsObjective.ACTION_CONDITIONED_VIDEO),
        ("video_conditioned_action", DynamicsObjective.VIDEO_CONDITIONED_ACTION),
        (
            DynamicsObjective.JOINT,
            DynamicsObjective.JOINT,
        ),
        (
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
    ],
)
def test_resolve_action_conditioning_mode_supports_canonical_objectives(
    value: DynamicsObjective | str,
    expected: DynamicsObjective,
) -> None:
    assert resolve_dynamics_objective(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "vanilla_joint_rollout",
        "clean_action_feedback",
        "forced_action_joint_fdm",
        "fdm",
        "idm",
        "unknown",
    ],
)
def test_resolve_action_conditioning_mode_rejects_noncanonical_labels(
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported dynamics objective",
    ):
        resolve_dynamics_objective(value)


@pytest.mark.parametrize(
    ("mode", "is_conditional", "window_size", "chunk_size"),
    [
        (DynamicsObjective.JOINT, False, 30, 4),
        (
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            True,
            3,
            1,
        ),
        (
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            True,
            3,
            1,
        ),
    ],
)
def test_conditional_rollout_geometry_matches_shared_gjd_contract(
    mode: DynamicsObjective,
    is_conditional: bool,
    window_size: int,
    chunk_size: int,
) -> None:
    assert is_conditional_dynamics_objective(mode) is is_conditional
    assert (
        dynamics_objective_attention_window_size(
            mode,
            fallback_window_size=30,
        )
        == window_size
    )
    assert (
        dynamics_objective_rollout_chunk_size(
            mode,
            fallback_chunk_size=4,
        )
        == chunk_size
    )


@pytest.mark.parametrize(
    "mode",
    [
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
    ],
)
def test_conditional_modes_force_video_only_history(
    mode: DynamicsObjective,
) -> None:
    config = _policy_config(
        history_stream_visibility=(HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY)
    )
    geometry = _rollout_geometry(mode, config)

    assert geometry.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert (
        dynamics_rollout_prefix_visibility_mode(geometry, config)
        == "video_history_only"
    )


def test_joint_mode_preserves_policy_history_contract() -> None:
    config = _policy_config(
        history_stream_visibility=(HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY)
    )
    geometry = _rollout_geometry(DynamicsObjective.JOINT, config)

    assert (
        geometry.history_stream_visibility
        == HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
    )
    assert (
        dynamics_rollout_prefix_visibility_mode(
            geometry,
            config,
        )
        == "video_queries_video_only"
    )


def test_joint_warmup_history_is_returned_by_identity() -> None:
    video_latents = torch.randn(1, 2, 5, 2, 2)
    action_latents = torch.randn(1, 3, 5, 4, 1)
    config = _policy_config()

    selected_video, selected_action, frame_start, dropped = (
        select_dynamics_warmup_history_suffix(
            video_latents=video_latents,
            action_latents=action_latents,
            frame_start=7,
            geometry=_rollout_geometry(
                DynamicsObjective.JOINT,
                config,
                frame_chunk_size=2,
            ),
        )
    )

    assert selected_video is video_latents
    assert selected_action is action_latents
    assert frame_start == 7
    assert dropped == 0


def test_conditional_warmup_keeps_latest_local_chunk() -> None:
    video_latents = torch.arange(6.0).reshape(1, 1, 6, 1, 1)
    action_latents = torch.arange(60.0, 66.0).reshape(1, 1, 6, 1, 1)
    config = _policy_config()

    selected_video, selected_action, frame_start, dropped = (
        select_dynamics_warmup_history_suffix(
            video_latents=video_latents,
            action_latents=action_latents,
            frame_start=7,
            geometry=_rollout_geometry(
                DynamicsObjective.ACTION_CONDITIONED_VIDEO,
                config,
                frame_chunk_size=2,
            ),
        )
    )

    torch.testing.assert_close(
        selected_video,
        video_latents[:, :, -1:],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        selected_action,
        action_latents[:, :, -1:],
        rtol=0.0,
        atol=0.0,
    )
    assert selected_video.is_contiguous()
    assert selected_action.is_contiguous()
    assert frame_start == 12
    assert dropped == 5


def test_conditional_warmup_handles_empty_history() -> None:
    video_latents = torch.empty(1, 2, 0, 2, 2)
    action_latents = torch.empty(1, 3, 0, 4, 1)
    config = _policy_config()

    selected_video, selected_action, frame_start, dropped = (
        select_dynamics_warmup_history_suffix(
            video_latents=video_latents,
            action_latents=action_latents,
            frame_start=7,
            geometry=_rollout_geometry(
                DynamicsObjective.VIDEO_CONDITIONED_ACTION,
                config,
            ),
        )
    )

    assert selected_video.shape[2] == 0
    assert selected_action.shape[2] == 0
    assert frame_start == 7
    assert dropped == 0


def test_slice_conditioning_chunk_preserves_values_and_gradients() -> None:
    value = torch.arange(6.0).reshape(1, 1, 6, 1, 1)
    value.requires_grad_(True)

    selected = slice_dynamics_conditioning_chunk(
        value,
        target_frames=2,
        source="forced actions",
    )

    assert selected is not None
    torch.testing.assert_close(
        selected,
        value[:, :, :2],
        rtol=0.0,
        atol=0.0,
    )
    assert selected.is_contiguous()
    selected.square().sum().backward()
    expected_grad = torch.zeros_like(value)
    expected_grad[:, :, :2] = 2.0 * value.detach()[:, :, :2]
    torch.testing.assert_close(value.grad, expected_grad, rtol=0.0, atol=0.0)


def test_slice_conditioning_chunk_preserves_none_and_exact_tensor_identity() -> None:
    value = torch.randn(1, 2, 3, 2, 2)

    assert (
        slice_dynamics_conditioning_chunk(
            None,
            target_frames=3,
            source="condition latents",
        )
        is None
    )
    assert (
        slice_dynamics_conditioning_chunk(
            value,
            target_frames=3,
            source="condition latents",
        )
        is value
    )


def test_slice_conditioning_chunk_rejects_insufficient_frames() -> None:
    with pytest.raises(
        ValueError,
        match="condition latents provides 2 frames.*needs 3",
    ):
        slice_dynamics_conditioning_chunk(
            torch.randn(1, 2, 2, 2, 2),
            target_frames=3,
            source="condition latents",
        )


def test_generalist_mode_text_token_selection_is_config_owned() -> None:
    assert not uses_dynamics_mode_text_token(_policy_config())
    assert uses_dynamics_mode_text_token(
        _policy_config(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
            generalist_mode_text_token=True,
        )
    )
