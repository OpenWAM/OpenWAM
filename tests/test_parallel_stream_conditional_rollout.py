from __future__ import annotations

import pytest
import torch

from open_wam.configs import (
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    ParallelHistoryStreamVisibility,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    ParallelStreamVariantProfile,
)
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.conditional_rollout import (
    generalist_conditioning_chunk_size,
    generalist_conditioning_history_stream_visibility,
    generalist_conditioning_prefix_visibility_mode,
    generalist_conditioning_window_size,
    is_conditional_joint_denoise_mode,
    resolve_action_conditioning_mode,
    select_conditional_warmup_history_suffix,
    slice_conditioning_chunk,
    uses_generalist_mode_text_token,
)


def _policy_config(**overrides: object) -> ParallelStreamPolicyConfig:
    return ParallelStreamPolicyConfig(hidden_size=32, **overrides)


def test_reference_runtime_conditional_rollout_names_alias_canonical_contract() -> None:
    aliases = {
        "_chunk_size_for_generalist_conditioning": (
            generalist_conditioning_chunk_size
        ),
        "_generalist_mode_for_action_conditioning": (
            resolve_action_conditioning_mode
        ),
        "_history_stream_visibility_for_generalist_conditioning": (
            generalist_conditioning_history_stream_visibility
        ),
        "_is_conditional_joint_denoise_mode": is_conditional_joint_denoise_mode,
        "_prefix_visibility_mode_for_generalist_conditioning": (
            generalist_conditioning_prefix_visibility_mode
        ),
        "_select_conditional_warmup_history_suffix": (
            select_conditional_warmup_history_suffix
        ),
        "_slice_conditioning_chunk": slice_conditioning_chunk,
        "_uses_generalist_mode_text_token": uses_generalist_mode_text_token,
        "_window_size_for_generalist_conditioning": (
            generalist_conditioning_window_size
        ),
    }

    for legacy_name, canonical in aliases.items():
        assert getattr(reference_runtime, legacy_name) is canonical


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("joint", JointDenoiseTrainingMode.JOINT),
        ("vanilla_joint_rollout", JointDenoiseTrainingMode.JOINT),
        ("clean_action_feedback", JointDenoiseTrainingMode.JOINT),
        ("fdm", JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO),
        (
            "forced_action_joint_fdm",
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        ),
        ("idm", JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION),
        (
            JointDenoiseTrainingMode.JOINT,
            JointDenoiseTrainingMode.JOINT,
        ),
        (
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        ),
        (
            JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
            JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
        ),
    ],
)
def test_resolve_action_conditioning_mode_supports_runtime_labels(
    value: JointDenoiseTrainingMode | str,
    expected: JointDenoiseTrainingMode,
) -> None:
    assert resolve_action_conditioning_mode(value) == expected


def test_resolve_action_conditioning_mode_rejects_unknown_label() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported joint-denoise rollout mode 'unknown'",
    ):
        resolve_action_conditioning_mode("unknown")


@pytest.mark.parametrize(
    ("mode", "is_conditional", "window_size", "chunk_size"),
    [
        (JointDenoiseTrainingMode.JOINT, False, 30, 4),
        (
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
            True,
            3,
            1,
        ),
        (
            JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
            True,
            3,
            1,
        ),
    ],
)
def test_conditional_rollout_geometry_matches_shared_gjd_contract(
    mode: JointDenoiseTrainingMode,
    is_conditional: bool,
    window_size: int,
    chunk_size: int,
) -> None:
    assert is_conditional_joint_denoise_mode(mode) is is_conditional
    assert (
        generalist_conditioning_window_size(
            mode,
            fallback_window_size=30,
        )
        == window_size
    )
    assert (
        generalist_conditioning_chunk_size(
            mode,
            fallback_chunk_size=4,
        )
        == chunk_size
    )


@pytest.mark.parametrize(
    "mode",
    [
        JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
    ],
)
def test_conditional_modes_force_video_only_history(
    mode: JointDenoiseTrainingMode,
) -> None:
    config = _policy_config(
        history_stream_visibility=(
            ParallelHistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
        )
    )

    assert (
        generalist_conditioning_history_stream_visibility(mode, config)
        == ParallelHistoryStreamVisibility.VIDEO_ONLY
    )
    assert (
        generalist_conditioning_prefix_visibility_mode(mode, config)
        == "video_history_only"
    )


def test_joint_mode_preserves_policy_history_contract() -> None:
    config = _policy_config(
        history_stream_visibility=(
            ParallelHistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
        )
    )

    assert (
        generalist_conditioning_history_stream_visibility(
            JointDenoiseTrainingMode.JOINT,
            config,
        )
        == ParallelHistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
    )
    assert (
        generalist_conditioning_prefix_visibility_mode(
            JointDenoiseTrainingMode.JOINT,
            config,
        )
        == "preserve_video_pretrain_history"
    )


def test_joint_warmup_history_is_returned_by_identity() -> None:
    video_latents = torch.randn(1, 2, 5, 2, 2)
    action_latents = torch.randn(1, 3, 5, 4, 1)

    selected_video, selected_action, frame_start, dropped = (
        select_conditional_warmup_history_suffix(
            video_latents=video_latents,
            action_latents=action_latents,
            frame_start=7,
            frame_chunk_size=2,
            mode=JointDenoiseTrainingMode.JOINT,
        )
    )

    assert selected_video is video_latents
    assert selected_action is action_latents
    assert frame_start == 7
    assert dropped == 0


def test_conditional_warmup_keeps_latest_local_chunk() -> None:
    video_latents = torch.arange(6.0).reshape(1, 1, 6, 1, 1)
    action_latents = torch.arange(60.0, 66.0).reshape(1, 1, 6, 1, 1)

    selected_video, selected_action, frame_start, dropped = (
        select_conditional_warmup_history_suffix(
            video_latents=video_latents,
            action_latents=action_latents,
            frame_start=7,
            frame_chunk_size=2,
            mode=JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        )
    )

    torch.testing.assert_close(
        selected_video,
        video_latents[:, :, -2:],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        selected_action,
        action_latents[:, :, -2:],
        rtol=0.0,
        atol=0.0,
    )
    assert selected_video.is_contiguous()
    assert selected_action.is_contiguous()
    assert frame_start == 11
    assert dropped == 4


def test_conditional_warmup_handles_empty_history() -> None:
    video_latents = torch.empty(1, 2, 0, 2, 2)
    action_latents = torch.empty(1, 3, 0, 4, 1)

    selected_video, selected_action, frame_start, dropped = (
        select_conditional_warmup_history_suffix(
            video_latents=video_latents,
            action_latents=action_latents,
            frame_start=7,
            frame_chunk_size=0,
            mode=JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
        )
    )

    assert selected_video.shape[2] == 0
    assert selected_action.shape[2] == 0
    assert frame_start == 7
    assert dropped == 0


def test_slice_conditioning_chunk_preserves_values_and_gradients() -> None:
    value = torch.arange(6.0).reshape(1, 1, 6, 1, 1)
    value.requires_grad_(True)

    selected = slice_conditioning_chunk(
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
        slice_conditioning_chunk(
            None,
            target_frames=3,
            source="condition latents",
        )
        is None
    )
    assert (
        slice_conditioning_chunk(
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
        slice_conditioning_chunk(
            torch.randn(1, 2, 2, 2, 2),
            target_frames=3,
            source="condition latents",
        )


def test_generalist_mode_text_token_selection_is_config_owned() -> None:
    assert not uses_generalist_mode_text_token(_policy_config())
    assert uses_generalist_mode_text_token(
        _policy_config(
            runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            variant_profile=(
                ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
            ),
            current_block_coupling=CurrentBlockCoupling.JOINT,
            video_condition_on_action=True,
            generalist_mode_text_token=True,
        )
    )
