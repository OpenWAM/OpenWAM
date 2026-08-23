from __future__ import annotations

import pytest
import torch

from open_wam.configs import (
    DualExpertPolicyConfig,
    ParallelStreamPolicyConfig,
    VideoActionProgram,
)
from open_wam.models.common.video_conditioning import (
    build_repeated_first_frame_condition,
    resolve_full_window_condition_latents,
    select_first_frame_condition_latents,
)
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.models.policy_variants.dual_expert.conditioning import (
    DualExpertConditioning,
)
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.conditioning import (
    ParallelStreamConditioning,
)


def _training_conditioners():
    shared = {
        "program": VideoActionProgram.VIDEO_THEN_ACTION,
        "use_condition_latents": True,
        "require_condition_latents": True,
    }
    return (
        DualExpertConditioning(DualExpertPolicyConfig(**shared)),
        ParallelStreamConditioning(ParallelStreamPolicyConfig(**shared)),
    )


def test_policy_architectures_accept_the_same_canonical_condition_layout() -> None:
    video_latents = torch.zeros(1, 3, 4, 2, 2, dtype=torch.float64)
    condition_latents = torch.ones(1, 3, 1, 2, 2, dtype=torch.float32)
    batch = PolicyTrainBatch(
        actions=torch.zeros(1, 1, 1),
        extra={"condition_latents": condition_latents},
    )

    resolved = tuple(
        conditioning.resolve_train_condition_latents(
            batch,
            video_latents=video_latents,
        )
        for conditioning in _training_conditioners()
    )

    for value in resolved:
        assert value is not None
        assert value.dtype == video_latents.dtype
        torch.testing.assert_close(
            value,
            condition_latents.to(dtype=video_latents.dtype),
            rtol=0.0,
            atol=0.0,
        )


def test_policy_architectures_reject_noncanonical_condition_layout_equally() -> None:
    video_latents = torch.zeros(1, 3, 4, 2, 2)
    time_first_condition = torch.zeros(1, 4, 3, 2, 2)
    batch = PolicyTrainBatch(
        actions=torch.zeros(1, 1, 1),
        extra={"condition_latents": time_first_condition},
    )

    for conditioning in _training_conditioners():
        with pytest.raises(
            ValueError,
            match="batch/channel dimensions must match video_latents",
        ):
            conditioning.resolve_train_condition_latents(
                batch,
                video_latents=video_latents,
            )


def test_reference_runtime_video_conditioning_names_alias_canonical_contract() -> None:
    assert (
        reference_runtime._build_clean_video_condition_from_anchor
        is build_repeated_first_frame_condition
    )
    assert (
        reference_runtime._resolve_full_condition_latents
        is resolve_full_window_condition_latents
    )
    assert (
        reference_runtime._select_first_frame_condition_latents
        is select_first_frame_condition_latents
    )


def test_first_frame_fallback_is_a_view_and_repeat_preserves_exact_gradients() -> None:
    video_latents = torch.arange(
        2 * 3 * 4 * 2 * 2,
        dtype=torch.float64,
    ).reshape(2, 3, 4, 2, 2)
    video_latents.requires_grad_()

    selected, source = select_first_frame_condition_latents(
        video_latents,
        label="Test",
    )

    assert source == "video_latents"
    assert selected.untyped_storage().data_ptr() == video_latents.untyped_storage().data_ptr()
    torch.testing.assert_close(
        selected,
        video_latents[:, :, :1],
        rtol=0.0,
        atol=0.0,
    )

    repeated = build_repeated_first_frame_condition(
        video_latents,
        target_frames=3,
    )
    torch.testing.assert_close(
        repeated,
        video_latents[:, :, :1].repeat(1, 1, 3, 1, 1),
        rtol=0.0,
        atol=0.0,
    )
    repeated.sum().backward()

    expected_gradient = torch.zeros_like(video_latents)
    expected_gradient[:, :, :1] = 3
    torch.testing.assert_close(
        video_latents.grad,
        expected_gradient,
        rtol=0.0,
        atol=0.0,
    )


def test_first_frame_explicit_condition_casts_and_preserves_exact_gradients() -> None:
    video_latents = torch.zeros(1, 2, 3, 2, 2, dtype=torch.float64)
    condition_latents = torch.arange(
        1 * 2 * 2 * 2 * 2,
        dtype=torch.float32,
    ).reshape(1, 2, 2, 2, 2)
    condition_latents.requires_grad_()

    selected, source = select_first_frame_condition_latents(
        video_latents,
        condition_latents=condition_latents,
        label="Explicit",
    )

    assert source == "condition_latents"
    assert selected.dtype == video_latents.dtype
    assert selected.device == video_latents.device
    torch.testing.assert_close(
        selected,
        condition_latents[:, :, :1].to(dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )

    weights = torch.arange(selected.numel(), dtype=torch.float64).reshape_as(selected)
    (selected * weights).sum().backward()
    expected_gradient = torch.zeros_like(condition_latents)
    expected_gradient[:, :, :1] = weights.to(dtype=condition_latents.dtype)
    torch.testing.assert_close(
        condition_latents.grad,
        expected_gradient,
        rtol=0.0,
        atol=0.0,
    )


def test_full_window_condition_preserves_fallback_and_explicit_tensor_identity() -> None:
    video_latents = torch.zeros(1, 2, 3, 2, 2)

    fallback, fallback_source = resolve_full_window_condition_latents(
        video_latents,
        None,
        label="Full",
    )
    assert fallback is None
    assert fallback_source == "video_latents"

    condition_latents = torch.ones_like(video_latents, requires_grad=True)
    explicit, explicit_source = resolve_full_window_condition_latents(
        video_latents,
        condition_latents,
        label="Full",
    )
    assert explicit is condition_latents
    assert explicit_source == "condition_latents"

    explicit.square().sum().backward()
    torch.testing.assert_close(
        condition_latents.grad,
        torch.full_like(condition_latents, 2),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("condition_latents", "message"),
    [
        (
            torch.zeros(1, 2, 3, 2),
            r"First condition_latents must have shape `\[B, C, T, H, W\]`",
        ),
        (
            torch.zeros(2, 2, 3, 2, 2),
            "First condition_latents batch/channel dimensions must match",
        ),
        (
            torch.zeros(1, 2, 0, 2, 2),
            "First condition_latents must contain at least one latent frame",
        ),
        (
            torch.zeros(1, 2, 3, 3, 2),
            "First condition_latents spatial shape must match",
        ),
    ],
)
def test_first_frame_condition_rejects_incompatible_tensors(
    condition_latents: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_first_frame_condition_latents(
            torch.zeros(1, 2, 3, 2, 2),
            condition_latents=condition_latents,
            label="First",
        )


def test_first_frame_condition_rejects_invalid_video_rank() -> None:
    with pytest.raises(
        ValueError,
        match=r"Expected video latents shaped \[B, C, T, H, W\]",
    ):
        select_first_frame_condition_latents(
            torch.zeros(1, 2, 3, 2),
            label="First",
        )


@pytest.mark.parametrize(
    "condition_latents",
    [
        torch.zeros(1, 2, 3, 2),
        torch.zeros(1, 2, 4, 2, 2),
    ],
)
def test_full_window_condition_rejects_incompatible_tensors(
    condition_latents: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="Full condition_latents"):
        resolve_full_window_condition_latents(
            torch.zeros(1, 2, 3, 2, 2),
            condition_latents,
            label="Full",
        )


@pytest.mark.parametrize("target_frames", [0, -1])
def test_repeated_first_frame_condition_requires_positive_target_frames(
    target_frames: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Current-frame action chunks require positive target_frames",
    ):
        build_repeated_first_frame_condition(
            torch.zeros(1, 2, 3, 2, 2),
            target_frames=target_frames,
        )
