from __future__ import annotations

import pytest
import torch

from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.proprio_conditioning import (
    apply_parallel_chunk_proprio_context,
    build_single_stream_hidden_proprio_context,
    inject_deprecated_proprio_text_context,
)


def test_reference_runtime_proprio_names_alias_canonical_contract() -> None:
    assert (
        reference_runtime._apply_parallel_chunk_proprio_context
        is apply_parallel_chunk_proprio_context
    )
    assert (
        reference_runtime._single_stream_hidden_proprio_context
        is build_single_stream_hidden_proprio_context
    )
    assert (
        reference_runtime._inject_proprio_text_context
        is inject_deprecated_proprio_text_context
    )


def test_deprecated_text_context_preserves_cfg_values_and_gradients() -> None:
    class _TextContextTransformer:
        @staticmethod
        def append_proprio_context_tokens(
            text_emb: torch.Tensor,
            proprio_state: torch.Tensor,
        ) -> torch.Tensor:
            return torch.cat([text_emb, proprio_state[:, None, :]], dim=1)

    text_emb = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
    negative_text_emb = torch.arange(3, dtype=torch.float64).reshape(1, 1, 3)
    proprio_state = torch.tensor([[7.0, 8.0, 9.0]], dtype=torch.float64)
    text_emb.requires_grad_()
    negative_text_emb.requires_grad_()
    proprio_state.requires_grad_()

    conditioned, negative = inject_deprecated_proprio_text_context(
        _TextContextTransformer(),
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        proprio_state=proprio_state,
    )

    assert negative is not None
    torch.testing.assert_close(
        conditioned,
        torch.cat([text_emb, proprio_state[:, None, :]], dim=1),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        negative,
        torch.cat([negative_text_emb, proprio_state[:, None, :]], dim=1),
        rtol=0.0,
        atol=0.0,
    )
    (conditioned.sum() + 2.0 * negative.sum()).backward()
    torch.testing.assert_close(text_emb.grad, torch.ones_like(text_emb))
    torch.testing.assert_close(
        negative_text_emb.grad,
        torch.full_like(negative_text_emb, 2.0),
    )
    torch.testing.assert_close(
        proprio_state.grad,
        torch.full_like(proprio_state, 3.0),
    )


def test_absent_proprio_context_is_an_identity_operation() -> None:
    text_emb = torch.zeros(1, 2, 3)
    negative_text_emb = torch.ones(1, 2, 3)
    conditioned, negative = inject_deprecated_proprio_text_context(
        torch.nn.Identity(),
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        proprio_state=None,
    )
    assert conditioned is text_emb
    assert negative is negative_text_emb

    hidden_states = torch.zeros(1, 4, 3)
    output = apply_parallel_chunk_proprio_context(
        torch.nn.Identity(),
        hidden_states=hidden_states,
        split_list=(1, 1, 1, 1),
        input_dict={},
    )
    assert output is hidden_states

    stream_context = build_single_stream_hidden_proprio_context(
        torch.nn.Identity(),
        proprio_state=None,
        stream_latents=torch.zeros(1, 2, 3, 4, 4),
        action_mode=False,
    )
    assert stream_context is None


def test_single_stream_video_context_uses_latest_anchor_and_patch_geometry() -> None:
    class _HiddenContextTransformer:
        patch_size = (2, 2, 3)

        def __init__(self) -> None:
            self.calls: list[tuple[torch.device, torch.dtype]] = []

        def encode_proprio_hidden_context(
            self,
            frame_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            self.calls.append((device, dtype))
            return frame_state.to(device=device, dtype=dtype)

    transformer = _HiddenContextTransformer()
    proprio_state = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    stream_latents = torch.zeros(1, 2, 4, 4, 6, dtype=torch.float64)

    output = build_single_stream_hidden_proprio_context(
        transformer,
        proprio_state=proprio_state,
        stream_latents=stream_latents,
        action_mode=False,
    )

    assert output is not None
    assert transformer.calls == [(stream_latents.device, stream_latents.dtype)]
    assert output.shape == (1, 8, 2)
    torch.testing.assert_close(
        output,
        torch.tensor([3.0, 4.0], dtype=torch.float64)
        .reshape(1, 1, 2)
        .expand(1, 8, 2),
        rtol=0.0,
        atol=0.0,
    )
    output.sum().backward()
    expected_gradient = torch.zeros_like(proprio_state)
    expected_gradient[:, -1, :] = 8.0
    torch.testing.assert_close(
        proprio_state.grad,
        expected_gradient,
        rtol=0.0,
        atol=0.0,
    )


def test_single_stream_action_context_uses_unpatched_action_tokens() -> None:
    class _HiddenContextTransformer:
        patch_size = (2, 2, 2)

        @staticmethod
        def encode_proprio_hidden_context(
            frame_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return frame_state.to(device=device, dtype=dtype)

    proprio_state = torch.tensor([[2.0, 5.0]], dtype=torch.float32)
    stream_latents = torch.zeros(1, 4, 3, 2, 4, dtype=torch.bfloat16)

    output = build_single_stream_hidden_proprio_context(
        _HiddenContextTransformer(),
        proprio_state=proprio_state,
        stream_latents=stream_latents,
        action_mode=True,
    )

    assert output is not None
    assert output.shape == (1, 24, 2)
    assert output.dtype == stream_latents.dtype
    torch.testing.assert_close(
        output,
        torch.tensor([2.0, 5.0], dtype=torch.bfloat16)
        .reshape(1, 1, 2)
        .expand(1, 24, 2),
        rtol=0.0,
        atol=0.0,
    )


def test_packed_context_preserves_values_and_exact_gradients() -> None:
    class _HiddenContextTransformer:
        patch_size = (1, 1, 1)

        @staticmethod
        def encode_proprio_hidden_context(
            frame_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return frame_state.to(device=device, dtype=dtype)

    hidden_states = torch.arange(
        16,
        dtype=torch.float64,
    ).reshape(1, 8, 2)
    hidden_states.requires_grad_()
    proprio_state = torch.tensor(
        [[[10.0, 20.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected_context = proprio_state.detach().expand(1, 2, 2)

    output = apply_parallel_chunk_proprio_context(
        _HiddenContextTransformer(),
        hidden_states=hidden_states,
        split_list=(2, 2, 2, 2),
        input_dict={
            "chunk_size": 2,
            "latent_dict": {
                "noisy_latents": torch.zeros(1, 1, 2, 1, 1),
            },
            "action_dict": {
                "noisy_latents": torch.zeros(1, 1, 2, 1, 1),
            },
            "per_chunk_proprio_state": proprio_state,
        },
    )

    for start in (0, 2, 4, 6):
        torch.testing.assert_close(
            output[:, start : start + 2, :],
            hidden_states[:, start : start + 2, :] + expected_context,
            rtol=0.0,
            atol=0.0,
        )
    output.sum().backward()
    torch.testing.assert_close(
        hidden_states.grad,
        torch.ones_like(hidden_states),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        proprio_state.grad,
        torch.full_like(proprio_state, 8.0),
        rtol=0.0,
        atol=0.0,
    )


def test_packed_context_applies_frame_boundaries_to_pre_target_prefix() -> None:
    class _ContextTransformer:
        patch_size = (1, 1, 1)

        @staticmethod
        def encode_proprio_hidden_context(
            frame_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return frame_state.to(device=device, dtype=dtype).expand(-1, -1, 4)

    output = apply_parallel_chunk_proprio_context(
        _ContextTransformer(),
        hidden_states=torch.zeros(1, 16, 4),
        split_list=(4, 4, 4, 4),
        input_dict={
            "chunk_size": 2,
            "chunk_origin_frame": 1,
            "per_chunk_proprio_state_granularity": "frame",
            "latent_dict": {
                "noisy_latents": torch.zeros(1, 1, 4, 1, 1),
            },
            "action_dict": {
                "noisy_latents": torch.zeros(1, 1, 4, 1, 1),
            },
            "per_chunk_proprio_state": torch.tensor(
                [[[1.0], [2.0], [3.0], [4.0]]]
            ),
        },
    )

    expected_context = torch.tensor(
        [[[[1.0] * 4, [1.0] * 4, [1.0] * 4, [3.0] * 4]]],
        dtype=output.dtype,
    ).reshape(1, 4, 4)
    for start in (0, 4, 8, 12):
        torch.testing.assert_close(
            output[:, start : start + 4, :],
            expected_context,
        )


def test_legacy_prefix_context_skips_video_branches() -> None:
    class _ContextTransformer:
        patch_size = (1, 1, 1)

        @staticmethod
        def encode_proprio_hidden_context(
            frame_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return frame_state.to(device=device, dtype=dtype).expand(-1, -1, 4)

    output = apply_parallel_chunk_proprio_context(
        _ContextTransformer(),
        hidden_states=torch.zeros(1, 18, 4),
        split_list=(5, 5, 4, 4),
        input_dict={
            "chunk_size": 2,
            "prefix_condition_frames": 1,
            "per_chunk_proprio_apply_to_video": False,
            "per_chunk_proprio_state_granularity": "frame",
            "latent_dict": {
                "noisy_latents": torch.zeros(1, 1, 5, 1, 1),
            },
            "action_dict": {
                "noisy_latents": torch.zeros(1, 1, 4, 1, 1),
            },
            "per_chunk_proprio_state": torch.tensor(
                [[[1.0], [2.0], [3.0], [4.0], [5.0]]]
            ),
        },
    )

    expected_action_context = torch.tensor(
        [[[[1.0] * 4, [1.0] * 4, [3.0] * 4, [3.0] * 4]]],
        dtype=output.dtype,
    ).reshape(1, 4, 4)
    torch.testing.assert_close(output[:, :10, :], torch.zeros(1, 10, 4))
    torch.testing.assert_close(output[:, 10:14, :], expected_action_context)
    torch.testing.assert_close(output[:, 14:18, :], expected_action_context)


def test_legacy_prefix_context_accepts_chunk_level_state() -> None:
    class _ContextTransformer:
        patch_size = (1, 1, 1)

        @staticmethod
        def encode_proprio_hidden_context(
            frame_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return frame_state.to(device=device, dtype=dtype).expand(-1, -1, 4)

    output = apply_parallel_chunk_proprio_context(
        _ContextTransformer(),
        hidden_states=torch.zeros(1, 18, 4),
        split_list=(5, 5, 4, 4),
        input_dict={
            "chunk_size": 2,
            "prefix_condition_frames": 1,
            "per_chunk_proprio_apply_to_video": False,
            "per_chunk_proprio_state_granularity": "chunk",
            "latent_dict": {
                "noisy_latents": torch.zeros(1, 1, 5, 1, 1),
            },
            "action_dict": {
                "noisy_latents": torch.zeros(1, 1, 4, 1, 1),
            },
            "per_chunk_proprio_state": torch.tensor(
                [[[1.0], [2.0], [4.0]]]
            ),
        },
    )

    expected_action_context = torch.tensor(
        [[[[2.0] * 4, [2.0] * 4, [4.0] * 4, [4.0] * 4]]],
        dtype=output.dtype,
    ).reshape(1, 4, 4)
    torch.testing.assert_close(output[:, :10, :], torch.zeros(1, 10, 4))
    torch.testing.assert_close(output[:, 10:14, :], expected_action_context)
    torch.testing.assert_close(output[:, 14:18, :], expected_action_context)


def test_chunk_size_one_treats_state_as_chunk_level() -> None:
    class _ContextTransformer:
        patch_size = (1, 1, 1)

        @staticmethod
        def encode_proprio_hidden_context(
            frame_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return frame_state.to(device=device, dtype=dtype).expand(-1, -1, 4)

    output = apply_parallel_chunk_proprio_context(
        _ContextTransformer(),
        hidden_states=torch.zeros(1, 12, 4),
        split_list=(3, 3, 3, 3),
        input_dict={
            "chunk_size": 1,
            "latent_dict": {
                "noisy_latents": torch.zeros(1, 1, 3, 1, 1),
            },
            "action_dict": {
                "noisy_latents": torch.zeros(1, 1, 3, 1, 1),
            },
            "per_chunk_proprio_state": torch.tensor(
                [[[10.0], [20.0], [30.0]]]
            ),
        },
    )

    expected_context = torch.tensor(
        [[[[10.0] * 4, [20.0] * 4, [30.0] * 4]]],
        dtype=output.dtype,
    ).reshape(1, 3, 4)
    for start in (0, 3, 6, 9):
        torch.testing.assert_close(
            output[:, start : start + 3, :],
            expected_context,
        )


def test_proprio_contract_rejects_missing_transformer_hooks() -> None:
    with pytest.raises(
        ValueError,
        match="support proprio context appending",
    ):
        inject_deprecated_proprio_text_context(
            torch.nn.Identity(),
            text_emb=torch.zeros(1, 2, 3),
            negative_text_emb=None,
            proprio_state=torch.zeros(1, 3),
        )

    with pytest.raises(
        ValueError,
        match="encode_proprio_hidden_context",
    ):
        build_single_stream_hidden_proprio_context(
            torch.nn.Identity(),
            proprio_state=torch.zeros(1, 3),
            stream_latents=torch.zeros(1, 2, 3, 4, 4),
            action_mode=False,
        )


@pytest.mark.parametrize(
    ("proprio_state", "expected_message"),
    [
        (torch.zeros(1, 2, 3, 4), "expects state with shape"),
        (torch.zeros(2, 3), "batch mismatch"),
    ],
)
def test_single_stream_context_rejects_invalid_state_layout(
    proprio_state: torch.Tensor,
    expected_message: str,
) -> None:
    class _HiddenContextTransformer:
        patch_size = (1, 1, 1)

        @staticmethod
        def encode_proprio_hidden_context(
            frame_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return frame_state.to(device=device, dtype=dtype)

    with pytest.raises(ValueError, match=expected_message):
        build_single_stream_hidden_proprio_context(
            _HiddenContextTransformer(),
            proprio_state=proprio_state,
            stream_latents=torch.zeros(1, 2, 3, 4, 4),
            action_mode=False,
        )


def test_packed_context_rejects_non_tensor_and_unknown_granularity() -> None:
    class _HiddenContextTransformer:
        patch_size = (1, 1, 1)

        @staticmethod
        def encode_proprio_hidden_context(
            frame_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return frame_state.to(device=device, dtype=dtype)

    base_input = {
        "chunk_size": 1,
        "latent_dict": {
            "noisy_latents": torch.zeros(1, 1, 1, 1, 1),
        },
        "action_dict": {
            "noisy_latents": torch.zeros(1, 1, 1, 1, 1),
        },
    }
    with pytest.raises(ValueError, match="must be a tensor"):
        apply_parallel_chunk_proprio_context(
            _HiddenContextTransformer(),
            hidden_states=torch.zeros(1, 4, 3),
            split_list=(1, 1, 1, 1),
            input_dict={
                **base_input,
                "per_chunk_proprio_state": "invalid",
            },
        )

    with pytest.raises(
        ValueError,
        match="per_chunk_proprio_state_granularity",
    ):
        apply_parallel_chunk_proprio_context(
            _HiddenContextTransformer(),
            hidden_states=torch.zeros(1, 4, 3),
            split_list=(1, 1, 1, 1),
            input_dict={
                **base_input,
                "per_chunk_proprio_state": torch.zeros(1, 1, 3),
                "per_chunk_proprio_state_granularity": "segment",
            },
        )
