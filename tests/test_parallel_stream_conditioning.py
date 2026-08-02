from __future__ import annotations

from types import SimpleNamespace

import torch

from open_wam.configs import ParallelRuntimeMode, ProprioContextMode
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.policy_variants.contracts import PolicyTrainBatch
from open_wam.models.policy_variants.parallel_stream.conditioning import (
    ParallelStreamConditioning,
)


def _conditioning(
    *,
    runtime_mode: ParallelRuntimeMode = ParallelRuntimeMode.LINGBOT_EXACT,
) -> ParallelStreamConditioning:
    return ParallelStreamConditioning(
        ParallelStreamPolicyConfig(
            runtime_mode=runtime_mode,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        )
    )


def test_train_hidden_proprio_context_preserves_runtime_source_priority_and_mask() -> None:
    frame_state = torch.tensor([[[1.0], [2.0]]], requires_grad=True)
    frame_mask = torch.tensor([[[1.0], [0.0]]])
    chunk_state = torch.tensor([[[3.0]]], requires_grad=True)
    batch = PolicyTrainBatch(
        actions=torch.zeros(1, 1, 1),
        extra={
            "proprio_context_frames": frame_state,
            "proprio_context_frames_mask": frame_mask,
            "proprio_context_state": chunk_state,
        },
    )

    standard_payload = _conditioning().resolve_train_hidden_proprio_context(
        batch,
        label="test",
    )
    fastwam_payload = _conditioning(
        runtime_mode=ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
    ).resolve_train_hidden_proprio_context(batch, label="test")

    assert standard_payload is not None
    assert fastwam_payload is not None
    standard_state, standard_granularity = standard_payload
    fastwam_state, fastwam_granularity = fastwam_payload
    torch.testing.assert_close(
        standard_state,
        torch.tensor([[[1.0], [0.0]]]),
        rtol=0.0,
        atol=0.0,
    )
    assert standard_granularity == "frame"
    assert fastwam_state is chunk_state
    assert fastwam_granularity == "chunk"

    standard_state.sum().backward()
    torch.testing.assert_close(frame_state.grad, frame_mask, rtol=0.0, atol=0.0)


def test_rollout_state_selection_and_cache_clone_preserve_m1_semantics() -> None:
    state = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    standard = _conditioning()
    fastwam = _conditioning(runtime_mode=ParallelRuntimeMode.FASTWAM_FIRST_FRAME)

    torch.testing.assert_close(
        standard.select_rollout_proprio_state(state),
        state[:, -1, :],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fastwam.select_rollout_proprio_state(state),
        state[:, 0, :],
        rtol=0.0,
        atol=0.0,
    )

    state.requires_grad_()
    cache: dict[str, torch.Tensor] = {}
    standard.cache_infer_proprio_state(cache, state)
    cached = cache["last_proprio_state"]
    assert cached.data_ptr() != state.data_ptr()
    assert cached.requires_grad is False
    torch.testing.assert_close(cached, state, rtol=0.0, atol=0.0)


def test_prefix_hidden_proprio_alignment_keeps_condition_frame_separate() -> None:
    conditioning = _conditioning()
    artifacts = SimpleNamespace(input_dict={"prefix_condition_frames": 1})
    batch = PolicyTrainBatch(
        actions=torch.zeros(1, 1, 1),
        state=torch.tensor([[[2.0], [4.0]]]),
    )
    target_state = torch.tensor([[[10.0], [20.0], [30.0]]])
    video_latents = torch.zeros(1, 1, 3, 1, 1, dtype=torch.float64)

    conditioning.attach_train_hidden_proprio_context(
        artifacts,
        batch=batch,
        video_latents=video_latents,
        payload=(target_state, "frame"),
    )

    torch.testing.assert_close(
        artifacts.input_dict["per_chunk_proprio_state"],
        torch.tensor([[[4.0], [10.0], [20.0], [30.0]]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert artifacts.input_dict["per_chunk_proprio_state_granularity"] == "frame"


def test_train_condition_latents_preserve_values_dtype_and_storage() -> None:
    conditioning = ParallelStreamConditioning(
        ParallelStreamPolicyConfig(use_condition_latents=True)
    )
    video_latents = torch.zeros(1, 2, 3, 4, 5, dtype=torch.float64)
    condition_latents = torch.arange(
        video_latents.numel(),
        dtype=torch.float64,
    ).reshape_as(video_latents)
    batch = PolicyTrainBatch(
        actions=torch.zeros(1, 1, 1),
        extra={"condition_latents": condition_latents},
    )

    resolved = conditioning.resolve_train_condition_latents(
        batch,
        video_latents=video_latents,
    )

    assert resolved is condition_latents
    torch.testing.assert_close(resolved, condition_latents, rtol=0.0, atol=0.0)
