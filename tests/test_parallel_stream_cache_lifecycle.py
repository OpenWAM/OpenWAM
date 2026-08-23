from __future__ import annotations

import pytest
import torch

from open_wam.configs import (
    InferenceConfig,
    ParallelExactCacheWriteMode,
    ParallelStreamPolicyConfig,
    VideoActionProgram,
)
from open_wam.models.policy_variants.parallel_stream import cache_lifecycle
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.cache_lifecycle import (
    commit_initial_observed_video_context,
    run_parallel_exact_cache_warmup,
)
from open_wam.models.policy_variants.parallel_stream.exact_cache import (
    ExactCacheInterfaceSpec,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig


def _backbone() -> SharedVideoTransformerConfig:
    return SharedVideoTransformerConfig(
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        attention_head_dim=8,
        ffn_dim=16,
        text_dim=4,
        freq_dim=4,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )


def _policy() -> ParallelStreamPolicyConfig:
    return ParallelStreamPolicyConfig(
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        hidden_size=8,
        frame_chunk_size=2,
        action_per_frame=2,
    )


def _commit_kwargs() -> dict[str, object]:
    return {
        "transformer": object(),
        "cache_spec": ExactCacheInterfaceSpec(
            write_mode=ParallelExactCacheWriteMode.JOINT_PACKED,
        ),
        "cache_name": "cache",
        "backbone_config": _backbone(),
        "policy_config": _policy(),
        "condition_latents": torch.ones(1, 3, 1, 2, 2),
        "text_emb": torch.zeros(1, 2, 4),
        "negative_text_emb": None,
        "use_cfg": False,
        "action_channel_mask": None,
        "action_dim": 5,
        "model_dtype": torch.float32,
        "current_frame_start": 0,
        "step_index": 0,
        "current_block_coupling": "joint",
        "window_size": 4,
    }


def test_reference_runtime_cache_lifecycle_names_alias_owner() -> None:
    assert (
        reference_runtime._maybe_commit_initial_observed_video_context
        is commit_initial_observed_video_context
    )
    assert (
        reference_runtime.run_parallel_exact_cache_warmup
        is run_parallel_exact_cache_warmup
    )


def test_initial_observation_commit_is_inert_without_cache() -> None:
    frame_start, committed = commit_initial_observed_video_context(
        inference_config=InferenceConfig(
            frame_chunk_size=2,
            use_cache=False,
        ),
        **_commit_kwargs(),
    )

    assert frame_start == 0
    assert committed is False


def test_initial_observation_commit_uses_staged_video_only_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cache_lifecycle,
        "_write_exact_cache_chunk",
        lambda **kwargs: calls.append(dict(kwargs)),
    )

    frame_start, committed = commit_initial_observed_video_context(
        inference_config=InferenceConfig(
            frame_chunk_size=2,
            use_cache=True,
        ),
        **_commit_kwargs(),
    )

    assert frame_start == 1
    assert committed is True
    assert len(calls) == 1
    call = calls[0]
    assert call["frame_start"] == 0
    assert call["update_cache"] == 2
    assert call["cache_spec"].write_mode == ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED
    assert call["video_latents"].shape == (1, 3, 1, 2, 2)
    assert call["action_latents"].shape == (1, 5, 0, 2, 1)
