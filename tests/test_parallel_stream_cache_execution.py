from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import (
    CurrentBlockCoupling,
    ParallelExactCacheWriteMode,
    SharedVideoTransformerConfig,
)
from open_wam.models.policy_variants.parallel_stream import cache_execution
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.cache_execution import (
    build_joint_clean_cache_attention_mask,
    build_joint_clean_cache_attention_profile,
    summarize_slot_pool_cache_state,
    write_exact_cache_chunk,
    write_joint_clean_tokens_to_exact_cache,
)
from open_wam.models.policy_variants.parallel_stream.exact_cache import (
    ExactCacheInterfaceSpec,
)


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


def _write_kwargs() -> dict[str, object]:
    return {
        "transformer": object(),
        "cache_name": "cache",
        "frame_start": 3,
        "backbone_config": _backbone(),
        "video_latents": torch.zeros(1, 2, 1, 1, 1),
        "action_latents": torch.zeros(1, 3, 1, 2, 1),
        "text_emb": torch.zeros(1, 2, 4),
        "negative_text_emb": None,
        "use_cfg": False,
        "action_channel_mask": None,
        "update_cache": 2,
        "chunk_size": 1,
        "window_size": 4,
        "current_block_coupling": CurrentBlockCoupling.JOINT,
        "preserve_video_pretrain_history": True,
    }


def test_reference_runtime_cache_execution_names_alias_canonical_contract() -> None:
    assert (
        reference_runtime._build_joint_clean_cache_attention_mask
        is build_joint_clean_cache_attention_mask
    )
    assert (
        reference_runtime._build_joint_clean_cache_attention_profile
        is build_joint_clean_cache_attention_profile
    )
    assert (
        reference_runtime._summarize_slot_pool_cache_state
        is summarize_slot_pool_cache_state
    )
    assert reference_runtime._write_exact_cache_chunk is write_exact_cache_chunk
    assert (
        reference_runtime._write_joint_clean_tokens_to_exact_cache
        is write_joint_clean_tokens_to_exact_cache
    )


def test_joint_packed_write_delegates_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_joint_write(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(
        cache_execution,
        "write_joint_clean_tokens_to_exact_cache",
        fake_joint_write,
    )
    kwargs = _write_kwargs()
    original_video = kwargs["video_latents"]
    original_actions = kwargs["action_latents"]

    write_exact_cache_chunk(
        cache_spec=ExactCacheInterfaceSpec(
            write_mode=ParallelExactCacheWriteMode.JOINT_PACKED
        ),
        **kwargs,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["latents"] is original_video
    assert call["actions"] is original_actions
    assert call["transformer"] is kwargs["transformer"]
    assert call["cache_name"] == "cache"
    assert call["frame_start"] == 3
    assert call["current_block_coupling"] == CurrentBlockCoupling.JOINT
    assert call["preserve_video_pretrain_history"] is True


def test_cache_write_rejects_unknown_interface_before_model_execution() -> None:
    kwargs = _write_kwargs()

    with pytest.raises(
        ValueError,
        match="Unsupported exact cache write_mode",
    ):
        write_exact_cache_chunk(
            cache_spec=ExactCacheInterfaceSpec(write_mode="unknown"),  # type: ignore[arg-type]
            **kwargs,
        )


def test_slot_pool_summary_reports_cached_and_prediction_tokens() -> None:
    layer_state = SimpleNamespace(
        slot_mask=torch.tensor([True, False, True, True]),
        prediction_mask=torch.tensor([False, True, True, False]),
    )
    cache_state = SimpleNamespace(
        backend_name="slot_pool_exact",
        backend_payload=SimpleNamespace(layer_states=[layer_state]),
    )

    class _Transformer:
        @staticmethod
        def _resolve_exact_cache_state(cache_name: str):
            assert cache_name == "cache"
            return cache_state

    assert summarize_slot_pool_cache_state(_Transformer(), "cache") == {
        "cached_tokens": 3,
        "prediction_tokens": 1,
        "total_slots": 4,
    }


@pytest.mark.parametrize(
    "transformer",
    [
        object(),
        SimpleNamespace(
            _resolve_exact_cache_state=lambda _name: SimpleNamespace(
                backend_name="merged_prefix",
                backend_payload=None,
            )
        ),
        SimpleNamespace(
            _resolve_exact_cache_state=lambda _name: SimpleNamespace(
                backend_name="slot_pool_exact",
                backend_payload=SimpleNamespace(layer_states=[]),
            )
        ),
    ],
)
def test_slot_pool_summary_returns_none_without_supported_state(
    transformer: object,
) -> None:
    assert summarize_slot_pool_cache_state(transformer, "cache") is None
