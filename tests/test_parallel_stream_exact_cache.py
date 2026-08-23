from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import (
    InferenceConfig,
    ParallelExactCacheWriteMode,
    ParallelStreamPolicyConfig,
    SharedVideoTransformerConfig,
    VideoActionProgram,
)
from open_wam.models.common import SlotPoolLayerState
from open_wam.models.policy_variants.parallel_stream import exact_cache
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.exact_cache import (
    ExactCacheContext,
    ExactCacheInterfaceSpec,
    build_clean_video_action_cache_stream_ids,
    build_dual_stream_cache_stream_ids,
    build_exact_cache_spec,
    count_single_stream_action_tokens,
    ensure_exact_cache_initialized,
    ensure_exact_text_embeddings,
    existing_exact_cache_attention_window,
    restore_slot_pool_layer_metadata,
    resolve_exact_cache_context,
    set_slot_pool_layer_metadata,
    validate_existing_exact_cache_attention_window,
)


def test_reference_runtime_exact_cache_names_alias_canonical_contract() -> None:
    assert reference_runtime.ExactCacheContext is ExactCacheContext
    assert reference_runtime.ExactCacheInterfaceSpec is ExactCacheInterfaceSpec
    assert (
        reference_runtime._stream_ids_for_clean_video_action_tokens
        is build_clean_video_action_cache_stream_ids
    )
    assert (
        reference_runtime._stream_ids_for_exact_dual_stream_split
        is build_dual_stream_cache_stream_ids
    )
    assert reference_runtime._build_exact_cache_spec is build_exact_cache_spec
    assert (
        reference_runtime._single_stream_action_token_count
        is count_single_stream_action_tokens
    )
    assert (
        reference_runtime._ensure_exact_cache_initialized
        is ensure_exact_cache_initialized
    )
    assert (
        reference_runtime.ensure_reference_text_embeddings
        is ensure_exact_text_embeddings
    )
    assert (
        reference_runtime._existing_exact_cache_attn_window
        is existing_exact_cache_attention_window
    )
    assert (
        reference_runtime._restore_slot_pool_layer_metadata
        is restore_slot_pool_layer_metadata
    )
    assert reference_runtime._resolve_exact_cache_context is resolve_exact_cache_context
    assert (
        reference_runtime._set_slot_pool_layer_metadata is set_slot_pool_layer_metadata
    )
    assert (
        reference_runtime._validate_existing_exact_cache_attn_window
        is validate_existing_exact_cache_attention_window
    )


def test_build_dual_stream_cache_stream_ids_preserves_packed_split_order() -> None:
    stream_ids = build_dual_stream_cache_stream_ids(
        [2, 1, 3, 2, 2],
        device=torch.device("cpu"),
    )

    assert stream_ids.dtype == torch.long
    assert stream_ids.device == torch.device("cpu")
    torch.testing.assert_close(
        stream_ids,
        torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, -1, -1]),
        rtol=0.0,
        atol=0.0,
    )


def test_build_clean_video_action_cache_stream_ids_preserves_token_order() -> None:
    stream_ids = build_clean_video_action_cache_stream_ids(
        video_token_count=2,
        action_token_count=3,
        device=torch.device("cpu"),
    )

    assert stream_ids.dtype == torch.long
    torch.testing.assert_close(
        stream_ids,
        torch.tensor([0, 0, 1, 1, 1]),
        rtol=0.0,
        atol=0.0,
    )


def test_count_single_stream_action_tokens_uses_frame_action_width() -> None:
    actions = torch.zeros(2, 7, 3, 4, 2)

    assert count_single_stream_action_tokens(actions) == 24

    with pytest.raises(
        ValueError,
        match=r"Expected action latents shaped \[B, C, F, A, W\]",
    ):
        count_single_stream_action_tokens(torch.zeros(2, 3, 4, 5))


def test_slot_pool_layer_metadata_round_trips_existing_and_new_keys() -> None:
    layers = (
        SlotPoolLayerState(metadata={"existing": "first", "untouched": 1}),
        SlotPoolLayerState(metadata={"existing": "second"}),
    )

    class _Transformer(torch.nn.Module):
        def get_runtime_cache_state(self, cache_name: str) -> object:
            assert cache_name == "session"
            return SimpleNamespace(
                backend_name="slot_pool_exact",
                backend_payload=SimpleNamespace(layer_states=layers),
            )

    previous = set_slot_pool_layer_metadata(
        _Transformer(),
        cache_name="session",
        updates={"existing": "temporary", "added": True},
    )

    assert [layer.metadata for layer in layers] == [
        {"existing": "temporary", "untouched": 1, "added": True},
        {"existing": "temporary", "added": True},
    ]
    restore_slot_pool_layer_metadata(previous)
    assert [layer.metadata for layer in layers] == [
        {"existing": "first", "untouched": 1},
        {"existing": "second"},
    ]


@pytest.mark.parametrize(
    "transformer",
    [
        SimpleNamespace(get_runtime_cache_state=lambda _name: None),
        SimpleNamespace(
            get_runtime_cache_state=lambda _name: SimpleNamespace(
                backend_name="merged_prefix",
                backend_payload=SimpleNamespace(layer_states=()),
            )
        ),
        SimpleNamespace(
            get_runtime_cache_state=lambda _name: SimpleNamespace(
                backend_name="slot_pool_exact",
                backend_payload=SimpleNamespace(),
            )
        ),
    ],
)
def test_slot_pool_layer_metadata_is_noop_without_compatible_cache(
    transformer: object,
) -> None:
    assert (
        set_slot_pool_layer_metadata(
            transformer,
            cache_name="session",
            updates={"temporary": True},
        )
        == []
    )


def test_resolve_exact_cache_context_materializes_cfg_inputs() -> None:
    transformer = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
    backbone_config = SharedVideoTransformerConfig(
        max_text_tokens=3,
        text_dim=2,
    )

    context, text_emb, negative_text_emb = resolve_exact_cache_context(
        transformer=transformer,
        backbone_config=backbone_config,
        inference_config=InferenceConfig(guidance_scale=2.0),
        infer_cache={},
        batch_size=2,
        latent_height=4,
        latent_width=5,
        device=torch.device("cpu"),
        text_emb=None,
        negative_text_emb=None,
    )

    assert context == ExactCacheContext(
        cache_name="open_wam_exact",
        cache_backend_name="slot_pool_exact",
        cache_initialized=False,
        batch_size=2,
        latent_height=4,
        latent_width=5,
        use_cfg=True,
        device=torch.device("cpu"),
        model_dtype=torch.float64,
    )
    assert text_emb.shape == (2, 3, 2)
    assert text_emb.dtype == torch.float64
    torch.testing.assert_close(text_emb, torch.zeros_like(text_emb), rtol=0.0, atol=0.0)
    assert negative_text_emb is not None
    torch.testing.assert_close(
        negative_text_emb,
        torch.zeros_like(negative_text_emb),
        rtol=0.0,
        atol=0.0,
    )


def test_build_exact_cache_spec_coerces_public_write_mode() -> None:
    spec = build_exact_cache_spec(
        write_mode="joint_packed",
        batch_size=4,
        use_cfg=True,
        prefix_visibility_mode="video_history_only",
    )

    assert spec == ExactCacheInterfaceSpec(
        write_mode=ParallelExactCacheWriteMode.JOINT_PACKED,
        prefix_visibility_mode="video_history_only",
    )


def test_ensure_exact_cache_initialized_delegates_shared_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _record_initialization(
        transformer: torch.nn.Module,
        **kwargs: object,
    ) -> None:
        calls.append({"transformer": transformer, **kwargs})

    monkeypatch.setattr(
        exact_cache,
        "initialize_exact_runtime_cache",
        _record_initialization,
    )
    transformer = torch.nn.Linear(1, 1, bias=False)
    context = ExactCacheContext(
        cache_name="session",
        cache_backend_name="slot_pool_exact",
        cache_initialized=False,
        batch_size=2,
        latent_height=4,
        latent_width=5,
        use_cfg=True,
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )

    initialized = ensure_exact_cache_initialized(
        transformer=transformer,
        policy_config=ParallelStreamPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            action_per_frame=4,
            attn_window=12,
        ),
        inference_config=InferenceConfig(frame_chunk_size=3),
        cache_context=context,
        cache_spec=ExactCacheInterfaceSpec(
            write_mode=ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED,
            token_batch_factor=2,
            prefix_visibility_mode="video_history_only",
        ),
    )

    assert initialized.cache_initialized
    assert initialized != context
    assert calls == [
        {
            "transformer": transformer,
            "cache_name": "session",
            "attn_window": 12,
            "batch_size": 2,
            "frame_chunk_size": 3,
            "latent_height": 4,
            "latent_width": 5,
            "device": torch.device("cpu"),
            "action_per_frame": 4,
            "use_cfg": True,
            "cache_backend_name": "slot_pool_exact",
            "cache_batch_size_override": None,
            "token_batch_factor": 2,
            "prefix_visibility_mode": "video_history_only",
        }
    ]


@pytest.mark.parametrize(
    ("cache_state", "expected"),
    [
        (SimpleNamespace(payload={"attn_window": 7}), 7),
        (
            SimpleNamespace(
                payload={},
                backend_payload=SimpleNamespace(metadata={"attn_window": 9}),
            ),
            9,
        ),
        (None, None),
    ],
)
def test_existing_exact_cache_attention_window_supports_both_backends(
    cache_state: object,
    expected: int | None,
) -> None:
    class _Transformer(torch.nn.Module):
        def get_runtime_cache_state(self, cache_name: str) -> object:
            assert cache_name == "session"
            return cache_state

    assert (
        existing_exact_cache_attention_window(
            _Transformer(),
            cache_name="session",
        )
        == expected
    )


def test_existing_exact_cache_attention_window_rejects_contract_change() -> None:
    class _Transformer(torch.nn.Module):
        def get_runtime_cache_state(self, cache_name: str) -> object:
            assert cache_name == "session"
            return SimpleNamespace(payload={"attn_window": 7})

    with pytest.raises(ValueError, match="existing=7, requested=8"):
        validate_existing_exact_cache_attention_window(
            _Transformer(),
            cache_name="session",
            requested_attn_window=8,
        )
