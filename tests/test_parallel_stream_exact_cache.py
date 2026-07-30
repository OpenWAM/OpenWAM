from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import (
    InferenceConfig,
    ParallelExactCacheWriteMode,
    ParallelStreamPolicyConfig,
    SharedVideoTransformerConfig,
)
from open_wam.models.policy_variants.parallel_stream import exact_cache
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.exact_cache import (
    ExactCacheContext,
    ExactCacheInterfaceSpec,
    build_exact_cache_spec,
    ensure_exact_cache_initialized,
    ensure_exact_text_embeddings,
    existing_exact_cache_attention_window,
    resolve_exact_cache_context,
    validate_existing_exact_cache_attention_window,
)


def test_reference_runtime_exact_cache_names_alias_canonical_contract() -> None:
    assert reference_runtime.ExactCacheContext is ExactCacheContext
    assert reference_runtime.ExactCacheInterfaceSpec is ExactCacheInterfaceSpec
    assert reference_runtime._build_exact_cache_spec is build_exact_cache_spec
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
    assert reference_runtime._resolve_exact_cache_context is resolve_exact_cache_context
    assert (
        reference_runtime._validate_existing_exact_cache_attn_window
        is validate_existing_exact_cache_attention_window
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
        def _resolve_exact_cache_state(self, cache_name: str) -> object:
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
        def _resolve_exact_cache_state(self, cache_name: str) -> object:
            assert cache_name == "session"
            return SimpleNamespace(payload={"attn_window": 7})

    with pytest.raises(ValueError, match="existing=7, requested=8"):
        validate_existing_exact_cache_attention_window(
            _Transformer(),
            cache_name="session",
            requested_attn_window=8,
        )
