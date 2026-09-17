"""A cache partition reuses features, never a different attention law."""

import os
from dataclasses import replace

import pytest
import torch

from open_wam.configs import CurrentBlockCoupling
from open_wam.models.common.denoising_cache import (
    DenoisingCache,
    attention_dependency_closure,
)
from open_wam.models.common.packed_token_layout import PackedTokenStream
from open_wam.models.policy_variants.dual_expert.attention_packed import (
    build_dual_expert_packed_coupling_attention_profile,
)
from tests.test_causal_video_prediction import _tiny_chunked_conditioned_video_pipeline
from tests.test_denoising_cache_invariants import _case


def _profile(coupling, device):
    return build_dual_expert_packed_coupling_attention_profile(
        num_video_frames=9,
        video_tokens_per_frame=1,
        num_action_frames=9,
        action_tokens_per_frame=1,
        chunk_size_frames=4,
        chunk_origin_frame=1,
        device=device,
        current_block_coupling=coupling,
        build_dense_masks=True,
        build_flex_masks=False,
    )


@pytest.mark.parametrize("coupling", tuple(CurrentBlockCoupling))
def test_partition_uses_original_query_and_key_positions(coupling):
    profile = _profile(coupling, torch.device("cpu"))
    cache = DenoisingCache()
    cache.bind(
        profile=profile,
        invariant_tokens=profile.token_layout.noise_id == 1,
        stream_lengths=(18, 18),
        num_layers=2,
    )
    active = torch.cat((torch.arange(9), torch.arange(18, 27)))
    assert torch.equal(cache.query_indices, active)
    expected = profile.self_attention_mask[active]
    assert torch.equal(cache.query_profile.self_attention_mask, expected)
    assert all(len(layer) == 2 for layer in cache.layers)


def test_fixed_inputs_are_rejected_when_their_features_depend_on_live_tokens():
    profile = _profile(CurrentBlockCoupling.JOINT, torch.device("cpu"))
    fixed = profile.token_layout.noise_id == 1
    fixed[:9] = True  # IDM's supplied video remains in the live attention slot.
    with pytest.raises(ValueError, match="attend to changing tokens"):
        DenoisingCache().bind(
            profile=profile,
            invariant_tokens=fixed,
            stream_lengths=(18, 18),
            num_layers=2,
        )


@pytest.mark.parametrize("coupling", tuple(CurrentBlockCoupling))
@torch.no_grad()
def test_cached_block_stack_matches_recomputation_over_successive_steps(coupling):
    stack, inputs = _case(coupling, torch.device("cpu"))
    profile = _profile(coupling, torch.device("cpu"))
    cache = DenoisingCache()
    cache.bind(
        profile=profile,
        invariant_tokens=profile.token_layout.noise_id == 1,
        stream_lengths=(18, 18),
        num_layers=len(stack.packed_blocks),
    )
    for step in range(3):
        changed = {**inputs}
        for name in ("video_hidden_states", "action_hidden_states"):
            changed[name] = inputs[name].clone()
            changed[name][:, :9] += step / 3
        reference = stack(**changed)
        actual = stack(**changed, denoising_cache=cache)
        for expected, observed in zip(reference, actual, strict=True):
            # Incremental projections can choose a different GEMM geometry.
            # The attention law and unchanged training path remain binding.
            torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    assert all(entry.ready for layer in cache.layers for entry in layer)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@torch.no_grad()
def test_conditioned_video_cache_matches_full_shape_execution(device):
    if device == "cuda" and (
        not torch.cuda.is_available() or os.getenv("OPEN_WAM_RUN_GPU_SANITY") != "1"
    ):
        pytest.skip("Set OPEN_WAM_RUN_GPU_SANITY=1 with an allocated GPU")
    _, pipeline = _tiny_chunked_conditioned_video_pipeline()
    tower = pipeline.visual_tower.to(device).eval()
    inputs = {
        "noisy_latents": torch.randn(1, 48, 5, 2, 4, device=device),
        "condition_latents": torch.randn(1, 48, 5, 2, 4, device=device),
        "timesteps": torch.ones(1, 5, device=device),
        "condition_timesteps": torch.zeros(1, 5, device=device),
        "text_context": torch.randn(1, 3, 8, device=device),
        "chunk_size": 2,
        "window_size": 4,
        "frame_start": 7,
        "prefix_condition_frames": 1,
        "stage": "infer",
    }
    cache = DenoisingCache()
    for step in range(3):
        inputs["noisy_latents"][:, :, -2:] += step / 3
        inputs["timesteps"][:, -2:] += step
        expected = tower.predict_chunked_conditioned_video_flow(**inputs)
        actual = tower.predict_chunked_conditioned_video_flow(
            **inputs, denoising_cache=cache
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert cache.bound


def test_cache_scope_cannot_be_rebound():
    profile = _profile(CurrentBlockCoupling.JOINT, torch.device("cpu"))
    cache = DenoisingCache()
    args = {
        "profile": profile,
        "invariant_tokens": profile.token_layout.noise_id == 1,
        "stream_lengths": (18, 18),
        "num_layers": 2,
    }
    cache.bind(**args)
    with pytest.raises(ValueError, match="bound once"):
        cache.bind(**args)


@pytest.mark.parametrize("coupling", tuple(CurrentBlockCoupling))
@pytest.mark.parametrize("stream", (PackedTokenStream.VIDEO, PackedTokenStream.ACTION))
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@torch.no_grad()
def test_dependency_pruned_features_match_full_graph(coupling, stream, device):
    if device == "cuda" and (
        not torch.cuda.is_available() or os.getenv("OPEN_WAM_RUN_GPU_SANITY") != "1"
    ):
        pytest.skip("Set OPEN_WAM_RUN_GPU_SANITY=1 with an allocated GPU")
    device = torch.device(device)
    stack, inputs = _case(coupling, device)
    profile = _profile(coupling, device)
    layout = profile.token_layout
    roots = (
        (layout.noise_id == 0) & (layout.stream_id == stream) & (layout.frame_id >= 5)
    )
    required = attention_dependency_closure(profile, roots)
    cache = DenoisingCache()
    cache.bind(
        profile=profile,
        invariant_tokens=layout.noise_id == 1,
        stream_lengths=(18, 18),
        num_layers=2,
        required_tokens=required,
    )
    for step in range(3):
        changed = {**inputs}
        for name in ("video_hidden_states", "action_hidden_states"):
            changed[name] = inputs[name].clone()
            changed[name][:, :9] += step / 3
        expected = torch.cat(stack(**changed), dim=1)[:, roots]
        actual = torch.cat(stack(**changed, denoising_cache=cache), dim=1)[:, roots]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert cache.key_indices.numel() < layout.token_count


def test_dependency_pruning_rejects_an_incomplete_graph():
    profile = _profile(CurrentBlockCoupling.JOINT, torch.device("cpu"))
    roots = profile.token_layout.noise_id == 0
    with pytest.raises(ValueError, match="omit attention dependencies"):
        DenoisingCache().bind(
            profile=profile,
            invariant_tokens=~roots,
            stream_lengths=(18, 18),
            num_layers=2,
            required_tokens=roots,
        )


@torch.no_grad()
def test_cache_skips_invariant_projections_and_ffns_without_changing_kernel(
    monkeypatch,
):
    stack, inputs = _case(CurrentBlockCoupling.JOINT, torch.device("cpu"))
    profile = _profile(CurrentBlockCoupling.JOINT, torch.device("cpu"))
    cache = DenoisingCache()
    cache.bind(
        profile=profile,
        invariant_tokens=profile.token_layout.noise_id == 1,
        stream_lengths=(18, 18),
        num_layers=2,
    )
    # An available but unselected Flex mask must not override dense execution.
    cache.query_profile = replace(
        cache.query_profile, self_attention_block_mask=object()
    )
    queries, projection_rows, ffn_rows = [], [], []
    attention = torch.nn.functional.scaled_dot_product_attention

    def record_attention(query, key, value, **kwargs):
        if key.shape[2] == 36:
            queries.append(query.shape[2])
        return attention(query, key, value, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", record_attention
    )
    block = stack.packed_blocks[0].video_block
    handles = [
        block.attn1.to_q.register_forward_pre_hook(
            lambda module, args: projection_rows.append(args[0].shape[1])
        ),
        block.ffn.register_forward_pre_hook(
            lambda module, args: ffn_rows.append(args[0].shape[1])
        ),
    ]
    try:
        stack(**inputs, denoising_cache=cache)
        stack(**inputs, denoising_cache=cache)
    finally:
        for handle in handles:
            handle.remove()
    assert queries == [18] * 4 + [9] * 4
    assert projection_rows == ffn_rows == [18, 9]
