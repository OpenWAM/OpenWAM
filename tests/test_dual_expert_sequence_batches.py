"""Real-token batching parity against independent B1 transformer execution."""

import os

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import SharedVideoTransformerConfig
from open_wam.models.policy_variants.dual_expert.attention_packed import (
    build_dual_expert_packed_coupling_attention_profile,
)
from open_wam.models.policy_variants.dual_expert.batch_attention import (
    build_sequence_batch_cross_attention,
    build_sequence_batch_self_attention,
)
from open_wam.models.policy_variants.dual_expert.batch_execution import (
    DualExpertDenoiseRequest,
    forward_dual_expert_sequence_batch,
)
from open_wam.models.policy_variants.dual_expert.dual_stream_execution import (
    forward_dual_expert_packed_coupling_denoise,
)
from open_wam.models.policy_variants.dual_expert.modules import DualExpertActionExpert
from open_wam.models.policy_variants.dual_expert.packed_block import (
    DualExpertPackedBlockStack,
)
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore


def _fixture(program, device="cpu"):
    torch.manual_seed(72)
    config = SharedVideoTransformerConfig(
        latent_channels=2,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
    )
    core = SharedVideoTransformerCore(config, action_dim=4, state_dim=4).to(device)
    tower = SimpleNamespace(core=core, config=config)
    action = DualExpertActionExpert(
        hidden_size=32,
        action_dim=4,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
    )
    action = action.to(device)
    stack = DualExpertPackedBlockStack(core.blocks, action.blocks)
    prefix = 1 if program == "video_then_action" else 0
    requests = []
    for i, target_frames in enumerate((3, 5)):
        frames = target_frames + prefix
        text = torch.randn(1, 2 + i, 16, device=device)
        noisy = torch.randn(1, 2, frames, 2, 2, device=device)
        clean = torch.randn_like(noisy)
        times = torch.rand(1, frames, device=device) * 1000
        if prefix:
            noisy[:, :, :1] = clean[:, :, :1]
            times[:, :1] = 0
        action_tokens = target_frames * 2 * 2
        pre = action.pre_dit(
            action_tokens=torch.randn(1, action_tokens, 4, device=device),
            timestep=torch.rand(1, action_tokens, device=device) * 1000,
            context=text,
        )
        profile = build_dual_expert_packed_coupling_attention_profile(
            num_video_frames=frames,
            video_tokens_per_frame=1,
            num_action_frames=target_frames,
            action_tokens_per_frame=2,
            chunk_size_frames=2,
            attention_window_size=4,
            device=torch.device(device),
            build_dense_masks=True,
            build_flex_masks=False,
            current_block_coupling=program,
            prefix_condition_frames=prefix,
            chunk_origin_frame=i,
            history_stream_visibility="video_only" if prefix else "full",
        )
        requests.append(
            DualExpertDenoiseRequest(
                noisy_video_latents=noisy,
                clean_video_latents=clean,
                noisy_video_timesteps=times,
                clean_video_timesteps=torch.zeros_like(times),
                packed_action_pre=pre,
                attention_profile=profile,
                text_context=text,
                frame_start=i * 7 - prefix,
                video_hidden_context=None
                if prefix
                else torch.randn(1, frames * 2, 32, device=device),
            )
        )
    return tower, action, stack, requests


@pytest.mark.parametrize("program", ["video_then_action", "joint"])
@pytest.mark.parametrize("padded", [False, True])
def test_sequence_batch_forward_gradient_parity_and_single_heavy_call(program, padded):
    tower, action, stack, requests = _fixture(program)
    original_keys = (
        tuple(tower.core.state_dict())
        + tuple(action.state_dict())
        + tuple(stack.state_dict())
    )
    reference = [
        forward_dual_expert_packed_coupling_denoise(
            visual_tower=tower,
            action_expert=action,
            packed_block_stack=stack,
            prefer_flex_attention=False,
            **request.as_kwargs(),
        )
        for request in requests
    ]
    calls = []
    handles = [
        block.register_forward_pre_hook(
            lambda module, args: calls.append((args[0].shape[1], args[1].shape[1]))
        )
        for block in stack.packed_blocks
    ]
    actual = forward_dual_expert_sequence_batch(
        visual_tower=tower,
        action_expert=action,
        packed_block_stack=stack,
        requests=requests,
        padded=padded,
    )
    for handle in handles:
        handle.remove()
    assert original_keys == tuple(tower.core.state_dict()) + tuple(
        action.state_dict()
    ) + tuple(stack.state_dict())
    assert len(calls) == 2  # two layers, not samples x layers
    video_sizes = [request.noisy_video_latents.shape[2] * 2 for request in requests]
    action_sizes = [request.packed_action_pre.tokens.shape[1] for request in requests]
    assert calls[0] == (
        max(video_sizes) * 2 if padded else sum(video_sizes),
        max(action_sizes) * 2 if padded else sum(action_sizes),
    )
    for expected, result in zip(reference, actual, strict=True):
        for lhs, rhs in zip(expected, result, strict=True):
            torch.testing.assert_close(lhs, rhs, atol=2e-6, rtol=2e-5)
    parameters = tuple(tower.core.parameters()) + tuple(action.parameters())

    def loss(outputs):
        return torch.stack(
            [v.square().mean() + a.square().mean() for v, a in outputs]
        ).mean()

    expected_grads = torch.autograd.grad(
        loss(reference), parameters, allow_unused=True, retain_graph=True
    )
    actual_grads = torch.autograd.grad(loss(actual), parameters, allow_unused=True)
    for expected, result in zip(expected_grads, actual_grads, strict=True):
        if expected is None:
            assert result is None
        else:
            torch.testing.assert_close(expected, result, atol=3e-6, rtol=3e-4)


@pytest.mark.parametrize("program", ["video_then_action", "joint"])
@pytest.mark.parametrize("padded", [False, True])
def test_sequence_batch_cannot_read_another_samples_video_action_or_text(
    program, padded
):
    tower, action, stack, requests = _fixture(program)

    def execute(items):
        return forward_dual_expert_sequence_batch(
            visual_tower=tower,
            action_expert=action,
            packed_block_stack=stack,
            requests=items,
            padded=padded,
        )

    before = execute(requests)
    other = requests[1]
    changed = replace(
        other,
        noisy_video_latents=other.noisy_video_latents + 100,
        clean_video_latents=other.clean_video_latents - 100,
        text_context=other.text_context * 100,
        packed_action_pre=replace(
            other.packed_action_pre,
            tokens=other.packed_action_pre.tokens + 100,
            context=other.packed_action_pre.context * 100,
        ),
    )
    after = execute([requests[0], changed])
    for lhs, rhs in zip(before[0], after[0], strict=True):
        torch.testing.assert_close(lhs, rhs, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(before[1][0], after[1][0])


@pytest.mark.parametrize("program", ["video_then_action", "joint"])
def test_batched_layout_is_exact_b1_mask_with_isolated_dummy_slots(program):
    _, _, _, requests = _fixture(program)
    profiles = [request.attention_profile for request in requests]
    nv = [request.noisy_video_latents.shape[2] * 2 for request in requests]
    na = [request.packed_action_pre.tokens.shape[1] for request in requests]
    sv, sa = [max(nv)] * 2, [max(na)] * 2
    dense, sparse, layout = build_sequence_batch_self_attention(
        profiles,
        nv,
        na,
        sv,
        sa,
        device=torch.device("cpu"),
    )
    assert sparse is None
    for i, profile in enumerate(profiles):
        ids = torch.where(layout.seq_id == i)[0]
        torch.testing.assert_close(
            dense[ids[:, None], ids[None, :]], profile.self_attention_mask
        )
        assert not dense[ids][:, layout.seq_id != i].any()
    dummy = torch.where(layout.seq_id < 0)[0]
    assert torch.equal(dense[dummy].sum(dim=1), torch.ones_like(dummy))
    assert dense[dummy, dummy].all()


@pytest.mark.parametrize("program", ["video_then_action", "joint"])
def test_real_flex_block_mask_predicates_match_dense_and_bound_partial_tiles(
    monkeypatch, program
):
    from torch.nn.attention.flex_attention import create_mask

    import open_wam.models.policy_variants.dual_expert.batch_attention as attention

    _, _, _, requests = _fixture(program)
    profiles = [request.attention_profile for request in requests]
    nv = [request.noisy_video_latents.shape[2] * 2 for request in requests]
    na = [request.packed_action_pre.tokens.shape[1] for request in requests]
    sv, sa = [max(nv)] * 2, [max(na)] * 2
    args = (profiles, nv, na, sv, sa)
    expected, _, _ = build_sequence_batch_self_attention(
        *args, device=torch.device("cpu")
    )
    local = torch.ones((1, nv[0], 2), dtype=torch.bool)
    local[:, :, 1] = False
    expected_cross, _ = build_sequence_batch_cross_attention(
        nv,
        sv,
        [2, 3],
        [local, None],
        device=torch.device("cpu"),
    )
    backend = attention._backend_mask
    monkeypatch.setattr(
        attention,
        "_backend_mask",
        lambda p, q, k, d: backend(p, q, k, d, build_sparse=True),
    )
    dense, sparse, _ = build_sequence_batch_self_attention(
        *args, device=torch.device("cpu")
    )
    assert dense is None
    actual = create_mask(
        sparse.mask_mod, 1, 1, expected.shape[0], expected.shape[1], device="cpu"
    )[0, 0]
    torch.testing.assert_close(expected, actual)
    # A real Flex tile may run its predicate beyond the final partial block.
    assert not sparse.mask_mod(
        None, None, torch.tensor(expected.shape[0] + 5), torch.tensor(0)
    )
    assert not sparse.mask_mod(
        None, None, torch.tensor(0), torch.tensor(expected.shape[1] + 5)
    )
    _, cross = build_sequence_batch_cross_attention(
        nv,
        sv,
        [2, 3],
        [local, None],
        device=torch.device("cpu"),
    )
    actual_cross = create_mask(cross.mask_mod, 1, 1, sum(sv), 5, device="cpu")[0, 0]
    torch.testing.assert_close(expected_cross, actual_cross)
    assert not cross.mask_mod(None, None, torch.tensor(0), torch.tensor(128))


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires a CUDA GPU and compiled FlexAttention kernels",
)
@pytest.mark.parametrize("program", ["video_then_action", "joint"])
def test_cuda_packed_flex_forward_backward_and_sequence_isolation(program):
    """Deployment gate: compiled CUDA self/cross Flex kernels, not CPU mask emulation."""
    if os.getenv("OPEN_WAM_RUN_GPU_SANITY") != "1":
        pytest.skip("Set OPEN_WAM_RUN_GPU_SANITY=1 with an intentionally allocated GPU.")
    tower, action, stack, requests = _fixture(program, device="cuda")
    reference = [
        forward_dual_expert_packed_coupling_denoise(
            visual_tower=tower,
            action_expert=action,
            packed_block_stack=stack,
            prefer_flex_attention=False,
            **request.as_kwargs(),
        )
        for request in requests
    ]
    actual = forward_dual_expert_sequence_batch(
        visual_tower=tower,
        action_expert=action,
        packed_block_stack=stack,
        requests=requests,
        padded=False,
        use_activation_checkpointing=True,
    )
    for expected, result in zip(reference, actual, strict=True):
        for lhs, rhs in zip(expected, result, strict=True):
            torch.testing.assert_close(lhs, rhs, atol=3e-4, rtol=3e-3)
    parameters = tuple(tower.core.parameters()) + tuple(action.parameters())

    def loss(outputs):
        return torch.stack(
            [v.square().mean() + a.square().mean() for v, a in outputs]
        ).mean()

    expected_grads = torch.autograd.grad(
        loss(reference), parameters, allow_unused=True, retain_graph=True
    )
    actual_grads = torch.autograd.grad(
        loss(actual), parameters, allow_unused=True, retain_graph=True
    )
    for expected, result in zip(expected_grads, actual_grads, strict=True):
        if expected is None:
            assert result is None
        else:
            assert torch.isfinite(result).all()
            torch.testing.assert_close(expected, result, atol=8e-4, rtol=8e-3)
    other = requests[1]
    changed = replace(
        other,
        noisy_video_latents=other.noisy_video_latents + 100,
        clean_video_latents=other.clean_video_latents - 100,
        text_context=other.text_context * 100,
        packed_action_pre=replace(
            other.packed_action_pre,
            tokens=other.packed_action_pre.tokens + 100,
            context=other.packed_action_pre.context * 100,
        ),
    )
    after = forward_dual_expert_sequence_batch(
        visual_tower=tower,
        action_expert=action,
        packed_block_stack=stack,
        requests=[requests[0], changed],
        padded=False,
    )
    for lhs, rhs in zip(actual[0], after[0], strict=True):
        torch.testing.assert_close(lhs, rhs, atol=3e-4, rtol=3e-3)
