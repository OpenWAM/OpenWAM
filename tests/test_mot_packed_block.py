"""CPU parity + smoke tests for ``MoTPackedBlock`` (Step A1).

These tests do not exercise FSDP. They confirm that running joint
``[V_noisy, V_clean, A_noisy, A_clean]`` attention through the new wrapper
module produces the same numerical output as the existing ``_packed_block_step``
inline logic when both consume the same underlying ``video_block`` and
``action_block``.

Step A1 keeps using the legacy ``prepare_self_attention_inputs`` /
``apply_post_attention`` helpers, which call ``_*_with_materialized_params``
internally; on CPU without FSDP these helpers are identity over the param
shards, so parity holds by construction. The FSDP-correctness fix lives in
Step A3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import open_wam.configs  # ensure typed-config + backbone modules import order
from open_wam.models.policy_variants.mot.modules import MoTActionExpert
from open_wam.models.policy_variants.mot.packed_block import (
    MoTPackedBlock,
    MoTPackedBlockStack,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore


# --- shared fixture builders -------------------------------------------------


_HIDDEN = 32
_NUM_HEADS = 4
_HEAD_DIM = 8
_FFN_DIM = 64
_TEXT_DIM = 16
_FREQ_DIM = 8
_NUM_LAYERS = 2


def _make_video_core() -> SharedVideoTransformerCore:
    return SharedVideoTransformerCore(
        SharedVideoTransformerConfig(
            hidden_size=_HIDDEN,
            num_layers=_NUM_LAYERS,
            num_heads=_NUM_HEADS,
            attention_head_dim=_HEAD_DIM,
            ffn_dim=_FFN_DIM,
            text_dim=_TEXT_DIM,
            freq_dim=_FREQ_DIM,
        ),
        action_dim=4,
        state_dim=4,
    )


def _make_action_expert() -> MoTActionExpert:
    return MoTActionExpert(
        hidden_size=_HIDDEN,
        action_dim=4,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        attention_head_dim=_HEAD_DIM,
        ffn_dim=_FFN_DIM,
        text_dim=_TEXT_DIM,
        freq_dim=_FREQ_DIM,
    )


def _packed_block_step_reference(
    video_block,
    action_block,
    video_hidden_states: torch.Tensor,
    action_hidden_states: torch.Tensor,
    *,
    video_timestep_proj: torch.Tensor,
    video_rotary_emb: torch.Tensor | None,
    action_temb: torch.Tensor,
    action_rotary_emb: torch.Tensor | None,
    video_attention_mask: torch.Tensor,
    action_attention_mask: torch.Tensor,
    video_text_hidden_states: torch.Tensor,
    action_text_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: structurally identical to ``_packed_block_step`` closure.

    Mirrors `runtime.py:_packed_block_step` byte-for-byte. Used to verify the
    new ``MoTPackedBlock`` produces the same output.
    """
    video_attn_inputs = video_block.prepare_self_attention_inputs(
        video_hidden_states,
        temb=video_timestep_proj,
        rotary_emb=video_rotary_emb,
    )
    action_attn_inputs = action_block.prepare_self_attention_inputs(
        action_hidden_states,
        temb=action_temb,
        rotary_emb=action_rotary_emb,
    )
    joint_key = torch.cat([video_attn_inputs["key"], action_attn_inputs["key"]], dim=2)
    joint_value = torch.cat([video_attn_inputs["value"], action_attn_inputs["value"]], dim=2)
    mixed_video = (
        F.scaled_dot_product_attention(
            video_attn_inputs["query"], joint_key, joint_value,
            attn_mask=video_attention_mask, dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).flatten(2, 3)
    )
    mixed_action = (
        F.scaled_dot_product_attention(
            action_attn_inputs["query"], joint_key, joint_value,
            attn_mask=action_attention_mask, dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).flatten(2, 3)
    )
    video_self_out = video_block.attn1.to_out[1](video_block.attn1.to_out[0](mixed_video))
    action_self_out = action_block.attn1.to_out[1](action_block.attn1.to_out[0](mixed_action))
    new_video, _ = video_block.apply_post_attention(
        video_attn_inputs["hidden_states"],
        mixed_attn_output=video_self_out,
        encoder_hidden_states=video_text_hidden_states,
        gate_msa=video_attn_inputs["gate_msa"],
        c_shift_msa=video_attn_inputs["c_shift_msa"],
        c_scale_msa=video_attn_inputs["c_scale_msa"],
        c_gate_msa=video_attn_inputs["c_gate_msa"],
    )
    new_action, _ = action_block.apply_post_attention(
        action_attn_inputs["hidden_states"],
        mixed_attn_output=action_self_out,
        encoder_hidden_states=action_text_hidden_states,
        gate_msa=action_attn_inputs["gate_msa"],
        c_shift_msa=action_attn_inputs["c_shift_msa"],
        c_scale_msa=action_attn_inputs["c_scale_msa"],
        c_gate_msa=action_attn_inputs["c_gate_msa"],
    )
    return new_video, new_action


def _build_inputs(
    *,
    batch: int = 1,
    video_seq_len: int = 6,
    action_seq_len: int = 4,
    seed: int = 0,
) -> dict:
    """Random inputs shaped to match a small joint forward call."""
    g = torch.Generator().manual_seed(seed)
    video_h = torch.randn(batch, video_seq_len, _HIDDEN, generator=g)
    action_h = torch.randn(batch, action_seq_len, _HIDDEN, generator=g)
    # video timestep proj: [B, video_seq, 6, hidden] (per-token; shaped to match
    # `_select_chunk_slices(temb_scale_shift_table, 6)` consumer in
    # `prepare_self_attention_inputs`).
    video_timestep_proj = torch.randn(batch, video_seq_len, 6, _HIDDEN, generator=g)
    action_temb = torch.randn(batch, action_seq_len, 6, _HIDDEN, generator=g)
    # `rotary_emb=None` skips RoPE inside `prepare_self_attention_inputs`.
    # We're testing packed-attention plumbing, not RoPE itself; using None
    # avoids constructing complex-typed freqs of the right shape (those come
    # from `core.rope(grid_ids)` in real runtime, not naive randn).
    video_rotary = None
    action_rotary = None
    # Cross-attn `encoder_hidden_states` arrive ALREADY embedded to
    # ``hidden_size`` (text encoder runs upstream of these blocks). The
    # `apply_post_attention` cross-attn `to_k/to_v` linears expect
    # ``Linear(hidden_size, hidden_size)`` inputs.
    video_text = torch.randn(batch, 4, _HIDDEN, generator=g)
    action_text = torch.randn(batch, 4, _HIDDEN, generator=g)
    # joint mask: full True over packed [video | action] keys for trivially
    # comparing forward parity. Shape: [B, num_heads, q_seq, kv_seq].
    kv_seq = video_seq_len + action_seq_len
    video_mask = torch.ones(1, 1, video_seq_len, kv_seq, dtype=torch.bool)
    action_mask = torch.ones(1, 1, action_seq_len, kv_seq, dtype=torch.bool)
    return {
        "video_hidden_states": video_h,
        "action_hidden_states": action_h,
        "video_timestep_proj": video_timestep_proj,
        "video_rotary_emb": video_rotary,
        "action_temb": action_temb,
        "action_rotary_emb": action_rotary,
        "video_attention_mask": video_mask,
        "action_attention_mask": action_mask,
        "video_text_hidden_states": video_text,
        "action_text_hidden_states": action_text,
    }


# --- tests -------------------------------------------------------------------


def test_packed_block_forward_parity_with_inline_step() -> None:
    """``MoTPackedBlock.forward`` matches the inline ``_packed_block_step`` logic."""
    torch.manual_seed(0)
    video_core = _make_video_core()
    action_expert = _make_action_expert()
    video_block = video_core.blocks[0]
    action_block = action_expert.blocks[0]
    inputs = _build_inputs()

    packed = MoTPackedBlock(video_block, action_block).eval()
    with torch.no_grad():
        wrapped_video, wrapped_action = packed(**inputs)
        ref_video, ref_action = _packed_block_step_reference(
            video_block, action_block, **inputs
        )

    assert wrapped_video.shape == ref_video.shape
    assert wrapped_action.shape == ref_action.shape
    assert torch.allclose(wrapped_video, ref_video, atol=1e-6, rtol=1e-5)
    assert torch.allclose(wrapped_action, ref_action, atol=1e-6, rtol=1e-5)


def test_packed_block_backward_produces_finite_gradients() -> None:
    """Backward through ``MoTPackedBlock`` yields finite gradients on both experts.

    Smoke check that no buffer-lifetime issues exist outside FSDP. The whole
    point of replacing ``linear_with_materialized_params`` with native nn.Linear
    calls is that backward through standard autograd graph just works.
    """
    torch.manual_seed(0)
    video_core = _make_video_core()
    action_expert = _make_action_expert()
    video_block = video_core.blocks[0]
    action_block = action_expert.blocks[0]
    inputs = _build_inputs()

    packed = MoTPackedBlock(video_block, action_block).train()
    new_video, new_action = packed(**inputs)
    loss = new_video.float().pow(2).mean() + new_action.float().pow(2).mean()
    loss.backward()

    video_grad_count = 0
    for param in video_block.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), "video block grad has non-finite entries"
            video_grad_count += 1
    action_grad_count = 0
    for param in action_block.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), "action block grad has non-finite entries"
            action_grad_count += 1
    assert video_grad_count > 0
    assert action_grad_count > 0


def test_packed_block_stack_forward_matches_manual_loop() -> None:
    """``MoTPackedBlockStack`` running 2 layers matches a manual block-by-block loop."""
    torch.manual_seed(0)
    video_core = _make_video_core()
    action_expert = _make_action_expert()
    video_blocks = list(video_core.blocks)
    action_blocks = list(action_expert.blocks)
    inputs = _build_inputs()

    stack = MoTPackedBlockStack(video_blocks, action_blocks).eval()
    with torch.no_grad():
        stack_video, stack_action = stack(**inputs)

        manual_video = inputs["video_hidden_states"]
        manual_action = inputs["action_hidden_states"]
        loop_kwargs = {
            key: value
            for key, value in inputs.items()
            if key not in {"video_hidden_states", "action_hidden_states"}
        }
        for video_block, action_block in zip(video_blocks, action_blocks, strict=True):
            manual_video, manual_action = _packed_block_step_reference(
                video_block,
                action_block,
                manual_video,
                manual_action,
                **loop_kwargs,
            )

    assert torch.allclose(stack_video, manual_video, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stack_action, manual_action, atol=1e-6, rtol=1e-5)


def test_packed_block_stack_rejects_mismatched_block_counts() -> None:
    """Stack constructor must reject unequal video/action block counts."""
    video_core = _make_video_core()
    action_expert = _make_action_expert()
    # Provide 2 video blocks but only 1 action block.
    try:
        MoTPackedBlockStack(list(video_core.blocks), list(action_expert.blocks)[:1])
    except ValueError as exc:
        assert "equal video/action block counts" in str(exc)
    else:
        raise AssertionError("Expected ValueError on mismatched block counts.")


def test_packed_block_registers_video_and_action_as_children() -> None:
    """Both blocks must be discoverable via ``nn.Module.children()`` for FSDP."""
    video_core = _make_video_core()
    action_expert = _make_action_expert()
    packed = MoTPackedBlock(video_core.blocks[0], action_expert.blocks[0])
    children = dict(packed.named_children())
    assert "video_block" in children
    assert "action_block" in children
    # Sanity: parameters of video/action blocks both reachable via packed.
    packed_param_ids = {id(p) for p in packed.parameters()}
    video_param_ids = {id(p) for p in video_core.blocks[0].parameters()}
    action_param_ids = {id(p) for p in action_expert.blocks[0].parameters()}
    assert video_param_ids.issubset(packed_param_ids)
    assert action_param_ids.issubset(packed_param_ids)
