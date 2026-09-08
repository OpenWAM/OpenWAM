from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from open_wam.models.common.attention_backends import (
    apply_attention_backend,
    select_attention_profile_mask,
)
from open_wam.models.common.attention_contracts import PreparedAttentionProfile
from open_wam.models.common.sharded_execution import (
    checkpoint_unshard_context as _checkpoint_summon_context,
)
from open_wam.models.common.sharded_execution import (
    unshard_runtime_parameters as _unshard_runtime_params,
)
from open_wam.models.common.video_geometry import (
    unpatchify_video_sequence,
    video_token_grid_from_latent_shape,
)
from open_wam.models.visual_tower.grid_ids import build_video_grid_ids
from open_wam.models.visual_tower.runtime_parameter_ops import (
    layer_norm_with_materialized_params,
    linear_with_materialized_params,
    materialize_runtime_parameter,
)
from open_wam.models.visual_tower.shared_transformer_layout import (
    select_chunk_slices,
)

from .modules import DualExpertActionExpert, DualExpertActionPreprocessOutput


def _video_token_grid_for_latents(visual_tower, video_latents: torch.Tensor):
    return video_token_grid_from_latent_shape(
        video_latents,
        patch_size=visual_tower.core.patch_size,
    )


def prepare_packed_video_inputs(
    *,
    visual_tower,
    noisy_video_latents: torch.Tensor,
    clean_video_latents: torch.Tensor,
    noisy_video_timesteps: torch.Tensor,
    clean_video_timesteps: torch.Tensor | None,
    text_context: torch.Tensor | None,
    frame_start: int = 0,
    video_hidden_context: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare the same visual stream for single-sequence and batched execution."""
    if noisy_video_latents.shape != clean_video_latents.shape:
        raise ValueError(
            "dual-expert packed coupling expects matching noisy/clean video shapes, "
            f"got noisy={tuple(noisy_video_latents.shape)}, clean={tuple(clean_video_latents.shape)}."
        )
    batch_size = noisy_video_latents.shape[0]
    effective_clean_video_timesteps = (
        torch.zeros_like(noisy_video_timesteps) if clean_video_timesteps is None else clean_video_timesteps
    )
    if noisy_video_timesteps.shape != effective_clean_video_timesteps.shape:
        raise ValueError(
            "dual-expert packed coupling expects matching noisy/clean video timestep shapes, "
            f"got noisy={tuple(noisy_video_timesteps.shape)}, clean={tuple(effective_clean_video_timesteps.shape)}."
        )

    resolved_text = (
        torch.zeros(
            batch_size,
            visual_tower.config.max_text_tokens,
            visual_tower.config.text_dim,
            device=noisy_video_latents.device,
            dtype=noisy_video_latents.dtype,
        )
        if text_context is None
        else text_context.to(device=noisy_video_latents.device, dtype=noisy_video_latents.dtype)
    )
    packed_video_latents = torch.cat([noisy_video_latents, clean_video_latents], dim=2)
    packed_video_timesteps = torch.cat([noisy_video_timesteps, effective_clean_video_timesteps], dim=1)
    video_prepared = visual_tower.core.prepare_exact_single_stream_inputs(
        {
            "noisy_latents": packed_video_latents,
            "text_emb": resolved_text,
            "grid_id": torch.cat(
                [
                    build_video_grid_ids(
                        _video_token_grid_for_latents(visual_tower, noisy_video_latents),
                        device=noisy_video_latents.device,
                        frame_shift=float(frame_start),
                    ),
                    build_video_grid_ids(
                        _video_token_grid_for_latents(visual_tower, clean_video_latents),
                        device=clean_video_latents.device,
                        frame_shift=float(frame_start),
                    ),
                ],
                dim=1,
            )[None].expand(batch_size, -1, -1),
            "timesteps": packed_video_timesteps,
        },
        action_mode=False,
    )
    video_hidden_states = video_prepared["hidden_states"]
    if video_hidden_context is not None:
        if tuple(video_hidden_context.shape) != tuple(video_hidden_states.shape):
            raise ValueError(
                "dual-expert packed video hidden_context must match embedded video hidden states, "
                f"got hidden_context={tuple(video_hidden_context.shape)}, "
                f"hidden_states={tuple(video_hidden_states.shape)}."
            )
        video_hidden_states = video_hidden_states + video_hidden_context.to(
            device=video_hidden_states.device,
            dtype=video_hidden_states.dtype,
        )
    video_prepared["hidden_states"] = video_hidden_states
    return video_prepared


def finish_packed_video(
    visual_tower,
    video_hidden_states: torch.Tensor,
    video_temb: torch.Tensor,
    latent_shape: tuple[int, ...],
) -> torch.Tensor:
    """Project transformer outputs into the original noisy-video latent extent."""
    batch_size, _, num_frames, latent_height, latent_width = latent_shape
    shift, scale = select_chunk_slices(
        materialize_runtime_parameter(
            visual_tower.core.scale_shift_table,
            device=video_temb.device,
            dtype=video_temb.dtype,
        )[None]
        + video_temb[:, :, None, ...],
        2,
    )
    shift = shift.to(video_hidden_states.device)
    scale = scale.to(video_hidden_states.device)
    video_hidden_states = (
        layer_norm_with_materialized_params(visual_tower.core.norm_out, video_hidden_states.float())
        * (1.0 + scale)
        + shift
    ).type_as(video_hidden_states)
    packed_video_flow = linear_with_materialized_params(visual_tower.core.proj_out, video_hidden_states)
    packed_video_flow = unpatchify_video_sequence(
        visual_tower.core.patch_size,
        packed_video_flow,
        num_frames * 2,
        latent_height,
        latent_width,
        batch_size=batch_size,
    )
    video_flow = packed_video_flow[:, :, :num_frames].contiguous()
    return video_flow


def forward_dual_expert_packed_coupling_denoise(
    *,
    visual_tower,
    noisy_video_latents: torch.Tensor,
    clean_video_latents: torch.Tensor,
    noisy_video_timesteps: torch.Tensor,
    clean_video_timesteps: torch.Tensor | None,
    action_expert: DualExpertActionExpert,
    packed_action_pre: DualExpertActionPreprocessOutput,
    attention_profile: PreparedAttentionProfile,
    text_context: torch.Tensor | None,
    frame_start: int = 0,
    use_activation_checkpointing: bool = False,
    packed_block_stack=None,
    prefer_flex_attention: bool = True,
    video_cross_attention_mask: torch.Tensor | None = None,
    video_hidden_context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run dual-expert's native four-stream packed coupling forward.

    Video/action experts execute separate blocks, but every block attends over
    concatenated K/V from ``[V_noisy, V_clean, A_noisy, A_clean]`` using the
    supplied coupling mask. Returns the V_noisy flow and packed action hidden
    states; callers take loss on the first action half.
    """

    video_prepared = prepare_packed_video_inputs(
        visual_tower=visual_tower,
        noisy_video_latents=noisy_video_latents,
        clean_video_latents=clean_video_latents,
        noisy_video_timesteps=noisy_video_timesteps,
        clean_video_timesteps=clean_video_timesteps,
        text_context=text_context,
        frame_start=frame_start,
        video_hidden_context=video_hidden_context,
    )
    video_hidden_states = video_prepared["hidden_states"]
    video_text_hidden_states = video_prepared["text_hidden_states"]
    video_rotary_emb = video_prepared["rotary_emb"]
    video_temb = video_prepared["temb"]
    video_timestep_proj = video_prepared["timestep_proj"]

    action_hidden_states = packed_action_pre.tokens
    action_rotary_emb = packed_action_pre.freqs[:, :, None]
    video_seq_len = int(video_hidden_states.shape[1])
    action_seq_len = int(action_hidden_states.shape[1])
    expected_total = video_seq_len + action_seq_len
    profile_attention_mask, profile_block_mask = select_attention_profile_mask(
        attention_profile,
        device=video_hidden_states.device,
        prefer_flex=prefer_flex_attention,
        is_cross_attention=False,
    )
    if profile_block_mask is None:
        if profile_attention_mask is None or profile_attention_mask.shape != (expected_total, expected_total):
            raise ValueError(
                "dual-expert packed coupling requires a dense or flex attention profile matching packed video+action length, "
                f"got dense_mask={None if profile_attention_mask is None else tuple(profile_attention_mask.shape)}, "
                f"expected=({expected_total}, {expected_total})."
            )
        video_attention_mask = profile_attention_mask[:video_seq_len, :expected_total][None, None, :, :]
        action_attention_mask = profile_attention_mask[video_seq_len:, :expected_total][None, None, :, :]
    else:
        video_attention_mask = None
        action_attention_mask = None

    def _packed_block_step(
        video_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        video_block,
        action_block,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_attn_inputs = video_block.prepare_self_attention_inputs(
            video_hidden_states,
            temb=video_timestep_proj,
            rotary_emb=video_rotary_emb,
        )
        action_attn_inputs = action_block.prepare_self_attention_inputs(
            action_hidden_states,
            temb=packed_action_pre.t_mod,
            rotary_emb=action_rotary_emb,
        )
        joint_query = torch.cat([video_attn_inputs["query"], action_attn_inputs["query"]], dim=2)
        joint_key = torch.cat([video_attn_inputs["key"], action_attn_inputs["key"]], dim=2)
        joint_value = torch.cat([video_attn_inputs["value"], action_attn_inputs["value"]], dim=2)
        if profile_block_mask is not None:
            mixed = apply_attention_backend(
                query=joint_query,
                key=joint_key,
                value=joint_value,
                block_mask=profile_block_mask,
                kernel_options={
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 32,
                },
            )
            mixed_video, mixed_action = torch.split(mixed, [video_seq_len, action_seq_len], dim=2)
            mixed_video = mixed_video.transpose(1, 2).flatten(2, 3)
            mixed_action = mixed_action.transpose(1, 2).flatten(2, 3)
        else:
            mixed_video = F.scaled_dot_product_attention(
                video_attn_inputs["query"],
                joint_key,
                joint_value,
                attn_mask=video_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
            mixed_action = F.scaled_dot_product_attention(
                action_attn_inputs["query"],
                joint_key,
                joint_value,
                attn_mask=action_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
        new_video, _ = video_block.apply_post_attention(
            video_attn_inputs["hidden_states"],
            mixed_attn_output=video_block.attn1.to_out[1](
                linear_with_materialized_params(video_block.attn1.to_out[0], mixed_video)
            ),
            encoder_hidden_states=video_text_hidden_states,
            gate_msa=video_attn_inputs["gate_msa"],
            c_shift_msa=video_attn_inputs["c_shift_msa"],
            c_scale_msa=video_attn_inputs["c_scale_msa"],
            c_gate_msa=video_attn_inputs["c_gate_msa"],
            cross_attention_mask=video_cross_attention_mask,
        )
        new_action, _ = action_block.apply_post_attention(
            action_attn_inputs["hidden_states"],
            mixed_attn_output=action_block.attn1.to_out[1](
                linear_with_materialized_params(action_block.attn1.to_out[0], mixed_action)
            ),
            encoder_hidden_states=packed_action_pre.context,
            gate_msa=action_attn_inputs["gate_msa"],
            c_shift_msa=action_attn_inputs["c_shift_msa"],
            c_scale_msa=action_attn_inputs["c_scale_msa"],
            c_gate_msa=action_attn_inputs["c_gate_msa"],
            cross_attention_mask=packed_action_pre.cross_attention_mask,
        )
        return new_video, new_action

    checkpoint_active = bool(use_activation_checkpointing) and torch.is_grad_enabled()
    if packed_block_stack is not None:
        # FSDP-friendly path: each DualExpertPackedBlock owns its (video_block,
        # action_block) pair and is its own FSDP unit. Calling the wrapper's
        # forward triggers FSDP's standard pre/post-forward hooks; backward
        # gather is also FSDP-managed. No manual `summon_full_parameters` /
        # `linear_with_materialized_params` calls in the per-block path,
        # so backward no longer hits "setStorage out of bounds" from
        # resharded buffers. Both the dense (video/action_attention_mask) and
        # flex (profile_block_mask) profile paths are supported.
        flex_kernel_options = (
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            }
            if profile_block_mask is not None
            else None
        )
        kwargs = dict(
            video_timestep_proj=video_timestep_proj,
            video_rotary_emb=video_rotary_emb,
            action_temb=packed_action_pre.t_mod,
            action_rotary_emb=action_rotary_emb,
            video_attention_mask=video_attention_mask,
            action_attention_mask=action_attention_mask,
            video_text_hidden_states=video_text_hidden_states,
            action_text_hidden_states=packed_action_pre.context,
            video_cross_attention_mask=video_cross_attention_mask,
            action_cross_attention_mask=packed_action_pre.cross_attention_mask,
            block_mask=profile_block_mask,
            flex_kernel_options=flex_kernel_options,
        )
        for packed_block in packed_block_stack.packed_blocks:
            if checkpoint_active:
                video_hidden_states, action_hidden_states = torch.utils.checkpoint.checkpoint(
                    packed_block,
                    video_hidden_states,
                    action_hidden_states,
                    use_reentrant=False,
                    **kwargs,
                )
            else:
                video_hidden_states, action_hidden_states = packed_block(
                    video_hidden_states,
                    action_hidden_states,
                    **kwargs,
                )
    else:
        for video_block, action_block in zip(visual_tower.core.blocks, action_expert.blocks, strict=True):
            if checkpoint_active:
                video_hidden_states, action_hidden_states = torch.utils.checkpoint.checkpoint(
                    _packed_block_step,
                    video_hidden_states,
                    action_hidden_states,
                    video_block,
                    action_block,
                    use_reentrant=False,
                    context_fn=lambda vb=video_block, ab=action_block: _checkpoint_summon_context(vb, ab),
                )
            else:
                with _unshard_runtime_params(video_block, action_block):
                    video_hidden_states, action_hidden_states = _packed_block_step(
                        video_hidden_states,
                        action_hidden_states,
                        video_block,
                        action_block,
                    )

    video_flow = finish_packed_video(
        visual_tower, video_hidden_states, video_temb, tuple(noisy_video_latents.shape)
    )
    return video_flow, action_hidden_states


def forward_joint_video_action_denoise(
    *,
    visual_tower,
    noisy_video_latents: torch.Tensor,
    video_timesteps: torch.Tensor,
    action_expert: DualExpertActionExpert,
    action_pre: DualExpertActionPreprocessOutput,
    text_context: torch.Tensor | None,
    attention_mask: torch.Tensor,
    frame_start: int = 0,
    use_activation_checkpointing: bool = False,
    video_cross_attention_mask: torch.Tensor | None = None,
    video_hidden_context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run true joint video+action denoising with cross-stream attention on every layer.

    When `use_activation_checkpointing` is true and grad is enabled, each per-block
    (video, action) joint step is wrapped in `torch.utils.checkpoint.checkpoint`
    so the intermediate activations are freed after forward and recomputed at
    backward time. Trades ~1.3x forward compute for a large activation-memory win
    (useful on the two-stream training path where both experts are trainable).
    """

    batch_size = noisy_video_latents.shape[0]
    _, _, num_frames, latent_height, latent_width = noisy_video_latents.shape
    resolved_text = (
        torch.zeros(
            batch_size,
            visual_tower.config.max_text_tokens,
            visual_tower.config.text_dim,
            device=noisy_video_latents.device,
            dtype=noisy_video_latents.dtype,
        )
        if text_context is None
        else text_context.to(device=noisy_video_latents.device, dtype=noisy_video_latents.dtype)
    )
    video_prepared = visual_tower.core.prepare_exact_single_stream_inputs(
        {
            "noisy_latents": noisy_video_latents,
            "text_emb": resolved_text,
            "grid_id": build_video_grid_ids(
                _video_token_grid_for_latents(visual_tower, noisy_video_latents),
                device=noisy_video_latents.device,
                frame_shift=float(frame_start),
            )[None].expand(batch_size, -1, -1),
            "timesteps": video_timesteps,
        },
        action_mode=False,
    )
    video_hidden_states = video_prepared["hidden_states"]
    if video_hidden_context is not None:
        if tuple(video_hidden_context.shape) != tuple(video_hidden_states.shape):
            raise ValueError(
                "DualExpert joint video hidden_context must match embedded video hidden states, "
                f"got hidden_context={tuple(video_hidden_context.shape)}, "
                f"hidden_states={tuple(video_hidden_states.shape)}."
            )
        video_hidden_states = video_hidden_states + video_hidden_context.to(
            device=video_hidden_states.device,
            dtype=video_hidden_states.dtype,
        )
    video_text_hidden_states = video_prepared["text_hidden_states"]
    video_rotary_emb = video_prepared["rotary_emb"]
    video_temb = video_prepared["temb"]
    video_timestep_proj = video_prepared["timestep_proj"]

    action_hidden_states = action_pre.tokens
    action_rotary_emb = action_pre.freqs[:, :, None]
    video_seq_len = int(video_hidden_states.shape[1])
    action_seq_len = int(action_hidden_states.shape[1])
    expected_total = video_seq_len + action_seq_len
    if attention_mask.shape != (expected_total, expected_total):
        raise ValueError(
            "DualExpert joint runtime requires a square joint attention mask matching video+action length, "
            f"got attention_mask={tuple(attention_mask.shape)}, expected=({expected_total}, {expected_total})."
        )
    video_attention_mask = attention_mask[:video_seq_len, :expected_total][None, None, :, :]
    action_attention_mask = attention_mask[video_seq_len:, :expected_total][None, None, :, :]

    def _joint_block_step(
        video_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        video_block,
        action_block,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_attn_inputs = video_block.prepare_self_attention_inputs(
            video_hidden_states,
            temb=video_timestep_proj,
            rotary_emb=video_rotary_emb,
        )
        action_attn_inputs = action_block.prepare_self_attention_inputs(
            action_hidden_states,
            temb=action_pre.t_mod,
            rotary_emb=action_rotary_emb,
        )
        joint_key = torch.cat([video_attn_inputs["key"], action_attn_inputs["key"]], dim=2)
        joint_value = torch.cat([video_attn_inputs["value"], action_attn_inputs["value"]], dim=2)

        mixed_video = F.scaled_dot_product_attention(
            video_attn_inputs["query"],
            joint_key,
            joint_value,
            attn_mask=video_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).flatten(2, 3)
        mixed_action = F.scaled_dot_product_attention(
            action_attn_inputs["query"],
            joint_key,
            joint_value,
            attn_mask=action_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).flatten(2, 3)

        new_video, _ = video_block.apply_post_attention(
            video_attn_inputs["hidden_states"],
            mixed_attn_output=video_block.attn1.to_out[1](
                linear_with_materialized_params(video_block.attn1.to_out[0], mixed_video)
            ),
            encoder_hidden_states=video_text_hidden_states,
            gate_msa=video_attn_inputs["gate_msa"],
            c_shift_msa=video_attn_inputs["c_shift_msa"],
            c_scale_msa=video_attn_inputs["c_scale_msa"],
            c_gate_msa=video_attn_inputs["c_gate_msa"],
            cross_attention_mask=video_cross_attention_mask,
        )
        new_action, _ = action_block.apply_post_attention(
            action_attn_inputs["hidden_states"],
            mixed_attn_output=action_block.attn1.to_out[1](
                linear_with_materialized_params(action_block.attn1.to_out[0], mixed_action)
            ),
            encoder_hidden_states=action_pre.context,
            gate_msa=action_attn_inputs["gate_msa"],
            c_shift_msa=action_attn_inputs["c_shift_msa"],
            c_scale_msa=action_attn_inputs["c_scale_msa"],
            c_gate_msa=action_attn_inputs["c_gate_msa"],
            cross_attention_mask=action_pre.cross_attention_mask,
        )
        return new_video, new_action

    checkpoint_active = bool(use_activation_checkpointing) and torch.is_grad_enabled()
    for video_block, action_block in zip(visual_tower.core.blocks, action_expert.blocks, strict=True):
        if checkpoint_active:
            video_hidden_states, action_hidden_states = torch.utils.checkpoint.checkpoint(
                _joint_block_step,
                video_hidden_states,
                action_hidden_states,
                video_block,
                action_block,
                use_reentrant=False,
                context_fn=lambda vb=video_block, ab=action_block: _checkpoint_summon_context(vb, ab),
            )
        else:
            with _unshard_runtime_params(video_block, action_block):
                video_hidden_states, action_hidden_states = _joint_block_step(
                    video_hidden_states,
                    action_hidden_states,
                    video_block,
                    action_block,
                )

    shift, scale = select_chunk_slices(
        materialize_runtime_parameter(
            visual_tower.core.scale_shift_table,
            device=video_temb.device,
            dtype=video_temb.dtype,
        )[None]
        + video_temb[:, :, None, ...],
        2,
    )
    shift = shift.to(video_hidden_states.device)
    scale = scale.to(video_hidden_states.device)
    video_hidden_states = (
        layer_norm_with_materialized_params(visual_tower.core.norm_out, video_hidden_states.float())
        * (1.0 + scale)
        + shift
    ).type_as(video_hidden_states)
    video_flow = linear_with_materialized_params(visual_tower.core.proj_out, video_hidden_states)
    video_flow = unpatchify_video_sequence(
        visual_tower.core.patch_size,
        video_flow,
        num_frames,
        latent_height,
        latent_width,
        batch_size=batch_size,
    )
    return video_flow, action_hidden_states
