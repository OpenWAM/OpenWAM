from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm
from torch import nn

from open_wam.models.visual_tower.grid_ids import build_sequence_grid_ids
from open_wam.models.visual_tower.shared_transformer_support import (
    SharedTransformerAttention,
    SharedTransformerRotaryPositionalEmbedding,
    SharedTransformerTimeEmbedding,
    apply_rotary_emb,
    feed_forward_with_materialized_params,
    layer_norm_with_materialized_params,
    linear_with_materialized_params,
    materialize_runtime_parameter,
    rms_norm_with_materialized_weight,
    select_chunk_slices,
)


@dataclass(frozen=True)
class ActionExpertPreprocessOutput:
    """Action-expert inputs aligned to the shared action-transformer contract."""

    tokens: torch.Tensor
    freqs: torch.Tensor
    t_mod: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None
    cross_attention_mask: torch.Tensor | None
    timesteps: torch.Tensor


class ConditionedActionTransformerBlock(nn.Module):
    """Action-side transformer block with self-attn over actions and cross-attn to context."""

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        attention_head_dim: int,
        cross_attn_norm: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = SharedTransformerAttention(
            dim=dim,
            heads=num_heads,
            dim_head=attention_head_dim,
            eps=eps,
            cross_attention_dim_head=None,
        )
        self.attn2 = SharedTransformerAttention(
            dim=dim,
            heads=num_heads,
            dim_head=attention_head_dim,
            eps=eps,
            cross_attention_dim_head=attention_head_dim,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def prepare_self_attention_inputs(
        self,
        hidden_states: torch.Tensor,
        *,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        temb_scale_shift_table = materialize_runtime_parameter(
            self.scale_shift_table,
            device=temb.device,
            dtype=temb.dtype,
        )[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = select_chunk_slices(
            temb_scale_shift_table,
            6,
        )
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
        query = rms_norm_with_materialized_weight(
            self.attn1.norm_q,
            linear_with_materialized_params(self.attn1.to_q, norm_hidden_states),
        ).unflatten(2, (self.attn1.heads, -1))
        key = rms_norm_with_materialized_weight(
            self.attn1.norm_k,
            linear_with_materialized_params(self.attn1.to_k, norm_hidden_states),
        ).unflatten(2, (self.attn1.heads, -1))
        value = linear_with_materialized_params(self.attn1.to_v, norm_hidden_states).unflatten(
            2,
            (self.attn1.heads, -1),
        )
        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        return {
            "query": query.transpose(1, 2).contiguous(),
            "key": key.transpose(1, 2).contiguous(),
            "value": value.transpose(1, 2).contiguous(),
            "gate_msa": gate_msa,
            "c_shift_msa": c_shift_msa,
            "c_scale_msa": c_scale_msa,
            "c_gate_msa": c_gate_msa,
            "hidden_states": hidden_states,
        }

    def apply_post_attention(
        self,
        hidden_states: torch.Tensor,
        *,
        mixed_attn_output: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        gate_msa: torch.Tensor,
        c_shift_msa: torch.Tensor,
        c_scale_msa: torch.Tensor,
        c_gate_msa: torch.Tensor,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = (hidden_states.float() + mixed_attn_output.float() * gate_msa).type_as(hidden_states)
        norm_hidden_states = (
            layer_norm_with_materialized_params(self.norm2, hidden_states.float())
            if isinstance(self.norm2, nn.LayerNorm)
            else self.norm2(hidden_states.float())
        ).type_as(hidden_states)
        attn_output, _ = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            rotary_emb=None,
            attention_mask=cross_attention_mask,
            is_cross_attention=True,
            cache_current_token_count=encoder_hidden_states.shape[1],
        )
        hidden_states = hidden_states + attn_output
        norm_hidden_states = (
            layer_norm_with_materialized_params(self.norm3, hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = feed_forward_with_materialized_params(self.ffn, norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states, None


class VideoConditionedActionExpert(nn.Module):
    """Reusable action expert for video-conditioned chunk decoding."""

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        num_layers: int,
        num_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        freq_dim: int,
        context_dim: int | None = None,
        text_dim: int | None = None,
        hidden_context_dim: int | None = None,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        resolved_context_dim = context_dim if context_dim is not None else text_dim
        if resolved_context_dim is None:
            raise ValueError("VideoConditionedActionExpert requires `context_dim` or legacy `text_dim`.")
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attention_head_dim)
        self.ffn_dim = int(ffn_dim)
        self.context_dim = int(resolved_context_dim)
        self.text_dim = self.context_dim
        self.hidden_context_dim = int(hidden_context_dim) if hidden_context_dim is not None else self.hidden_size
        self.freq_dim = int(freq_dim)
        self.cross_attn_norm = bool(cross_attn_norm)
        self.eps = float(eps)

        self.action_embedder = nn.Linear(self.action_dim, self.hidden_size)
        self.time_conditioner = SharedTransformerTimeEmbedding(self.hidden_size, self.freq_dim)
        self.context_proj = nn.Linear(self.context_dim, self.hidden_size)
        self.hidden_context_proj = (
            nn.Identity()
            if self.hidden_context_dim == self.hidden_size
            else nn.Linear(self.hidden_context_dim, self.hidden_size)
        )
        self.rope = SharedTransformerRotaryPositionalEmbedding(self.attn_head_dim)
        self.blocks = nn.ModuleList(
            [
                ConditionedActionTransformerBlock(
                    dim=self.hidden_size,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    attention_head_dim=self.attn_head_dim,
                    cross_attn_norm=self.cross_attn_norm,
                    eps=self.eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(self.hidden_size, self.eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, self.hidden_size) / self.hidden_size**0.5)
        self.action_proj_out = nn.Linear(self.hidden_size, self.action_dim)

    def pre_dit(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        action_grid_ids: torch.Tensor | None = None,
        hidden_context: torch.Tensor | None = None,
    ) -> ActionExpertPreprocessOutput:
        if action_tokens.ndim != 3:
            raise ValueError(
                "VideoConditionedActionExpert expects `action_tokens` with shape [B, T, A], "
                f"got {tuple(action_tokens.shape)}."
            )
        if context.ndim != 3:
            raise ValueError(
                "VideoConditionedActionExpert expects `context` with shape [B, L, D], "
                f"got {tuple(context.shape)}."
            )
        batch_size, seq_len, _ = action_tokens.shape
        if context.shape[0] != batch_size:
            raise ValueError(
                "VideoConditionedActionExpert requires action/context batch sizes to match, "
                f"got action batch={batch_size}, context batch={context.shape[0]}."
            )
        if context.shape[2] != self.context_dim:
            raise ValueError(
                "VideoConditionedActionExpert requires context last dim to match `context_dim`, "
                f"got context.shape[2]={context.shape[2]}, context_dim={self.context_dim}."
            )
        if timestep.ndim == 1:
            if timestep.shape[0] != batch_size:
                raise ValueError(
                    "VideoConditionedActionExpert expects scalar-per-batch timesteps with shape [B] or dense [B, T], "
                    f"got {tuple(timestep.shape)} for batch_size={batch_size}."
                )
            timestep = timestep[:, None].expand(-1, seq_len)
        elif timestep.ndim == 2:
            if timestep.shape != (batch_size, seq_len):
                raise ValueError(
                    "VideoConditionedActionExpert expects dense timesteps with shape [B, T], "
                    f"got {tuple(timestep.shape)} for action shape {tuple(action_tokens.shape)}."
                )
        else:
            raise ValueError(
                "VideoConditionedActionExpert expects timestep rank 1 or 2, "
                f"got {tuple(timestep.shape)}."
            )
        if context_mask is not None and context_mask.shape != context.shape[:2]:
            raise ValueError(
                "VideoConditionedActionExpert expects context_mask with shape [B, L], "
                f"got {tuple(context_mask.shape)} for context {tuple(context.shape)}."
            )
        if cross_attention_mask is not None and cross_attention_mask.shape != (batch_size, seq_len, context.shape[1]):
            raise ValueError(
                "VideoConditionedActionExpert expects cross_attention_mask with shape [B, T, L], "
                f"got {tuple(cross_attention_mask.shape)} for action/context shapes "
                f"{tuple(action_tokens.shape)} / {tuple(context.shape)}."
            )

        tokens = self.action_embedder(action_tokens)
        if hidden_context is not None:
            expected_hidden_shape = (batch_size, seq_len, self.hidden_context_dim)
            if tuple(hidden_context.shape) != expected_hidden_shape:
                raise ValueError(
                    "VideoConditionedActionExpert expects hidden_context to match action token sequence and configured "
                    "hidden_context_dim, "
                    f"got hidden_context={tuple(hidden_context.shape)}, expected={expected_hidden_shape}."
                )
            projected_hidden_context = self.hidden_context_proj(
                hidden_context.to(device=tokens.device, dtype=tokens.dtype)
            )
            tokens = tokens + projected_hidden_context.to(device=tokens.device, dtype=tokens.dtype)
        _, t_mod = self.time_conditioner(timestep.to(device=tokens.device, dtype=torch.float32), dtype=tokens.dtype)
        projected_context = self.context_proj(context.to(device=tokens.device, dtype=tokens.dtype))
        if action_grid_ids is not None:
            if action_grid_ids.shape != (batch_size, 4, seq_len):
                raise ValueError(
                    "VideoConditionedActionExpert expects action_grid_ids with shape [B, 4, T], "
                    f"got {tuple(action_grid_ids.shape)} for action shape {tuple(action_tokens.shape)}."
                )
            grid_ids = action_grid_ids.to(device=tokens.device, dtype=torch.float32)
        else:
            grid_ids = build_sequence_grid_ids(seq_len, device=tokens.device)[None].expand(batch_size, -1, -1)
        freqs = self.rope(grid_ids)
        resolved_context_mask = None if context_mask is None else context_mask.to(device=tokens.device, dtype=torch.bool)
        resolved_cross_attention_mask = (
            None
            if cross_attention_mask is None
            else cross_attention_mask.to(device=tokens.device, dtype=torch.bool)
        )
        if resolved_cross_attention_mask is None and resolved_context_mask is not None:
            resolved_cross_attention_mask = resolved_context_mask[:, None, :].expand(-1, seq_len, -1)
        return ActionExpertPreprocessOutput(
            tokens=tokens,
            freqs=freqs,
            t_mod=t_mod,
            context=projected_context,
            context_mask=resolved_context_mask,
            cross_attention_mask=resolved_cross_attention_mask,
            timesteps=timestep.to(device=tokens.device, dtype=torch.float32),
        )

    def forward_layers(self, preprocessed: ActionExpertPreprocessOutput) -> torch.Tensor:
        hidden_states = preprocessed.tokens
        rotary_emb = preprocessed.freqs[:, :, None]
        for block in self.blocks:
            attn_inputs = block.prepare_self_attention_inputs(
                hidden_states,
                temb=preprocessed.t_mod,
                rotary_emb=rotary_emb,
            )
            mixed = F.scaled_dot_product_attention(
                attn_inputs["query"],
                attn_inputs["key"],
                attn_inputs["value"],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
            hidden_states, _ = block.apply_post_attention(
                attn_inputs["hidden_states"],
                mixed_attn_output=block.attn1.to_out[1](linear_with_materialized_params(block.attn1.to_out[0], mixed)),
                encoder_hidden_states=preprocessed.context,
                gate_msa=attn_inputs["gate_msa"],
                c_shift_msa=attn_inputs["c_shift_msa"],
                c_scale_msa=attn_inputs["c_scale_msa"],
                c_gate_msa=attn_inputs["c_gate_msa"],
                cross_attention_mask=preprocessed.cross_attention_mask,
            )
        return hidden_states

    def forward_conditioned(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        action_grid_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preprocessed = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            cross_attention_mask=cross_attention_mask,
            action_grid_ids=action_grid_ids,
        )
        hidden_states = self.forward_layers(preprocessed)
        return self.post_dit(hidden_states, preprocessed)

    def post_dit(
        self,
        hidden_states: torch.Tensor,
        preprocessed: ActionExpertPreprocessOutput,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "VideoConditionedActionExpert expects hidden_states with shape [B, T, D], "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[:2] != preprocessed.tokens.shape[:2]:
            raise ValueError(
                "VideoConditionedActionExpert post_dit requires sequence shape to match pre_dit output, "
                f"got hidden_states={tuple(hidden_states.shape)}, preprocessed={tuple(preprocessed.tokens.shape)}."
            )
        shift, scale = select_chunk_slices(
            materialize_runtime_parameter(
                self.scale_shift_table,
                device=preprocessed.t_mod.device,
                dtype=preprocessed.t_mod.dtype,
            )[None]
            + preprocessed.t_mod[:, :, :2, :],
            2,
        )
        hidden_states = (
            layer_norm_with_materialized_params(self.norm_out, hidden_states.float()) * (1.0 + scale) + shift
        ).type_as(hidden_states)
        return linear_with_materialized_params(self.action_proj_out, hidden_states)


def init_conditioned_action_expert_from_video_core(
    *,
    action_expert: VideoConditionedActionExpert,
    video_core,
    mode: str = "video_weight_copy",
) -> None:
    """Initialize a conditioned action expert from the shared video core."""

    if mode == "random":
        return
    if mode not in {"video_weight_copy", "video_weight_interpolate"}:
        raise ValueError(f"Unsupported action expert init mode {mode!r}.")
    video_blocks = _select_video_blocks_for_action_expert(
        video_blocks=video_core.blocks,
        target_layer_count=len(action_expert.blocks),
        mode=mode,
    )
    if len(action_expert.blocks) != len(video_blocks):
        raise ValueError(
            "Action expert initialization resolved the wrong number of source layers, "
            f"got action={len(action_expert.blocks)}, selected_video={len(video_blocks)}."
        )
    if action_expert.num_heads != int(video_core.config.num_heads):
        raise ValueError(
            "Action expert initialization requires matching head counts, "
            f"got action={action_expert.num_heads}, video={video_core.config.num_heads}."
        )
    resolved_video_head_dim = int(
        video_core.config.attention_head_dim or (video_core.config.hidden_size // video_core.config.num_heads)
    )
    if action_expert.attn_head_dim != resolved_video_head_dim:
        raise ValueError(
            "Action expert initialization requires matching head dims, "
            f"got action={action_expert.attn_head_dim}, video={resolved_video_head_dim}."
        )

    with torch.no_grad():
        _load_resized_state_dict(
            action_expert.time_conditioner,
            video_core.time_conditioner.state_dict(),
            allow_resize=(mode == "video_weight_interpolate"),
        )
        if tuple(action_expert.scale_shift_table.shape) == tuple(video_core.scale_shift_table.shape):
            action_expert.scale_shift_table.copy_(video_core.scale_shift_table)
        elif mode == "video_weight_interpolate":
            action_expert.scale_shift_table.copy_(
                _resize_tensor_to_shape(video_core.scale_shift_table, tuple(action_expert.scale_shift_table.shape)).to(
                    device=action_expert.scale_shift_table.device,
                    dtype=action_expert.scale_shift_table.dtype,
                )
            )
        else:
            raise ValueError(
                "Action expert copy initialization requires matching `scale_shift_table` shapes, "
                f"got action={tuple(action_expert.scale_shift_table.shape)}, "
                f"video={tuple(video_core.scale_shift_table.shape)}."
            )
        for action_block, video_block in zip(action_expert.blocks, video_blocks, strict=True):
            _load_resized_state_dict(
                action_block,
                video_block.state_dict(),
                allow_resize=(mode == "video_weight_interpolate"),
            )


def _select_video_blocks_for_action_expert(
    *,
    video_blocks: nn.ModuleList,
    target_layer_count: int,
    mode: str,
) -> list[nn.Module]:
    source_layer_count = len(video_blocks)
    target_layer_count = int(target_layer_count)
    if target_layer_count <= 0:
        raise ValueError("Action expert initialization requires at least one action layer.")
    if source_layer_count == target_layer_count:
        return list(video_blocks)
    if mode != "video_weight_interpolate":
        raise ValueError(
            "Action expert copy initialization requires matching layer counts. "
            "Use `action_expert_init_mode = video_weight_interpolate` for a shallower action expert, "
            f"got action={target_layer_count}, video={source_layer_count}."
        )
    if source_layer_count <= 0:
        raise ValueError("Action expert initialization requires at least one source video layer.")
    if target_layer_count > source_layer_count:
        raise ValueError(
            "`action_expert_init_mode = video_weight_interpolate` supports matching or shallower action experts only, "
            f"got action={target_layer_count}, video={source_layer_count}."
        )
    if target_layer_count == 1:
        return [video_blocks[-1]]
    selected_indices = [
        round(index * (source_layer_count - 1) / (target_layer_count - 1))
        for index in range(target_layer_count)
    ]
    return [video_blocks[int(index)] for index in selected_indices]


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}."
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [index for index in range(out.ndim) if index != dim] + [dim]
        inv_perm = [0] * out.ndim
        for index, value in enumerate(perm):
            inv_perm[value] = index
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}."
        )
    return out.to(dtype=src.dtype)


def _load_resized_state_dict(
    module: nn.Module,
    source_state_dict: dict[str, torch.Tensor],
    *,
    allow_resize: bool,
) -> None:
    target_state = module.state_dict()
    merged_state = dict(target_state)
    for key, target in target_state.items():
        if key not in source_state_dict:
            raise ValueError(f"Missing source parameter `{key}` during action expert initialization.")
        source = source_state_dict[key]
        if tuple(source.shape) == tuple(target.shape):
            value = source
        elif allow_resize:
            value = _resize_tensor_to_shape(source, tuple(target.shape))
            if source.ndim >= 2 and source.shape[-1] != target.shape[-1]:
                alpha = (float(source.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
        else:
            raise ValueError(
                "Action expert copy initialization requires matching parameter shapes, "
                f"got key={key!r}, action={tuple(target.shape)}, video={tuple(source.shape)}."
            )
        merged_state[key] = value.to(device=target.device, dtype=target.dtype)
    module.load_state_dict(merged_state, strict=True)
