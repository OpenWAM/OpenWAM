from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm
from torch import nn

from open_wam.models.visual_tower.grid_ids import build_sequence_grid_ids
from open_wam.models.visual_tower.replica_core import (
    SharedTransformerAttention,
    SharedTransformerRotaryPositionalEmbedding,
    SharedTransformerTimeEmbedding,
    _feed_forward_with_materialized_params,
    _layer_norm_with_materialized_params,
    _linear_with_materialized_params,
    _materialize_runtime_parameter,
    _rms_norm_with_materialized_weight,
    _apply_rotary_emb,
    _select_chunk_slices,
)


@dataclass(frozen=True)
class MoTActionPreprocessOutput:
    """Action-expert inputs aligned to the video expert runtime contract."""

    tokens: torch.Tensor
    freqs: torch.Tensor
    t_mod: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None
    timesteps: torch.Tensor


class MoTActionTransformerBlock(nn.Module):
    """Action-side transformer block with shared attention geometry but independent hidden size."""

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
        temb_scale_shift_table = _materialize_runtime_parameter(
            self.scale_shift_table,
            device=temb.device,
            dtype=temb.dtype,
        )[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = _select_chunk_slices(
            temb_scale_shift_table,
            6,
        )
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
        query = _rms_norm_with_materialized_weight(
            self.attn1.norm_q,
            _linear_with_materialized_params(self.attn1.to_q, norm_hidden_states),
        ).unflatten(2, (self.attn1.heads, -1))
        key = _rms_norm_with_materialized_weight(
            self.attn1.norm_k,
            _linear_with_materialized_params(self.attn1.to_k, norm_hidden_states),
        ).unflatten(2, (self.attn1.heads, -1))
        value = _linear_with_materialized_params(self.attn1.to_v, norm_hidden_states).unflatten(
            2,
            (self.attn1.heads, -1),
        )
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)
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
    ) -> tuple[torch.Tensor, None]:
        hidden_states = (hidden_states.float() + mixed_attn_output.float() * gate_msa).type_as(hidden_states)
        norm_hidden_states = (
            _layer_norm_with_materialized_params(self.norm2, hidden_states.float())
            if isinstance(self.norm2, nn.LayerNorm)
            else self.norm2(hidden_states.float())
        ).type_as(hidden_states)
        attn_output, _ = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            rotary_emb=None,
            attention_mask=None,
            is_cross_attention=True,
            cache_current_token_count=encoder_hidden_states.shape[1],
        )
        hidden_states = hidden_states + attn_output
        norm_hidden_states = (
            _layer_norm_with_materialized_params(self.norm3, hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = _feed_forward_with_materialized_params(self.ffn, norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states, None


class MoTActionExpert(nn.Module):
    """Method-5 action-side expert aligned to the shared video transformer.

    This mirrors the high-level Wan/FastWAM split:
    - `pre_dit(...)` maps noisy actions into expert tokens plus RoPE/time state
    - `blocks` owns the layer stack used by the later mixed-attention runtime
    - `post_dit(...)` projects action tokens back to flow/noise predictions

    The implementation intentionally keeps only the minimal contract needed by
    the upcoming MoT runtime and does not yet couple itself to a specific
    train/infer loop.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        num_layers: int,
        num_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attention_head_dim)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.cross_attn_norm = bool(cross_attn_norm)
        self.eps = float(eps)

        self.action_embedder = nn.Linear(self.action_dim, self.hidden_size)
        self.time_conditioner = SharedTransformerTimeEmbedding(self.hidden_size, self.freq_dim)
        self.context_proj = nn.Linear(self.text_dim, self.hidden_size)
        self.rope = SharedTransformerRotaryPositionalEmbedding(self.attn_head_dim)
        self.blocks = nn.ModuleList(
            [
                MoTActionTransformerBlock(
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
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, self.hidden_size) / self.hidden_size**0.5
        )
        self.action_proj_out = nn.Linear(self.hidden_size, self.action_dim)

    def pre_dit(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> MoTActionPreprocessOutput:
        if action_tokens.ndim != 3:
            raise ValueError(
                "MoT action expert expects `action_tokens` with shape [B, T, A], "
                f"got {tuple(action_tokens.shape)}."
            )
        if context.ndim != 3:
            raise ValueError(
                "MoT action expert expects `context` with shape [B, L, D], "
                f"got {tuple(context.shape)}."
            )
        batch_size, seq_len, _ = action_tokens.shape
        if context.shape[0] != batch_size:
            raise ValueError(
                "MoT action expert requires action/context batch sizes to match, "
                f"got action batch={batch_size}, context batch={context.shape[0]}."
            )
        if context.shape[2] != self.text_dim:
            raise ValueError(
                "MoT action expert requires context last dim to match `text_dim`, "
                f"got context.shape[2]={context.shape[2]}, text_dim={self.text_dim}."
            )
        if timestep.ndim == 1:
            if timestep.shape[0] != batch_size:
                raise ValueError(
                    "MoT action expert expects scalar-per-batch timesteps with shape [B] or dense [B, T], "
                    f"got {tuple(timestep.shape)} for batch_size={batch_size}."
                )
            timestep = timestep[:, None].expand(-1, seq_len)
        elif timestep.ndim == 2:
            if timestep.shape != (batch_size, seq_len):
                raise ValueError(
                    "MoT action expert expects dense timesteps with shape [B, T], "
                    f"got {tuple(timestep.shape)} for action shape {tuple(action_tokens.shape)}."
                )
        else:
            raise ValueError(
                "MoT action expert expects timestep rank 1 or 2, "
                f"got {tuple(timestep.shape)}."
            )
        if context_mask is not None and context_mask.shape != context.shape[:2]:
            raise ValueError(
                "MoT action expert expects context_mask with shape [B, L], "
                f"got {tuple(context_mask.shape)} for context {tuple(context.shape)}."
            )

        tokens = self.action_embedder(action_tokens)
        _, t_mod = self.time_conditioner(timestep.to(device=tokens.device, dtype=torch.float32), dtype=tokens.dtype)
        projected_context = self.context_proj(context.to(device=tokens.device, dtype=tokens.dtype))
        grid_ids = build_sequence_grid_ids(seq_len, device=tokens.device)[None].expand(batch_size, -1, -1)
        freqs = self.rope(grid_ids)
        resolved_context_mask = (
            None if context_mask is None else context_mask.to(device=tokens.device, dtype=torch.bool)
        )
        return MoTActionPreprocessOutput(
            tokens=tokens,
            freqs=freqs,
            t_mod=t_mod,
            context=projected_context,
            context_mask=resolved_context_mask,
            timesteps=timestep.to(device=tokens.device, dtype=torch.float32),
        )

    def post_dit(
        self,
        hidden_states: torch.Tensor,
        preprocessed: MoTActionPreprocessOutput,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "MoT action expert expects hidden_states with shape [B, T, D], "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[:2] != preprocessed.tokens.shape[:2]:
            raise ValueError(
                "MoT action expert post_dit requires sequence shape to match pre_dit output, "
                f"got hidden_states={tuple(hidden_states.shape)}, preprocessed={tuple(preprocessed.tokens.shape)}."
            )
        shift, scale = _select_chunk_slices(
            _materialize_runtime_parameter(
                self.scale_shift_table,
                device=preprocessed.t_mod.device,
                dtype=preprocessed.t_mod.dtype,
            )[None]
            + preprocessed.t_mod[:, :, :2, :],
            2,
        )
        hidden_states = (
            _layer_norm_with_materialized_params(self.norm_out, hidden_states.float()) * (1.0 + scale) + shift
        ).type_as(hidden_states)
        return _linear_with_materialized_params(self.action_proj_out, hidden_states)


def init_action_expert_from_video_core(
    *,
    action_expert: MoTActionExpert,
    video_core,
    mode: str = "video_weight_copy",
) -> None:
    """Initialize the MoT action expert from the shared video core.

    The intended default is a blockwise copy for all shape-compatible weights.
    Action-specific input/output layers remain untouched because they do not
    share the same tensor semantics as video patch embeddings.
    """

    if mode == "random":
        return
    if mode not in {"video_weight_copy", "video_weight_interpolate"}:
        raise ValueError(f"Unsupported MoT action expert init mode {mode!r}.")
    if len(action_expert.blocks) != len(video_core.blocks):
        raise ValueError(
            "MoT action expert initialization requires matching layer counts, "
            f"got action={len(action_expert.blocks)}, video={len(video_core.blocks)}."
        )
    if action_expert.num_heads != video_core.config.num_heads:
        raise ValueError(
            "MoT action expert initialization requires matching head counts, "
            f"got action={action_expert.num_heads}, video={video_core.config.num_heads}."
        )
    if action_expert.attn_head_dim != video_core.config.attention_head_dim:
        raise ValueError(
            "MoT action expert initialization requires matching head dims, "
            f"got action={action_expert.attn_head_dim}, video={video_core.config.attention_head_dim}."
        )

    with torch.no_grad():
        _load_resized_state_dict(
            action_expert.time_conditioner,
            video_core.action_time_conditioner.state_dict(),
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
                "MoT action expert copy initialization requires matching `scale_shift_table` shapes, "
                f"got action={tuple(action_expert.scale_shift_table.shape)}, "
                f"video={tuple(video_core.scale_shift_table.shape)}."
            )
        for action_block, video_block in zip(action_expert.blocks, video_core.blocks, strict=True):
            _load_resized_state_dict(
                action_block,
                video_block.state_dict(),
                allow_resize=(mode == "video_weight_interpolate"),
            )


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
            raise ValueError(f"Missing source parameter `{key}` during MoT action expert initialization.")
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
                "MoT action expert copy initialization requires matching parameter shapes, "
                f"got key={key!r}, action={tuple(target.shape)}, video={tuple(source.shape)}."
            )
        merged_state[key] = value.to(device=target.device, dtype=target.dtype)
    module.load_state_dict(merged_state, strict=True)
