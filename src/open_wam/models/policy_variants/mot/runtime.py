from __future__ import annotations

import math
from contextlib import ExitStack

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.configs import MoTConditionMode
from open_wam.models.visual_tower.grid_ids import build_video_grid_ids
from open_wam.models.policy_variants.parallel_stream.reference_runtime import data_seq_to_patch
from open_wam.models.visual_tower.shared_transformer_support import (
    layer_norm_with_materialized_params,
    linear_with_materialized_params,
    materialize_runtime_parameter,
    select_chunk_slices,
)

from .contracts import MoTVideoCache, MoTVideoLayerCache
from .modules import MoTActionExpert, MoTActionPreprocessOutput

try:  # pragma: no cover - import surface depends on torch build
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover - CPU-only or non-FSDP env
    FSDP = None


def move_mot_video_cache(
    video_cache: MoTVideoCache,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> MoTVideoCache:
    target_device = torch.device(device)
    moved_layers: list[MoTVideoLayerCache] = []
    for layer in video_cache.layers:
        moved_layers.append(
            MoTVideoLayerCache(
                key=layer.key.to(device=target_device, dtype=dtype if dtype is not None else layer.key.dtype),
                value=layer.value.to(device=target_device, dtype=dtype if dtype is not None else layer.value.dtype),
            )
        )
    return MoTVideoCache(
        layers=tuple(moved_layers),
        video_seq_len=video_cache.video_seq_len,
    )


def build_mot_attention_mask(
    *,
    video_seq_len: int,
    action_seq_len: int,
    device: torch.device,
    condition_mode: MoTConditionMode | str,
    video_tokens_per_frame: int | None = None,
    video_can_attend_action: bool = False,
) -> torch.Tensor:
    """Build a shared MoT mask for the FastWAM conditioning variants."""

    if video_seq_len <= 0 or action_seq_len <= 0:
        raise ValueError(
            "MoT attention mask requires positive video and action lengths, "
            f"got video_seq_len={video_seq_len}, action_seq_len={action_seq_len}."
        )
    resolved_mode = MoTConditionMode(condition_mode)
    total_seq_len = video_seq_len + action_seq_len
    mask = torch.zeros(total_seq_len, total_seq_len, device=device, dtype=torch.bool)
    mask[:video_seq_len, :video_seq_len] = True
    mask[video_seq_len:, video_seq_len:] = True
    if resolved_mode == MoTConditionMode.FIRST_FRAME:
        if video_tokens_per_frame is None:
            raise ValueError("MoT first-frame conditioning requires `video_tokens_per_frame`.")
        visible_video = min(video_tokens_per_frame, video_seq_len)
    elif resolved_mode in {MoTConditionMode.FULL_VIDEO, MoTConditionMode.TEACHER_FORCING_COND_VIDEO}:
        visible_video = video_seq_len
    else:  # pragma: no cover - enum guard
        raise ValueError(f"Unsupported MoT condition mode {resolved_mode!r}.")
    mask[video_seq_len:, :visible_video] = True
    if video_can_attend_action:
        mask[:video_seq_len, video_seq_len:] = True
    return mask


def prefill_video_kv_cache(
    *,
    visual_tower,
    observed_prefix: torch.Tensor,
    text_context: torch.Tensor | None,
    frame_start: int = 0,
) -> MoTVideoCache:
    """Run the observed video prefix once and cache per-layer self-attention K/V."""

    if observed_prefix.ndim != 5:
        raise ValueError(
            "MoT video prefill expects observed_prefix with shape [B, C, T, H, W], "
            f"got {tuple(observed_prefix.shape)}."
        )
    cache_state = visual_tower.prefill_exact_video_cache(
        observed_prefix=observed_prefix,
        text_context=text_context,
        frame_start=frame_start,
        cache_name="mot_video_prefill",
    )
    cache_layers: list[MoTVideoLayerCache] = []
    for layer_index, entry in enumerate(cache_state.self_attention_kv):
        if entry.key is None or entry.value is None:
            raise ValueError(
                "MoT video prefill expected materialized self-attention K/V entries, "
                f"but layer {layer_index} was empty."
            )
        cache_layers.append(
            MoTVideoLayerCache(
                key=entry.key.detach(),
                value=entry.value.detach(),
            )
        )
    if not cache_layers:
        raise ValueError("MoT video prefill did not materialize any layer cache entries.")
    return MoTVideoCache(layers=tuple(cache_layers), video_seq_len=int(cache_layers[0].key.shape[2]))


def resolve_mot_condition_latents(
    *,
    video_latents: torch.Tensor,
    condition_mode: MoTConditionMode | str,
    video_prefix_frames: int,
    teacher_forcing_video_noise_prob: float,
    training: bool,
    scheduler=None,
) -> torch.Tensor:
    """Select the video branch used to condition the MoT action expert."""

    resolved_mode = MoTConditionMode(condition_mode)
    if resolved_mode == MoTConditionMode.FIRST_FRAME:
        return video_latents[:, :, :1]
    if resolved_mode == MoTConditionMode.FULL_VIDEO:
        return video_latents
    if resolved_mode == MoTConditionMode.TEACHER_FORCING_COND_VIDEO:
        cond_latents = video_latents[:, :, : max(1, video_prefix_frames)].clone()
        if (
            training
            and scheduler is not None
            and teacher_forcing_video_noise_prob > 0.0
            and torch.rand(1, device=video_latents.device).item() < teacher_forcing_video_noise_prob
        ):
            batch_size = cond_latents.shape[0]
            timestep_ids = torch.randint(
                low=0,
                high=len(scheduler.timesteps),
                size=(batch_size, cond_latents.shape[2]),
                device=video_latents.device,
            )
            timesteps = scheduler.timesteps.to(device=video_latents.device)[timestep_ids]
            noise = torch.randn_like(cond_latents)
            cond_latents = scheduler.add_noise(cond_latents, noise, timesteps, t_dim=2)
        return cond_latents
    raise ValueError(f"Unsupported MoT condition mode {resolved_mode!r}.")


def forward_action_with_video_cache(
    *,
    action_expert: MoTActionExpert,
    action_pre: MoTActionPreprocessOutput,
    video_cache: MoTVideoCache,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the action expert against a cached video prefix.

    This mirrors the FastWAM `forward_action_with_video_cache` contract:
    action queries are recomputed every denoise step, while video K/V come from
    a prefilled observed-prefix cache. The helper deliberately leaves video
    cache construction to a later stage because that requires exposing
    additional blockwise helpers from the shared video core.
    """

    if len(video_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "MoT action runtime requires one cached video K/V pair per action layer, "
            f"got cache_layers={len(video_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    action_seq_len = int(action_pre.tokens.shape[1])
    expected_total = video_cache.video_seq_len + action_seq_len
    if attention_mask.shape != (expected_total, expected_total):
        raise ValueError(
            "MoT action runtime requires a square joint attention mask matching cache+action length, "
            f"got attention_mask={tuple(attention_mask.shape)}, expected=({expected_total}, {expected_total})."
        )

    hidden_states = action_pre.tokens
    action_attention_mask = attention_mask[video_cache.video_seq_len:, :expected_total][None, None, :, :]
    action_rotary_emb = action_pre.freqs[:, :, None]
    with _summon_full_params(action_expert):
        for block, layer_cache in zip(action_expert.blocks, video_cache.layers, strict=True):
            attn_inputs = block.prepare_self_attention_inputs(
                hidden_states,
                temb=action_pre.t_mod,
                rotary_emb=action_rotary_emb,
            )
            mixed = F.scaled_dot_product_attention(
                attn_inputs["query"],
                torch.cat([layer_cache.key, attn_inputs["key"]], dim=2),
                torch.cat([layer_cache.value, attn_inputs["value"]], dim=2),
                attn_mask=action_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
            hidden_states, _ = block.apply_post_attention(
                attn_inputs["hidden_states"],
                mixed_attn_output=block.attn1.to_out[1](linear_with_materialized_params(block.attn1.to_out[0], mixed)),
                encoder_hidden_states=action_pre.context,
                gate_msa=attn_inputs["gate_msa"],
                c_shift_msa=attn_inputs["c_shift_msa"],
                c_scale_msa=attn_inputs["c_scale_msa"],
                c_gate_msa=attn_inputs["c_gate_msa"],
            )
    return hidden_states


def forward_joint_video_action_denoise(
    *,
    visual_tower,
    noisy_video_latents: torch.Tensor,
    video_timesteps: torch.Tensor,
    action_expert: MoTActionExpert,
    action_pre: MoTActionPreprocessOutput,
    text_context: torch.Tensor | None,
    attention_mask: torch.Tensor,
    frame_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run true joint video+action denoising with cross-stream attention on every layer."""

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
                visual_tower.frontend.tokenize_video_latents(noisy_video_latents)[1],
                device=noisy_video_latents.device,
                frame_shift=float(frame_start),
            )[None].expand(batch_size, -1, -1),
            "timesteps": video_timesteps,
        },
        action_mode=False,
    )
    video_hidden_states = video_prepared["hidden_states"]
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
            "MoT joint runtime requires a square joint attention mask matching video+action length, "
            f"got attention_mask={tuple(attention_mask.shape)}, expected=({expected_total}, {expected_total})."
        )
    video_attention_mask = attention_mask[:video_seq_len, :expected_total][None, None, :, :]
    action_attention_mask = attention_mask[video_seq_len:, :expected_total][None, None, :, :]

    with _summon_full_params(visual_tower.core, action_expert):
        for video_block, action_block in zip(visual_tower.core.blocks, action_expert.blocks, strict=True):
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

            video_hidden_states, _ = video_block.apply_post_attention(
                video_attn_inputs["hidden_states"],
                mixed_attn_output=video_block.attn1.to_out[1](
                    linear_with_materialized_params(video_block.attn1.to_out[0], mixed_video)
                ),
                encoder_hidden_states=video_text_hidden_states,
                gate_msa=video_attn_inputs["gate_msa"],
                c_shift_msa=video_attn_inputs["c_shift_msa"],
                c_scale_msa=video_attn_inputs["c_scale_msa"],
                c_gate_msa=video_attn_inputs["c_gate_msa"],
            )
            action_hidden_states, _ = action_block.apply_post_attention(
                action_attn_inputs["hidden_states"],
                mixed_attn_output=action_block.attn1.to_out[1](
                    linear_with_materialized_params(action_block.attn1.to_out[0], mixed_action)
                ),
                encoder_hidden_states=action_pre.context,
                gate_msa=action_attn_inputs["gate_msa"],
                c_shift_msa=action_attn_inputs["c_shift_msa"],
                c_scale_msa=action_attn_inputs["c_scale_msa"],
                c_gate_msa=action_attn_inputs["c_gate_msa"],
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
        video_flow = data_seq_to_patch(
            visual_tower.core.patch_size,
            video_flow,
            num_frames,
            latent_height,
            latent_width,
            batch_size=batch_size,
        )
    return video_flow, action_hidden_states


def _summon_full_params(*modules):
    stack = ExitStack()
    if FSDP is None:
        return stack
    seen_ids: set[int] = set()
    for module in modules:
        fsdp_modules = tuple(FSDP.fsdp_modules(module, root_only=False))
        if not fsdp_modules:
            continue
        for fsdp_module in fsdp_modules:
            module_id = id(fsdp_module)
            if module_id in seen_ids:
                continue
            seen_ids.add(module_id)
            stack.enter_context(FSDP.summon_full_params(fsdp_module, recurse=False, writeback=False))
    return stack
