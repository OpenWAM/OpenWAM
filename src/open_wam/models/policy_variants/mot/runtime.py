from __future__ import annotations

import math
from contextlib import ExitStack

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from open_wam.configs import MoTConditionMode
from open_wam.models.common.attention_profiles import (
    PreparedAttentionProfile,
    apply_attention_backend,
    select_attention_profile_mask,
)
from open_wam.models.common.video_geometry import (
    unpatchify_video_sequence,
    video_token_grid_from_latent_shape,
)
from open_wam.models.visual_tower.grid_ids import build_video_grid_ids
from open_wam.models.visual_tower.shared_transformer_support import (
    layer_norm_with_materialized_params,
    linear_with_materialized_params,
    materialize_runtime_parameter,
    select_chunk_slices,
)

from .attention import (
    build_chunk_causal_video_mask as build_chunk_causal_video_mask,
    build_mot_attention_mask as build_mot_attention_mask,
    build_mot_inference_action_attention_mask as build_mot_inference_action_attention_mask,
    build_mot_packed_coupling_attention_mask as build_mot_packed_coupling_attention_mask,
    build_mot_packed_coupling_attention_profile as build_mot_packed_coupling_attention_profile,
    build_packed_action_attention_mask as build_packed_action_attention_mask,
)
from .cache_state import (
    append_mot_action_cache as append_mot_action_cache,
    move_mot_action_cache as move_mot_action_cache,
    move_mot_video_cache as move_mot_video_cache,
    rewind_mot_runtime_action_cache_to_frame as rewind_mot_runtime_action_cache_to_frame,
    trim_mot_action_cache_prefix as trim_mot_action_cache_prefix,
    trim_mot_action_cache_tail as trim_mot_action_cache_tail,
    trim_mot_video_cache_tail as trim_mot_video_cache_tail,
)
from .contracts import (
    MoTActionCache,
    MoTActionLayerCache,
    MoTVideoCache,
    MoTVideoLayerCache,
)
from .modules import MoTActionExpert, MoTActionPreprocessOutput

try:  # pragma: no cover - import surface depends on torch build
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover - CPU-only or non-FSDP env
    FSDP = None


def _video_token_grid_for_latents(visual_tower, video_latents: torch.Tensor):
    return video_token_grid_from_latent_shape(
        video_latents,
        patch_size=visual_tower.core.patch_size,
    )


def prefill_video_kv_cache(
    *,
    visual_tower,
    observed_prefix: torch.Tensor,
    text_context: torch.Tensor | None,
    frame_start: int = 0,
    attention_mask: torch.Tensor | None = None,
    cross_attention_mask: torch.Tensor | None = None,
    detach_cache: bool = True,
) -> MoTVideoCache:
    """Run the observed video prefix once and cache per-layer self-attention K/V.

    ``attention_mask``: optional square ``[S, S]`` mask over the post-patch
    video token sequence; used to make the prefill chunk-causal when
    teacher-forcing a full clean video during training. Default ``None``
    preserves the fully bidirectional prefill used at inference over
    observed history.

    ``detach_cache``: when True (inference default), K/V are detached so they
    can cross device/dtype boundaries without participating in autograd. Set
    to False during training if you want action-loss gradients to flow back
    through the shared video core (matches Method 1 shared-backbone behavior).
    """

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
        attention_mask=attention_mask,
        cross_attention_mask=cross_attention_mask,
        detach_cache=detach_cache,
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
                key=entry.key.detach() if detach_cache else entry.key,
                value=entry.value.detach() if detach_cache else entry.value,
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


def mot_scheduler_next_sigma(scheduler, step_index: int) -> torch.Tensor:
    """Resolve the next MoT integration sigma, ending every schedule at zero."""

    if int(step_index) + 1 >= len(scheduler.sigmas):
        return scheduler.sigmas.new_tensor(0.0)
    return scheduler.sigmas[int(step_index) + 1]


def step_mot_flow_with_sigmas(
    sample: torch.Tensor,
    flow_pred: torch.Tensor,
    *,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Apply one explicit-sigma Euler flow step."""

    return sample + flow_pred * (
        sigma_next.to(device=sample.device, dtype=sample.dtype)
        - sigma.to(device=sample.device, dtype=sample.dtype)
    )


def expand_mot_scalar_timestep(
    value: torch.Tensor | float,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Materialize a scalar timestep over a requested MoT stream shape."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar timestep value, got shape {tuple(value.shape)}.")
        return value.to(device=device, dtype=torch.float32).reshape(()).expand(shape).clone()
    return torch.full(shape, float(value), device=device, dtype=torch.float32)


def forward_action_with_video_and_action_cache(
    *,
    action_expert: MoTActionExpert,
    action_pre: MoTActionPreprocessOutput,
    video_cache: MoTVideoCache,
    action_cache: MoTActionCache | None,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, MoTActionCache]:
    """Run the action expert against cached video AND cached past-action K/V.

    Method-1-aligned variant of `forward_action_with_video_cache`. Past
    action chunks contribute per-layer clean K/V via `action_cache`; the
    current (fresh, noisy) chunk's K/V is recomputed from `action_pre`
    each call. Returns the action hidden states plus the per-layer fresh
    K/V, so the caller can append to the running `MoTActionCache` at the
    last denoise step (matching Method 1's `update_cache=1` semantics).
    """

    if len(video_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "MoT action runtime requires one cached video K/V pair per action layer, "
            f"got video_cache_layers={len(video_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    if action_cache is not None and len(action_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "MoT action runtime requires one cached action K/V pair per action layer, "
            f"got action_cache_layers={len(action_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    fresh_action_seq_len = int(action_pre.tokens.shape[1])
    action_cache_seq_len = int(action_cache.action_seq_len) if action_cache is not None else 0
    expected_total = video_cache.video_seq_len + action_cache_seq_len + fresh_action_seq_len
    if attention_mask.shape != (expected_total, expected_total):
        raise ValueError(
            "MoT action runtime requires a square joint attention mask matching video+action_cache+fresh length, "
            f"got attention_mask={tuple(attention_mask.shape)}, expected=({expected_total}, {expected_total})."
        )

    hidden_states = action_pre.tokens
    fresh_q_start = video_cache.video_seq_len + action_cache_seq_len
    action_attention_mask = attention_mask[fresh_q_start:, :expected_total][None, None, :, :]
    action_rotary_emb = action_pre.freqs[:, :, None]
    fresh_kv_layers: list[MoTActionLayerCache] = []
    with _summon_full_params(action_expert):
        for layer_index, (block, vid_layer) in enumerate(
            zip(action_expert.blocks, video_cache.layers, strict=True)
        ):
            attn_inputs = block.prepare_self_attention_inputs(
                hidden_states,
                temb=action_pre.t_mod,
                rotary_emb=action_rotary_emb,
            )
            key_parts: list[torch.Tensor] = [vid_layer.key]
            value_parts: list[torch.Tensor] = [vid_layer.value]
            if action_cache is not None:
                act_layer = action_cache.layers[layer_index]
                key_parts.append(act_layer.key)
                value_parts.append(act_layer.value)
            key_parts.append(attn_inputs["key"])
            value_parts.append(attn_inputs["value"])
            mixed = F.scaled_dot_product_attention(
                attn_inputs["query"],
                torch.cat(key_parts, dim=2),
                torch.cat(value_parts, dim=2),
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
                cross_attention_mask=action_pre.cross_attention_mask,
            )
            fresh_kv_layers.append(
                MoTActionLayerCache(
                    key=attn_inputs["key"].detach(),
                    value=attn_inputs["value"].detach(),
                )
            )
    fresh_cache = MoTActionCache(
        layers=tuple(fresh_kv_layers),
        action_seq_len=int(fresh_kv_layers[0].key.shape[2]),
    )
    return hidden_states, fresh_cache


def forward_mot_packed_coupling_denoise(
    *,
    visual_tower,
    noisy_video_latents: torch.Tensor,
    clean_video_latents: torch.Tensor,
    noisy_video_timesteps: torch.Tensor,
    clean_video_timesteps: torch.Tensor | None,
    action_expert: MoTActionExpert,
    packed_action_pre: MoTActionPreprocessOutput,
    attention_profile: PreparedAttentionProfile,
    text_context: torch.Tensor | None,
    frame_start: int = 0,
    use_activation_checkpointing: bool = False,
    packed_block_stack=None,
    prefer_flex_attention: bool = True,
    video_cross_attention_mask: torch.Tensor | None = None,
    video_hidden_context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run M5's native four-stream packed coupling forward.

    Video/action experts execute separate blocks, but every block attends over
    concatenated K/V from ``[V_noisy, V_clean, A_noisy, A_clean]`` using the
    supplied coupling mask. Returns the V_noisy flow and packed action hidden
    states; callers take loss on the first action half.
    """

    if noisy_video_latents.shape != clean_video_latents.shape:
        raise ValueError(
            "M5 packed coupling expects matching noisy/clean video shapes, "
            f"got noisy={tuple(noisy_video_latents.shape)}, clean={tuple(clean_video_latents.shape)}."
        )
    batch_size = noisy_video_latents.shape[0]
    _, _, num_frames, latent_height, latent_width = noisy_video_latents.shape
    effective_clean_video_timesteps = (
        torch.zeros_like(noisy_video_timesteps) if clean_video_timesteps is None else clean_video_timesteps
    )
    if noisy_video_timesteps.shape != effective_clean_video_timesteps.shape:
        raise ValueError(
            "M5 packed coupling expects matching noisy/clean video timestep shapes, "
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
                "M5 packed video hidden_context must match embedded video hidden states, "
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
                "M5 packed coupling requires a dense or flex attention profile matching packed video+action length, "
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
        # FSDP-friendly path: each MoTPackedBlock owns its (video_block,
        # action_block) pair and is its own FSDP unit. Calling the wrapper's
        # forward triggers FSDP's standard pre/post-forward hooks; backward
        # gather is also FSDP-managed. No manual `_summon_full_params` /
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
    return video_flow, action_hidden_states


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
                cross_attention_mask=action_pre.cross_attention_mask,
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
                "MoT joint video hidden_context must match embedded video hidden states, "
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
            "MoT joint runtime requires a square joint attention mask matching video+action length, "
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


class _DummyCtx:
    """No-op context manager used when we want to skip ``summon_full_params``."""

    def __enter__(self) -> "_DummyCtx":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


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


class _FSDP2UnshardCtx:
    def __init__(self, *modules) -> None:
        self._modules = modules
        self._unsharded: list[object] = []

    def __enter__(self) -> "_FSDP2UnshardCtx":
        seen_ids: set[int] = set()
        for module in self._modules:
            for submodule in module.modules():
                module_id = id(submodule)
                if module_id in seen_ids:
                    continue
                seen_ids.add(module_id)
                unshard = getattr(submodule, "unshard", None)
                reshard = getattr(submodule, "reshard", None)
                if not callable(unshard) or not callable(reshard):
                    continue
                unshard()
                self._unsharded.append(submodule)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        for submodule in reversed(self._unsharded):
            reshard = getattr(submodule, "reshard", None)
            if callable(reshard):
                reshard()
        self._unsharded.clear()
        return False


def _unshard_runtime_params(*modules):
    stack = ExitStack()
    stack.enter_context(_summon_full_params(*modules))
    stack.enter_context(_FSDP2UnshardCtx(*modules))
    return stack


def _checkpoint_summon_context(video_block, action_block):
    return (
        _unshard_runtime_params(video_block, action_block),
        _unshard_runtime_params(video_block, action_block),
    )
