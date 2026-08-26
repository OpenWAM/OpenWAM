from __future__ import annotations

import math

import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig

from .reference_transformer import preferred_reference_dtype
from .runtime_programs import (
    RuntimeStepInput,
    build_single_stream_exact_runtime_program,
)


def resolve_runtime_module_dtype(module: torch.nn.Module) -> torch.dtype:
    """Resolve the floating-point dtype used by an exact runtime module."""

    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    try:
        device = next(module.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return preferred_reference_dtype(device)


def build_reference_mesh_id(
    f: int,
    h: int,
    w: int,
    *,
    t: int,
    f_w: int = 1,
    f_shift: int = 0,
    action: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the exact integer-preserving LingBot reference mesh."""

    f_idx = torch.arange(f_shift, f + f_shift, device=device) * f_w
    h_idx = torch.arange(h, device=device)
    w_idx = torch.arange(w, device=device)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h], device=device).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1
    grid_id = torch.cat(
        [ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0
    ).flatten(1)
    return torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)


def prepare_exact_single_stream_input(
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor | float,
    text_emb: torch.Tensor,
    frame_st_id: int,
    backbone_config: SharedVideoTransformerConfig,
    action_mode: bool,
    cond: torch.Tensor | None = None,
    action_channel_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build the exact LingBot single-stream input payload."""

    batch_size, _, num_frames, height, width = latents.shape
    device = latents.device
    if isinstance(timestep, torch.Tensor):
        timestep_value = (
            float(timestep.item())
            if timestep.ndim == 0
            else timestep.to(device=device, dtype=torch.float32)
        )
    else:
        timestep_value = float(timestep)
    if isinstance(timestep_value, float):
        timesteps = (
            torch.ones(num_frames, device=device, dtype=torch.float32) * timestep_value
        )
    else:
        timesteps = timestep_value

    if action_mode:
        grid_id = build_reference_mesh_id(
            num_frames,
            height,
            width,
            t=1,
            f_w=1,
            f_shift=frame_st_id,
            action=True,
            device=device,
        )[None].repeat(batch_size, 1, 1)
    else:
        grid_id = build_reference_mesh_id(
            num_frames // backbone_config.patch_size_t,
            height // backbone_config.patch_size_h,
            width // backbone_config.patch_size_w,
            t=0,
            f_w=1,
            f_shift=frame_st_id,
            action=False,
            device=device,
        )[None].repeat(batch_size, 1, 1)
    input_dict = {
        "noisy_latents": latents.clone(),
        "timesteps": timesteps[None].repeat(batch_size, 1),
        "grid_id": grid_id,
        "text_emb": text_emb,
    }
    if cond is not None:
        condition_frames = int(cond.shape[2])
        input_dict["noisy_latents"][:, :, :condition_frames] = cond[
            :, :, :condition_frames
        ]
        input_dict["timesteps"][:, :condition_frames] *= 0
    if action_mode and action_channel_mask is not None:
        input_dict["noisy_latents"] = input_dict[
            "noisy_latents"
        ] * action_channel_mask.to(
            device=input_dict["noisy_latents"].device,
            dtype=input_dict["noisy_latents"].dtype,
        )
    return input_dict


def repeat_exact_single_stream_input_for_cfg(
    input_dict: dict[str, torch.Tensor],
    *,
    negative_text_emb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Duplicate one exact single-stream payload for classifier-free guidance."""

    repeated = {
        "noisy_latents": input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1),
        "text_emb": torch.cat([input_dict["text_emb"], negative_text_emb], dim=0),
        "grid_id": input_dict["grid_id"].repeat(2, 1, 1),
        "timesteps": input_dict["timesteps"].repeat(2, 1),
    }
    attention_mask = input_dict.get("attention_mask")
    if attention_mask is not None:
        if (
            attention_mask.ndim in {3, 4}
            and attention_mask.shape[0] == input_dict["noisy_latents"].shape[0]
        ):
            repeat_shape = (2,) + (1,) * (attention_mask.ndim - 1)
            attention_mask = attention_mask.repeat(*repeat_shape)
        repeated["attention_mask"] = attention_mask
    hidden_context = input_dict.get("hidden_context")
    if hidden_context is not None:
        repeated["hidden_context"] = hidden_context.repeat(2, 1, 1)
    return repeated


def prepare_exact_single_stream_forward_input(
    input_dict: dict[str, torch.Tensor],
    *,
    transformer: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Move exact single-stream floating tensors to the runtime dtype."""

    model_dtype = resolve_runtime_module_dtype(transformer)
    prepared = {
        "noisy_latents": input_dict["noisy_latents"].to(model_dtype),
        "text_emb": input_dict["text_emb"].to(model_dtype),
        "grid_id": input_dict["grid_id"],
        "timesteps": input_dict["timesteps"],
    }
    attention_mask = input_dict.get("attention_mask")
    if attention_mask is not None:
        prepared["attention_mask"] = attention_mask
    cross_attention_mask = input_dict.get("cross_attention_mask")
    if cross_attention_mask is not None:
        prepared["cross_attention_mask"] = cross_attention_mask
    hidden_context = input_dict.get("hidden_context")
    if hidden_context is not None:
        prepared["hidden_context"] = hidden_context.to(model_dtype)
    return prepared


def run_exact_single_stream_forward(
    transformer: torch.nn.Module,
    *,
    input_dict: dict[str, torch.Tensor],
    update_cache: int,
    cache_name: str,
    action_mode: bool,
    guidance_scale: float,
    negative_text_emb: torch.Tensor | None,
    combine_cfg: bool = True,
    force_cfg_batch: bool = False,
) -> torch.Tensor:
    """Execute one exact single-stream video or action forward."""

    batch_size = input_dict["noisy_latents"].shape[0]
    effective_input = input_dict
    use_cfg = negative_text_emb is not None and (
        force_cfg_batch or guidance_scale > 1.0
    )
    if use_cfg:
        effective_input = repeat_exact_single_stream_input_for_cfg(
            input_dict,
            negative_text_emb=negative_text_emb,
        )
    effective_input = prepare_exact_single_stream_forward_input(
        effective_input,
        transformer=transformer,
    )
    with torch.inference_mode():
        step_output = transformer.execute_runtime_step(
            RuntimeStepInput(
                program=build_single_stream_exact_runtime_program(),
                payload=effective_input,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            )
        )
        output = step_output.tokens
    if output is None:
        raise ValueError(
            "Exact single-stream runtime execution did not return token predictions."
        )
    if use_cfg and combine_cfg:
        cond_output = output[:batch_size]
        uncond_output = output[batch_size:]
        return uncond_output + guidance_scale * (cond_output - uncond_output)
    return output


def initialize_exact_runtime_cache(
    transformer: torch.nn.Module,
    *,
    cache_name: str,
    attn_window: int,
    batch_size: int,
    frame_chunk_size: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    action_per_frame: int,
    use_cfg: bool,
    cache_backend_name: str = "slot_pool_exact",
    cache_batch_size_override: int | None = None,
    token_batch_factor: int = 1,
    prefix_visibility_mode: str = "full_history",
) -> None:
    """Allocate the shared exact-runtime cache backend."""

    effective_batch_size = batch_size * (2 if use_cfg else 1)
    latent_token_per_chunk = (
        frame_chunk_size * latent_height * latent_width
    ) // math.prod(transformer.patch_size)
    latent_token_per_chunk *= max(1, int(token_batch_factor))
    action_token_per_chunk = (
        frame_chunk_size
        * action_per_frame
        * max(
            1,
            int(token_batch_factor),
        )
    )
    cache_batch_size = (
        int(cache_batch_size_override)
        if cache_batch_size_override is not None
        else effective_batch_size
    )
    transformer.clear_runtime_cache_state(cache_name)
    transformer.initialize_runtime_cache_backend(
        cache_name,
        attn_window=attn_window,
        latent_token_per_chunk=latent_token_per_chunk,
        action_token_per_chunk=action_token_per_chunk,
        device=device,
        dtype=resolve_runtime_module_dtype(transformer),
        batch_size=cache_batch_size,
        backend_name=cache_backend_name,
        prefix_visibility_mode=prefix_visibility_mode,
    )


def clear_exact_prediction_cache(
    transformer: torch.nn.Module,
    *,
    cache_name: str,
) -> None:
    """Clear transient prediction K/V while retaining stable exact history."""

    transformer.clear_runtime_prediction_cache(cache_name)
