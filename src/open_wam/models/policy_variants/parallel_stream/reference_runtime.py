from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.configs.inference import InferenceConfig
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.visual_tower.reference_transformer import preferred_reference_dtype


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
        exponential_shift: bool = False,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = None,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: float | None = None,
    ) -> None:
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.exponential_shift_mu if self.exponential_shift_mu is not None else 0.0
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())
            self.training = True
        else:
            self.training = False

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor, t_dim: int = 2) -> torch.Tensor:
        timestep = timestep.cpu()
        timestep = timestep[None]
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep).abs(), dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = timestep_id.shape[0]
        sigma = self.sigmas[timestep_id].to(original_samples).view(shape)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep_id = torch.argmin((self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0)
        return self.linear_timesteps_weights.to(timestep.device)[timestep_id].to(timestep.device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        to_final: bool = False,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_next = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_next - sigma)


def sample_timestep_id(
    batch_size: int,
    *,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
    device: torch.device | None = None,
) -> torch.Tensor:
    u = torch.rand(size=[batch_size], device=device)
    u = u * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    return (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)


def get_mesh_id(
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
    f_idx = torch.arange(f_shift, f + f_shift, device=device) * f_w
    h_idx = torch.arange(h, device=device)
    w_idx = torch.arange(w, device=device)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h], device=device).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1
    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    return torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)


def data_seq_to_patch(
    patch_size: tuple[int, int, int],
    data_seq: torch.Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    batch_size: int,
) -> torch.Tensor:
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w
    data_patch = data_seq.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)




@dataclass
class LingbotParallelTrainArtifacts:
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    latent_scheduler: FlowMatchScheduler
    action_scheduler: FlowMatchScheduler


@dataclass
class LingbotParallelInferArtifacts:
    action_pred: torch.Tensor
    predicted_latents: torch.Tensor
    next_cache: dict[str, Any]
    debug: dict[str, Any]


def _add_noise(
    latent: torch.Tensor,
    *,
    train_scheduler: FlowMatchScheduler,
    action_mask: torch.Tensor | None,
    action_mode: bool,
    noisy_cond_prob: float,
    patch_size: tuple[int, int, int],
) -> dict[str, torch.Tensor]:
    batch_size, _, num_frames, height, width = latent.shape
    # LingBot samples one timestep per frame, then broadcasts that scalar across
    # every channel/spatial location inside that frame. For video latents the
    # tensor is `[B, C_latent, F, H_latent, W_latent]`; for action latents it is
    # `[B, D_action, F, action_per_frame, 1]`.
    timestep_ids = sample_timestep_id(
        batch_size=num_frames,
        num_train_timesteps=train_scheduler.num_train_timesteps,
        device=latent.device,
    )
    noise = torch.zeros_like(latent).normal_()
    timesteps = train_scheduler.timesteps[timestep_ids].to(device=latent.device)
    noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
    targets = train_scheduler.training_target(latent, noise, timesteps)

    patch_f, patch_h, patch_w = patch_size
    if action_mode:
        patch_f = patch_h = patch_w = 1

    # Grid ids stay flattened to match the reference transformer input after
    # patchification:
    # - video: `[B, 4, T_video]` where `T_video = F/p_t * H/p_h * W/p_w`
    # - action: `[B, 4, T_action]` where `T_action = F * action_per_frame`
    latent_grid_id = get_mesh_id(
        latent.shape[-3] // patch_f,
        latent.shape[-2] // patch_h,
        latent.shape[-1] // patch_w,
        t=1 if action_mode else 0,
        f_w=1,
        f_shift=0,
        action=action_mode,
        device=latent.device,
    )[None].repeat(batch_size, 1, 1)

    if noisy_cond_prob > 0.0 and torch.rand(1, device=latent.device).item() < noisy_cond_prob:
        cond_timestep_ids = sample_timestep_id(
            batch_size=num_frames,
            min_timestep_bd=0.5,
            max_timestep_bd=1.0,
            num_train_timesteps=train_scheduler.num_train_timesteps,
            device=latent.device,
        )
        cond_noise = torch.zeros_like(latent).normal_()
        cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=latent.device)
        latent = train_scheduler.add_noise(latent, cond_noise, cond_timesteps, t_dim=2)
    else:
        cond_timesteps = torch.zeros_like(timesteps)

    if action_mask is not None:
        noisy_latents = noisy_latents * action_mask.float()
        targets = targets * action_mask.float()
        latent = latent * action_mask.float()

    return {
        "timesteps": timesteps[None].repeat(batch_size, 1),
        "noisy_latents": noisy_latents,
        "targets": targets,
        "latent": latent,
        "cond_timesteps": cond_timesteps[None].repeat(batch_size, 1),
        "grid_id": latent_grid_id,
    }


def prepare_parallel_exact_train_artifacts(
    *,
    backbone_config: LingbotCompatibleVideoBackboneConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    text_emb: torch.Tensor | None,
) -> LingbotParallelTrainArtifacts:
    batch_size, _, num_frames, _, _ = video_latents.shape
    # Exact parallel-stream training keeps video and action in the same frame
    # count. Actions are reshaped from `[B, F * A, D]` into
    # `[B, D, F, A, 1]` so the reference transformer can treat them like a
    # narrow latent volume with one "width" slot per action token.
    action_latents = rearrange(
        actions,
        "b (f a) c -> b c f a 1",
        f=num_frames,
        a=policy_config.action_per_frame,
    )
    action_mask_latents = None
    if action_mask is not None:
        action_mask_latents = rearrange(
            action_mask,
            "b (f a) c -> b c f a 1",
            f=num_frames,
            a=policy_config.action_per_frame,
        )

    latent_scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    latent_scheduler.set_timesteps(training_config.video_num_train_timesteps, training=True)
    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    action_scheduler.set_timesteps(training_config.action_num_train_timesteps, training=True)

    latent_dict = _add_noise(
        video_latents,
        train_scheduler=latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=policy_config.noisy_video_condition_prob,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
    )
    action_dict = _add_noise(
        action_latents,
        train_scheduler=action_scheduler,
        action_mask=action_mask_latents,
        action_mode=True,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
    )

    model_dtype = preferred_reference_dtype(video_latents.device)
    if text_emb is None:
        text_emb = torch.zeros(
            batch_size,
            backbone_config.max_text_tokens,
            backbone_config.text_dim,
            device=video_latents.device,
            dtype=model_dtype,
        )
    else:
        text_emb = text_emb.to(device=video_latents.device, dtype=model_dtype)

    latent_dict["text_emb"] = text_emb
    action_dict["text_emb"] = text_emb
    action_dict["actions_mask"] = (
        action_mask_latents
        if action_mask_latents is not None
        else torch.ones_like(action_latents, device=video_latents.device)
    )
    # LingBot varies the effective chunk and window during training. Those
    # values are carried through as metadata because later layout/mask builders
    # need them to reproduce the same local-attention regime.
    chunk_size = max(1, int(training_config.chunk_size))
    sampled_chunk_size = int(torch.randint(1, chunk_size + 1, (1,), device=video_latents.device).item())
    if training_config.window_size >= 4:
        sampled_window_size = int(
            torch.randint(4, int(training_config.window_size) + 1, (1,), device=video_latents.device).item()
        )
    else:
        sampled_window_size = max(1, int(training_config.window_size))
    return LingbotParallelTrainArtifacts(
        input_dict={
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": sampled_chunk_size,
            "window_size": sampled_window_size,
        },
        latent_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
    )


def ensure_reference_text_embeddings(
    text_emb: torch.Tensor | None,
    *,
    batch_size: int,
    backbone_config: LingbotCompatibleVideoBackboneConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if text_emb is None:
        return torch.zeros(
            batch_size,
            backbone_config.max_text_tokens,
            backbone_config.text_dim,
            device=device,
            dtype=dtype,
        )
    return text_emb.to(device=device, dtype=dtype)


def prepare_reference_single_stream_input(
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor | float,
    text_emb: torch.Tensor,
    frame_st_id: int,
    backbone_config: LingbotCompatibleVideoBackboneConfig,
    action_mode: bool,
    cond: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    batch_size, _, num_frames, height, width = latents.shape
    device = latents.device
    if isinstance(timestep, torch.Tensor):
        timestep_value = float(timestep.item()) if timestep.ndim == 0 else timestep.to(device=device, dtype=torch.float32)
    else:
        timestep_value = float(timestep)
    if isinstance(timestep_value, float):
        timesteps = torch.ones(num_frames, device=device, dtype=torch.float32) * timestep_value
    else:
        timesteps = timestep_value
    # This helper produces the exact single-stream dict the LingBot reference
    # transformer expects. Before patch embedding:
    # - video stream latents: `[B, C_latent, F, H_latent, W_latent]`
    # - action stream latents: `[B, D_action, F, action_per_frame, 1]`
    # The paired `grid_id` encodes where every future token belongs in frame
    # time and whether it came from the video or action stream.
    if action_mode:
        grid_id = get_mesh_id(
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
        grid_id = get_mesh_id(
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
        input_dict["noisy_latents"][:, :, 0:1] = cond[:, :, 0:1]
        input_dict["timesteps"][:, 0:1] *= 0
    return input_dict


def repeat_input_for_cfg(
    input_dict: dict[str, torch.Tensor],
    *,
    negative_text_emb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "noisy_latents": input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1),
        "text_emb": torch.cat([input_dict["text_emb"], negative_text_emb], dim=0),
        "grid_id": input_dict["grid_id"].repeat(2, 1, 1),
        "timesteps": input_dict["timesteps"].repeat(2, 1),
    }


def prepare_reference_forward_input(
    input_dict: dict[str, torch.Tensor],
    *,
    transformer: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    model_dtype = next(transformer.parameters()).dtype
    return {
        "noisy_latents": input_dict["noisy_latents"].to(model_dtype),
        "text_emb": input_dict["text_emb"].to(model_dtype),
        "grid_id": input_dict["grid_id"],
        "timesteps": input_dict["timesteps"],
    }


def run_reference_single_stream_forward(
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
    batch_size = input_dict["noisy_latents"].shape[0]
    effective_input = input_dict
    use_cfg = negative_text_emb is not None and (force_cfg_batch or guidance_scale > 1.0)
    if use_cfg:
        effective_input = repeat_input_for_cfg(input_dict, negative_text_emb=negative_text_emb)
    effective_input = prepare_reference_forward_input(effective_input, transformer=transformer)
    output = transformer(
        effective_input,
        update_cache=update_cache,
        cache_name=cache_name,
        action_mode=action_mode,
    )
    if use_cfg and combine_cfg:
        cond_output = output[:batch_size]
        uncond_output = output[batch_size:]
        return uncond_output + guidance_scale * (cond_output - uncond_output)
    return output


def initialize_reference_cache(
    transformer: torch.nn.Module,
    *,
    cache_name: str,
    attn_window: int,
    video_latents: torch.Tensor,
    action_per_frame: int,
    use_cfg: bool,
) -> None:
    batch_size = video_latents.shape[0] * (2 if use_cfg else 1)
    latent_token_per_chunk = (
        video_latents.shape[2] * video_latents.shape[3] * video_latents.shape[4]
    ) // math.prod(transformer.patch_size)
    action_token_per_chunk = video_latents.shape[2] * action_per_frame
    transformer.clear_cache(cache_name)
    transformer.create_empty_cache(
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        device=video_latents.device,
        dtype=next(transformer.parameters()).dtype,
        batch_size=batch_size,
    )


def run_parallel_exact_cache_warmup(
    *,
    transformer: torch.nn.Module,
    backbone_config: LingbotCompatibleVideoBackboneConfig,
    policy_config: ParallelStreamPolicyConfig,
    inference_config: InferenceConfig,
    observed_video_latents: torch.Tensor,
    observed_action_latents: torch.Tensor,
    text_emb: torch.Tensor | None,
    infer_cache: dict[str, Any],
) -> dict[str, Any]:
    device = observed_video_latents.device
    model_dtype = next(transformer.parameters()).dtype
    batch_size, _, observed_frames, latent_height, latent_width = observed_video_latents.shape
    text_emb = ensure_reference_text_embeddings(
        text_emb,
        batch_size=batch_size,
        backbone_config=backbone_config,
        device=device,
        dtype=model_dtype,
    )
    negative_text_emb = torch.zeros_like(text_emb)
    use_cfg = bool(
        infer_cache.get(
            "use_cfg",
            inference_config.guidance_scale > 1.0 or inference_config.action_guidance_scale > 1.0,
        )
    )
    cache_name = str(infer_cache.get("cache_name", "open_wam_exact"))
    current_frame_start = int(infer_cache.get("frame_start", 0))
    cache_initialized = bool(infer_cache.get("cache_initialized", False))
    cached_batch_size = int(infer_cache.get("batch_size", batch_size))
    cached_latent_height = int(infer_cache.get("latent_height", latent_height))
    cached_latent_width = int(infer_cache.get("latent_width", latent_width))

    if inference_config.use_cache and (
        not cache_initialized
        or cached_batch_size != batch_size
        or cached_latent_height != latent_height
        or cached_latent_width != latent_width
    ):
        initialize_reference_cache(
            transformer,
            cache_name=cache_name,
            attn_window=policy_config.attn_window,
            video_latents=observed_video_latents,
            action_per_frame=policy_config.action_per_frame,
            use_cfg=use_cfg,
        )
        cache_initialized = True
        current_frame_start = 0

    if inference_config.use_cache:
        transformer.clear_pred_cache(cache_name)

    # Warmup pushes already-observed history into the transformer cache without
    # denoising it. Both streams therefore use timestep `0.0`, and the
    # resulting KV cache represents the observed prefix before generation
    # starts at `frame_start_after`.
    cache_video_input = prepare_reference_single_stream_input(
        latents=observed_video_latents.to(dtype=model_dtype),
        timestep=0.0,
        text_emb=text_emb,
        frame_st_id=current_frame_start,
        backbone_config=backbone_config,
        action_mode=False,
    )
    cache_action_input = prepare_reference_single_stream_input(
        latents=observed_action_latents.to(device=device, dtype=model_dtype),
        timestep=0.0,
        text_emb=text_emb,
        frame_st_id=current_frame_start,
        backbone_config=backbone_config,
        action_mode=True,
    )
    run_reference_single_stream_forward(
        transformer,
        input_dict=cache_video_input,
        update_cache=2 if inference_config.use_cache else 0,
        cache_name=cache_name,
        action_mode=False,
        guidance_scale=inference_config.guidance_scale,
        negative_text_emb=negative_text_emb,
        combine_cfg=False,
        force_cfg_batch=use_cfg and inference_config.use_cache,
    )
    run_reference_single_stream_forward(
        transformer,
        input_dict=cache_action_input,
        update_cache=2 if inference_config.use_cache else 0,
        cache_name=cache_name,
        action_mode=True,
        guidance_scale=inference_config.action_guidance_scale,
        negative_text_emb=negative_text_emb,
        combine_cfg=False,
        force_cfg_batch=use_cfg and inference_config.use_cache,
    )
    debug = {
        "cache_name": cache_name,
        "use_cfg": use_cfg,
        "batch_size": batch_size,
        "observed_frames": observed_frames,
        "frame_start_before": int(infer_cache.get("frame_start", 0)),
        "frame_start_after": current_frame_start + observed_frames,
    }
    return {
        "runtime_mode": "lingbot_exact",
        "cache_name": cache_name,
        "cache_initialized": cache_initialized and inference_config.use_cache,
        "frame_start": current_frame_start + observed_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "batch_size": batch_size,
        "step_index": int(infer_cache.get("step_index", 0)),
        "use_cfg": use_cfg,
        "debug_last_warmup": debug,
    }


def run_parallel_exact_inference_rollout(
    *,
    transformer: torch.nn.Module,
    backbone_config: LingbotCompatibleVideoBackboneConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    action_dim: int,
    condition_latents: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    infer_cache: dict[str, Any],
    advance_frame_start: bool = False,
) -> LingbotParallelInferArtifacts:
    if condition_latents is not None:
        device = condition_latents.device
        batch_size = condition_latents.shape[0]
        latent_height = condition_latents.shape[-2]
        latent_width = condition_latents.shape[-1]
    else:
        if "batch_size" not in infer_cache or "latent_height" not in infer_cache or "latent_width" not in infer_cache:
            raise ValueError(
                "Exact LingBot inference without condition latents requires cached batch/latent shape metadata."
            )
        device = next(transformer.parameters()).device
        batch_size = int(infer_cache["batch_size"])
        latent_height = int(infer_cache["latent_height"])
        latent_width = int(infer_cache["latent_width"])
    model_dtype = next(transformer.parameters()).dtype
    text_emb = ensure_reference_text_embeddings(
        text_emb,
        batch_size=batch_size,
        backbone_config=backbone_config,
        device=device,
        dtype=model_dtype,
    )
    negative_text_emb = torch.zeros_like(text_emb)
    use_cfg = bool(
        infer_cache.get(
            "use_cfg",
            inference_config.guidance_scale > 1.0 or inference_config.action_guidance_scale > 1.0,
        )
    )
    cache_name = str(infer_cache.get("cache_name", "open_wam_exact"))
    current_frame_start = int(infer_cache.get("frame_start", 0))
    cache_initialized = bool(infer_cache.get("cache_initialized", False))
    if inference_config.use_cache and not cache_initialized:
        if condition_latents is None:
            raise ValueError("Exact LingBot inference requires condition latents on the first chunk when cache is empty.")
        initialize_reference_cache(
            transformer,
            cache_name=cache_name,
            attn_window=policy_config.attn_window,
            video_latents=condition_latents,
            action_per_frame=policy_config.action_per_frame,
            use_cfg=use_cfg,
        )
        cache_initialized = True
    generation_frame_start = current_frame_start
    latent_cond = None
    if infer_cache.get("step_index", 0) == 0 and condition_latents is not None and current_frame_start == 0:
        latent_cond = condition_latents[:, :, 0:1].to(dtype=model_dtype)

    latents = torch.randn(
        batch_size,
        backbone_config.latent_channels,
        inference_config.frame_chunk_size,
        latent_height,
        latent_width,
        device=device,
        dtype=model_dtype,
    )
    # One generated chunk always has aligned video/action frame count:
    # - `latents`: `[B, C_latent, F_chunk, H_latent, W_latent]`
    # - `actions`: `[B, D_action, F_chunk, action_per_frame, 1]`
    # Both streams share `F_chunk = inference_config.frame_chunk_size`.
    actions = torch.randn(
        batch_size,
        action_dim,
        inference_config.frame_chunk_size,
        policy_config.action_per_frame,
        1,
        device=device,
        dtype=model_dtype,
    )

    video_scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    video_scheduler.set_timesteps(inference_config.video_num_inference_steps)
    action_scheduler.set_timesteps(inference_config.action_num_inference_steps)
    video_timesteps = F.pad(video_scheduler.timesteps.to(device=device), (0, 1), mode="constant", value=0)
    if inference_config.video_exec_step != -1:
        video_timesteps = video_timesteps[: inference_config.video_exec_step]
    action_timesteps = F.pad(action_scheduler.timesteps.to(device=device), (0, 1), mode="constant", value=0)

    # Video denoising always runs before action denoising so the action stream
    # can condition on the final visual chunk, matching the LingBot server
    # rollout order.
    for index, timestep in enumerate(video_timesteps):
        last_step = index == len(video_timesteps) - 1
        video_input = prepare_reference_single_stream_input(
            latents=latents,
            timestep=timestep,
            text_emb=text_emb,
            frame_st_id=generation_frame_start,
            backbone_config=backbone_config,
            action_mode=False,
            cond=latent_cond,
        )
        video_noise_pred = run_reference_single_stream_forward(
            transformer,
            input_dict=video_input,
            update_cache=1 if (last_step and inference_config.use_cache) else 0,
            cache_name=cache_name,
            action_mode=False,
            guidance_scale=inference_config.guidance_scale,
            negative_text_emb=negative_text_emb,
            force_cfg_batch=use_cfg and inference_config.use_cache,
        )
        if not last_step or inference_config.video_exec_step != -1:
            video_noise_pred = data_seq_to_patch(
                transformer.patch_size,
                video_noise_pred,
                inference_config.frame_chunk_size,
                latent_height,
                latent_width,
                batch_size=batch_size,
            )
            latents = video_scheduler.step(video_noise_pred, timestep, latents)
        if latent_cond is not None:
            latents[:, :, 0:1] = latent_cond

    action_cond = None
    if infer_cache.get("step_index", 0) == 0:
        action_cond = torch.zeros(
            batch_size,
            actions.shape[1],
            1,
            policy_config.action_per_frame,
            1,
            device=device,
            dtype=model_dtype,
        )
    # Actions are denoised in their native `[B, D_action, F_chunk, A, 1]`
    # volume and converted back to `[B, F_chunk * A, D_action]` only once the
    # chunk is complete.
    for index, timestep in enumerate(action_timesteps):
        last_step = index == len(action_timesteps) - 1
        action_input = prepare_reference_single_stream_input(
            latents=actions,
            timestep=timestep,
            text_emb=text_emb,
            frame_st_id=generation_frame_start,
            backbone_config=backbone_config,
            action_mode=True,
            cond=action_cond,
        )
        action_noise_pred = run_reference_single_stream_forward(
            transformer,
            input_dict=action_input,
            update_cache=1 if (last_step and inference_config.use_cache) else 0,
            cache_name=cache_name,
            action_mode=True,
            guidance_scale=inference_config.action_guidance_scale,
            negative_text_emb=negative_text_emb,
            force_cfg_batch=use_cfg and inference_config.use_cache,
        )
        if not last_step:
            action_noise_pred = rearrange(
                action_noise_pred,
                "b (f n) c -> b c f n 1",
                f=inference_config.frame_chunk_size,
            )
            actions = action_scheduler.step(action_noise_pred, timestep, actions)
        if action_cond is not None:
            actions[:, :, 0:1] = action_cond

    next_cache = {
        "runtime_mode": "lingbot_exact",
        "cache_name": cache_name,
        "cache_initialized": cache_initialized and inference_config.use_cache,
        "frame_start": int(
            current_frame_start + inference_config.frame_chunk_size if advance_frame_start else current_frame_start
        ),
        "latent_height": latent_height,
        "latent_width": latent_width,
        "batch_size": batch_size,
        "step_index": int(infer_cache.get("step_index", 0) + 1),
        "use_cfg": use_cfg,
    }
    debug = {
        "cache_name": cache_name,
        "use_cfg": use_cfg,
        "generation_frame_start": generation_frame_start,
        "advance_frame_start": advance_frame_start,
        "video_timesteps": video_timesteps.tolist(),
        "action_timesteps": action_timesteps.tolist(),
        "video_guidance_scale": float(inference_config.guidance_scale),
        "action_guidance_scale": float(inference_config.action_guidance_scale),
    }
    output_dtype = condition_latents.dtype if condition_latents is not None else model_dtype
    action_pred = rearrange(actions, "b c f n 1 -> b (f n) c").to(dtype=output_dtype)
    return LingbotParallelInferArtifacts(
        action_pred=action_pred,
        predicted_latents=latents.to(dtype=output_dtype),
        next_cache=next_cache,
        debug=debug,
    )


def run_parallel_exact_train(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_dict = input_dict["latent_dict"]
    action_dict = input_dict["action_dict"]
    model_dtype = next(transformer.parameters()).dtype
    latent_dict["noisy_latents"] = latent_dict["noisy_latents"].to(model_dtype)
    latent_dict["latent"] = latent_dict["latent"].to(model_dtype)
    action_dict["noisy_latents"] = action_dict["noisy_latents"].to(model_dtype)
    action_dict["latent"] = action_dict["latent"].to(model_dtype)
    latent_dict["text_emb"] = latent_dict["text_emb"].to(model_dtype)
    action_dict["text_emb"] = action_dict["text_emb"].to(model_dtype)

    batch_size = latent_dict["noisy_latents"].shape[0]
    latent_hidden_states = transformer._input_embed(latent_dict["noisy_latents"], input_type="latent").flatten(0, 1)[None]
    action_hidden_states = transformer._input_embed(action_dict["noisy_latents"], input_type="action").flatten(0, 1)[None]
    text_hidden_states = transformer._input_embed(latent_dict["text_emb"], input_type="text").flatten(0, 1)[None]
    condition_latent_hidden_states = transformer._input_embed(latent_dict["latent"], input_type="latent").flatten(0, 1)[None]
    condition_action_hidden_states = transformer._input_embed(action_dict["latent"], input_type="action").flatten(0, 1)[None]

    hidden_states = torch.cat(
        [
            latent_hidden_states,
            condition_latent_hidden_states,
            action_hidden_states,
            condition_action_hidden_states,
        ],
        dim=1,
    )
    latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
    action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
    full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)
    rotary_emb = transformer.rope(full_grid_id)[:, :, None]

    latent_time_steps = torch.cat([latent_dict["timesteps"].flatten(0, 1), latent_dict["cond_timesteps"].flatten(0, 1)])[None]
    action_time_steps = torch.cat([action_dict["timesteps"].flatten(0, 1), action_dict["cond_timesteps"].flatten(0, 1)])[None]
    latent_temb, latent_timestep_proj = transformer._time_embed(
        latent_time_steps,
        latent_dict["noisy_latents"].shape[-2],
        latent_dict["noisy_latents"].shape[-1],
        dtype=hidden_states.dtype,
        action_mode=False,
    )
    action_temb, action_timestep_proj = transformer._time_embed(
        action_time_steps,
        action_dict["noisy_latents"].shape[-2],
        action_dict["noisy_latents"].shape[-1],
        dtype=hidden_states.dtype,
        action_mode=True,
    )
    temb = torch.cat([latent_temb, action_temb], dim=1)
    timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

    total_length = hidden_states.shape[1]
    padded_length = (128 - total_length % 128) % 128
    hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
    rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
    temb = F.pad(temb, (0, 0, 0, padded_length))
    timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

    split_list = [
        latent_hidden_states.shape[1],
        condition_latent_hidden_states.shape[1],
        action_hidden_states.shape[1],
        condition_action_hidden_states.shape[1],
        padded_length,
    ]

    attn_mode = getattr(transformer.blocks[0], "attn_mode", "torch") if transformer.blocks else "torch"
    if attn_mode == "flex":
        module = sys.modules[transformer.__class__.__module__]
        module.FlexAttnFunc.init_mask(
            latent_dict["noisy_latents"].shape,
            action_dict["noisy_latents"].shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict["window_size"],
            patch_size=transformer.patch_size,
            device=hidden_states.device,
        )

    for block in transformer.blocks:
        hidden_states = block(
            hidden_states,
            text_hidden_states,
            timestep_proj,
            rotary_emb,
            update_cache=False,
        )

    temb_scale_shift_table = transformer.scale_shift_table[None] + temb[:, :, None, ...]
    shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
    shift = shift.to(hidden_states.device).squeeze(1)
    scale = scale.to(hidden_states.device).squeeze(1)
    hidden_states = (transformer.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
    latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
    latent_hidden_states = transformer.proj_out(latent_hidden_states)
    latent_hidden_states = rearrange(
        latent_hidden_states,
        "1 (b l) (n c) -> b (l n) c",
        n=math.prod(transformer.patch_size),
        b=batch_size,
    )
    action_hidden_states = transformer.action_proj_out(action_hidden_states)
    action_hidden_states = rearrange(
        action_hidden_states,
        "1 (b l) c -> b l c",
        b=batch_size,
    )
    return latent_hidden_states, action_hidden_states
