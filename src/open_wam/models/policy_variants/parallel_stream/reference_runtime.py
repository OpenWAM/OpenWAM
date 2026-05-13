from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.configs.enums import (
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    ParallelExactCacheWriteMode,
    ParallelRuntimeMode,
    ParallelStreamVariantProfile,
)
from open_wam.configs.variant_semantics import GENERALIST_TRAINING_SOURCE_METADATA_KEY
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common import (
    PreparedAttentionProfile,
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    build_chunked_temporal_exact_attention_profile,
    cache_backend_uses_slot_pool,
    chunked_temporal_exact_profile_name_for_coupling,
    materialize_cache_backend_entries,
)
from open_wam.models.common.flow_matching import FlowMatchScheduler
from open_wam.models.common.flow_noise_plan import (
    clean_timestep_values,
    sample_coupled_timestep_values as sample_shared_coupled_timestep_values,
    sample_timestep_values as sample_shared_timestep_values,
)
from open_wam.models.common.joint_conditioning import (
    sample_conditioning_mode,
    should_drop_text_for_conditioning_mode,
)
from open_wam.models.common.modality_slots import force_clean_noisy_slot, zero_condition_slot
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig, resolve_stage_attention_mode
from open_wam.models.video_backbone.contracts import CacheState
from open_wam.models.visual_tower import (
    RuntimeStepInput,
    build_chunked_dual_stream_exact_inference_program,
    build_chunked_dual_stream_exact_train_program,
    build_single_stream_exact_runtime_program,
)
from open_wam.models.visual_tower.sequence_adapters import prepare_exact_dual_stream_train_sequence
from open_wam.models.visual_tower.reference_transformer import preferred_reference_dtype


def reference_runtime_dtype(transformer: torch.nn.Module) -> torch.dtype:
    for parameter in transformer.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    try:
        device = next(transformer.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return preferred_reference_dtype(device)


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


@dataclass(frozen=True)
class ExactCacheInterfaceSpec:
    """Unified exact-runtime cache interface independent of rollout style."""

    write_mode: ParallelExactCacheWriteMode
    cache_batch_size_override: int | None = None
    token_batch_factor: int = 1
    prefix_visibility_mode: str = "full_history"


@dataclass(frozen=True)
class ExactCacheContext:
    """Resolved cache metadata shared across exact-runtime rollout paths."""

    cache_name: str
    cache_backend_name: str
    cache_initialized: bool
    batch_size: int
    latent_height: int
    latent_width: int
    use_cfg: bool
    device: torch.device
    model_dtype: torch.dtype


def _prefix_visibility_mode_for_policy(policy_config: ParallelStreamPolicyConfig) -> str:
    return (
        "preserve_video_pretrain_history"
        if bool(getattr(policy_config, "preserve_video_pretrain_history", False))
        else "full_history"
    )


def _stream_ids_for_exact_dual_stream_split(
    split_list: list[int] | tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(int(split_list[0]), device=device, dtype=torch.long),
            torch.zeros(int(split_list[1]), device=device, dtype=torch.long),
            torch.ones(int(split_list[2]), device=device, dtype=torch.long),
            torch.ones(int(split_list[3]), device=device, dtype=torch.long),
            torch.full((int(split_list[4]),), -1, device=device, dtype=torch.long),
        ],
        dim=0,
    )


def _stream_ids_for_clean_video_action_tokens(
    *,
    video_token_count: int,
    action_token_count: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(int(video_token_count), device=device, dtype=torch.long),
            torch.ones(int(action_token_count), device=device, dtype=torch.long),
        ],
        dim=0,
    )


def _single_stream_action_token_count(actions: torch.Tensor) -> int:
    if actions.ndim != 5:
        raise ValueError(f"Expected action latents shaped [B, C, F, A, W], got {tuple(actions.shape)}.")
    return int(actions.shape[2]) * int(actions.shape[3]) * int(actions.shape[4])


def _set_slot_pool_layer_metadata(
    transformer: torch.nn.Module,
    *,
    cache_name: str,
    updates: dict[str, Any],
) -> list[tuple[Any, dict[str, tuple[bool, Any]]]]:
    if not updates or not hasattr(transformer, "_resolve_exact_cache_state"):
        return []
    cache_state = transformer._resolve_exact_cache_state(cache_name)
    if cache_state is None or not cache_backend_uses_slot_pool(cache_state.backend_name):
        return []
    cache_payload = cache_state.backend_payload
    layer_states = getattr(cache_payload, "layer_states", None)
    if layer_states is None:
        return []
    previous: list[tuple[Any, dict[str, tuple[bool, Any]]]] = []
    for layer_state in layer_states:
        layer_previous: dict[str, tuple[bool, Any]] = {}
        for key, value in updates.items():
            layer_previous[key] = (key in layer_state.metadata, layer_state.metadata.get(key))
            layer_state.metadata[key] = value
        previous.append((layer_state, layer_previous))
    return previous


def _restore_slot_pool_layer_metadata(
    previous: list[tuple[Any, dict[str, tuple[bool, Any]]]],
) -> None:
    for layer_state, layer_previous in previous:
        for key, (was_present, value) in layer_previous.items():
            if was_present:
                layer_state.metadata[key] = value
            else:
                layer_state.metadata.pop(key, None)


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


def resolve_parallel_current_block_coupling(
    policy_config: ParallelStreamPolicyConfig,
) -> CurrentBlockCoupling:
    """Resolve legacy M1 runtime knobs into an explicit current-block mode."""

    if policy_config.current_block_coupling is not None:
        return CurrentBlockCoupling(policy_config.current_block_coupling)
    if policy_config.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
        return CurrentBlockCoupling.JOINT
    return CurrentBlockCoupling.VIDEO_THEN_ACTION


def should_couple_action_to_video_timesteps(
    policy_config: ParallelStreamPolicyConfig,
) -> bool:
    """Return whether this exact-runtime program should share video/action sigmas."""

    if not bool(policy_config.couple_action_to_video_timesteps):
        return False
    return resolve_parallel_current_block_coupling(policy_config) in {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }


def _attention_profile_name_for_current_block_coupling(
    coupling: CurrentBlockCoupling,
) -> str:
    return chunked_temporal_exact_profile_name_for_coupling(coupling.value)


def _add_noise(
    latent: torch.Tensor,
    *,
    train_scheduler: FlowMatchScheduler,
    action_mask: torch.Tensor | None,
    action_mode: bool,
    noisy_cond_prob: float,
    patch_size: tuple[int, int, int],
    frame_shift: int = 0,
    timestep_values: torch.Tensor | None = None,
    sigma_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    batch_size, _, num_frames, height, width = latent.shape
    # LingBot samples one timestep per frame, then broadcasts that scalar across
    # every channel/spatial location inside that frame. For video latents the
    # tensor is `[B, C_latent, F, H_latent, W_latent]`; for action latents it is
    # `[B, D_action, F, action_per_frame, 1]`.
    noise = torch.zeros_like(latent).normal_()
    scheduler_timesteps = train_scheduler.timesteps.to(device=latent.device)
    if timestep_values is None:
        timestep_ids = sample_timestep_id(
            batch_size=num_frames,
            num_train_timesteps=train_scheduler.num_train_timesteps,
            device=latent.device,
        )
        timesteps = scheduler_timesteps[timestep_ids]
    else:
        timesteps = timestep_values.to(device=latent.device, dtype=scheduler_timesteps.dtype)
        if timesteps.ndim != 1 or timesteps.shape[0] != num_frames:
            raise ValueError(
                "Explicit denoise timestep values must be one scalar per frame, "
                f"got shape={tuple(timesteps.shape)} and num_frames={num_frames}."
            )
    if sigma_values is None:
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
    else:
        sigmas = sigma_values.to(device=latent.device, dtype=latent.dtype)
        if sigmas.ndim != 1 or sigmas.shape[0] != num_frames:
            raise ValueError(
                "Explicit denoise sigma values must be one scalar per frame, "
                f"got shape={tuple(sigmas.shape)} and num_frames={num_frames}."
            )
        shape = [1] * noise.ndim
        shape[2] = num_frames
        sigmas = sigmas.view(shape)
        noisy_latents = (1 - sigmas) * latent + sigmas * noise
    targets = train_scheduler.training_target(latent, noise, timesteps)

    patch_f, patch_h, patch_w = patch_size
    if action_mode:
        patch_f = patch_h = patch_w = 1

    # Grid ids stay flattened to match the shared exact-runtime backbone input
    # after patchification:
    # - video: `[B, 4, T_video]` where `T_video = F/p_t * H/p_h * W/p_w`
    # - action: `[B, 4, T_action]` where `T_action = F * action_per_frame`
    latent_grid_id = get_mesh_id(
        latent.shape[-3] // patch_f,
        latent.shape[-2] // patch_h,
        latent.shape[-1] // patch_w,
        t=1 if action_mode else 0,
        f_w=1,
        f_shift=frame_shift,
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
        cond_timesteps = scheduler_timesteps[cond_timestep_ids]
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


def _sample_joint_denoise_training_mode(
    policy_config: ParallelStreamPolicyConfig,
    *,
    device: torch.device,
) -> JointDenoiseTrainingMode:
    probs = policy_config.joint_denoise_training_mode_probs
    if probs is None:
        return JointDenoiseTrainingMode.JOINT
    return sample_conditioning_mode(
        probs,
        enum_cls=JointDenoiseTrainingMode,
        device=device,
        error_label="Generalist joint-denoise training mode",
    )


def _sample_timestep_values(
    scheduler: FlowMatchScheduler,
    *,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    return sample_shared_timestep_values(
        scheduler,
        num_frames=num_frames,
        device=device,
    )


def _sample_coupled_timestep_values(
    *,
    latent_scheduler: FlowMatchScheduler,
    action_scheduler: FlowMatchScheduler,
    num_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = sample_shared_coupled_timestep_values(
        video_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
        num_frames=num_frames,
        device=device,
    )
    return values.video_timesteps, values.action_timesteps, values.sigma_values


def _apply_generalist_joint_denoise_training_mode(
    *,
    artifacts: LingbotParallelTrainArtifacts,
    policy_config: ParallelStreamPolicyConfig,
    backbone_config: SharedVideoTransformerConfig,
    video_latents: torch.Tensor,
    action_latents: torch.Tensor,
    action_mask_latents: torch.Tensor | None,
    frame_shift: int,
    training_mode_override: JointDenoiseTrainingMode | str | None = None,
    drop_text_conditioning: bool | None = None,
    training_source: str | None = None,
) -> None:
    if int(video_latents.shape[0]) != 1:
        raise ValueError(
            "`generalist_joint_denoising` currently samples one conditioning mode per runtime batch. "
            "Use train_batch_size=1 to preserve the intended one-mode-per-segment contract."
        )
    mode = (
        JointDenoiseTrainingMode(training_mode_override)
        if training_mode_override is not None
        else _sample_joint_denoise_training_mode(policy_config, device=video_latents.device)
    )
    num_frames = int(video_latents.shape[2])
    clean_zero_timesteps = clean_timestep_values(num_frames=num_frames, device=video_latents.device)
    shared_sigma_values: torch.Tensor | None = None
    if mode == JointDenoiseTrainingMode.JOINT and policy_config.couple_action_to_video_timesteps:
        latent_timestep_values, action_timestep_values, shared_sigma_values = _sample_coupled_timestep_values(
            latent_scheduler=artifacts.latent_scheduler,
            action_scheduler=artifacts.action_scheduler,
            num_frames=num_frames,
            device=video_latents.device,
        )
    else:
        latent_timestep_values = (
            clean_zero_timesteps
            if mode == JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION
            else _sample_timestep_values(
                artifacts.latent_scheduler,
                num_frames=num_frames,
                device=video_latents.device,
            )
        )
        action_timestep_values = (
            clean_zero_timesteps
            if mode == JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
            else _sample_timestep_values(
                artifacts.action_scheduler,
                num_frames=num_frames,
                device=video_latents.device,
            )
        )

    latent_dict = _add_noise(
        video_latents,
        train_scheduler=artifacts.latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
        timestep_values=latent_timestep_values,
        sigma_values=shared_sigma_values,
    )
    action_dict = _add_noise(
        action_latents,
        train_scheduler=artifacts.action_scheduler,
        action_mask=action_mask_latents,
        action_mode=True,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
        timestep_values=action_timestep_values,
        sigma_values=shared_sigma_values,
    )
    zero_condition_slot(latent_dict)
    zero_condition_slot(action_dict)

    if mode == JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO:
        force_clean_noisy_slot(action_dict, action_latents, action_mask=action_mask_latents)
        action_dict["loss_mask"] = torch.zeros_like(artifacts.input_dict["action_dict"]["loss_mask"])
    else:
        action_dict["loss_mask"] = artifacts.input_dict["action_dict"]["loss_mask"]

    if mode == JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION:
        force_clean_noisy_slot(latent_dict, video_latents)
        latent_dict["loss_mask"] = torch.zeros_like(artifacts.input_dict["latent_dict"]["loss_mask"])
    else:
        latent_dict["loss_mask"] = artifacts.input_dict["latent_dict"]["loss_mask"]

    text_emb = artifacts.input_dict["latent_dict"]["text_emb"]
    text_dropped = should_drop_text_for_conditioning_mode(
        mode,
        joint_mode=JointDenoiseTrainingMode.JOINT,
        drop_text_conditioning=drop_text_conditioning,
    )
    if text_dropped:
        text_emb = torch.zeros_like(text_emb)
    latent_dict["text_emb"] = text_emb
    action_dict["text_emb"] = text_emb
    action_dict["actions_mask"] = artifacts.input_dict["action_dict"]["actions_mask"]
    if mode == JointDenoiseTrainingMode.JOINT:
        latent_dict["loss_mask"] = artifacts.input_dict["latent_dict"]["loss_mask"]
        action_dict["loss_mask"] = artifacts.input_dict["action_dict"]["loss_mask"]

    artifacts.input_dict["latent_dict"] = latent_dict
    artifacts.input_dict["action_dict"] = action_dict
    artifacts.input_dict["variant_profile"] = policy_config.variant_profile.value
    artifacts.input_dict["generalist_training_paradigm"] = policy_config.generalist_training_paradigm.value
    artifacts.input_dict[GENERALIST_TRAINING_SOURCE_METADATA_KEY] = training_source
    artifacts.input_dict["joint_denoise_training_mode"] = mode.value
    artifacts.input_dict["joint_denoise_training_mode_override"] = (
        None if training_mode_override is None else mode.value
    )
    artifacts.input_dict["joint_denoise_text_dropped"] = bool(text_dropped)
    artifacts.input_dict["joint_denoise_training_mode_probs"] = {
        mode_key.value: float(prob)
        for mode_key, prob in (policy_config.joint_denoise_training_mode_probs or {}).items()
    }
    if shared_sigma_values is not None:
        artifacts.input_dict["joint_denoise_shared_sigmas"] = shared_sigma_values.detach().clone()


def prepare_parallel_exact_train_artifacts(
    *,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    chunk_size_override: int | None = None,
    window_size_override: int | None = None,
    loss_frame_start: int | None = None,
    loss_frame_end: int | None = None,
    latent_loss_frame_start: int | None = None,
    latent_loss_frame_end: int | None = None,
    action_loss_frame_start: int | None = None,
    action_loss_frame_end: int | None = None,
    frame_shift: int = 0,
    force_clean_video_condition: bool = False,
) -> LingbotParallelTrainArtifacts:
    batch_size, _, num_frames, _, _ = video_latents.shape
    train_attn_mode = resolve_stage_attention_mode(backbone_config, stage="train", exact_runtime=True)
    # Exact parallel-stream training keeps video and action in the same frame
    # count. Actions are reshaped from `[B, F * A, D]` into
    # `[B, D, F, A, 1]` so the shared exact-runtime backbone can treat them
    # like a narrow latent volume with one "width" slot per action token.
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

    coupled_action_video_timesteps = (
        should_couple_action_to_video_timesteps(policy_config)
        and policy_config.variant_profile != ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
    )
    shared_sigma_values: torch.Tensor | None = None
    latent_timestep_values: torch.Tensor | None = None
    action_timestep_values: torch.Tensor | None = None
    if coupled_action_video_timesteps:
        latent_timestep_values, action_timestep_values, shared_sigma_values = _sample_coupled_timestep_values(
            latent_scheduler=latent_scheduler,
            action_scheduler=action_scheduler,
            num_frames=num_frames,
            device=video_latents.device,
        )

    # FDM/IDM-style objectives need clean condition streams to be marked as
    # clean-from-start, not "almost denoised" targets. Keep the legacy joint
    # policy augmentation by default, but allow objective-specific callers to
    # force zero condition timesteps for the video condition copy.
    latent_dict = _add_noise(
        video_latents,
        train_scheduler=latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0 if force_clean_video_condition else policy_config.noisy_video_condition_prob,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
        timestep_values=latent_timestep_values,
        sigma_values=shared_sigma_values,
    )
    action_dict = _add_noise(
        action_latents,
        train_scheduler=action_scheduler,
        action_mask=action_mask_latents,
        action_mode=True,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
        timestep_values=action_timestep_values,
        sigma_values=shared_sigma_values,
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
    def _resolve_frame_range(
        *,
        start: int | None,
        end: int | None,
        default_start: int | None = None,
        default_end: int | None = None,
        label: str,
    ) -> tuple[int, int]:
        start_value = default_start if start is None else start
        end_value = default_end if end is None else end
        resolved_start = 0 if start_value is None else int(start_value)
        resolved_end = num_frames if end_value is None else int(end_value)
        if resolved_start < 0 or resolved_end < resolved_start or resolved_end > num_frames:
            raise ValueError(
                f"Invalid {label} frame range for parallel exact training, "
                f"got start={resolved_start}, end={resolved_end}, num_frames={num_frames}."
            )
        return resolved_start, resolved_end

    resolved_loss_frame_start, resolved_loss_frame_end = _resolve_frame_range(
        start=loss_frame_start,
        end=loss_frame_end,
        label="current-loss",
    )
    resolved_latent_loss_frame_start, resolved_latent_loss_frame_end = _resolve_frame_range(
        start=latent_loss_frame_start,
        end=latent_loss_frame_end,
        default_start=loss_frame_start,
        default_end=loss_frame_end,
        label="latent-loss",
    )
    resolved_action_loss_frame_start, resolved_action_loss_frame_end = _resolve_frame_range(
        start=action_loss_frame_start,
        end=action_loss_frame_end,
        default_start=loss_frame_start,
        default_end=loss_frame_end,
        label="action-loss",
    )
    latent_loss_mask = torch.zeros_like(video_latents, device=video_latents.device)
    latent_loss_mask[:, :, resolved_latent_loss_frame_start:resolved_latent_loss_frame_end] = 1.0
    action_loss_mask = torch.zeros_like(action_latents, device=video_latents.device)
    action_loss_mask[:, :, resolved_action_loss_frame_start:resolved_action_loss_frame_end] = 1.0
    latent_dict["loss_mask"] = latent_loss_mask
    action_dict["loss_mask"] = action_loss_mask

    # LingBot varies the effective chunk and window during training. Those
    # values are carried through as metadata because later layout/mask builders
    # need them to reproduce the same local-attention regime.
    if chunk_size_override is not None:
        sampled_chunk_size = max(1, int(chunk_size_override))
    else:
        chunk_size = max(1, int(training_config.chunk_size))
        sampled_chunk_size = int(torch.randint(1, chunk_size + 1, (1,), device=video_latents.device).item())
    if window_size_override is not None:
        sampled_window_size = max(1, int(window_size_override))
    elif training_config.window_size >= 4:
        sampled_window_size = int(
            torch.randint(4, int(training_config.window_size) + 1, (1,), device=video_latents.device).item()
        )
    else:
        sampled_window_size = max(1, int(training_config.window_size))
    attention_profile_name = None
    if train_attn_mode == "flex":
        attention_profile_name = _attention_profile_name_for_current_block_coupling(
            resolve_parallel_current_block_coupling(policy_config)
        )

    return LingbotParallelTrainArtifacts(
        input_dict={
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": sampled_chunk_size,
            "window_size": sampled_window_size,
            "loss_frame_start": resolved_loss_frame_start,
            "loss_frame_end": resolved_loss_frame_end,
            "latent_loss_frame_start": resolved_latent_loss_frame_start,
            "latent_loss_frame_end": resolved_latent_loss_frame_end,
            "action_loss_frame_start": resolved_action_loss_frame_start,
            "action_loss_frame_end": resolved_action_loss_frame_end,
            "frame_shift": int(frame_shift),
            "attention_profile_name": attention_profile_name,
            "preserve_video_pretrain_history": bool(
                getattr(policy_config, "preserve_video_pretrain_history", False)
            ),
            "force_clean_video_condition": bool(force_clean_video_condition),
            "coupled_action_video_timesteps": bool(coupled_action_video_timesteps),
        },
        latent_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
    )


def prepare_parallel_action_conditioned_train_artifacts(
    *,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    chunk_size_override: int | None = None,
    window_size_override: int | None = None,
    loss_frame_start: int | None = None,
    loss_frame_end: int | None = None,
    latent_loss_frame_start: int | None = None,
    latent_loss_frame_end: int | None = None,
    action_loss_frame_start: int | None = None,
    action_loss_frame_end: int | None = None,
    frame_shift: int = 0,
    force_clean_video_condition: bool = False,
    generalist_training_mode_override: JointDenoiseTrainingMode | str | None = None,
    generalist_drop_text_conditioning: bool | None = None,
    generalist_training_source: str | None = None,
) -> LingbotParallelTrainArtifacts:
    coupling = resolve_parallel_current_block_coupling(policy_config)
    if (
        coupling
        in {
            CurrentBlockCoupling.JOINT,
            CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
            CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
        }
        and policy_config.current_block_coupling is None
        and not policy_config.video_condition_on_action
    ):
        raise ValueError(
            "`lingbot_exact_action_conditioned` requires `video_condition_on_action = true`."
        )
    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        text_emb=text_emb,
        chunk_size_override=chunk_size_override,
        window_size_override=window_size_override,
        loss_frame_start=loss_frame_start,
        loss_frame_end=loss_frame_end,
        latent_loss_frame_start=latent_loss_frame_start,
        latent_loss_frame_end=latent_loss_frame_end,
        action_loss_frame_start=action_loss_frame_start,
        action_loss_frame_end=action_loss_frame_end,
        frame_shift=frame_shift,
        force_clean_video_condition=force_clean_video_condition,
    )
    if policy_config.variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING:
        _, _, num_frames, _, _ = video_latents.shape
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
        _apply_generalist_joint_denoise_training_mode(
            artifacts=artifacts,
            policy_config=policy_config,
            backbone_config=backbone_config,
            video_latents=video_latents,
            action_latents=action_latents,
            action_mask_latents=action_mask_latents,
            frame_shift=frame_shift,
            training_mode_override=generalist_training_mode_override,
            drop_text_conditioning=generalist_drop_text_conditioning,
            training_source=generalist_training_source,
        )
    return artifacts


def ensure_reference_text_embeddings(
    text_emb: torch.Tensor | None,
    *,
    batch_size: int,
    backbone_config: SharedVideoTransformerConfig,
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


def _resolve_exact_cache_context(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    inference_config: InferenceConfig,
    infer_cache: dict[str, Any],
    batch_size: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
) -> tuple[ExactCacheContext, torch.Tensor, torch.Tensor | None]:
    model_dtype = reference_runtime_dtype(transformer)
    resolved_text_emb = ensure_reference_text_embeddings(
        text_emb,
        batch_size=batch_size,
        backbone_config=backbone_config,
        device=device,
        dtype=model_dtype,
    )
    use_cfg = bool(
        infer_cache.get(
            "use_cfg",
            inference_config.guidance_scale > 1.0 or inference_config.action_guidance_scale > 1.0,
        )
    )
    if negative_text_emb is not None:
        resolved_negative_text_emb = ensure_reference_text_embeddings(
            negative_text_emb,
            batch_size=batch_size,
            backbone_config=backbone_config,
            device=device,
            dtype=model_dtype,
        )
    elif use_cfg:
        resolved_negative_text_emb = torch.zeros_like(resolved_text_emb)
    else:
        resolved_negative_text_emb = None
    return (
        ExactCacheContext(
            cache_name=str(infer_cache.get("cache_name", "open_wam_exact")),
            cache_backend_name=str(infer_cache.get("cache_backend_name", "slot_pool_exact")),
            cache_initialized=bool(infer_cache.get("cache_initialized", False)),
            batch_size=batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
            use_cfg=use_cfg,
            device=device,
            model_dtype=model_dtype,
        ),
        resolved_text_emb,
        resolved_negative_text_emb,
    )


def _build_exact_cache_spec(
    *,
    write_mode: ParallelExactCacheWriteMode | str,
    batch_size: int,
    use_cfg: bool,
    prefix_visibility_mode: str = "full_history",
) -> ExactCacheInterfaceSpec:
    write_mode = ParallelExactCacheWriteMode(write_mode)
    if write_mode == ParallelExactCacheWriteMode.JOINT_PACKED:
        # The shared exact runtime keeps batch as the cache batch dimension.
        # Overriding cache batch to 1 and folding batch into token count no
        # longer matches the runtime-step execution path after the shared-core
        # refactors, and it causes slot-pool cache writes to receive `[B, T]`
        # tensors for a `[1, ...]` cache allocation.
        return ExactCacheInterfaceSpec(
            write_mode=write_mode,
            prefix_visibility_mode=prefix_visibility_mode,
        )
    return ExactCacheInterfaceSpec(
        write_mode=write_mode,
        prefix_visibility_mode=prefix_visibility_mode,
    )


def _ensure_exact_cache_initialized(
    *,
    transformer: torch.nn.Module,
    policy_config: ParallelStreamPolicyConfig,
    inference_config: InferenceConfig,
    cache_context: ExactCacheContext,
    cache_spec: ExactCacheInterfaceSpec,
) -> ExactCacheContext:
    if not inference_config.use_cache or cache_context.cache_initialized:
        return cache_context
    initialize_reference_cache(
        transformer,
        cache_name=cache_context.cache_name,
        attn_window=policy_config.attn_window,
        batch_size=cache_context.batch_size,
        frame_chunk_size=inference_config.frame_chunk_size,
        latent_height=cache_context.latent_height,
        latent_width=cache_context.latent_width,
        device=cache_context.device,
        action_per_frame=policy_config.action_per_frame,
        use_cfg=cache_context.use_cfg,
        cache_backend_name=cache_context.cache_backend_name,
        cache_batch_size_override=cache_spec.cache_batch_size_override,
        token_batch_factor=cache_spec.token_batch_factor,
        prefix_visibility_mode=cache_spec.prefix_visibility_mode,
    )
    return ExactCacheContext(
        cache_name=cache_context.cache_name,
        cache_backend_name=cache_context.cache_backend_name,
        cache_initialized=True,
        batch_size=cache_context.batch_size,
        latent_height=cache_context.latent_height,
        latent_width=cache_context.latent_width,
        use_cfg=cache_context.use_cfg,
        device=cache_context.device,
        model_dtype=cache_context.model_dtype,
    )


def _clear_exact_prediction_cache(transformer: torch.nn.Module, *, cache_name: str) -> None:
    if hasattr(transformer, "clear_runtime_prediction_cache"):
        transformer.clear_runtime_prediction_cache(cache_name)
    else:
        transformer.clear_pred_cache(cache_name)


def _build_next_exact_cache_state(
    *,
    runtime_mode: str,
    cache_context: ExactCacheContext,
    infer_cache: dict[str, Any],
    frame_start: int,
    advance_frame_start: bool,
    frame_chunk_size: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    next_cache = {
        "runtime_mode": runtime_mode,
        "cache_name": cache_context.cache_name,
        "cache_backend_name": cache_context.cache_backend_name,
        "cache_initialized": cache_context.cache_initialized,
        "frame_start": int(frame_start + frame_chunk_size if advance_frame_start else frame_start),
        "latent_height": cache_context.latent_height,
        "latent_width": cache_context.latent_width,
        "batch_size": cache_context.batch_size,
        "step_index": int(infer_cache.get("step_index", 0) + 1),
        "use_cfg": cache_context.use_cfg,
    }
    if extra:
        next_cache.update(extra)
    return next_cache


def prepare_reference_single_stream_input(
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
    if action_mode and action_channel_mask is not None:
        input_dict["noisy_latents"] = input_dict["noisy_latents"] * action_channel_mask.to(
            device=input_dict["noisy_latents"].device,
            dtype=input_dict["noisy_latents"].dtype,
        )
    return input_dict


def repeat_input_for_cfg(
    input_dict: dict[str, torch.Tensor],
    *,
    negative_text_emb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    repeated = {
        "noisy_latents": input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1),
        "text_emb": torch.cat([input_dict["text_emb"], negative_text_emb], dim=0),
        "grid_id": input_dict["grid_id"].repeat(2, 1, 1),
        "timesteps": input_dict["timesteps"].repeat(2, 1),
    }
    attention_mask = input_dict.get("attention_mask")
    if attention_mask is not None:
        if attention_mask.ndim in {3, 4} and attention_mask.shape[0] == input_dict["noisy_latents"].shape[0]:
            repeat_shape = (2,) + (1,) * (attention_mask.ndim - 1)
            attention_mask = attention_mask.repeat(*repeat_shape)
        repeated["attention_mask"] = attention_mask
    return repeated


def _repeat_joint_input_for_cfg(
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    *,
    negative_text_emb: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    latent_dict = dict(input_dict["latent_dict"])  # type: ignore[index]
    action_dict = dict(input_dict["action_dict"])  # type: ignore[index]
    repeated_latent_dict = {
        **latent_dict,
        "noisy_latents": latent_dict["noisy_latents"].repeat(2, 1, 1, 1, 1),
        "latent": latent_dict["latent"].repeat(2, 1, 1, 1, 1),
        "grid_id": latent_dict["grid_id"].repeat(2, 1, 1),
        "timesteps": latent_dict["timesteps"].repeat(2, 1),
        "cond_timesteps": latent_dict["cond_timesteps"].repeat(2, 1),
        "text_emb": torch.cat([latent_dict["text_emb"], negative_text_emb], dim=0),
    }
    repeated_action_dict = {
        **action_dict,
        "noisy_latents": action_dict["noisy_latents"].repeat(2, 1, 1, 1, 1),
        "latent": action_dict["latent"].repeat(2, 1, 1, 1, 1),
        "grid_id": action_dict["grid_id"].repeat(2, 1, 1),
        "timesteps": action_dict["timesteps"].repeat(2, 1),
        "cond_timesteps": action_dict["cond_timesteps"].repeat(2, 1),
        "text_emb": torch.cat([action_dict["text_emb"], negative_text_emb], dim=0),
    }
    if "actions_mask" in action_dict:
        repeated_action_dict["actions_mask"] = action_dict["actions_mask"].repeat(2, 1, 1, 1, 1)
    if "loss_mask" in latent_dict:
        repeated_latent_dict["loss_mask"] = latent_dict["loss_mask"].repeat(2, 1, 1, 1, 1)
    if "loss_mask" in action_dict:
        repeated_action_dict["loss_mask"] = action_dict["loss_mask"].repeat(2, 1, 1, 1, 1)
    return {
        **input_dict,
        "latent_dict": repeated_latent_dict,
        "action_dict": repeated_action_dict,
    }


def prepare_reference_forward_input(
    input_dict: dict[str, torch.Tensor],
    *,
    transformer: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    model_dtype = reference_runtime_dtype(transformer)
    prepared = {
        "noisy_latents": input_dict["noisy_latents"].to(model_dtype),
        "text_emb": input_dict["text_emb"].to(model_dtype),
        "grid_id": input_dict["grid_id"],
        "timesteps": input_dict["timesteps"],
    }
    attention_mask = input_dict.get("attention_mask")
    if attention_mask is not None:
        prepared["attention_mask"] = attention_mask
    return prepared


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
    with torch.inference_mode():
        if hasattr(transformer, "execute_runtime_step"):
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
        else:
            output = transformer(
                effective_input,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            )
    if output is None:
        raise ValueError("Exact single-stream runtime execution did not return token predictions.")
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
    effective_batch_size = batch_size * (2 if use_cfg else 1)
    latent_token_per_chunk = (
        frame_chunk_size * latent_height * latent_width
    ) // math.prod(transformer.patch_size)
    latent_token_per_chunk *= max(1, int(token_batch_factor))
    action_token_per_chunk = frame_chunk_size * action_per_frame * max(1, int(token_batch_factor))
    cache_batch_size = (
        int(cache_batch_size_override)
        if cache_batch_size_override is not None
        else effective_batch_size
    )
    if hasattr(transformer, "clear_runtime_cache_state"):
        transformer.clear_runtime_cache_state(cache_name)
    else:
        transformer.clear_cache(cache_name)
    if hasattr(transformer, "initialize_runtime_cache_backend"):
        transformer.initialize_runtime_cache_backend(
            cache_name,
            attn_window=attn_window,
            latent_token_per_chunk=latent_token_per_chunk,
            action_token_per_chunk=action_token_per_chunk,
            device=device,
            dtype=reference_runtime_dtype(transformer),
            batch_size=cache_batch_size,
            backend_name=cache_backend_name,
            prefix_visibility_mode=prefix_visibility_mode,
        )
    else:
        transformer.create_empty_cache(
            cache_name,
            attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=device,
            dtype=reference_runtime_dtype(transformer),
            batch_size=cache_batch_size,
            backend_name=cache_backend_name,
            prefix_visibility_mode=prefix_visibility_mode,
        )


def run_parallel_exact_cache_warmup(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    inference_config: InferenceConfig,
    observed_video_latents: torch.Tensor,
    observed_action_latents: torch.Tensor,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
    action_channel_mask: torch.Tensor | None,
    infer_cache: dict[str, Any],
    cache_write_mode: ParallelExactCacheWriteMode | str = ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED,
    frame_start_override: int | None = None,
) -> dict[str, Any]:
    device = observed_video_latents.device
    batch_size, _, observed_frames, latent_height, latent_width = observed_video_latents.shape
    cache_context, text_emb, negative_text_emb = _resolve_exact_cache_context(
        transformer=transformer,
        backbone_config=backbone_config,
        inference_config=inference_config,
        infer_cache=infer_cache,
        batch_size=batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        device=device,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
    )
    cache_spec = _build_exact_cache_spec(
        write_mode=cache_write_mode,
        batch_size=batch_size,
        use_cfg=cache_context.use_cfg,
        prefix_visibility_mode=_prefix_visibility_mode_for_policy(policy_config),
    )
    current_frame_start = (
        int(infer_cache.get("frame_start", 0))
        if frame_start_override is None
        else int(frame_start_override)
    )
    cached_batch_size = int(infer_cache.get("batch_size", batch_size))
    cached_latent_height = int(infer_cache.get("latent_height", latent_height))
    cached_latent_width = int(infer_cache.get("latent_width", latent_width))

    if inference_config.use_cache and (
        not cache_context.cache_initialized
        or cached_batch_size != batch_size
        or cached_latent_height != latent_height
        or cached_latent_width != latent_width
    ):
        cache_context = _ensure_exact_cache_initialized(
            transformer=transformer,
            policy_config=policy_config,
            inference_config=inference_config,
            cache_context=cache_context,
            cache_spec=cache_spec,
        )
        if frame_start_override is None:
            current_frame_start = 0

    if inference_config.use_cache:
        _clear_exact_prediction_cache(transformer, cache_name=cache_context.cache_name)

    # Warmup pushes already-observed history into the transformer cache without
    # denoising it. Both streams therefore use timestep `0.0`, and the
    # resulting KV cache represents the observed prefix before generation
    # starts at `frame_start_after`.
    _write_exact_cache_chunk(
        transformer=transformer,
        cache_spec=cache_spec,
        cache_name=cache_context.cache_name,
        frame_start=current_frame_start,
        backbone_config=backbone_config,
        video_latents=observed_video_latents.to(dtype=cache_context.model_dtype),
        action_latents=observed_action_latents.to(device=device, dtype=cache_context.model_dtype),
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        use_cfg=cache_context.use_cfg and inference_config.use_cache,
        action_channel_mask=action_channel_mask,
        update_cache=2 if inference_config.use_cache else 0,
        chunk_size=inference_config.frame_chunk_size,
        window_size=policy_config.attn_window,
        current_block_coupling=resolve_parallel_current_block_coupling(policy_config),
        preserve_video_pretrain_history=bool(
            getattr(policy_config, "preserve_video_pretrain_history", False)
        ),
    )
    debug = {
        "cache_name": cache_context.cache_name,
        "cache_backend_name": cache_context.cache_backend_name,
        "use_cfg": cache_context.use_cfg,
        "batch_size": batch_size,
        "observed_frames": observed_frames,
        "frame_start_before": int(infer_cache.get("frame_start", 0)),
        "frame_start_override": None if frame_start_override is None else int(frame_start_override),
        "frame_start_after": current_frame_start + observed_frames,
        "cache_write_mode": str(cache_spec.write_mode),
    }
    return {
        "runtime_mode": "lingbot_exact",
        "cache_name": cache_context.cache_name,
        "cache_backend_name": cache_context.cache_backend_name,
        "cache_initialized": cache_context.cache_initialized and inference_config.use_cache,
        "frame_start": current_frame_start + observed_frames,
        "latent_height": cache_context.latent_height,
        "latent_width": cache_context.latent_width,
        "batch_size": cache_context.batch_size,
        "step_index": int(infer_cache.get("step_index", 0)),
        "use_cfg": cache_context.use_cfg,
        "debug_last_warmup": debug,
    }


def run_parallel_exact_inference_rollout(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    action_dim: int,
    condition_latents: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
    action_channel_mask: torch.Tensor | None,
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
    cache_context, text_emb, negative_text_emb = _resolve_exact_cache_context(
        transformer=transformer,
        backbone_config=backbone_config,
        inference_config=inference_config,
        infer_cache=infer_cache,
        batch_size=batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        device=device,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
    )
    model_dtype = cache_context.model_dtype
    cache_name = cache_context.cache_name
    cache_backend_name = cache_context.cache_backend_name
    current_frame_start = int(infer_cache.get("frame_start", 0))
    current_block_coupling = resolve_parallel_current_block_coupling(policy_config)
    joint_packed_couplings = {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }
    if current_block_coupling in joint_packed_couplings:
        raise ValueError(
            "Joint-like M1 coupling must use `run_parallel_action_conditioned_inference_rollout`; "
            "the staged exact rollout only supports ordered or decoupled same-step coupling."
        )
    cache_spec = _build_exact_cache_spec(
        write_mode=ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED,
        batch_size=batch_size,
        use_cfg=cache_context.use_cfg,
        prefix_visibility_mode=_prefix_visibility_mode_for_policy(policy_config),
    )
    if inference_config.use_cache and not cache_context.cache_initialized:
        if condition_latents is None:
            raise ValueError("Exact LingBot inference requires condition latents on the first chunk when cache is empty.")
        cache_context = _ensure_exact_cache_initialized(
            transformer=transformer,
            policy_config=policy_config,
            inference_config=inference_config,
            cache_context=cache_context,
            cache_spec=cache_spec,
        )
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

    action_cond = None
    if generation_frame_start == 0:
        action_cond = torch.zeros(
            batch_size,
            actions.shape[1],
            1,
            policy_config.action_per_frame,
            1,
            device=device,
            dtype=model_dtype,
        )

    def denoise_video_chunk(*, commit_to_cache: bool) -> None:
        nonlocal latents
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
                update_cache=1 if (last_step and commit_to_cache and inference_config.use_cache) else 0,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=inference_config.guidance_scale,
                negative_text_emb=negative_text_emb,
                force_cfg_batch=cache_context.use_cfg and inference_config.use_cache,
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

    def denoise_action_chunk(*, commit_to_cache: bool) -> None:
        nonlocal actions
        # Actions are denoised in their native `[B, D_action, F_chunk, A, 1]`
        # volume and converted back to `[B, F_chunk * A, D_action]` once the
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
                action_channel_mask=action_channel_mask,
            )
            action_noise_pred = run_reference_single_stream_forward(
                transformer,
                input_dict=action_input,
                update_cache=1 if (last_step and commit_to_cache and inference_config.use_cache) else 0,
                cache_name=cache_name,
                action_mode=True,
                guidance_scale=inference_config.action_guidance_scale,
                negative_text_emb=negative_text_emb,
                force_cfg_batch=cache_context.use_cfg and inference_config.use_cache,
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

    if current_block_coupling == CurrentBlockCoupling.VIDEO_THEN_ACTION:
        cache_commit_strategy = "video_then_action_staged"
        denoise_video_chunk(commit_to_cache=True)
        denoise_action_chunk(commit_to_cache=True)
    elif current_block_coupling == CurrentBlockCoupling.ACTION_THEN_VIDEO:
        cache_commit_strategy = "action_then_video_staged"
        denoise_action_chunk(commit_to_cache=True)
        metadata_previous = _set_slot_pool_layer_metadata(
            transformer,
            cache_name=cache_name,
            updates={
                SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS: _single_stream_action_token_count(actions),
            },
        )
        try:
            denoise_video_chunk(commit_to_cache=True)
        finally:
            _restore_slot_pool_layer_metadata(metadata_previous)
    elif current_block_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
        cache_commit_strategy = "decoupled_same_step_deferred"
        denoise_video_chunk(commit_to_cache=False)
        denoise_action_chunk(commit_to_cache=False)
        if inference_config.use_cache:
            _write_exact_cache_chunk(
                transformer=transformer,
                cache_spec=cache_spec,
                cache_name=cache_name,
                frame_start=generation_frame_start,
                backbone_config=backbone_config,
                video_latents=latents,
                action_latents=actions,
                text_emb=text_emb,
                negative_text_emb=negative_text_emb,
                use_cfg=cache_context.use_cfg,
                action_channel_mask=action_channel_mask,
                update_cache=1,
                chunk_size=inference_config.frame_chunk_size,
                window_size=policy_config.attn_window,
                current_block_coupling=current_block_coupling,
                preserve_video_pretrain_history=bool(
                    getattr(policy_config, "preserve_video_pretrain_history", False)
                ),
            )
    else:  # pragma: no cover - enum guard
        raise ValueError(f"Unsupported M1 current-block coupling: {current_block_coupling!r}")

    next_cache = {
        "runtime_mode": "lingbot_exact",
        "cache_name": cache_name,
        "cache_backend_name": cache_backend_name,
        "cache_initialized": cache_context.cache_initialized and inference_config.use_cache,
        "frame_start": int(
            current_frame_start + inference_config.frame_chunk_size if advance_frame_start else current_frame_start
        ),
        "latent_height": latent_height,
        "latent_width": latent_width,
        "batch_size": batch_size,
        "step_index": int(infer_cache.get("step_index", 0) + 1),
        "use_cfg": cache_context.use_cfg,
    }
    debug = {
        "cache_name": cache_name,
        "cache_backend_name": cache_backend_name,
        "use_cfg": cache_context.use_cfg,
        "generation_frame_start": generation_frame_start,
        "advance_frame_start": advance_frame_start,
        "video_timesteps": video_timesteps.tolist(),
        "action_timesteps": action_timesteps.tolist(),
        "current_block_coupling": current_block_coupling.value,
        "cache_commit_strategy": cache_commit_strategy,
        "video_guidance_scale": float(inference_config.guidance_scale),
        "action_guidance_scale": float(inference_config.action_guidance_scale),
        "cache_write_mode": str(cache_spec.write_mode),
    }
    output_dtype = condition_latents.dtype if condition_latents is not None else model_dtype
    action_pred = rearrange(actions, "b c f n 1 -> b (f n) c").to(dtype=output_dtype)
    return LingbotParallelInferArtifacts(
        action_pred=action_pred,
        predicted_latents=latents.to(dtype=output_dtype),
        next_cache=next_cache,
        debug=debug,
    )


def _run_parallel_exact_joint_forward_manual(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    *,
    update_cache: int = 0,
    cache_name: str = "open_wam_exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = prepare_exact_dual_stream_train_sequence(
        input_dict,
        config=transformer.config,
        patch_size=transformer.patch_size,
        model_dtype=reference_runtime_dtype(transformer),
        input_embed=lambda tensor, input_type: transformer._input_embed(tensor, input_type=input_type),
        exact_text_hidden_states=lambda text_emb: transformer._exact_text_hidden_states(
            text_emb,
            dtype=reference_runtime_dtype(transformer),
        ),
        time_embed=lambda timesteps, height, width, dtype, action_mode: transformer._time_embed(
            timesteps,
            height,
            width,
            dtype=dtype,
            action_mode=action_mode,
        ),
        rope=transformer.rope,
    )
    batch_size = prepared.batch_size
    hidden_states = prepared.hidden_states
    text_hidden_states = prepared.text_hidden_states
    rotary_emb = prepared.rotary_emb
    temb = prepared.temb
    timestep_proj = prepared.timestep_proj
    split_list = prepared.split_list
    exact_attention_profile = prepared.attention_profile
    cache_stream_ids = _stream_ids_for_exact_dual_stream_split(
        split_list,
        device=hidden_states.device,
    )
    cache_state = transformer._resolve_exact_cache_state(cache_name)
    cache_backend_name = cache_state.backend_name if cache_state is not None else None
    cache_backend_payload = cache_state.backend_payload if cache_state is not None else None
    if cache_backend_uses_slot_pool(cache_backend_name):
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        assert isinstance(latent_dict, dict)
        assert isinstance(action_dict, dict)
        attention_profile_name = input_dict.get("attention_profile_name")
        rebuilt_dense_profile = build_chunked_temporal_exact_attention_profile(
            latent_shape=tuple(int(dim) for dim in latent_dict["noisy_latents"].shape),
            action_shape=tuple(int(dim) for dim in action_dict["noisy_latents"].shape),
            padded_length=int(hidden_states.shape[1] - sum(int(length) for length in split_list[:4])),
            chunk_size=int(input_dict["chunk_size"]),
            window_size=int(input_dict["window_size"]),
            patch_size=transformer.patch_size,
            text_token_count=int(latent_dict["text_emb"].shape[1]),
            device=hidden_states.device,
            build_dense_masks=True,
            build_flex_masks=False,
            current_block_coupling=(
                str(attention_profile_name)
                if attention_profile_name not in (None, "none")
                else None
            ),
            preserve_video_pretrain_history=bool(
                input_dict.get("preserve_video_pretrain_history", False)
            ),
        )
        exact_attention_profile = PreparedAttentionProfile(
            spec=rebuilt_dense_profile.spec,
            self_attention_mask=rebuilt_dense_profile.self_attention_mask,
            cross_attention_mask=rebuilt_dense_profile.cross_attention_mask,
            self_attention_block_mask=None,
            cross_attention_block_mask=None,
            metadata=dict(rebuilt_dense_profile.metadata),
        )

    for layer_index, block in enumerate(transformer.blocks):
        hidden_states, _, _ = block(
            hidden_states,
            encoder_hidden_states=text_hidden_states,
            temb=timestep_proj,
            rotary_emb=rotary_emb,
            attention_profile=exact_attention_profile,
            self_attention_cache_backend_name=cache_backend_name,
            self_attention_cache_backend_state=(
                cache_backend_payload.layer_states[layer_index]
                if cache_backend_uses_slot_pool(cache_backend_name)
                and cache_backend_payload is not None
                and layer_index < len(cache_backend_payload.layer_states)
                else None
            ),
            self_attention_cache_update_mode=update_cache,
            self_attention_cache_stream_ids=cache_stream_ids,
        )

    temb_scale_shift_table = transformer.scale_shift_table[None] + temb[:, :, None, ...]
    shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
    shift = shift.to(hidden_states.device).squeeze(1)
    scale = scale.to(hidden_states.device).squeeze(1)
    hidden_states = (transformer.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
    if cache_state is not None and cache_backend_uses_slot_pool(cache_backend_name):
        materialized_entries = materialize_cache_backend_entries(cache_backend_payload)
        transformer._exact_runtime_caches[cache_name] = CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=cache_state.cached_frames,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_backend_payload,
            payload=dict(cache_state.payload),
            self_attention_kv=materialized_entries,
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=cache_state.update_metadata,
        )
    latent_hidden_states, _, action_hidden_states, _, _ = torch.split(
        hidden_states,
        tuple(int(length) for length in split_list),
        dim=1,
    )
    effective_batch_size = int(input_dict["latent_dict"]["noisy_latents"].shape[0])  # type: ignore[index]
    latent_hidden_states = transformer.proj_out(latent_hidden_states)
    if latent_hidden_states.shape[0] == 1:
        latent_hidden_states = rearrange(
            latent_hidden_states,
            "1 (b l) c -> b l c",
            b=effective_batch_size,
        )
    elif latent_hidden_states.shape[0] == effective_batch_size:
        latent_hidden_states = latent_hidden_states.contiguous()
    else:
        raise ValueError(
            "Unexpected exact joint latent output layout: expected leading dimension to be 1 "
            f"or effective_batch_size={effective_batch_size}, got {latent_hidden_states.shape[0]}."
        )
    action_hidden_states = transformer.action_proj_out(action_hidden_states)
    if action_hidden_states.shape[0] == 1:
        action_hidden_states = rearrange(
            action_hidden_states,
            "1 (b l) c -> b l c",
            b=effective_batch_size,
        )
    elif action_hidden_states.shape[0] != effective_batch_size:
        raise ValueError(
            "Unexpected exact joint action output layout: expected leading dimension to be 1 "
            f"or effective_batch_size={effective_batch_size}, got {action_hidden_states.shape[0]}."
        )
    return latent_hidden_states, action_hidden_states


def _run_parallel_action_conditioned_forward(
    transformer: torch.nn.Module,
    *,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    video_guidance_scale: float,
    action_guidance_scale: float,
    negative_text_emb: torch.Tensor | None,
    update_cache: int = 0,
    cache_name: str = "open_wam_exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    def _split_cfg_prediction(
        prediction: torch.Tensor,
        *,
        logical_batch_size: int,
        expected_tokens: int,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction.ndim != 3:
            raise ValueError(f"Expected {name} prediction rank 3, got shape {tuple(prediction.shape)}.")
        if prediction.shape[0] == logical_batch_size * 2 and prediction.shape[1] == expected_tokens:
            return prediction[:logical_batch_size], prediction[logical_batch_size:]
        if prediction.shape[0] == logical_batch_size * 2 and prediction.shape[1] == logical_batch_size * 2 * expected_tokens:
            packed = rearrange(
                prediction,
                "(g b_row) (h b_seq l) c -> g b_row h b_seq l c",
                g=2,
                h=2,
                b_row=logical_batch_size,
                b_seq=logical_batch_size,
                l=expected_tokens,
            )
            batch_index = torch.arange(logical_batch_size, device=prediction.device)
            cond = packed[0, batch_index, 0, batch_index]
            uncond = packed[1, batch_index, 1, batch_index]
            return cond.contiguous(), uncond.contiguous()
        if prediction.shape[0] == logical_batch_size and prediction.shape[1] == expected_tokens * 2:
            return prediction[:, :expected_tokens], prediction[:, expected_tokens:]
        if prediction.shape[0] == 1 and prediction.shape[1] == logical_batch_size * expected_tokens * 2:
            unpacked = rearrange(
                prediction,
                "1 (g b l) c -> (g b) l c",
                g=2,
                b=logical_batch_size,
                l=expected_tokens,
            )
            return unpacked[:logical_batch_size], unpacked[logical_batch_size:]
        raise ValueError(
            f"Unable to split CFG {name} prediction with shape {tuple(prediction.shape)}; "
            f"expected logical_batch_size={logical_batch_size}, expected_tokens={expected_tokens}."
        )

    batch_size = input_dict["latent_dict"]["noisy_latents"].shape[0]  # type: ignore[index]
    latent_noisy = input_dict["latent_dict"]["noisy_latents"]  # type: ignore[index]
    action_noisy = input_dict["action_dict"]["noisy_latents"]  # type: ignore[index]
    expected_video_tokens = (
        int(latent_noisy.shape[2]) // transformer.patch_size[0]
    ) * (
        int(latent_noisy.shape[3]) // transformer.patch_size[1]
    ) * (
        int(latent_noisy.shape[4]) // transformer.patch_size[2]
    )
    expected_action_tokens = int(action_noisy.shape[2]) * int(action_noisy.shape[3])
    use_cfg = negative_text_emb is not None and (video_guidance_scale > 1.0 or action_guidance_scale > 1.0)
    effective_input = input_dict
    if use_cfg:
        effective_input = _repeat_joint_input_for_cfg(input_dict, negative_text_emb=negative_text_emb)
    with torch.inference_mode():
        video_pred, action_pred = _run_parallel_exact_joint_forward_manual(
            transformer,
            effective_input,
            update_cache=update_cache,
            cache_name=cache_name,
        )
    if not use_cfg:
        return video_pred, action_pred
    cond_video_pred, uncond_video_pred = _split_cfg_prediction(
        video_pred,
        logical_batch_size=batch_size,
        expected_tokens=expected_video_tokens,
        name="video",
    )
    cond_action_pred, uncond_action_pred = _split_cfg_prediction(
        action_pred,
        logical_batch_size=batch_size,
        expected_tokens=expected_action_tokens,
        name="action",
    )
    combined_video_pred = uncond_video_pred + video_guidance_scale * (cond_video_pred - uncond_video_pred)
    combined_action_pred = uncond_action_pred + action_guidance_scale * (cond_action_pred - uncond_action_pred)
    return combined_video_pred, combined_action_pred


def _expand_condition_video_latents(
    condition_latents: torch.Tensor,
    *,
    target_frames: int,
) -> torch.Tensor:
    if condition_latents.shape[2] >= target_frames:
        return condition_latents[:, :, -target_frames:]
    pad_frames = target_frames - condition_latents.shape[2]
    pad = condition_latents[:, :, -1:].repeat(1, 1, pad_frames, 1, 1)
    return torch.cat([condition_latents, pad], dim=2)


def _build_action_condition_volume(
    *,
    batch_size: int,
    action_dim: int,
    frame_chunk_size: int,
    action_per_frame: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros(
        batch_size,
        action_dim,
        frame_chunk_size,
        action_per_frame,
        1,
        device=device,
        dtype=dtype,
    )


def _build_joint_clean_cache_attention_mask(
    *,
    latents: torch.Tensor,
    actions: torch.Tensor,
    text_token_count: int,
    backbone_config: SharedVideoTransformerConfig,
    chunk_size: int,
    window_size: int,
    current_block_coupling: CurrentBlockCoupling | str,
    preserve_video_pretrain_history: bool,
) -> torch.Tensor:
    profile = _build_joint_clean_cache_attention_profile(
        latents=latents,
        actions=actions,
        text_token_count=text_token_count,
        backbone_config=backbone_config,
        chunk_size=chunk_size,
        window_size=window_size,
        current_block_coupling=current_block_coupling,
        preserve_video_pretrain_history=preserve_video_pretrain_history,
    )
    if profile.self_attention_mask is None:
        raise ValueError("Joint clean cache attention profile did not materialize a clean self-attention mask.")
    return profile.self_attention_mask


def _build_joint_clean_cache_attention_profile(
    *,
    latents: torch.Tensor,
    actions: torch.Tensor,
    text_token_count: int,
    backbone_config: SharedVideoTransformerConfig,
    chunk_size: int,
    window_size: int,
    current_block_coupling: CurrentBlockCoupling | str,
    preserve_video_pretrain_history: bool,
) -> PreparedAttentionProfile:
    # The clean-cache writer keeps batch as the real batch dimension. Build a
    # batch-local mask that can broadcast across CFG/batch rows instead of a
    # flattened `[B * tokens, B * tokens]` mask.
    batch_local_latent_shape = (1, *tuple(int(dim) for dim in latents.shape[1:]))
    batch_local_action_shape = (1, *tuple(int(dim) for dim in actions.shape[1:]))
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=batch_local_latent_shape,
        action_shape=batch_local_action_shape,
        padded_length=0,
        chunk_size=max(1, int(chunk_size)),
        window_size=max(1, int(window_size)),
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
        text_token_count=int(text_token_count),
        device=latents.device,
        build_dense_masks=True,
        build_flex_masks=False,
        current_block_coupling=CurrentBlockCoupling(current_block_coupling).value,
        preserve_video_pretrain_history=bool(preserve_video_pretrain_history),
    )
    if profile.self_attention_mask is None or profile.cross_attention_mask is None:
        raise ValueError("Joint clean cache attention profile did not materialize dense masks.")

    video_token_count = (
        int(latents.shape[2])
        // max(1, int(backbone_config.patch_size_t))
        * (int(latents.shape[3]) // max(1, int(backbone_config.patch_size_h)))
        * (int(latents.shape[4]) // max(1, int(backbone_config.patch_size_w)))
    )
    action_token_count = (
        int(actions.shape[2])
        * int(actions.shape[3])
        * int(actions.shape[4])
    )
    clean_indices = torch.cat(
        [
            torch.arange(
                video_token_count,
                2 * video_token_count,
                device=profile.self_attention_mask.device,
            ),
            torch.arange(
                2 * video_token_count + action_token_count,
                2 * video_token_count + 2 * action_token_count,
                device=profile.self_attention_mask.device,
            ),
        ],
        dim=0,
    )
    return PreparedAttentionProfile(
        spec=profile.spec,
        self_attention_mask=profile.self_attention_mask.index_select(0, clean_indices).index_select(1, clean_indices),
        cross_attention_mask=profile.cross_attention_mask.index_select(0, clean_indices),
        metadata={
            **profile.metadata,
            "clean_cache_commit": True,
        },
    )


def _write_joint_clean_tokens_to_exact_cache(
    *,
    transformer: torch.nn.Module,
    cache_name: str,
    frame_start: int,
    latents: torch.Tensor,
    actions: torch.Tensor,
    text_emb: torch.Tensor,
    negative_text_emb: torch.Tensor | None,
    use_cfg: bool,
    action_channel_mask: torch.Tensor | None,
    update_cache: int,
    backbone_config: SharedVideoTransformerConfig,
    chunk_size: int,
    window_size: int,
    current_block_coupling: CurrentBlockCoupling | str,
    preserve_video_pretrain_history: bool,
) -> None:
    model_dtype = reference_runtime_dtype(transformer)
    video_cache_input = prepare_reference_single_stream_input(
        latents=latents,
        timestep=0.0,
        text_emb=text_emb,
        frame_st_id=frame_start,
        backbone_config=backbone_config,
        action_mode=False,
    )
    action_cache_input = prepare_reference_single_stream_input(
        latents=actions,
        timestep=0.0,
        text_emb=text_emb,
        frame_st_id=frame_start,
        backbone_config=backbone_config,
        action_mode=True,
        action_channel_mask=action_channel_mask,
    )
    if use_cfg:
        if negative_text_emb is None:
            raise ValueError("Joint cache commit with CFG requires negative_text_emb.")
        video_cache_input = repeat_input_for_cfg(video_cache_input, negative_text_emb=negative_text_emb)
        action_cache_input = repeat_input_for_cfg(action_cache_input, negative_text_emb=negative_text_emb)

    latent_hidden_states = transformer._input_embed(
        video_cache_input["noisy_latents"].to(dtype=model_dtype),
        input_type="latent",
    ).contiguous().clone()
    action_hidden_states = transformer._input_embed(
        action_cache_input["noisy_latents"].to(dtype=model_dtype),
        input_type="action",
    ).contiguous().clone()
    hidden_states = torch.cat([latent_hidden_states, action_hidden_states], dim=1)
    cache_stream_ids = _stream_ids_for_clean_video_action_tokens(
        video_token_count=int(latent_hidden_states.shape[1]),
        action_token_count=int(action_hidden_states.shape[1]),
        device=hidden_states.device,
    )

    text_hidden_states = transformer._exact_text_hidden_states(
        video_cache_input["text_emb"],
        dtype=model_dtype,
    ).contiguous().clone()
    latent_grid_id = video_cache_input["grid_id"].contiguous().clone()
    action_grid_id = action_cache_input["grid_id"].contiguous().clone()
    rotary_emb = transformer.rope(torch.cat([latent_grid_id, action_grid_id], dim=2))[:, :, None]

    latent_time_steps = video_cache_input["timesteps"].contiguous().clone()
    action_time_steps = action_cache_input["timesteps"].contiguous().clone()
    _, latent_timestep_proj = transformer._time_embed(
        latent_time_steps,
        int(latents.shape[-2]),
        int(latents.shape[-1]),
        dtype=model_dtype,
        action_mode=False,
    )
    _, action_timestep_proj = transformer._time_embed(
        action_time_steps,
        int(actions.shape[-2]),
        int(actions.shape[-1]),
        dtype=model_dtype,
        action_mode=True,
    )
    timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)
    attention_profile = _build_joint_clean_cache_attention_profile(
        latents=video_cache_input["noisy_latents"],
        actions=action_cache_input["noisy_latents"],
        text_token_count=int(video_cache_input["text_emb"].shape[1]),
        backbone_config=backbone_config,
        chunk_size=chunk_size,
        window_size=window_size,
        current_block_coupling=current_block_coupling,
        preserve_video_pretrain_history=preserve_video_pretrain_history,
    )

    cache_state = transformer._resolve_exact_cache_state(cache_name)
    cache_backend_name = cache_state.backend_name if cache_state is not None else None
    cache_backend_payload = cache_state.backend_payload if cache_state is not None else None
    for layer_index, block in enumerate(transformer.blocks):
        hidden_states, _, _ = block(
            hidden_states,
            encoder_hidden_states=text_hidden_states,
            temb=timestep_proj,
            rotary_emb=rotary_emb,
            attention_profile=attention_profile,
            self_attention_cache_backend_name=cache_backend_name,
            self_attention_cache_backend_state=(
                cache_backend_payload.layer_states[layer_index]
                if cache_backend_uses_slot_pool(cache_backend_name)
                and cache_backend_payload is not None
                and layer_index < len(cache_backend_payload.layer_states)
                else None
            ),
            self_attention_cache_update_mode=update_cache,
            self_attention_cache_stream_ids=cache_stream_ids,
        )

    if cache_state is not None and cache_backend_uses_slot_pool(cache_backend_name):
        materialized_entries = materialize_cache_backend_entries(cache_backend_payload)
        transformer._exact_runtime_caches[cache_name] = CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=cache_state.cached_frames,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_backend_payload,
            payload=dict(cache_state.payload),
            self_attention_kv=materialized_entries,
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=cache_state.update_metadata,
        )


def _write_exact_cache_chunk(
    *,
    transformer: torch.nn.Module,
    cache_spec: ExactCacheInterfaceSpec,
    cache_name: str,
    frame_start: int,
    backbone_config: SharedVideoTransformerConfig,
    video_latents: torch.Tensor,
    action_latents: torch.Tensor,
    text_emb: torch.Tensor,
    negative_text_emb: torch.Tensor | None,
    use_cfg: bool,
    action_channel_mask: torch.Tensor | None,
    update_cache: int,
    chunk_size: int,
    window_size: int,
    current_block_coupling: CurrentBlockCoupling | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
    preserve_video_pretrain_history: bool = False,
) -> None:
    if cache_spec.write_mode == ParallelExactCacheWriteMode.JOINT_PACKED:
        _write_joint_clean_tokens_to_exact_cache(
            transformer=transformer,
            cache_name=cache_name,
            frame_start=frame_start,
            latents=video_latents,
            actions=action_latents,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            use_cfg=use_cfg,
            action_channel_mask=action_channel_mask,
            update_cache=update_cache,
            backbone_config=backbone_config,
            chunk_size=chunk_size,
            window_size=window_size,
            current_block_coupling=current_block_coupling,
            preserve_video_pretrain_history=preserve_video_pretrain_history,
        )
        return
    if cache_spec.write_mode == ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED:
        current_block_coupling = CurrentBlockCoupling(current_block_coupling)
        chunk_size = max(1, int(chunk_size))
        video_frames = int(video_latents.shape[2])
        action_frames = int(action_latents.shape[2])
        total_frames = max(video_frames, action_frames)

        def _write_video_cache(video_chunk: torch.Tensor, *, chunk_frame_start: int) -> None:
            video_cache_input = prepare_reference_single_stream_input(
                latents=video_chunk,
                timestep=0.0,
                text_emb=text_emb,
                frame_st_id=chunk_frame_start,
                backbone_config=backbone_config,
                action_mode=False,
            )
            run_reference_single_stream_forward(
                transformer,
                input_dict=video_cache_input,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=1.0,
                negative_text_emb=negative_text_emb,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )

        def _write_action_cache(action_chunk: torch.Tensor, *, chunk_frame_start: int) -> None:
            action_cache_input = prepare_reference_single_stream_input(
                latents=action_chunk,
                timestep=0.0,
                text_emb=text_emb,
                frame_st_id=chunk_frame_start,
                backbone_config=backbone_config,
                action_mode=True,
                action_channel_mask=action_channel_mask,
            )
            run_reference_single_stream_forward(
                transformer,
                input_dict=action_cache_input,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=True,
                guidance_scale=1.0,
                negative_text_emb=negative_text_emb,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )

        for chunk_offset in range(0, total_frames, chunk_size):
            chunk_frame_start = int(frame_start + chunk_offset)
            chunk_end = chunk_offset + chunk_size
            video_chunk = video_latents[:, :, chunk_offset:min(chunk_end, video_frames)]
            action_chunk = action_latents[:, :, chunk_offset:min(chunk_end, action_frames)]
            has_video = int(video_chunk.shape[2]) > 0
            has_action = int(action_chunk.shape[2]) > 0

            if current_block_coupling == CurrentBlockCoupling.VIDEO_THEN_ACTION:
                if has_video:
                    _write_video_cache(video_chunk, chunk_frame_start=chunk_frame_start)
                if has_action:
                    _write_action_cache(action_chunk, chunk_frame_start=chunk_frame_start)
            elif current_block_coupling == CurrentBlockCoupling.ACTION_THEN_VIDEO:
                if has_action:
                    _write_action_cache(action_chunk, chunk_frame_start=chunk_frame_start)
                if has_video and has_action:
                    metadata_previous = _set_slot_pool_layer_metadata(
                        transformer,
                        cache_name=cache_name,
                        updates={
                            SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS: _single_stream_action_token_count(
                                action_chunk
                            ),
                        },
                    )
                    try:
                        _write_video_cache(video_chunk, chunk_frame_start=chunk_frame_start)
                    finally:
                        _restore_slot_pool_layer_metadata(metadata_previous)
                elif has_video:
                    _write_video_cache(video_chunk, chunk_frame_start=chunk_frame_start)
            elif current_block_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
                overlap_frames = min(int(video_chunk.shape[2]), int(action_chunk.shape[2]))
                if overlap_frames > 0:
                    _write_joint_clean_tokens_to_exact_cache(
                        transformer=transformer,
                        cache_name=cache_name,
                        frame_start=chunk_frame_start,
                        latents=video_chunk[:, :, :overlap_frames],
                        actions=action_chunk[:, :, :overlap_frames],
                        text_emb=text_emb,
                        negative_text_emb=negative_text_emb,
                        use_cfg=use_cfg,
                        action_channel_mask=action_channel_mask,
                        update_cache=update_cache,
                        backbone_config=backbone_config,
                        chunk_size=chunk_size,
                        window_size=window_size,
                        current_block_coupling=current_block_coupling,
                        preserve_video_pretrain_history=preserve_video_pretrain_history,
                    )
                if int(video_chunk.shape[2]) > overlap_frames:
                    _write_video_cache(
                        video_chunk[:, :, overlap_frames:],
                        chunk_frame_start=chunk_frame_start + overlap_frames,
                    )
                if int(action_chunk.shape[2]) > overlap_frames:
                    _write_action_cache(
                        action_chunk[:, :, overlap_frames:],
                        chunk_frame_start=chunk_frame_start + overlap_frames,
                    )
            else:
                raise ValueError(
                    "Single-stream staged cache writes only support ordered staged or decoupled couplings, "
                    f"got {current_block_coupling.value!r}."
                )
        return
    raise ValueError(f"Unsupported exact cache write_mode: {cache_spec.write_mode!r}")


def _commit_joint_chunk_to_exact_cache(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    inference_config: InferenceConfig,
    policy_config: ParallelStreamPolicyConfig | None = None,
    cache_name: str,
    frame_start: int,
    latents: torch.Tensor,
    actions: torch.Tensor,
    text_emb: torch.Tensor,
    negative_text_emb: torch.Tensor | None,
    use_cfg: bool,
    action_channel_mask: torch.Tensor | None,
) -> None:
    _write_joint_clean_tokens_to_exact_cache(
        transformer=transformer,
        cache_name=cache_name,
        frame_start=frame_start,
        latents=latents,
        actions=actions,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        use_cfg=use_cfg,
        action_channel_mask=action_channel_mask,
        update_cache=1,
        backbone_config=backbone_config,
        chunk_size=inference_config.frame_chunk_size,
        window_size=(
            int(policy_config.attn_window)
            if policy_config is not None
            else int(inference_config.frame_chunk_size)
        ),
        current_block_coupling=(
            resolve_parallel_current_block_coupling(policy_config)
            if policy_config is not None
            else CurrentBlockCoupling.JOINT
        ),
        preserve_video_pretrain_history=bool(
            getattr(policy_config, "preserve_video_pretrain_history", False)
        )
        if policy_config is not None
        else False,
    )


def _summarize_slot_pool_cache_state(
    transformer: torch.nn.Module,
    cache_name: str,
) -> dict[str, int] | None:
    if not hasattr(transformer, "_resolve_exact_cache_state"):
        return None
    cache_state = transformer._resolve_exact_cache_state(cache_name)
    if cache_state is None or not cache_backend_uses_slot_pool(cache_state.backend_name):
        return None
    backend_payload = cache_state.backend_payload
    if backend_payload is None or not getattr(backend_payload, "layer_states", None):
        return None
    layer_state = backend_payload.layer_states[0]
    if layer_state.slot_mask is None:
        return None
    cached_tokens = int(layer_state.slot_mask.sum().item())
    prediction_tokens = (
        int(layer_state.prediction_mask[layer_state.slot_mask].sum().item())
        if layer_state.prediction_mask is not None
        else 0
    )
    return {
        "cached_tokens": cached_tokens,
        "prediction_tokens": prediction_tokens,
        "total_slots": int(layer_state.slot_mask.numel()),
    }


def _run_parallel_action_conditioned_inference_rollout_impl(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    action_dim: int,
    condition_latents: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
    action_channel_mask: torch.Tensor | None,
    infer_cache: dict[str, Any],
    advance_frame_start: bool = False,
    forced_action_latents: torch.Tensor | None = None,
    commit_action_latents: torch.Tensor | None = None,
    forced_action_noise: torch.Tensor | None = None,
    action_conditioning_mode: str = "vanilla_joint_rollout",
) -> LingbotParallelInferArtifacts:
    current_block_coupling = resolve_parallel_current_block_coupling(policy_config)
    joint_packed_couplings = {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }
    if current_block_coupling not in joint_packed_couplings:
        raise ValueError(
            "`run_parallel_action_conditioned_inference_rollout` only implements packed noisy same-step coupling; "
            f"got {current_block_coupling.value!r}."
        )
    if policy_config.current_block_coupling is None and not policy_config.video_condition_on_action:
        raise ValueError(
            "`lingbot_exact_action_conditioned` requires `video_condition_on_action = true`."
        )
    if condition_latents is not None:
        device = condition_latents.device
        batch_size = condition_latents.shape[0]
        latent_height = condition_latents.shape[-2]
        latent_width = condition_latents.shape[-1]
    else:
        if "batch_size" not in infer_cache or "latent_height" not in infer_cache or "latent_width" not in infer_cache:
            raise ValueError(
                "Joint exact inference without current condition latents requires cached batch/latent shape metadata."
            )
        device = next(transformer.parameters()).device
        batch_size = int(infer_cache["batch_size"])
        latent_height = int(infer_cache["latent_height"])
        latent_width = int(infer_cache["latent_width"])
    cache_context, text_emb, negative_text_emb = _resolve_exact_cache_context(
        transformer=transformer,
        backbone_config=backbone_config,
        inference_config=inference_config,
        infer_cache=infer_cache,
        batch_size=batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        device=device,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
    )
    model_dtype = cache_context.model_dtype
    generation_frame_start = int(infer_cache.get("frame_start", 0))
    cache_name = cache_context.cache_name
    cache_backend_name = cache_context.cache_backend_name
    cache_spec = _build_exact_cache_spec(
        write_mode=ParallelExactCacheWriteMode.JOINT_PACKED,
        batch_size=batch_size,
        use_cfg=cache_context.use_cfg,
        prefix_visibility_mode=_prefix_visibility_mode_for_policy(policy_config),
    )
    if inference_config.use_cache and not cache_context.cache_initialized:
        if condition_latents is None:
            raise ValueError(
                "Joint exact inference requires condition latents on the first chunk when cache is empty."
            )
        cache_context = _ensure_exact_cache_initialized(
            transformer=transformer,
            policy_config=policy_config,
            inference_config=inference_config,
            cache_context=cache_context,
            cache_spec=cache_spec,
        )
    latents = torch.randn(
        batch_size,
        backbone_config.latent_channels,
        inference_config.frame_chunk_size,
        latent_height,
        latent_width,
        device=device,
        dtype=model_dtype,
    )
    actions = torch.randn(
        batch_size,
        action_dim,
        inference_config.frame_chunk_size,
        policy_config.action_per_frame,
        1,
        device=device,
        dtype=model_dtype,
    )
    action_denoise_mask = None
    if action_channel_mask is not None:
        action_denoise_mask = action_channel_mask.to(device=device, dtype=model_dtype)
        actions = actions * action_denoise_mask
    if forced_action_latents is not None:
        forced_action_latents = forced_action_latents.to(device=device, dtype=model_dtype)
        if tuple(forced_action_latents.shape) != tuple(actions.shape):
            raise ValueError(
                "Forced joint-denoise action latents must match the generated action chunk shape, "
                f"got forced={tuple(forced_action_latents.shape)} and expected={tuple(actions.shape)}."
            )
        if action_denoise_mask is not None:
            forced_action_latents = forced_action_latents * action_denoise_mask
        if forced_action_noise is None:
            forced_action_noise = torch.randn_like(forced_action_latents)
        else:
            forced_action_noise = forced_action_noise.to(device=device, dtype=model_dtype)
            if tuple(forced_action_noise.shape) != tuple(actions.shape):
                raise ValueError(
                    "Forced joint-denoise action noise must match the generated action chunk shape, "
                    f"got noise={tuple(forced_action_noise.shape)} and expected={tuple(actions.shape)}."
                )
        if action_denoise_mask is not None:
            forced_action_noise = forced_action_noise * action_denoise_mask
    if commit_action_latents is not None:
        commit_action_latents = commit_action_latents.to(device=device, dtype=model_dtype)
        if tuple(commit_action_latents.shape) != tuple(actions.shape):
            raise ValueError(
                "Committed joint-denoise action latents must match the generated action chunk shape, "
                f"got commit={tuple(commit_action_latents.shape)} and expected={tuple(actions.shape)}."
            )
        if action_denoise_mask is not None:
            commit_action_latents = commit_action_latents * action_denoise_mask
    initial_observed_video_anchor = None
    if infer_cache.get("step_index", 0) == 0 and condition_latents is not None and generation_frame_start == 0:
        initial_observed_video_anchor = condition_latents[:, :, 0:1].to(device=device, dtype=model_dtype)
    # Keep the packed four-branch sequence contract for compatibility with the
    # trained backbone, but do not provide any explicit clean conditioning
    # signal at inference time. History should come only from the runtime
    # cache; the clean branches are zero placeholders.
    condition_video_latents = torch.zeros_like(latents)
    condition_action_latents = torch.zeros(
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
    if len(video_scheduler.timesteps) != len(action_scheduler.timesteps):
        raise ValueError(
            "Joint LingBot denoising expects matched video/action inference step counts; "
            "set `video_num_inference_steps == action_num_inference_steps` for this mode."
        )

    couple_action_video_timesteps = should_couple_action_to_video_timesteps(policy_config)
    action_timestep_lookup_scheduler: FlowMatchScheduler | None = None
    if couple_action_video_timesteps:
        action_timestep_lookup_scheduler = FlowMatchScheduler(
            shift=training_config.action_sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=training_config.action_num_train_timesteps,
        )
        action_timestep_lookup_scheduler.set_timesteps(training_config.action_num_train_timesteps)
        action_timestep_lookup_scheduler.sigmas = action_timestep_lookup_scheduler.sigmas.to(device=device)
        action_timestep_lookup_scheduler.timesteps = action_timestep_lookup_scheduler.timesteps.to(device=device)
    attention_profile_name = None
    if str(policy_config.video_action_attention_scope) == "block_local":
        if resolve_stage_attention_mode(backbone_config, stage="train", exact_runtime=True) == "flex":
            attention_profile_name = _attention_profile_name_for_current_block_coupling(current_block_coupling)

    video_timestep_values_list = list(video_scheduler.timesteps.to(device=device, dtype=torch.float32))
    action_timestep_values_list = list(action_scheduler.timesteps.to(device=device, dtype=torch.float32))
    video_sigma_values_list = list(video_scheduler.sigmas.to(device=device, dtype=torch.float32))
    for index, (video_timestep, action_timestep) in enumerate(
        zip(video_timestep_values_list, action_timestep_values_list)
    ):
        video_timestep_values = video_timestep.expand(batch_size, inference_config.frame_chunk_size)
        if initial_observed_video_anchor is not None:
            latents[:, :, 0:1] = initial_observed_video_anchor
            video_timestep_values = video_timestep_values.clone()
            video_timestep_values[:, 0] = 0.0
        if couple_action_video_timesteps:
            if action_timestep_lookup_scheduler is None:  # pragma: no cover - defensive guard
                raise RuntimeError("Coupled joint denoise requires an action timestep lookup scheduler.")
            shared_sigma = video_sigma_values_list[index]
            shared_sigma_next = video_scheduler.next_sigma(index).to(device=device, dtype=torch.float32)
            action_timestep = action_timestep_lookup_scheduler.timestep_matching_sigma(shared_sigma).to(
                device=device,
                dtype=torch.float32,
            )
            action_timestep_values = action_timestep.expand(batch_size, inference_config.frame_chunk_size)
        else:
            shared_sigma = None
            shared_sigma_next = None
            action_timestep_values = action_timestep.expand(batch_size, inference_config.frame_chunk_size)
        if forced_action_latents is not None:
            if couple_action_video_timesteps:
                sigma = shared_sigma.to(device=device, dtype=model_dtype).view(1, 1, 1, 1, 1)
                actions = (1 - sigma) * forced_action_latents + sigma * forced_action_noise
            else:
                actions = action_scheduler.add_noise(
                    forced_action_latents,
                    forced_action_noise,
                    action_timestep,
                    t_dim=2,
                )
            if action_denoise_mask is not None:
                actions = actions * action_denoise_mask
        action_mask_latents = (
            action_denoise_mask.expand_as(actions)
            if action_denoise_mask is not None
            else torch.ones_like(actions)
        )
        latent_grid_id = get_mesh_id(
            inference_config.frame_chunk_size // backbone_config.patch_size_t,
            latent_height // backbone_config.patch_size_h,
            latent_width // backbone_config.patch_size_w,
            t=0,
            f_w=1,
            f_shift=generation_frame_start,
            action=False,
            device=device,
        )[None].repeat(batch_size, 1, 1)
        action_grid_id = get_mesh_id(
            inference_config.frame_chunk_size,
            policy_config.action_per_frame,
            1,
            t=1,
            f_w=1,
            f_shift=generation_frame_start,
            action=True,
            device=device,
        )[None].repeat(batch_size, 1, 1)
        input_dict = {
            "latent_dict": {
                "noisy_latents": latents,
                "latent": condition_video_latents,
                "text_emb": text_emb,
                "grid_id": latent_grid_id,
                "timesteps": video_timestep_values,
                "cond_timesteps": torch.zeros_like(video_timestep_values),
            },
            "action_dict": {
                "noisy_latents": actions,
                "latent": condition_action_latents,
                "text_emb": text_emb,
                "grid_id": action_grid_id,
                "timesteps": action_timestep_values,
                "cond_timesteps": torch.zeros_like(action_timestep_values),
                "actions_mask": action_mask_latents,
            },
            "chunk_size": max(1, int(inference_config.frame_chunk_size)),
            "window_size": max(1, int(policy_config.attn_window)),
            "attention_profile_name": attention_profile_name,
            "preserve_video_pretrain_history": bool(
                getattr(policy_config, "preserve_video_pretrain_history", False)
            ),
        }
        video_noise_pred, action_noise_pred = _run_parallel_action_conditioned_forward(
            transformer,
            input_dict=input_dict,
            video_guidance_scale=float(inference_config.guidance_scale),
            action_guidance_scale=float(inference_config.action_guidance_scale),
            negative_text_emb=negative_text_emb,
            update_cache=0,
            cache_name=cache_name,
        )
        video_noise_pred = data_seq_to_patch(
            transformer.patch_size,
            video_noise_pred,
            inference_config.frame_chunk_size,
            latent_height,
            latent_width,
            batch_size=batch_size,
        )
        if couple_action_video_timesteps:
            latents = video_scheduler.step_with_sigmas(
                video_noise_pred,
                sigma=shared_sigma,
                sigma_next=shared_sigma_next,
                sample=latents,
            )
        else:
            latents = video_scheduler.step(video_noise_pred, video_timestep, latents)
        if initial_observed_video_anchor is not None:
            latents[:, :, 0:1] = initial_observed_video_anchor
        action_noise_pred = rearrange(
            action_noise_pred,
            "b (f n) c -> b c f n 1",
            f=inference_config.frame_chunk_size,
        )
        if forced_action_latents is None:
            if couple_action_video_timesteps:
                actions = action_scheduler.step_with_sigmas(
                    action_noise_pred,
                    sigma=shared_sigma,
                    sigma_next=shared_sigma_next,
                    sample=actions,
                )
            else:
                actions = action_scheduler.step(action_noise_pred, action_timestep, actions)
            if action_denoise_mask is not None:
                actions = actions * action_denoise_mask

    final_action_latents = (
        commit_action_latents
        if commit_action_latents is not None
        else (forced_action_latents if forced_action_latents is not None else actions)
    )

    if inference_config.use_cache:
        _write_exact_cache_chunk(
            transformer=transformer,
            cache_spec=cache_spec,
            cache_name=cache_name,
            frame_start=generation_frame_start,
            backbone_config=backbone_config,
            video_latents=latents,
            action_latents=final_action_latents,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            use_cfg=cache_context.use_cfg,
            action_channel_mask=action_channel_mask,
            update_cache=1,
            chunk_size=inference_config.frame_chunk_size,
            window_size=policy_config.attn_window,
            current_block_coupling=current_block_coupling,
            preserve_video_pretrain_history=bool(
                getattr(policy_config, "preserve_video_pretrain_history", False)
            ),
        )

    next_cache = {
        "runtime_mode": "lingbot_exact_action_conditioned",
        "cache_name": cache_name,
        "cache_backend_name": cache_backend_name,
        "cache_initialized": cache_context.cache_initialized and inference_config.use_cache,
        "frame_start": int(
            generation_frame_start + inference_config.frame_chunk_size if advance_frame_start else generation_frame_start
        ),
        "latent_height": latent_height,
        "latent_width": latent_width,
        "batch_size": batch_size,
        "step_index": int(infer_cache.get("step_index", 0) + 1),
        "use_cfg": cache_context.use_cfg,
    }
    debug = {
        "runtime_mode": "lingbot_exact_action_conditioned",
        "cache_name": cache_name,
        "cache_backend_name": cache_backend_name,
        "generation_frame_start": generation_frame_start,
        "advance_frame_start": advance_frame_start,
        "video_condition_on_action": bool(policy_config.video_condition_on_action),
        "video_action_condition_source": str(policy_config.video_action_condition_source),
        "video_action_attention_scope": str(policy_config.video_action_attention_scope),
        "current_block_coupling": current_block_coupling.value,
        "couple_action_to_video_timesteps": bool(couple_action_video_timesteps),
        "joint_denoise": True,
        "uses_explicit_clean_condition": False,
        "use_cache": bool(inference_config.use_cache),
        "cache_commit_mode": str(cache_spec.write_mode),
        "use_cfg": cache_context.use_cfg,
        "video_num_inference_steps": int(inference_config.video_num_inference_steps),
        "action_num_inference_steps": int(inference_config.action_num_inference_steps),
        "action_conditioning_mode": action_conditioning_mode,
        "initial_observed_video_anchor": initial_observed_video_anchor is not None,
        "forced_action_denoise": forced_action_latents is not None,
        "commit_action_override": commit_action_latents is not None,
    }
    cache_summary = _summarize_slot_pool_cache_state(transformer, cache_name)
    if cache_summary is not None:
        debug.update(cache_summary)
    output_dtype = condition_latents.dtype if condition_latents is not None else model_dtype
    action_pred = rearrange(final_action_latents, "b c f n 1 -> b (f n) c").to(dtype=output_dtype)
    return LingbotParallelInferArtifacts(
        action_pred=action_pred,
        predicted_latents=latents.to(dtype=output_dtype),
        next_cache=next_cache,
        debug=debug,
    )


def run_parallel_action_conditioned_inference_rollout(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    action_dim: int,
    condition_latents: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
    action_channel_mask: torch.Tensor | None,
    infer_cache: dict[str, Any],
    advance_frame_start: bool = False,
) -> LingbotParallelInferArtifacts:
    return _run_parallel_action_conditioned_inference_rollout_impl(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=action_dim,
        condition_latents=condition_latents,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        action_channel_mask=action_channel_mask,
        infer_cache=infer_cache,
        advance_frame_start=advance_frame_start,
    )


def run_parallel_action_conditioned_action_override_inference_rollout(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    action_dim: int,
    condition_latents: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
    action_channel_mask: torch.Tensor | None,
    infer_cache: dict[str, Any],
    advance_frame_start: bool,
    forced_action_latents: torch.Tensor | None = None,
    commit_action_latents: torch.Tensor | None = None,
    forced_action_noise: torch.Tensor | None = None,
    action_conditioning_mode: str = "forced_action_joint_fdm",
) -> LingbotParallelInferArtifacts:
    """Run joint-denoise inference with ablation-owned action overrides.

    `forced_action_latents` replaces the noisy action branch at every
    denoising timestep. `commit_action_latents` only changes the clean action
    tokens committed into history after the chunk is generated.
    """

    return _run_parallel_action_conditioned_inference_rollout_impl(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=action_dim,
        condition_latents=condition_latents,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        action_channel_mask=action_channel_mask,
        infer_cache=infer_cache,
        advance_frame_start=advance_frame_start,
        forced_action_latents=forced_action_latents,
        commit_action_latents=commit_action_latents,
        forced_action_noise=forced_action_noise,
        action_conditioning_mode=action_conditioning_mode,
    )


def run_parallel_exact_train(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(transformer, "execute_runtime_step"):
        step_output = transformer.execute_runtime_step(
            RuntimeStepInput(
                program=build_chunked_dual_stream_exact_train_program(
                    attention_profile_name=input_dict.get("attention_profile_name"),  # type: ignore[arg-type]
                    cache_backend_name="slot_pool_exact",
                ),
                payload=input_dict,
                train_mode=True,
            )
        )
        try:
            return (
                step_output.projected_outputs["video_prediction"],
                step_output.projected_outputs["action_prediction"],
            )
        except KeyError as exc:
            raise ValueError("Exact dual-stream runtime step did not return both video/action predictions.") from exc

    forward_train = getattr(transformer, "forward_train", None)
    if callable(forward_train):
        return forward_train(input_dict)
    return _run_parallel_exact_joint_forward_manual(transformer, input_dict)


def run_parallel_action_conditioned_train(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the LingBot exact train-time packed layout and shared runtime
    # execution path; the joint-denoise variant changes inference rollout
    # semantics, not the backbone's train-time sequence contract.
    return run_parallel_exact_train(transformer, input_dict)
