from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import (
    CFGMode,
    CacheUpdateMode,
    CacheWarmupSource,
    InferenceConfig,
    JointSampler,
    TrainingConfig,
    WarmupAnchor,
)
from open_wam.models.video_backbone.contracts import ConditioningState

from .flow_matching import (
    FlowMatchScheduler,
    build_action_flow_match_inference_scheduler,
    build_flow_unipc_inference_scheduler,
    build_video_flow_match_inference_scheduler,
)
from .flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler

InferenceScheduler = FlowMatchScheduler | FlowUniPCMultistepScheduler


@dataclass(frozen=True)
class RuntimeGuidanceConfig:
    """Shared CFG settings for rollout-time denoising."""

    enabled: bool
    cfg_mode: str
    video_guidance_scale: float
    action_guidance_scale: float
    video_mode: CFGMode
    action_mode: CFGMode
    conditioned_cache_branch: str = "conditioned"
    unconditioned_cache_branch: str = "unconditioned"


@dataclass(frozen=True)
class RuntimeCachePolicy:
    """Shared cache warmup/update policy for rollout-time inference."""

    warmup_before_denoise: bool
    warmup_source: CacheWarmupSource
    update_mode: CacheUpdateMode
    update_cross_attention_on_warmup: bool
    update_cross_attention_during_denoise: bool
    initial_warmup_anchor: WarmupAnchor
    initial_warmup_frames: int | None
    rollout_warmup_anchor: WarmupAnchor
    rollout_warmup_frames: int | None


@dataclass(frozen=True)
class RuntimeWarmupReference:
    """Resolved clean-reference slice used to prefill rollout cache."""

    frame_start: int
    frame_count: int


@dataclass(frozen=True)
class JointRuntimeSchedulers:
    """Shared sampler bundle for joint video/action rollout."""

    video_scheduler: InferenceScheduler
    action_scheduler: InferenceScheduler
    use_unipc: bool
    num_steps: int


def build_joint_video_timestep_grid(
    *,
    batch_size: int,
    num_video_frames: int,
    timestep_value: float,
    device: torch.device,
    observed_prefix_frames: int = 0,
    observed_timestep_value: float = 0.0,
) -> torch.Tensor:
    """Build a per-frame timestep grid for joint video/action rollout.

    Variants with an observed visual prefix can keep those frames at timestep 0
    while applying the active denoising timestep to the generated suffix.
    """

    grid = torch.full(
        (batch_size, num_video_frames),
        fill_value=float(timestep_value),
        device=device,
        dtype=torch.float32,
    )
    prefix_frames = max(0, min(int(observed_prefix_frames), num_video_frames))
    if prefix_frames:
        grid[:, :prefix_frames] = float(observed_timestep_value)
    return grid


def preserve_joint_observed_video_prefix(
    *,
    rollout_video_latents: torch.Tensor,
    observed_video_latents: torch.Tensor,
    observed_prefix_frames: int,
) -> torch.Tensor:
    """Copy the observed prefix frames back into a rollout latent window."""

    prefix_frames = max(
        0,
        min(
            int(observed_prefix_frames),
            rollout_video_latents.shape[2],
            observed_video_latents.shape[2],
        ),
    )
    if prefix_frames == 0:
        return rollout_video_latents
    preserved = rollout_video_latents.clone()
    preserved[:, :, :prefix_frames] = observed_video_latents[:, :, :prefix_frames]
    return preserved


def resolve_runtime_guidance(
    conditioning: ConditioningState | None,
    *,
    inference_config: InferenceConfig,
) -> RuntimeGuidanceConfig:
    """Resolve shared CFG settings from conditioning and inference config."""

    has_negative_text = (
        conditioning is not None
        and conditioning.negative_text_context is not None
        and conditioning.text_context is not None
    )
    enabled = has_negative_text and (
        inference_config.guidance_scale > 1.0 or inference_config.action_guidance_scale > 1.0
    )
    for field_name, mode in (
        ("video_cfg_mode", inference_config.video_cfg_mode),
        ("action_cfg_mode", inference_config.action_cfg_mode),
    ):
        if mode not in {CFGMode.GUIDED, CFGMode.CONDITIONED, CFGMode.UNCONDITIONED}:
            raise ValueError(f"Unsupported `{field_name}`, got {mode!r}.")
    return RuntimeGuidanceConfig(
        enabled=enabled,
        cfg_mode="joint_cfg" if enabled else "joint",
        video_guidance_scale=float(inference_config.guidance_scale),
        action_guidance_scale=float(inference_config.action_guidance_scale),
        video_mode=inference_config.video_cfg_mode,
        action_mode=inference_config.action_cfg_mode,
    )


def build_unconditional_conditioning(
    conditioning: ConditioningState | None,
) -> ConditioningState | None:
    """Build the unconditioned text bundle used by classifier-free guidance."""

    if conditioning is None or conditioning.negative_text_context is None:
        return None
    return ConditioningState(
        supported=conditioning.supported,
        text_context=conditioning.negative_text_context,
        negative_text_context=conditioning.negative_text_context,
        first_frame_context=conditioning.first_frame_context,
        metadata=dict(conditioning.metadata) if conditioning.metadata is not None else None,
    )


def combine_cfg_prediction(
    conditioned_prediction: torch.Tensor,
    unconditioned_prediction: torch.Tensor,
    *,
    guidance_scale: float,
) -> torch.Tensor:
    """Combine conditioned/unconditioned predictions with CFG."""

    return unconditioned_prediction + float(guidance_scale) * (
        conditioned_prediction - unconditioned_prediction
    )


def combine_joint_cfg_predictions(
    *,
    conditioned_video_prediction: torch.Tensor,
    unconditioned_video_prediction: torch.Tensor,
    conditioned_action_prediction: torch.Tensor,
    unconditioned_action_prediction: torch.Tensor,
    guidance: RuntimeGuidanceConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve per-stream CFG behavior on the video and action predictions."""

    if not guidance.enabled:
        return conditioned_video_prediction, conditioned_action_prediction
    return (
        _resolve_stream_cfg_prediction(
            conditioned_prediction=conditioned_video_prediction,
            unconditioned_prediction=unconditioned_video_prediction,
            guidance_scale=guidance.video_guidance_scale,
            mode=guidance.video_mode,
        ),
        _resolve_stream_cfg_prediction(
            conditioned_prediction=conditioned_action_prediction,
            unconditioned_prediction=unconditioned_action_prediction,
            guidance_scale=guidance.action_guidance_scale,
            mode=guidance.action_mode,
        ),
    )


def _resolve_stream_cfg_prediction(
    *,
    conditioned_prediction: torch.Tensor,
    unconditioned_prediction: torch.Tensor,
    guidance_scale: float,
    mode: CFGMode | str,
) -> torch.Tensor:
    if mode == CFGMode.GUIDED:
        return combine_cfg_prediction(
            conditioned_prediction,
            unconditioned_prediction,
            guidance_scale=guidance_scale,
        )
    if mode == CFGMode.CONDITIONED:
        return conditioned_prediction
    if mode == CFGMode.UNCONDITIONED:
        return unconditioned_prediction
    raise ValueError(f"Unsupported per-stream CFG mode {mode!r}.")


def resolve_runtime_cache_branches(
    guidance: RuntimeGuidanceConfig,
) -> tuple[str, ...]:
    """Return the named cache branches required by the current guidance mode."""

    if not guidance.enabled:
        return ("default",)
    return (guidance.conditioned_cache_branch, guidance.unconditioned_cache_branch)


def resolve_runtime_cache_branch(
    guidance: RuntimeGuidanceConfig,
    *,
    conditioned: bool,
) -> str:
    """Select the cache branch for one conditioned/unconditioned pass."""

    if not guidance.enabled:
        return "default"
    return guidance.conditioned_cache_branch if conditioned else guidance.unconditioned_cache_branch


def build_joint_runtime_schedulers(
    *,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    device: torch.device,
) -> JointRuntimeSchedulers:
    """Build the shared sampler bundle for DreamZero-style joint rollout."""

    if inference_config.joint_sampler == JointSampler.UNIPC:
        num_joint_steps = (
            inference_config.joint_num_inference_steps
            or inference_config.video_num_inference_steps
        )
        video_scheduler = build_flow_unipc_inference_scheduler(
            num_train_timesteps=training_config.video_num_train_timesteps,
            sigma_shift=training_config.video_sigma_shift,
            num_inference_steps=num_joint_steps,
            device=device,
        )
        action_scheduler = build_flow_unipc_inference_scheduler(
            num_train_timesteps=training_config.action_num_train_timesteps,
            sigma_shift=training_config.action_sigma_shift,
            num_inference_steps=num_joint_steps,
            device=device,
        )
        return JointRuntimeSchedulers(
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
            use_unipc=True,
            num_steps=num_joint_steps,
        )

    video_scheduler = build_video_flow_match_inference_scheduler(
        training_config=training_config,
        inference_config=inference_config,
        num_inference_steps_override=inference_config.joint_num_inference_steps,
    )
    action_scheduler = build_action_flow_match_inference_scheduler(
        training_config=training_config,
        inference_config=inference_config,
        num_inference_steps_override=inference_config.joint_num_inference_steps,
    )
    return JointRuntimeSchedulers(
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        use_unipc=False,
        num_steps=len(video_scheduler.timesteps),
    )


def resolve_runtime_cache_policy(
    *,
    inference_config: InferenceConfig,
) -> RuntimeCachePolicy:
    """Resolve the shared cache update policy for cache-aware rollout."""

    update_mode = inference_config.joint_cache_update_mode
    warmup_source = inference_config.joint_cache_warmup_source
    if update_mode not in {
        CacheUpdateMode.WARMUP_ONLY,
        CacheUpdateMode.FINAL_STEP,
        CacheUpdateMode.EVERY_STEP,
        CacheUpdateMode.NONE,
    }:
        raise ValueError(
            "Unsupported `inference.joint_cache_update_mode`, "
            f"got {update_mode!r}."
        )
    if warmup_source not in {CacheWarmupSource.REFERENCE_VIDEO, CacheWarmupSource.NONE}:
        raise ValueError(
            "Unsupported `inference.joint_cache_warmup_source`, "
            f"got {warmup_source!r}."
        )
    for field_name, anchor in (
        ("joint_cache_initial_warmup_anchor", inference_config.joint_cache_initial_warmup_anchor),
        ("joint_cache_rollout_warmup_anchor", inference_config.joint_cache_rollout_warmup_anchor),
    ):
        if anchor not in {WarmupAnchor.START, WarmupAnchor.END, WarmupAnchor.FULL}:
            raise ValueError(f"Unsupported `{field_name}`, got {anchor!r}.")
    warmup_before_denoise = (
        update_mode == CacheUpdateMode.WARMUP_ONLY
        and warmup_source != CacheWarmupSource.NONE
    )
    return RuntimeCachePolicy(
        warmup_before_denoise=warmup_before_denoise,
        warmup_source=warmup_source,
        update_mode=CacheUpdateMode.NONE if update_mode == CacheUpdateMode.WARMUP_ONLY else update_mode,
        update_cross_attention_on_warmup=warmup_before_denoise,
        update_cross_attention_during_denoise=False,
        initial_warmup_anchor=inference_config.joint_cache_initial_warmup_anchor,
        initial_warmup_frames=inference_config.joint_cache_initial_warmup_frames,
        rollout_warmup_anchor=inference_config.joint_cache_rollout_warmup_anchor,
        rollout_warmup_frames=inference_config.joint_cache_rollout_warmup_frames,
    )


def should_update_cache_during_denoise(
    policy: RuntimeCachePolicy,
    *,
    step_index: int,
    num_steps: int,
) -> bool:
    """Return whether the current denoising step should write KV cache."""

    if policy.update_mode == CacheUpdateMode.EVERY_STEP:
        return True
    if policy.update_mode == CacheUpdateMode.FINAL_STEP:
        return step_index == num_steps - 1
    return False


def resolve_runtime_warmup_reference(
    *,
    policy: RuntimeCachePolicy,
    current_start_frame: int,
    num_video_frames: int,
    num_frame_per_block: int,
) -> RuntimeWarmupReference | None:
    """Resolve which clean-reference frames should prefill cache.

    This stays shared and declarative so cache-aware variants can opt into
    DreamZero-style warmup scheduling without hard-coding it into one policy
    class.
    """

    if not policy.warmup_before_denoise:
        return None
    if policy.warmup_source == CacheWarmupSource.NONE:
        return None
    if policy.warmup_source == CacheWarmupSource.REFERENCE_VIDEO:
        if current_start_frame == 0:
            return _resolve_warmup_slice(
                anchor=policy.initial_warmup_anchor,
                frame_count=policy.initial_warmup_frames,
                num_video_frames=num_video_frames,
                default_frame_count=num_frame_per_block,
            )
        return _resolve_warmup_slice(
            anchor=policy.rollout_warmup_anchor,
            frame_count=policy.rollout_warmup_frames,
            num_video_frames=num_video_frames,
            default_frame_count=num_frame_per_block,
        )
    raise ValueError(f"Unsupported runtime warmup source {policy.warmup_source!r}.")


def _resolve_warmup_slice(
    *,
    anchor: WarmupAnchor | str,
    frame_count: int | None,
    num_video_frames: int,
    default_frame_count: int,
) -> RuntimeWarmupReference:
    if anchor == WarmupAnchor.FULL:
        return RuntimeWarmupReference(frame_start=0, frame_count=num_video_frames)
    resolved_frame_count = default_frame_count if frame_count is None else frame_count
    if resolved_frame_count <= 0:
        return RuntimeWarmupReference(frame_start=0, frame_count=0)
    resolved_frame_count = min(resolved_frame_count, num_video_frames)
    if anchor == WarmupAnchor.START:
        return RuntimeWarmupReference(frame_start=0, frame_count=resolved_frame_count)
    if anchor == WarmupAnchor.END:
        return RuntimeWarmupReference(
            frame_start=max(num_video_frames - resolved_frame_count, 0),
            frame_count=resolved_frame_count,
        )
    raise ValueError(f"Unsupported warmup anchor {anchor!r}.")
