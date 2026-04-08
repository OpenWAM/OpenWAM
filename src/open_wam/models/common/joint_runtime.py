from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F

from open_wam.models.video_backbone.contracts import CacheState, ConditioningState

from .register_sequence import RegisterSequenceLayout
from .runtime_controls import (
    build_joint_runtime_schedulers,
    build_unconditional_conditioning,
    combine_joint_cfg_predictions,
    resolve_runtime_cache_branch,
    resolve_runtime_cache_branches,
    resolve_runtime_cache_policy,
    resolve_runtime_guidance,
    resolve_runtime_warmup_reference,
    should_update_cache_during_denoise,
)

if TYPE_CHECKING:
    from open_wam.models.visual_tower import VisualStageOutputs, VisualTower


@dataclass(frozen=True)
class JointTrainFlowResult:
    video_flow_pred: torch.Tensor
    action_flow_pred: torch.Tensor
    denoised_video_latents: torch.Tensor
    denoised_actions: torch.Tensor
    latent_loss: torch.Tensor
    action_loss: torch.Tensor


@dataclass(frozen=True)
class JointInferenceLoopResult:
    noisy_video_latents: torch.Tensor
    noisy_actions: torch.Tensor
    latest_core_cache: CacheState
    layout: RegisterSequenceLayout | None
    core_aux: dict[str, object]
    guidance_enabled: bool
    guidance_cfg_mode: str
    video_num_inference_steps: int
    action_num_inference_steps: int


@dataclass
class JointPredictionReuseState:
    enabled: bool
    thresholds: tuple[float, ...]
    countdowns: tuple[int, ...]
    countdown: int = 0
    previous_predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None

    def __post_init__(self) -> None:
        if self.previous_predictions is None:
            self.previous_predictions = []


_DREAMZERO_DIT_STEP_MASKS: dict[int, tuple[bool, ...]] = {
    5: (True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False),
    6: (True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True),
    7: (True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True),
    8: (True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True),
}


def _build_joint_prediction_reuse_state(inference_config) -> JointPredictionReuseState:
    thresholds = tuple(float(value) for value in inference_config.joint_prediction_reuse_thresholds)
    countdowns = tuple(int(value) for value in inference_config.joint_prediction_reuse_countdowns)
    if len(thresholds) != len(countdowns):
        raise ValueError(
            "Expected `joint_prediction_reuse_thresholds` and `joint_prediction_reuse_countdowns` "
            f"to have the same length, got {len(thresholds)} and {len(countdowns)}."
        )
    return JointPredictionReuseState(
        enabled=bool(inference_config.joint_enable_prediction_reuse),
        thresholds=thresholds,
        countdowns=countdowns,
    )


def _resolve_joint_dit_step_mask(
    inference_config,
    *,
    num_inference_steps: int,
) -> tuple[bool, ...] | None:
    explicit_mask = inference_config.joint_dit_step_mask
    if explicit_mask is not None:
        resolved_mask = tuple(bool(flag) for flag in explicit_mask)
        if len(resolved_mask) != num_inference_steps:
            raise ValueError(
                "Expected `joint_dit_step_mask` to match the configured number of joint inference steps, "
                f"got mask length {len(resolved_mask)} for {num_inference_steps} steps."
            )
        if not resolved_mask[0]:
            raise ValueError("Expected `joint_dit_step_mask[0]` to be True so the first DiT step always runs.")
        return resolved_mask

    num_dit_steps = inference_config.joint_num_dit_steps
    if num_dit_steps is None:
        return None
    if num_dit_steps not in _DREAMZERO_DIT_STEP_MASKS:
        return tuple(True for _ in range(num_inference_steps))
    resolved_mask = _DREAMZERO_DIT_STEP_MASKS[num_dit_steps]
    if len(resolved_mask) != num_inference_steps:
        return tuple(True for _ in range(num_inference_steps))
    return resolved_mask


def _should_run_joint_model(
    inference_config,
    reuse_state: JointPredictionReuseState,
    *,
    step_index: int,
    dit_step_mask: tuple[bool, ...] | None,
) -> bool:
    if not bool(inference_config.joint_dynamic_cache_schedule):
        if dit_step_mask is None:
            return True
        return bool(dit_step_mask[step_index])

    if not reuse_state.enabled:
        return True
    if len(reuse_state.previous_predictions) < 2:
        return True
    if reuse_state.countdown > 1:
        reuse_state.countdown -= 1
        return False
    if reuse_state.countdown == 1:
        reuse_state.countdown = 0
        return True

    last_video_prediction = reuse_state.previous_predictions[-1][1].flatten(1).float()
    previous_video_prediction = reuse_state.previous_predictions[-2][1].flatten(1).float()
    similarity = F.cosine_similarity(last_video_prediction, previous_video_prediction, dim=1).mean()
    for threshold, countdown in zip(reuse_state.thresholds, reuse_state.countdowns):
        if float(similarity) > float(threshold):
            reuse_state.countdown = int(countdown)
            return False
    return True


def _record_joint_prediction(
    reuse_state: JointPredictionReuseState,
    *,
    timestep: torch.Tensor,
    video_flow_pred: torch.Tensor,
    action_flow_pred: torch.Tensor,
) -> None:
    reuse_state.previous_predictions.append(
        (
            timestep.detach().clone(),
            video_flow_pred.detach().clone(),
            action_flow_pred.detach().clone(),
        )
    )
    if len(reuse_state.previous_predictions) > 2:
        reuse_state.previous_predictions.pop(0)


def resolve_joint_train_flow_result(
    *,
    projected_outputs: dict[str, torch.Tensor],
    video_artifacts,
    action_artifacts,
    unpatchify_video_prediction: Callable[[torch.Tensor], torch.Tensor],
    denoised_video_latents_from_flow: Callable[..., torch.Tensor],
    denoised_actions_from_flow: Callable[..., torch.Tensor],
    reduce_video_flow_match_loss: Callable[..., torch.Tensor],
    reduce_slot_aligned_action_flow_match_loss: Callable[..., torch.Tensor],
) -> JointTrainFlowResult:
    video_flow_pred = unpatchify_video_prediction(projected_outputs["video_patch_flow"])
    action_flow_pred = projected_outputs["action_flow"]
    denoised_video_latents = denoised_video_latents_from_flow(
        noisy_latents=video_artifacts.noisy_latents,
        flow_pred=video_flow_pred,
        timesteps=video_artifacts.timesteps,
        scheduler=video_artifacts.scheduler,
    )
    denoised_actions = denoised_actions_from_flow(
        noisy_actions=action_artifacts.noisy_actions,
        flow_pred=action_flow_pred,
        timesteps=action_artifacts.timesteps,
        scheduler=action_artifacts.scheduler,
    )
    latent_loss = reduce_video_flow_match_loss(
        flow_pred=video_flow_pred,
        targets=video_artifacts.targets,
        timesteps=video_artifacts.timesteps,
        scheduler=video_artifacts.scheduler,
    )
    action_loss = reduce_slot_aligned_action_flow_match_loss(
        flow_pred=action_flow_pred,
        targets=action_artifacts.targets,
        timesteps=action_artifacts.timesteps,
        scheduler=action_artifacts.scheduler,
        action_mask=action_artifacts.action_mask,
    )
    return JointTrainFlowResult(
        video_flow_pred=video_flow_pred,
        action_flow_pred=action_flow_pred,
        denoised_video_latents=denoised_video_latents,
        denoised_actions=denoised_actions,
        latent_loss=latent_loss,
        action_loss=action_loss,
    )


def run_joint_inference_loop(
    *,
    visual_tower: VisualTower,
    visual_outputs: VisualStageOutputs,
    reference_visual_outputs: VisualStageOutputs | None,
    training_config,
    inference_config,
    action_horizon: int,
    action_dim: int,
    num_frame_per_block: int,
    cache_state: CacheState,
    state_inputs: torch.Tensor,
    current_start_frame: int,
    warmup_current_start_frame: int | None = None,
    observed_prefix_frames_override: int | None = None,
    build_noisy_visual_outputs: Callable[[torch.Tensor], VisualStageOutputs],
    preserve_observed_video_prefix: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    constant_future_video_timestep_grid: Callable[[int, int, float, torch.device, int], torch.Tensor],
    constant_action_timestep_grid: Callable[[int, float, torch.device], torch.Tensor],
    warmup_runtime_cache: Callable[[CacheState, torch.Tensor, str, tuple[int, int], str, ConditioningState | None], CacheState],
    run_conditioned_core: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CacheState, Any], tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], RegisterSequenceLayout, CacheState, dict[str, object]]],
    run_unconditioned_core: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CacheState, Any, ConditioningState], tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], RegisterSequenceLayout, CacheState, dict[str, object]]],
    unpatchify_video_prediction: Callable[[torch.Tensor], torch.Tensor],
) -> JointInferenceLoopResult:
    reference_visual_outputs = reference_visual_outputs or visual_outputs
    warmup_start_frame = int(current_start_frame) if warmup_current_start_frame is None else int(warmup_current_start_frame)
    batch_size = visual_outputs.frontend.video_tokens.shape[0]
    device = visual_outputs.frontend.video_tokens.device
    dtype = visual_outputs.frontend.video_tokens.dtype
    observed_video_latents = visual_outputs.frontend.video_latents
    observed_prefix_frames = max(
        0,
        min(
            int(inference_config.joint_observed_video_prefix_frames)
            if observed_prefix_frames_override is None
            else int(observed_prefix_frames_override),
            observed_video_latents.shape[2],
        ),
    )
    noisy_video_latents = preserve_observed_video_prefix(
        torch.randn_like(observed_video_latents),
        observed_video_latents,
        observed_prefix_frames,
    )
    scheduler_bundle = build_joint_runtime_schedulers(
        training_config=training_config,
        inference_config=inference_config,
        device=device,
    )
    cache_policy = resolve_runtime_cache_policy(inference_config=inference_config)
    guidance = resolve_runtime_guidance(
        visual_outputs.frontend.conditioning,
        inference_config=inference_config,
    )
    unconditional_conditioning = build_unconditional_conditioning(visual_outputs.frontend.conditioning)
    latest_core_cache = visual_tower.ensure_runtime_cache_branches(
        cache_state,
        branch_names=resolve_runtime_cache_branches(guidance),
    )
    conditioned_cache_branch = resolve_runtime_cache_branch(guidance, conditioned=True)
    unconditioned_cache_branch = resolve_runtime_cache_branch(guidance, conditioned=False)
    video_scheduler = scheduler_bundle.video_scheduler
    action_scheduler = scheduler_bundle.action_scheduler
    prediction_reuse_state = _build_joint_prediction_reuse_state(inference_config)
    dit_step_mask = _resolve_joint_dit_step_mask(
        inference_config,
        num_inference_steps=len(video_scheduler.timesteps),
    )
    if len(video_scheduler.timesteps) != len(action_scheduler.timesteps):
        raise ValueError(
            "Joint inference expects video/action schedulers with the same number of steps, "
            f"got video={len(video_scheduler.timesteps)} and action={len(action_scheduler.timesteps)}."
        )

    layout: RegisterSequenceLayout | None = None
    core_aux: dict[str, object] = {}
    step_debug: list[dict[str, float | int | bool]] = []
    warmup_reference = resolve_runtime_warmup_reference(
        policy=cache_policy,
        current_start_frame=warmup_start_frame,
        num_video_frames=reference_visual_outputs.frontend.token_grid.num_frames,
        num_frame_per_block=num_frame_per_block,
    )
    if warmup_reference is not None and warmup_start_frame != observed_prefix_frames:
        tokens_per_frame = reference_visual_outputs.frontend.token_grid.tokens_per_frame
        warmup_token_span = (
            warmup_reference.frame_start * tokens_per_frame,
            (warmup_reference.frame_start + warmup_reference.frame_count) * tokens_per_frame,
        )
        latest_core_cache = warmup_runtime_cache(
            latest_core_cache,
            state_inputs,
            guidance.cfg_mode,
            warmup_token_span,
            conditioned_cache_branch,
            None,
        )
        if guidance.enabled and unconditional_conditioning is not None:
            latest_core_cache = warmup_runtime_cache(
                latest_core_cache,
                state_inputs,
                guidance.cfg_mode,
                warmup_token_span,
                unconditioned_cache_branch,
                unconditional_conditioning,
            )

    noisy_actions = torch.randn(
        batch_size,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    )

    for step_index, (video_timestep, action_timestep) in enumerate(
        zip(video_scheduler.timesteps.to(device=device), action_scheduler.timesteps.to(device=device))
    ):
        input_cache_state = latest_core_cache
        noisy_visual_outputs = build_noisy_visual_outputs(noisy_video_latents)
        conditioned_video_flow_pred: torch.Tensor | None = None
        conditioned_action_flow_pred: torch.Tensor | None = None
        unconditioned_video_flow_pred: torch.Tensor | None = None
        unconditioned_action_flow_pred: torch.Tensor | None = None
        should_run_model = _should_run_joint_model(
            inference_config,
            prediction_reuse_state,
            step_index=step_index,
            dit_step_mask=dit_step_mask,
        )
        if should_run_model:
            step_cache_update = visual_tower.build_runtime_cache_update_metadata(
                input_cache_state,
                current_start_frame=int(current_start_frame),
                update_kv_cache=should_update_cache_during_denoise(
                    cache_policy,
                    step_index=step_index,
                    num_steps=len(video_scheduler.timesteps),
                ),
                update_cross_attention_cache=cache_policy.update_cross_attention_during_denoise,
                cfg_mode=guidance.cfg_mode,
                cache_branch=conditioned_cache_branch,
            )
            _, _, projected_outputs, layout, latest_core_cache, core_aux = run_conditioned_core(
                noisy_visual_outputs.frontend.video_tokens,
                noisy_actions,
                constant_future_video_timestep_grid(
                    batch_size,
                    visual_outputs.frontend.token_grid.num_frames,
                    float(video_timestep),
                    device,
                    observed_prefix_frames,
                ),
                constant_action_timestep_grid(
                    batch_size,
                    float(action_timestep),
                    device,
                ),
                input_cache_state,
                step_cache_update,
            )
            video_flow_pred = unpatchify_video_prediction(projected_outputs["video_patch_flow"])
            action_flow_pred = projected_outputs["action_flow"]
            conditioned_video_flow_pred = video_flow_pred
            conditioned_action_flow_pred = action_flow_pred

            if guidance.enabled and unconditional_conditioning is not None:
                _, _, uncond_projected_outputs, _, _, _ = run_unconditioned_core(
                    noisy_visual_outputs.frontend.video_tokens,
                    noisy_actions,
                    constant_future_video_timestep_grid(
                        batch_size,
                        visual_outputs.frontend.token_grid.num_frames,
                        float(video_timestep),
                        device,
                        observed_prefix_frames,
                    ),
                    constant_action_timestep_grid(
                        batch_size,
                        float(action_timestep),
                        device,
                    ),
                    input_cache_state,
                    visual_tower.build_runtime_cache_update_metadata(
                        input_cache_state,
                        current_start_frame=int(current_start_frame),
                        update_kv_cache=False,
                        update_cross_attention_cache=False,
                        cfg_mode=guidance.cfg_mode,
                        cache_branch=unconditioned_cache_branch,
                    ),
                    unconditional_conditioning,
                )
                uncond_video_flow_pred = unpatchify_video_prediction(uncond_projected_outputs["video_patch_flow"])
                uncond_action_flow_pred = uncond_projected_outputs["action_flow"]
                unconditioned_video_flow_pred = uncond_video_flow_pred
                unconditioned_action_flow_pred = uncond_action_flow_pred
                video_flow_pred, action_flow_pred = combine_joint_cfg_predictions(
                    conditioned_video_prediction=video_flow_pred,
                    unconditioned_video_prediction=uncond_video_flow_pred,
                    conditioned_action_prediction=action_flow_pred,
                    unconditioned_action_prediction=uncond_action_flow_pred,
                    guidance=guidance,
                )
            _record_joint_prediction(
                prediction_reuse_state,
                timestep=video_timestep,
                video_flow_pred=video_flow_pred,
                action_flow_pred=action_flow_pred,
            )
        else:
            assert prediction_reuse_state.previous_predictions, "Prediction reuse requires cached predictions."
            _, video_flow_pred, action_flow_pred = prediction_reuse_state.previous_predictions[-1]
            video_flow_pred = video_flow_pred.to(device=device, dtype=noisy_video_latents.dtype)
            action_flow_pred = action_flow_pred.to(device=device, dtype=noisy_actions.dtype)

        debug_entry: dict[str, float | int | bool] = {
            "step_index": int(step_index),
            "ran_model": bool(should_run_model),
            "video_timestep": float(video_timestep),
            "action_timestep": float(action_timestep),
            "video_flow_abs_mean": float(video_flow_pred.detach().float().abs().mean().item()),
            "video_flow_std": float(video_flow_pred.detach().float().std().item()),
            "video_flow_max_abs": float(video_flow_pred.detach().float().abs().max().item()),
            "action_flow_abs_mean": float(action_flow_pred.detach().float().abs().mean().item()),
            "action_flow_std": float(action_flow_pred.detach().float().std().item()),
            "action_flow_max_abs": float(action_flow_pred.detach().float().abs().max().item()),
            "noisy_video_abs_mean_before_step": float(noisy_video_latents.detach().float().abs().mean().item()),
            "noisy_video_std_before_step": float(noisy_video_latents.detach().float().std().item()),
            "noisy_actions_abs_mean_before_step": float(noisy_actions.detach().float().abs().mean().item()),
            "noisy_actions_std_before_step": float(noisy_actions.detach().float().std().item()),
        }
        if conditioned_video_flow_pred is not None:
            step_debug_entry_cond = conditioned_video_flow_pred.detach().float()
            debug_entry["conditioned_video_flow_abs_mean"] = float(step_debug_entry_cond.abs().mean().item())
            debug_entry["conditioned_video_flow_std"] = float(step_debug_entry_cond.std().item())
        if conditioned_action_flow_pred is not None:
            step_debug_entry_cond_action = conditioned_action_flow_pred.detach().float()
            debug_entry["conditioned_action_flow_abs_mean"] = float(step_debug_entry_cond_action.abs().mean().item())
            debug_entry["conditioned_action_flow_std"] = float(step_debug_entry_cond_action.std().item())
        if unconditioned_video_flow_pred is not None:
            step_debug_entry_uncond = unconditioned_video_flow_pred.detach().float()
            debug_entry["unconditioned_video_flow_abs_mean"] = float(step_debug_entry_uncond.abs().mean().item())
            debug_entry["unconditioned_video_flow_std"] = float(step_debug_entry_uncond.std().item())
            cfg_delta = (conditioned_video_flow_pred.detach().float() - step_debug_entry_uncond).abs()
            debug_entry["video_cfg_delta_abs_mean"] = float(cfg_delta.mean().item())
            debug_entry["video_cfg_delta_max_abs"] = float(cfg_delta.max().item())
        if unconditioned_action_flow_pred is not None:
            step_debug_entry_uncond_action = unconditioned_action_flow_pred.detach().float()
            debug_entry["unconditioned_action_flow_abs_mean"] = float(step_debug_entry_uncond_action.abs().mean().item())
            debug_entry["unconditioned_action_flow_std"] = float(step_debug_entry_uncond_action.std().item())
            cfg_delta_action = (conditioned_action_flow_pred.detach().float() - step_debug_entry_uncond_action).abs()
            debug_entry["action_cfg_delta_abs_mean"] = float(cfg_delta_action.mean().item())
            debug_entry["action_cfg_delta_max_abs"] = float(cfg_delta_action.max().item())

        if scheduler_bundle.use_unipc:
            noisy_video_latents = video_scheduler.step(
                video_flow_pred,
                video_timestep,
                noisy_video_latents,
                step_index=step_index,
                return_dict=False,
            )[0]
            noisy_video_latents = preserve_observed_video_prefix(
                noisy_video_latents,
                observed_video_latents,
                observed_prefix_frames,
            )
            noisy_actions = action_scheduler.step(
                action_flow_pred,
                action_timestep,
                noisy_actions,
                step_index=step_index,
                return_dict=False,
            )[0]
        else:
            noisy_video_latents = video_scheduler.step(
                video_flow_pred,
                video_timestep,
                noisy_video_latents,
                to_final=step_index == len(video_scheduler.timesteps) - 1,
            )
            noisy_video_latents = preserve_observed_video_prefix(
                noisy_video_latents,
                observed_video_latents,
                observed_prefix_frames,
            )
            noisy_actions = action_scheduler.step(
                action_flow_pred,
                action_timestep,
                noisy_actions,
                to_final=step_index == len(action_scheduler.timesteps) - 1,
            )
        debug_entry["noisy_video_abs_mean_after_step"] = float(noisy_video_latents.detach().float().abs().mean().item())
        debug_entry["noisy_video_std_after_step"] = float(noisy_video_latents.detach().float().std().item())
        debug_entry["noisy_actions_abs_mean_after_step"] = float(noisy_actions.detach().float().abs().mean().item())
        debug_entry["noisy_actions_std_after_step"] = float(noisy_actions.detach().float().std().item())
        step_debug.append(debug_entry)

    core_aux = {**core_aux, "joint_inference_step_debug": step_debug}
    return JointInferenceLoopResult(
        noisy_video_latents=noisy_video_latents,
        noisy_actions=noisy_actions,
        latest_core_cache=latest_core_cache,
        layout=layout,
        core_aux=core_aux,
        guidance_enabled=guidance.enabled,
        guidance_cfg_mode=guidance.cfg_mode,
        video_num_inference_steps=len(video_scheduler.timesteps),
        action_num_inference_steps=len(action_scheduler.timesteps),
    )
