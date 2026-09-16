"""Model-space denoising shared by video/action transformer architectures.

Programs own dependencies and schedules. The supplied predictor owns model
projections and transformer execution; it never advances a rollout timeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from open_wam.configs import (
    CurrentBlockCoupling,
    InferenceConfig,
    JointTimestepCoupling,
    TrainingConfig,
)
from open_wam.configs.enums import PolicyOutputModality
from open_wam.models.common.attention_contracts import PreparedAttentionProfile
from open_wam.models.common.denoising import (
    DenoisingStage,
    denoise_stages,
    resolve_denoising_stages,
)
from open_wam.models.common.denoising_cache import (
    DenoisingCache,
    attention_dependency_closure,
)
from open_wam.models.common.packed_token_layout import PackedTokenStream
from open_wam.models.common.dynamics_objectives import DynamicsRolloutPlan
from open_wam.models.common.flow_inference import (
    build_action_flow_match_inference_scheduler,
    build_video_flow_match_inference_scheduler,
)
from open_wam.models.common.flow_schedule import (
    explicit_sigma_euler_step,
    timesteps_matching_sigmas,
    zero_terminal_next_sigma,
)
from open_wam.models.common.runtime_controls import combine_cfg_prediction
from open_wam.configs.enums import CFGMode


@dataclass(frozen=True)
class VideoActionFlowInput:
    """One model evaluation, with all positions and conditioning explicit."""

    noisy_video: torch.Tensor
    clean_video: torch.Tensor
    video_timesteps: torch.Tensor
    noisy_action: torch.Tensor
    clean_action: torch.Tensor
    action_timesteps: torch.Tensor
    attention: PreparedAttentionProfile
    text_context: torch.Tensor
    frame_start: int
    proprio_frames: torch.Tensor | None = None
    required_tokens: torch.Tensor | None = None


FlowPrediction = tuple[torch.Tensor, torch.Tensor]
FlowPredictor = Callable[[VideoActionFlowInput, DenoisingCache | None], FlowPrediction]


@torch.no_grad()
def denoise_video_action(
    prepared: VideoActionFlowInput,
    *,
    predict: FlowPredictor,
    training: TrainingConfig,
    inference: InferenceConfig,
    coupling: CurrentBlockCoupling,
    timestep_coupling: JointTimestepCoupling,
    dynamics: DynamicsRolloutPlan,
    video_start: int,
    action_start: int,
    negative_text_context: torch.Tensor | None = None,
    requested: frozenset[PolicyOutputModality] | None = None,
    supplied: frozenset[PolicyOutputModality] = frozenset(),
    initial_video_noise: torch.Tensor | None = None,
    initial_action_noise: torch.Tensor | None = None,
) -> FlowPrediction:
    """Denoise prepared streams with a single stage/CFG/cache lifecycle.

    Start offsets exclude observed prefixes from integration. Clean-slot
    promotion is explicit for ordered programs; conditional objectives instead
    keep their fixed input in the live slot, as in training.
    """
    video, action = PolicyOutputModality.VIDEO, PolicyOutputModality.ACTION
    schedulers = {
        video: build_video_flow_match_inference_scheduler(
            training_config=training, inference_config=inference
        ),
        action: build_action_flow_match_inference_scheduler(
            training_config=training, inference_config=inference
        ),
    }
    semantic_stages = resolve_denoising_stages(coupling=coupling, dynamics=dynamics)
    shared_sigma_schedule = (
        any(len(stage.updates) > 1 for stage in semantic_stages)
        and timestep_coupling in {
            JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
            JointTimestepCoupling.MATCH_SIGMA,
        }
    )
    stages = resolve_denoising_stages(
        coupling=coupling,
        dynamics=dynamics,
        requested=requested,
        supplied=supplied,
    )
    for stage in semantic_stages:
        if len(stage.updates) > 1 and len(schedulers[video].timesteps) != len(
            schedulers[action].timesteps
        ):
            raise ValueError(
                "Interacting denoising streams require equal inference step counts."
            )
    lookup = (
        build_action_flow_match_inference_scheduler(
            training_config=training,
            inference_config=inference,
            num_inference_steps_override=training.action_num_train_timesteps,
        )
        if timestep_coupling is JointTimestepCoupling.MATCH_SIGMA
        else None
    )
    cache: DenoisingCache | None = None
    negative_cache: DenoisingCache | None = None
    clean_video, clean_action = prepared.clean_video, prepared.clean_action
    required_tokens = None
    guidance = {
        video: (inference.video_cfg_mode, inference.guidance_scale),
        action: (inference.action_cfg_mode, inference.action_guidance_scale),
    }

    def prepare_stage(stage, samples):
        nonlocal cache, negative_cache, clean_video, clean_action, required_tokens
        cache = DenoisingCache() if inference.use_cache else None
        negative_cache = DenoisingCache() if inference.use_cache else None
        clean_video = (
            samples[0] if video in stage.clean_conditions else prepared.clean_video
        )
        clean_action = (
            samples[1] if action in stage.clean_conditions else prepared.clean_action
        )
        layout = prepared.attention.token_layout
        if layout is None:
            raise ValueError(
                "Video/action execution requires explicit token layout metadata."
            )
        action_per_frame = (
            prepared.noisy_action.shape[1] // prepared.noisy_video.shape[2]
        )
        roots = (layout.noise_id == 0) & (
            (
                (layout.stream_id == PackedTokenStream.VIDEO)
                & (layout.frame_id >= video_start)
                & (video in stage.updates)
            )
            | (
                (layout.stream_id == PackedTokenStream.ACTION)
                & (layout.frame_id >= action_start // action_per_frame)
                & (action in stage.updates)
            )
        )
        required_tokens = attention_dependency_closure(prepared.attention, roots)
        initialized = []
        for modality, sample, start, dim, noise in zip(
            (video, action),
            samples,
            (video_start, action_start),
            (2, 1),
            (initial_video_noise, initial_action_noise),
            strict=True,
        ):
            if modality in stage.updates:
                prefix = sample.narrow(dim, 0, start)
                target = sample.narrow(dim, start, sample.shape[dim] - start)
                if noise is not None and noise.shape != target.shape:
                    raise ValueError(
                        "Explicit initial noise must match the generated suffix shape."
                    )
                sample = torch.cat(
                    (
                        prefix,
                        torch.randn_like(target) if noise is None else noise.to(target),
                    ),
                    dim=dim,
                )
            initialized.append(sample)
        return tuple(initialized)

    def schedule(stage):
        clock = video if shared_sigma_schedule else stage.updates[0]
        return range(len(schedulers[clock].timesteps))

    def times(stage, index):
        values = {
            modality: schedulers[modality].timesteps[index]
            for modality in stage.updates
        }
        sigmas = None
        # Output pruning removes work, not the original program's noise clock.
        if shared_sigma_schedule:
            sigma = schedulers[video].sigmas[index]
            sigma_next = zero_terminal_next_sigma(schedulers[video], index)
            sigmas = (sigma, sigma_next)
            if action in stage.updates:
                values[action] = (
                    timesteps_matching_sigmas(lookup, sigma.reshape(1))[0]
                    if lookup is not None
                    else schedulers[video].timesteps[index]
                )
        return values, sigmas

    def evaluate(stage: DenoisingStage, samples, index):
        values, sigmas = times(stage, index)
        vt = torch.zeros_like(prepared.video_timesteps)
        at = torch.zeros_like(prepared.action_timesteps)
        if video in values:
            vt[:, video_start:] = values[video]
        if action in values:
            at[:, action_start:] = values[action]
        step = replace(
            prepared,
            noisy_video=samples[0],
            noisy_action=samples[1],
            clean_video=clean_video,
            clean_action=clean_action,
            video_timesteps=vt,
            action_timesteps=at,
            required_tokens=required_tokens,
        )
        positive = predict(step, cache)
        needs_negative = (
            any(
                guidance[modality][0] is CFGMode.UNCONDITIONED
                or (
                    guidance[modality][0] is CFGMode.GUIDED
                    and guidance[modality][1] != 1.0
                )
                for modality in stage.updates
            )
            and not dynamics.semantics.drop_text_conditioning
        )
        if needs_negative:
            if negative_text_context is None:
                raise ValueError(
                    "Requested classifier-free guidance requires negative text context."
                )
            negative = predict(
                replace(step, text_context=negative_text_context),
                negative_cache,
            )
            predictions = []
            for modality, conditioned, unconditioned in zip(
                (video, action), positive, negative, strict=True
            ):
                mode, scale = guidance[modality]
                predictions.append(
                    unconditioned
                    if mode is CFGMode.UNCONDITIONED
                    else combine_cfg_prediction(
                        conditioned, unconditioned, guidance_scale=scale
                    )
                    if mode is CFGMode.GUIDED and modality in stage.updates
                    else conditioned
                )
            positive = tuple(predictions)
        return positive, values, sigmas

    def update(stage, prediction, index, samples):
        flows, values, sigmas = prediction
        results = []
        for modality, sample, flow, start, dim in zip(
            (video, action),
            samples,
            flows,
            (video_start, action_start),
            (2, 1),
            strict=True,
        ):
            if modality not in stage.updates:
                results.append(sample)
                continue
            suffix = [slice(None)] * sample.ndim
            suffix[dim] = slice(start, None)
            suffix = tuple(suffix)
            current = sample[suffix]
            if sigmas is None:
                current = schedulers[modality].step(
                    flow[suffix], values[modality], current
                )
            else:
                current = explicit_sigma_euler_step(
                    current, flow[suffix], sigma=sigmas[0], sigma_next=sigmas[1]
                )
            prefix = sample.narrow(dim, 0, start)
            results.append(torch.cat((prefix, current), dim=dim))
        return tuple(results)

    return denoise_stages(
        (prepared.noisy_video, prepared.noisy_action),
        stages,
        timesteps=schedule,
        prepare=prepare_stage,
        predict=evaluate,
        update=update,
    )
