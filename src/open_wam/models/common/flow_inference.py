"""Configured flow-matching inference scheduler construction."""

from __future__ import annotations

import torch

from open_wam.configs import InferenceConfig, TrainingConfig

from .flow_schedule import FlowMatchScheduler
from .flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler


def build_action_flow_match_inference_scheduler(
    *,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    num_inference_steps_override: int | None = None,
) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    scheduler.set_timesteps(num_inference_steps_override or inference_config.action_num_inference_steps)
    return scheduler


def build_video_flow_match_inference_scheduler(
    *,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    num_inference_steps_override: int | None = None,
) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    scheduler.set_timesteps(num_inference_steps_override or inference_config.video_num_inference_steps)
    return scheduler


def build_flow_unipc_inference_scheduler(
    *,
    num_train_timesteps: int,
    sigma_shift: float,
    num_inference_steps: int,
    device: torch.device,
) -> FlowUniPCMultistepScheduler:
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1.0,
    )
    scheduler.set_timesteps(
        num_inference_steps,
        device=device,
        shift=float(sigma_shift),
    )
    return scheduler


__all__ = [
    "build_action_flow_match_inference_scheduler",
    "build_video_flow_match_inference_scheduler",
    "build_flow_unipc_inference_scheduler",
]
