"""Stable compatibility facade for flow-matching utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from open_wam.configs import InferenceConfig, TrainingConfig

from .flow_inference import (
    build_action_flow_match_inference_scheduler as build_action_flow_match_inference_scheduler,
    build_flow_unipc_inference_scheduler as build_flow_unipc_inference_scheduler,
    build_video_flow_match_inference_scheduler as build_video_flow_match_inference_scheduler,
)
from .flow_schedule import (
    FlowMatchScheduler as FlowMatchScheduler,
    expand_scalar_timestep as expand_scalar_timestep,
    explicit_sigma_euler_step as explicit_sigma_euler_step,
    sample_timestep_id as sample_timestep_id,
    timesteps_matching_sigmas as timesteps_matching_sigmas,
    zero_terminal_next_sigma as zero_terminal_next_sigma,
)
from .flow_supervision import (
    denoised_actions_from_flow as denoised_actions_from_flow,
    denoised_video_latents_from_flow as denoised_video_latents_from_flow,
    reduce_frame_aligned_action_flow_match_loss as reduce_frame_aligned_action_flow_match_loss,
    reduce_slot_aligned_action_flow_match_loss as reduce_slot_aligned_action_flow_match_loss,
    reduce_video_flow_match_loss as reduce_video_flow_match_loss,
)
from .flow_training import (
    ActionFlowMatchTrainArtifacts as ActionFlowMatchTrainArtifacts,
    BlockCoupledActionFlowMatchTrainArtifacts as BlockCoupledActionFlowMatchTrainArtifacts,
    FrameAlignedActionFlowMatchTrainArtifacts as FrameAlignedActionFlowMatchTrainArtifacts,
    VideoFlowMatchTrainArtifacts as VideoFlowMatchTrainArtifacts,
    build_action_flow_match_train_artifacts as build_action_flow_match_train_artifacts,
    build_block_coupled_action_flow_match_train_artifacts as build_block_coupled_action_flow_match_train_artifacts,
    build_frame_aligned_action_flow_match_train_artifacts as build_frame_aligned_action_flow_match_train_artifacts,
    build_video_flow_match_train_artifacts as build_video_flow_match_train_artifacts,
)
from .flow_unipc_multistep_scheduler import (
    FlowUniPCMultistepScheduler as FlowUniPCMultistepScheduler,
)


# Preserve the historical wildcard-import surface.
__all__ = [
    "ActionFlowMatchTrainArtifacts",
    "BlockCoupledActionFlowMatchTrainArtifacts",
    "FlowMatchScheduler",
    "FlowUniPCMultistepScheduler",
    "FrameAlignedActionFlowMatchTrainArtifacts",
    "InferenceConfig",
    "TrainingConfig",
    "VideoFlowMatchTrainArtifacts",
    "annotations",
    "build_action_flow_match_inference_scheduler",
    "build_action_flow_match_train_artifacts",
    "build_block_coupled_action_flow_match_train_artifacts",
    "build_flow_unipc_inference_scheduler",
    "build_frame_aligned_action_flow_match_train_artifacts",
    "build_video_flow_match_inference_scheduler",
    "build_video_flow_match_train_artifacts",
    "dataclass",
    "denoised_actions_from_flow",
    "denoised_video_latents_from_flow",
    "expand_scalar_timestep",
    "explicit_sigma_euler_step",
    "math",
    "reduce_frame_aligned_action_flow_match_loss",
    "reduce_slot_aligned_action_flow_match_loss",
    "reduce_video_flow_match_loss",
    "sample_timestep_id",
    "timesteps_matching_sigmas",
    "torch",
    "zero_terminal_next_sigma",
]
