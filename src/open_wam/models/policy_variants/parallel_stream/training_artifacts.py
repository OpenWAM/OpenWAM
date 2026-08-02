"""Historical aggregate for parallel-stream training artifact builders.

Maintained code imports the layout-specific owners directly. This module keeps
the checkpoint-era import, wildcard, and pickle surface stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange

from open_wam.configs.backbone import (
    SharedVideoTransformerConfig,
    resolve_stage_attention_mode,
)
from open_wam.configs.enums import (
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelContextConditionLatentSource,
    ParallelStreamVariantProfile,
)
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common.flow_schedule import FlowMatchScheduler
from open_wam.models.common.flow_noise_plan import (
    clean_timestep_values,
    sample_joint_denoise_timestep_values,
)
from open_wam.models.common.modality_slots import (
    force_clean_noisy_slot,
    zero_condition_slot,
)
from open_wam.models.visual_tower.reference_transformer import preferred_reference_dtype

from .generalist_training import (
    apply_generalist_joint_denoise_training_mode as _apply_generalist_joint_denoise_training_mode,
    apply_generalist_legacy_prefix_joint_training_mode as _apply_generalist_legacy_prefix_joint_training_mode,
)
from .latent_conditioning import (
    resolve_full_window_condition_latents as _resolve_full_condition_latents,
    select_first_frame_condition_latents as _select_first_frame_condition_latents,
)
from .runtime_semantics import (
    attention_profile_name_for_current_block_coupling as _attention_profile_name_for_current_block_coupling,
    resolve_parallel_context_condition_latent_source,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
)
from .training_artifact_contracts import (
    LingbotParallelTrainArtifacts,
    ParallelTrainArtifacts,
)
from .training_exact_artifacts import (
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_exact_train_artifacts,
)
from .training_noise import (
    build_parallel_flow_noise_artifacts as _add_noise,
    sample_coupled_parallel_timestep_values as _sample_coupled_timestep_values,
    sample_index_matched_timestep_values as _sample_index_matched_timestep_values,
    sample_shared_video_schedule_timestep_values as _sample_shared_video_schedule_timestep_values,
    share_video_scheduler_grid_with_action_scheduler as _share_video_scheduler_grid_with_action_scheduler,
)
from .training_prefix_artifacts import (
    prepare_parallel_prefix_condition_exact_train_artifacts,
)
from .training_single_frame_artifacts import (
    prepare_parallel_current_frame_action_chunk_train_artifacts,
    prepare_parallel_fastwam_first_frame_train_artifacts,
)

(
    CurrentBlockCoupling,
    FlowMatchScheduler,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelContextConditionLatentSource,
    ParallelStreamPolicyConfig,
    ParallelStreamVariantProfile,
    SharedVideoTransformerConfig,
    TrainingConfig,
    _add_noise,
    _apply_generalist_joint_denoise_training_mode,
    _apply_generalist_legacy_prefix_joint_training_mode,
    _attention_profile_name_for_current_block_coupling,
    _resolve_full_condition_latents,
    _sample_coupled_timestep_values,
    _sample_index_matched_timestep_values,
    _sample_shared_video_schedule_timestep_values,
    _select_first_frame_condition_latents,
    _share_video_scheduler_grid_with_action_scheduler,
    clean_timestep_values,
    dataclass,
    force_clean_noisy_slot,
    preferred_reference_dtype,
    rearrange,
    resolve_parallel_context_condition_latent_source,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
    resolve_stage_attention_mode,
    sample_joint_denoise_timestep_values,
    torch,
    zero_condition_slot,
)

__all__ = [
    "LingbotParallelTrainArtifacts",
    "ParallelTrainArtifacts",
    "prepare_parallel_action_conditioned_train_artifacts",
    "prepare_parallel_current_frame_action_chunk_train_artifacts",
    "prepare_parallel_exact_train_artifacts",
    "prepare_parallel_fastwam_first_frame_train_artifacts",
    "prepare_parallel_prefix_condition_exact_train_artifacts",
]
