"""Historical MoT runtime-routing import surface.

Maintained code imports the role-owned modules directly. This facade preserves
existing direct imports, wildcard imports, and pickle lookup paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

from open_wam.configs import (
    CurrentBlockCoupling,
    JointTimestepCoupling,
    MoTRuntimeMode,
    PolicyVariantName,
)
from open_wam.configs.enums import RolloutContextPolicy, SampleTargetAlignment
from open_wam.configs.policy_mot import MoTPolicyConfig

from .coupling_semantics import (
    is_mot_same_step_coupling,
    resolve_mot_current_block_coupling,
    resolve_mot_joint_timestep_coupling,
    should_couple_mot_action_to_video_sigmas,
)
from .inference_backend import (
    ensure_mot_inference_backend,
    ensure_mot_policy_variant_inference_backend,
)
from .rollout_geometry import (
    MOT_ACTION_ONLY_ROLLOUT_COUPLINGS,
    _InferenceContextLike,
    mot_config_uses_strict_rollout_parity,
    resolve_mot_action_only_rollout,
    resolve_mot_inference_window_size,
    resolve_mot_rollout_cache_window_frames,
    resolve_mot_rollout_frame_chunk_size,
    resolve_mot_rollout_history_frames,
    resolve_mot_sequence_actions_per_frame,
    resolve_mot_sequence_execution_action_offset,
)
from .runtime_routes import (
    MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS,
    MoTRuntimeRoute,
    MoTRuntimeRouteKind,
    _coerce_current_block_coupling,
    _coerce_runtime_mode,
    _enum_value,
    _looks_like_mot_policy_config,
    _policy_config,
    _resolve_current_block_coupling,
    mot_policy_requires_legacy_split_cache_inference,
    resolve_mot_runtime_route,
    should_use_mot_legacy_split_cache_inference,
)

(
    Any,
    CurrentBlockCoupling,
    Enum,
    JointTimestepCoupling,
    MOT_ACTION_ONLY_ROLLOUT_COUPLINGS,
    MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS,
    Mapping,
    MoTPolicyConfig,
    MoTRuntimeMode,
    MoTRuntimeRoute,
    MoTRuntimeRouteKind,
    PolicyVariantName,
    Protocol,
    RolloutContextPolicy,
    SampleTargetAlignment,
    _InferenceContextLike,
    _coerce_current_block_coupling,
    _coerce_runtime_mode,
    _enum_value,
    _looks_like_mot_policy_config,
    _policy_config,
    _resolve_current_block_coupling,
    dataclass,
    ensure_mot_inference_backend,
    ensure_mot_policy_variant_inference_backend,
    is_mot_same_step_coupling,
    mot_config_uses_strict_rollout_parity,
    mot_policy_requires_legacy_split_cache_inference,
    resolve_mot_action_only_rollout,
    resolve_mot_current_block_coupling,
    resolve_mot_inference_window_size,
    resolve_mot_joint_timestep_coupling,
    resolve_mot_rollout_cache_window_frames,
    resolve_mot_rollout_frame_chunk_size,
    resolve_mot_rollout_history_frames,
    resolve_mot_runtime_route,
    resolve_mot_sequence_actions_per_frame,
    resolve_mot_sequence_execution_action_offset,
    should_couple_mot_action_to_video_sigmas,
    should_use_mot_legacy_split_cache_inference,
)
