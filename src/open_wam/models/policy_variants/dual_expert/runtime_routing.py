"""Historical DualExpert runtime-routing import surface.

Maintained code imports the role-owned modules directly. This facade preserves
existing direct imports, wildcard imports, and pickle lookup paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from open_wam.configs import (
    CurrentBlockCoupling,
    DualExpertRuntimeMode,
    JointTimestepCoupling,
    PolicyVariantName,
)
from open_wam.configs.enums import RolloutContextPolicy, SampleTargetAlignment
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig

from .coupling_semantics import (
    is_dual_expert_same_step_coupling,
    resolve_dual_expert_current_block_coupling,
    resolve_dual_expert_joint_timestep_coupling,
    should_couple_dual_expert_action_to_video_sigmas,
)
from .inference_backend import (
    ensure_dual_expert_inference_backend,
    ensure_dual_expert_policy_variant_inference_backend,
)
from .rollout_geometry import (
    DUAL_EXPERT_ACTION_ONLY_ROLLOUT_COUPLINGS,
    _InferenceContextLike,
    dual_expert_config_uses_strict_rollout_parity,
    resolve_dual_expert_action_only_rollout,
    resolve_dual_expert_inference_window_size,
    resolve_dual_expert_rollout_cache_window_frames,
    resolve_dual_expert_rollout_frame_chunk_size,
    resolve_dual_expert_rollout_history_frames,
    resolve_dual_expert_sequence_actions_per_frame,
    resolve_dual_expert_sequence_execution_action_offset,
)
from .runtime_routes import (
    DUAL_EXPERT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS,
    DualExpertRuntimeRoute,
    DualExpertRuntimeRouteKind,
    _coerce_current_block_coupling,
    _coerce_runtime_mode,
    _enum_value,
    _looks_like_dual_expert_policy_config,
    _policy_config,
    _resolve_current_block_coupling,
    dual_expert_policy_requires_legacy_split_cache_inference,
    resolve_dual_expert_runtime_route,
    should_use_dual_expert_legacy_split_cache_inference,
)

(
    Any,
    CurrentBlockCoupling,
    Enum,
    JointTimestepCoupling,
    DUAL_EXPERT_ACTION_ONLY_ROLLOUT_COUPLINGS,
    DUAL_EXPERT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS,
    Mapping,
    DualExpertPolicyConfig,
    DualExpertRuntimeMode,
    DualExpertRuntimeRoute,
    DualExpertRuntimeRouteKind,
    PolicyVariantName,
    Protocol,
    RolloutContextPolicy,
    SampleTargetAlignment,
    _InferenceContextLike,
    _coerce_current_block_coupling,
    _coerce_runtime_mode,
    _enum_value,
    _looks_like_dual_expert_policy_config,
    _policy_config,
    _resolve_current_block_coupling,
    dataclass,
    ensure_dual_expert_inference_backend,
    ensure_dual_expert_policy_variant_inference_backend,
    is_dual_expert_same_step_coupling,
    dual_expert_config_uses_strict_rollout_parity,
    dual_expert_policy_requires_legacy_split_cache_inference,
    resolve_dual_expert_action_only_rollout,
    resolve_dual_expert_current_block_coupling,
    resolve_dual_expert_inference_window_size,
    resolve_dual_expert_joint_timestep_coupling,
    resolve_dual_expert_rollout_cache_window_frames,
    resolve_dual_expert_rollout_frame_chunk_size,
    resolve_dual_expert_rollout_history_frames,
    resolve_dual_expert_runtime_route,
    resolve_dual_expert_sequence_actions_per_frame,
    resolve_dual_expert_sequence_execution_action_offset,
    should_couple_dual_expert_action_to_video_sigmas,
    should_use_dual_expert_legacy_split_cache_inference,
)
