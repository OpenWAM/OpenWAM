"""Compatibility imports for the historical attention-profile module.

New package code should import role owners from ``attention_contracts``,
``attention_backends``, or ``chunked_attention``.
"""

from __future__ import annotations

from dataclasses import dataclass as dataclass, field as field
from typing import Any as Any

import torch as torch

from open_wam.models.common.attention_backends import (
    BlockMask as BlockMask,
    _COMPILED_CREATE_BLOCK_MASK as _COMPILED_CREATE_BLOCK_MASK,
    _COMPILED_FLEX_ATTENTION as _COMPILED_FLEX_ATTENTION,
    _resolve_compiled_create_block_mask as _resolve_compiled_create_block_mask,
    _resolve_compiled_flex_attention as _resolve_compiled_flex_attention,
    apply_attention_backend as apply_attention_backend,
    create_block_mask as create_block_mask,
    flex_attention as flex_attention,
    resolve_attention_profile_backend as resolve_attention_profile_backend,
    select_attention_profile_mask as select_attention_profile_mask,
)
from open_wam.models.common.attention_contracts import (
    ACTION_NOISY_TO_VIDEO_COUPLING as ACTION_NOISY_TO_VIDEO_COUPLING,
    ACTION_THEN_VIDEO_COUPLING as ACTION_THEN_VIDEO_COUPLING,
    AttentionProfileSpec as AttentionProfileSpec,
    CONDITIONAL_HISTORY_POLICY_NONE as CONDITIONAL_HISTORY_POLICY_NONE,
    CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY as CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
    DECOUPLED_SAME_STEP_COUPLING as DECOUPLED_SAME_STEP_COUPLING,
    HISTORY_STREAM_VISIBILITY_FULL as HISTORY_STREAM_VISIBILITY_FULL,
    HISTORY_STREAM_VISIBILITY_VIDEO_ONLY as HISTORY_STREAM_VISIBILITY_VIDEO_ONLY,
    HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY as HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY,
    JOINT_COUPLING as JOINT_COUPLING,
    PreparedAttentionProfile as PreparedAttentionProfile,
    VIDEO_NOISY_TO_ACTION_COUPLING as VIDEO_NOISY_TO_ACTION_COUPLING,
    VIDEO_THEN_ACTION_COUPLING as VIDEO_THEN_ACTION_COUPLING,
    _ATTENTION_PROFILE_ALIASES as _ATTENTION_PROFILE_ALIASES,
    _CHUNKED_EXACT_COUPLING_BY_PROFILE as _CHUNKED_EXACT_COUPLING_BY_PROFILE,
    _CHUNKED_EXACT_PROFILE_BY_COUPLING as _CHUNKED_EXACT_PROFILE_BY_COUPLING,
    _CONDITIONAL_HISTORY_POLICY_VALUES as _CONDITIONAL_HISTORY_POLICY_VALUES,
    _HISTORY_STREAM_VISIBILITY_VALUES as _HISTORY_STREAM_VISIBILITY_VALUES,
    chunked_temporal_exact_coupling_from_profile_name as chunked_temporal_exact_coupling_from_profile_name,
    chunked_temporal_exact_profile_name_for_coupling as chunked_temporal_exact_profile_name_for_coupling,
    normalize_attention_profile_name as normalize_attention_profile_name,
    normalize_chunked_temporal_exact_coupling as normalize_chunked_temporal_exact_coupling,
    normalize_conditional_history_policy as normalize_conditional_history_policy,
    normalize_parallel_history_stream_visibility as normalize_parallel_history_stream_visibility,
)
from open_wam.models.common.chunked_attention import (
    _effective_frame_ids_for_singleton_cutoff as _effective_frame_ids_for_singleton_cutoff,
    _previous_boundary_frame_ids as _previous_boundary_frame_ids,
    build_chunked_temporal_exact_attention_profile as build_chunked_temporal_exact_attention_profile,
    build_chunked_text_context_cross_attention_mask as build_chunked_text_context_cross_attention_mask,
    build_lingbot_chunked_exact_attention_profile as build_lingbot_chunked_exact_attention_profile,
)
from open_wam.models.common.packed_token_layout import (
    PackedTokenStream as PackedTokenStream,
    build_exact_video_action_token_layout as build_exact_video_action_token_layout,
)

# Keep the established direct and wildcard import surface visible to static lint
# without adding a compatibility-only module global.
(
    ACTION_NOISY_TO_VIDEO_COUPLING,
    ACTION_THEN_VIDEO_COUPLING,
    Any,
    AttentionProfileSpec,
    BlockMask,
    CONDITIONAL_HISTORY_POLICY_NONE,
    CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
    DECOUPLED_SAME_STEP_COUPLING,
    HISTORY_STREAM_VISIBILITY_FULL,
    HISTORY_STREAM_VISIBILITY_VIDEO_ONLY,
    HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY,
    JOINT_COUPLING,
    PackedTokenStream,
    PreparedAttentionProfile,
    VIDEO_NOISY_TO_ACTION_COUPLING,
    VIDEO_THEN_ACTION_COUPLING,
    _ATTENTION_PROFILE_ALIASES,
    _CHUNKED_EXACT_COUPLING_BY_PROFILE,
    _CHUNKED_EXACT_PROFILE_BY_COUPLING,
    _COMPILED_CREATE_BLOCK_MASK,
    _COMPILED_FLEX_ATTENTION,
    _CONDITIONAL_HISTORY_POLICY_VALUES,
    _HISTORY_STREAM_VISIBILITY_VALUES,
    _effective_frame_ids_for_singleton_cutoff,
    _previous_boundary_frame_ids,
    _resolve_compiled_create_block_mask,
    _resolve_compiled_flex_attention,
    apply_attention_backend,
    build_chunked_temporal_exact_attention_profile,
    build_chunked_text_context_cross_attention_mask,
    build_exact_video_action_token_layout,
    build_lingbot_chunked_exact_attention_profile,
    chunked_temporal_exact_coupling_from_profile_name,
    chunked_temporal_exact_profile_name_for_coupling,
    create_block_mask,
    dataclass,
    field,
    flex_attention,
    normalize_attention_profile_name,
    normalize_chunked_temporal_exact_coupling,
    normalize_conditional_history_policy,
    normalize_parallel_history_stream_visibility,
    resolve_attention_profile_backend,
    select_attention_profile_mask,
    torch,
)
