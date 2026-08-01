"""Static validation rules for data, action, and sample contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .enums import (
    PaddedTargetPolicy,
    RolloutContextPolicy,
    SampleOrderMode,
    SampleStateAnchorMode,
    SampleTargetAlignment,
    SegmentContextPolicy,
    TailPaddingPolicy,
    WindowSamplingMode,
)
from .static_validation_contracts import _IssueBuilder
from .static_validation_primitives import _optional_int, _validate_enum, _validate_positive_ints


def _validate_action_mapping(
    action_mapping: Mapping[str, Any],
    action_schema: Mapping[str, Any] | None,
    issues: "_IssueBuilder",
) -> None:
    mode = action_mapping.get("mode", "none")
    if mode == "none":
        return
    source_dim = _optional_int(action_mapping.get("source_dim"))
    target_dim = _optional_int(action_mapping.get("target_dim"))
    if source_dim is None or source_dim <= 0:
        issues.error("data.action_mapping.source_dim", "Expected a positive integer when action mapping is active.")
    if target_dim is None or target_dim <= 0:
        issues.error("data.action_mapping.target_dim", "Expected a positive integer when action mapping is active.")
    indices = action_mapping.get("source_to_target_indices", ())
    if not isinstance(indices, list):
        issues.error("data.action_mapping.source_to_target_indices", "Expected a list of integer target indices.")
        return
    if source_dim is not None and len(indices) != source_dim:
        issues.error(
            "data.action_mapping.source_to_target_indices",
            f"Expected {source_dim} indices for source_dim={source_dim}, got {len(indices)}.",
        )
    if target_dim is not None:
        invalid = [value for value in indices if not isinstance(value, int) or value < 0 or value >= target_dim]
        if invalid:
            issues.error(
                "data.action_mapping.source_to_target_indices",
                f"Target indices outside target_dim={target_dim}: {invalid}.",
            )
    if len(set(indices)) != len(indices):
        issues.error("data.action_mapping.source_to_target_indices", "Target indices must be unique.")
    if action_schema is not None and target_dim is not None:
        schema_dim = _optional_int(action_schema.get("action_dim"))
        if schema_dim is not None and schema_dim != target_dim:
            issues.error(
                "data.action_mapping.target_dim",
                f"Expected target_dim to match data.action_schema.action_dim={schema_dim}.",
            )


def _validate_action_schema_compatibility(
    action_schema: Mapping[str, Any] | None,
    action_decoder: Mapping[str, Any] | None,
    action_head: Mapping[str, Any] | None,
    issues: "_IssueBuilder",
) -> None:
    if action_schema is None:
        return
    schema_dim = _optional_int(action_schema.get("action_dim"))
    schema_horizon = _optional_int(action_schema.get("action_horizon"))
    for section_name, section in (("action_decoder", action_decoder), ("action_head", action_head)):
        if section is None:
            continue
        decoder_dim = _optional_int(section.get("action_dim"))
        decoder_horizon = _optional_int(section.get("action_horizon"))
        if schema_dim is not None and decoder_dim is not None and decoder_dim != schema_dim:
            issues.warning(
                f"{section_name}.action_dim",
                f"Expected {section_name}.action_dim={decoder_dim} to match "
                f"data.action_schema.action_dim={schema_dim}.",
            )
        if schema_horizon is not None and decoder_horizon is not None and decoder_horizon != schema_horizon:
            issues.error(
                f"{section_name}.action_horizon",
                "Expected "
                f"{section_name}.action_horizon={decoder_horizon} to match "
                f"data.action_schema.action_horizon={schema_horizon}.",
            )


def _validate_sample_construction(
    sample_construction: Mapping[str, Any],
    issues: "_IssueBuilder",
) -> None:
    _validate_enum(sample_construction, "mode", WindowSamplingMode, issues, "data.sample_construction")
    _validate_enum(
        sample_construction,
        "context_prefix_policy",
        SegmentContextPolicy,
        issues,
        "data.sample_construction",
    )
    _validate_enum(sample_construction, "target_alignment", SampleTargetAlignment, issues, "data.sample_construction")
    _validate_enum(
        sample_construction,
        "rollout_context_policy",
        RolloutContextPolicy,
        issues,
        "data.sample_construction",
    )
    _validate_enum(sample_construction, "tail_padding_policy", TailPaddingPolicy, issues, "data.sample_construction")
    _validate_enum(sample_construction, "padded_target_policy", PaddedTargetPolicy, issues, "data.sample_construction")
    _validate_enum(sample_construction, "state_anchor_mode", SampleStateAnchorMode, issues, "data.sample_construction")
    _validate_enum(sample_construction, "sample_order_mode", SampleOrderMode, issues, "data.sample_construction")
    _validate_positive_ints(
        sample_construction,
        issues,
        "data.sample_construction",
        (
            "segment_frames",
            "segment_min_frames",
            "segment_max_frames",
            "segment_length_stride",
            "segment_locality_block_size",
        ),
    )
    if "start_padding_frames" in sample_construction and sample_construction["start_padding_frames"] is not None:
        value = _optional_int(sample_construction["start_padding_frames"])
        if value is None or value < 0:
            issues.error("data.sample_construction.start_padding_frames", "Expected a non-negative integer.")
    if (
        "condition_source_frame_offset" in sample_construction
        and sample_construction["condition_source_frame_offset"] is not None
    ):
        value = _optional_int(sample_construction["condition_source_frame_offset"])
        if value is None:
            issues.error("data.sample_construction.condition_source_frame_offset", "Expected an integer.")
    if "context_prefix_frames" in sample_construction and sample_construction["context_prefix_frames"] is not None:
        value = _optional_int(sample_construction["context_prefix_frames"])
        if value is None or value < 0:
            issues.error("data.sample_construction.context_prefix_frames", "Expected a non-negative integer.")
    if "rollout_context_frames" in sample_construction and sample_construction["rollout_context_frames"] is not None:
        value = _optional_int(sample_construction["rollout_context_frames"])
        if value is None or value <= 0:
            issues.error("data.sample_construction.rollout_context_frames", "Expected a positive integer or null.")
    mode = sample_construction.get("mode")
    if mode != WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT.value:
        return
    if "segment_frames" not in sample_construction:
        issues.error(
            "data.sample_construction.segment_frames",
            "Expected `segment_frames` when mode is `hierarchical_fixed_segment`.",
        )
    for legacy_key in (
        "segment_min_frames",
        "segment_max_frames",
        "randomize_segment_length",
        "randomize_segment_start",
        "require_full_segment",
        "sample_weight_mode",
        "sample_weight_length_power",
    ):
        if legacy_key in sample_construction:
            issues.error(
                f"data.sample_construction.{legacy_key}",
                "`hierarchical_fixed_segment` uses fixed segment and hierarchical power fields; "
                f"do not set `{legacy_key}`.",
            )
    if sample_construction.get("sample_order_mode") == SampleOrderMode.REPLACEMENT.value:
        issues.error(
            "data.sample_construction.sample_order_mode",
            "`hierarchical_fixed_segment` does not support replacement `sample_order_mode`.",
        )
    if sample_construction.get("target_alignment") == SampleTargetAlignment.NEXT_AFTER_CONTEXT.value:
        if sample_construction.get("randomize_geometry", True) and not sample_construction.get(
            "allow_next_after_context_random_geometry",
            False,
        ):
            issues.error(
                "data.sample_construction.randomize_geometry",
                "`target_alignment=next_after_context` requires fixed rollout chunking; set this to false "
                "unless allow_next_after_context_random_geometry is true.",
            )
        if sample_construction.get("start_padding_frames", 0) not in (0, None):
            issues.error(
                "data.sample_construction.start_padding_frames",
                "`target_alignment=next_after_context` deprecates virtual head padding; set this to 0.",
            )
        if sample_construction.get("chunk_size") not in (4, "4"):
            issues.error(
                "data.sample_construction.chunk_size",
                "`target_alignment=next_after_context` currently requires chunk_size=4.",
            )
        for legacy_context_key in ("context_prefix_policy", "context_prefix_frames"):
            if legacy_context_key in sample_construction:
                issues.error(
                    f"data.sample_construction.{legacy_context_key}",
                    "`target_alignment=next_after_context` uses rollout_context_policy/rollout_context_frames; "
                    f"do not set legacy `{legacy_context_key}`.",
                )


def _validate_generalist_dynamics_mixture(
    mixture: Mapping[str, Any],
    issues: "_IssueBuilder",
) -> None:
    weight_keys = (
        "real_joint_weight",
        "real_action_conditioned_video_weight",
        "real_video_conditioned_action_weight",
        "counterfactual_action_conditioned_video_weight",
        "counterfactual_video_conditioned_action_weight",
    )
    total = 0.0
    for key in weight_keys:
        if key not in mixture:
            continue
        value = mixture[key]
        if isinstance(value, bool):
            issues.error(f"data.generalist_dynamics_mixture.{key}", "Expected a numeric weight.")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            issues.error(f"data.generalist_dynamics_mixture.{key}", "Expected a numeric weight.")
            continue
        if not math.isfinite(numeric):
            issues.error(f"data.generalist_dynamics_mixture.{key}", "Expected a finite weight.")
            continue
        if numeric < 0.0:
            issues.error(f"data.generalist_dynamics_mixture.{key}", "Expected a non-negative weight.")
            continue
        total += numeric
    if total <= 0.0 and any(key in mixture for key in weight_keys):
        issues.error("data.generalist_dynamics_mixture", "Expected at least one positive mixture weight.")
    for key in ("train_latent_root", "val_latent_root"):
        if key in mixture and mixture[key] is not None and not isinstance(mixture[key], str):
            issues.error(f"data.generalist_dynamics_mixture.{key}", "Expected a string path.")
    if "allow_train_latent_root_for_val" in mixture and not isinstance(
        mixture["allow_train_latent_root_for_val"],
        bool,
    ):
        issues.error("data.generalist_dynamics_mixture.allow_train_latent_root_for_val", "Expected a boolean.")
    if "length_multiplier" in mixture:
        value = mixture["length_multiplier"]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            issues.error("data.generalist_dynamics_mixture.length_multiplier", "Expected a numeric value.")
            return
        if not math.isfinite(numeric) or numeric <= 0.0:
            issues.error("data.generalist_dynamics_mixture.length_multiplier", "Expected a finite positive value.")
    if "conditional_history_frames" in mixture and mixture["conditional_history_frames"] is not None:
        value = mixture["conditional_history_frames"]
        if isinstance(value, bool):
            issues.error("data.generalist_dynamics_mixture.conditional_history_frames", "Expected a positive integer or null.")
        else:
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                issues.error(
                    "data.generalist_dynamics_mixture.conditional_history_frames",
                    "Expected a positive integer or null.",
                )
                return
            if numeric <= 0:
                issues.error(
                    "data.generalist_dynamics_mixture.conditional_history_frames",
                    "Expected a positive integer or null.",
                )
