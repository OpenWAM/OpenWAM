"""Expansion and validation for policy sequence contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from . import enums
from .coercion import coerce_enum, coerce_strict_chunk_size, raw_enum_value
from .data_contracts import DataConfig
from .experiment import ExperimentConfig
from .policy_contracts import PolicyVariantConfig
from .policy_video_action import (
    VideoActionPolicyConfig,
    fixed_conditioning_mode_for_program,
    supports_dynamics_routing,
)

__all__ = [
    "apply_video_action_sequence_contract",
    "expand_video_action_sequence_contract",
    "validate_experiment_config_runtime_contract",
    "validate_policy_data_sequence_contract",
    "validate_video_action_sequence_contract_override_keys",
]


def _set_contract_default(
    mapping: dict[str, Any],
    *,
    key: str,
    value: Any,
    path: str,
    contract: enums.VideoActionSequenceContract,
) -> None:
    existing = mapping.get(key)
    if key in mapping and raw_enum_value(existing) != raw_enum_value(value):
        raise ValueError(
            f"`policy_variant.sequence_contract={contract.value}` requires "
            f"`{path}={raw_enum_value(value)}`, got {existing!r}."
        )
    mapping[key] = value


def expand_video_action_sequence_contract(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand sequence contracts into raw defaults before typed parsing."""

    normalized = dict(raw)
    policy_variant_raw = normalized.get("policy_variant")
    if not isinstance(policy_variant_raw, dict):
        return normalized
    policy_variant_raw = dict(policy_variant_raw)
    normalized["policy_variant"] = policy_variant_raw

    contract = coerce_enum(
        enums.VideoActionSequenceContract,
        policy_variant_raw.get(
            "sequence_contract",
            enums.VideoActionSequenceContract.DEFAULT,
        ),
    )
    if contract == enums.VideoActionSequenceContract.DEFAULT:
        return normalized

    if contract not in {
        enums.VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
        enums.VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    }:
        raise ValueError(
            f"Unsupported `policy_variant.sequence_contract={contract.value}`."
        )

    for key, value in (
        ("proprio_context_mode", enums.ProprioContextMode.PER_CHUNK_ADDITIVE),
        (
            "context_condition_latent_source",
            enums.ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        ),
        (
            "history_stream_visibility",
            enums.HistoryStreamVisibility.VIDEO_ONLY,
        ),
        ("use_condition_latents", True),
        ("require_condition_latents", True),
    ):
        _set_contract_default(
            policy_variant_raw,
            key=key,
            value=value,
            path=f"policy_variant.{key}",
            contract=contract,
        )

    data_raw = normalized.get("data")
    if not isinstance(data_raw, dict):
        data_raw = {}
    else:
        data_raw = dict(data_raw)
    normalized["data"] = data_raw
    sample_construction_raw = data_raw.get("sample_construction")
    if not isinstance(sample_construction_raw, dict):
        sample_construction_raw = {}
    else:
        sample_construction_raw = dict(sample_construction_raw)
    data_raw["sample_construction"] = sample_construction_raw

    common_sample_defaults = {
        "condition_source_frame_offset": -1,
        "start_padding_frames": 0,
    }
    if (
        contract
        == enums.VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
    ):
        sample_defaults = {
            **common_sample_defaults,
            "target_alignment": enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT,
            "rollout_context_policy": enums.RolloutContextPolicy.ONE_FRAME,
        }
    else:
        sample_defaults = {
            **common_sample_defaults,
            "target_alignment": enums.SampleTargetAlignment.LEGACY,
        }
    for key, value in sample_defaults.items():
        _set_contract_default(
            sample_construction_raw,
            key=key,
            value=value,
            path=f"data.sample_construction.{key}",
            contract=contract,
        )

    return normalized


def validate_policy_data_sequence_contract(
    *,
    data_config: DataConfig,
    policy_variant_config: PolicyVariantConfig,
) -> None:
    if not isinstance(policy_variant_config, VideoActionPolicyConfig):
        return
    if (
        policy_variant_config.context_condition_latent_source
        != enums.ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    ):
        return
    condition_source_frame_offset = int(
        data_config.sample_construction.condition_source_frame_offset
    )
    if condition_source_frame_offset != -1:
        raise ValueError(
            "`policy_variant.context_condition_latent_source=single_frame_condition_latent` requires "
            "`data.sample_construction.condition_source_frame_offset=-1` so the clean context latent is encoded "
            "from the raw frame immediately before the target latent span. Offset 0 can expose the first target "
            "raw frame and is not a safe default; use a separate explicit ablation path if that behavior is intended."
        )


def validate_experiment_config_runtime_contract(
    config: ExperimentConfig,
) -> ExperimentConfig:
    """Validate cross-section runtime contracts after YAML and CLI overrides."""

    policy_program = (
        config.policy_variant.program
        if isinstance(config.policy_variant, VideoActionPolicyConfig)
        else None
    )
    if supports_dynamics_routing(policy_program) and (
        int(config.data.train_batch_size) != 1
        or int(config.data.val_batch_size) != 1
    ):
        raise ValueError(
            "Generalist and conditional-dynamics programs currently require "
            "`data.train_batch_size = data.val_batch_size = 1` because one mode is applied per "
            "segment/forward pass and routed metadata is only unambiguous for rank-local batch size 1."
        )

    active_routes = config.data.dynamics_routing.active_routes
    fixed_mode = fixed_conditioning_mode_for_program(policy_program)
    if active_routes:
        if not supports_dynamics_routing(policy_program):
            raise ValueError(
                "Active `data.dynamics_routing.routes` require a generalist, "
                "forward-dynamics, or inverse-dynamics policy program."
            )
        conflicting_routes = tuple(
            route for route in active_routes if fixed_mode is not None and route.mode != fixed_mode
        )
        if conflicting_routes:
            route_labels = ", ".join(route.bucket_name for route in conflicting_routes)
            raise ValueError(
                f"`policy_variant.program={policy_program.value}` accepts only "
                f"{fixed_mode.value!r} routes; remove conflicting routes: {route_labels}."
            )
        if config.trainer.batch_adapter != enums.BatchAdapterName.LATENTS:
            raise ValueError(
                "Active `data.dynamics_routing.routes` require "
                "`trainer.batch_adapter=latents` because the dynamics source router wraps latent datasets."
            )
        sample_construction = config.data.sample_construction
        if (
            sample_construction.sample_order_mode
            != enums.SampleOrderMode.REPLACEMENT
        ):
            raise ValueError(
                "`data.sample_construction.sample_order_mode` must be `replacement` "
                "with active `data.dynamics_routing.routes` because route weights "
                "define replacement probabilities."
            )
        if sample_construction.sample_weight_mode != enums.SampleWeightMode.UNIFORM:
            raise ValueError(
                "`data.sample_construction.sample_weight_mode` must be `uniform` with "
                "active `data.dynamics_routing.routes` because the dynamics router "
                "owns source sampling and only preserves parity for uniform replacement draws."
            )
    elif fixed_mode is not None:
        raise ValueError(
            f"`policy_variant.program={policy_program.value}` requires at least one "
            "positive `data.dynamics_routing.routes` entry so every sample uses "
            "the target-only t0 contract."
        )

    if fixed_mode is not None:
        for task in config.validation.auxiliary_tasks:
            if not task.enabled or task.max_batches == 0:
                continue
            if task.mode_override is not None and task.mode_override != fixed_mode:
                raise ValueError(
                    f"Auxiliary validation task {task.name!r} requests mode "
                    f"{task.mode_override.value!r}, but policy program "
                    f"{policy_program.value!r} fixes mode {fixed_mode.value!r}."
                )

    for task in config.validation.auxiliary_tasks:
        effective_mode = (
            fixed_mode if task.mode_override is None else task.mode_override
        )
        if (
            effective_mode is not None
            and effective_mode.is_conditional
            and task.drop_text_conditioning is not None
        ):
            raise ValueError(
                f"Conditional auxiliary validation task {task.name!r} always "
                "removes task text; remove `drop_text_conditioning`."
            )
        if (
            not task.enabled
            or task.max_batches == 0
            or effective_mode is None
            or not effective_mode.is_conditional
        ):
            continue
        if not active_routes:
            raise ValueError(
                f"Conditional auxiliary validation task {task.name!r} requires "
                "active `data.dynamics_routing.routes` so samples receive the "
                "target-only t0 projection."
            )
        if (
            task.source == enums.AuxiliaryValidationSource.DATASET
            and any(route.mode != effective_mode for route in active_routes)
        ):
            raise ValueError(
                f"Conditional auxiliary validation task {task.name!r} cannot use "
                "`source = dataset` when active routes contain another objective. "
                "Select a named dynamics source or make the routed dataset "
                f"homogeneous for {effective_mode.value!r}."
            )

    if (
        config.data.sample_construction.target_alignment
        == enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT
    ):
        strict_chunk_sources = {
            "data.sample_construction.chunk_size": (
                config.data.sample_construction.chunk_size
            ),
            "training.chunk_size": config.training.chunk_size,
            "inference.frame_chunk_size": config.inference.frame_chunk_size,
        }
        policy_frame_chunk_size = getattr(
            config.policy_variant,
            "frame_chunk_size",
            None,
        )
        if policy_frame_chunk_size is not None:
            strict_chunk_sources["policy_variant.frame_chunk_size"] = (
                policy_frame_chunk_size
            )
        invalid_chunk_sources = {
            name: value
            for name, value in strict_chunk_sources.items()
            if coerce_strict_chunk_size(name, value) != 4
        }
        if invalid_chunk_sources:
            joined = ", ".join(
                f"{name}={value}"
                for name, value in sorted(invalid_chunk_sources.items())
            )
            raise ValueError(
                "`sample_construction.target_alignment=next_after_context` requires fixed 4-frame chunks "
                f"across data/training/inference/policy; got {joined}."
            )

    validate_policy_data_sequence_contract(
        data_config=config.data,
        policy_variant_config=config.policy_variant,
    )
    return config


def _materialize_video_action_sequence_contract(
    config: ExperimentConfig,
) -> ExperimentConfig:
    """Materialize fields owned by a typed sequence contract."""

    policy_variant = config.policy_variant
    if not isinstance(policy_variant, VideoActionPolicyConfig):
        return config

    contract = coerce_enum(
        enums.VideoActionSequenceContract,
        policy_variant.sequence_contract,
    )
    if contract == enums.VideoActionSequenceContract.DEFAULT:
        return config

    if contract not in {
        enums.VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
        enums.VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    }:
        raise ValueError(
            f"Unsupported `policy_variant.sequence_contract={contract.value}`."
        )

    policy_updates: dict[str, Any] = {
        "proprio_context_mode": enums.ProprioContextMode.PER_CHUNK_ADDITIVE,
        "context_condition_latent_source": (
            enums.ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
        ),
        "history_stream_visibility": (
            enums.HistoryStreamVisibility.VIDEO_ONLY
        ),
    }
    policy_updates["use_condition_latents"] = True
    policy_updates["require_condition_latents"] = True
    updated_policy_variant = replace(policy_variant, **policy_updates)

    sample_updates: dict[str, Any] = {
        "condition_source_frame_offset": -1,
        "start_padding_frames": 0,
    }
    if (
        contract
        == enums.VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
    ):
        sample_updates.update(
            {
                "target_alignment": enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT,
                "rollout_context_policy": enums.RolloutContextPolicy.ONE_FRAME,
            }
        )
    else:
        sample_updates.update(
            {
                "target_alignment": enums.SampleTargetAlignment.LEGACY,
            }
        )
    updated_sample_construction = replace(
        config.data.sample_construction,
        **sample_updates,
    )
    updated_data = replace(
        config.data,
        sample_construction=updated_sample_construction,
    )
    return replace(
        config,
        data=updated_data,
        policy_variant=updated_policy_variant,
    )


def apply_video_action_sequence_contract(config: ExperimentConfig) -> ExperimentConfig:
    """Materialize sequence defaults and validate the resulting runtime config."""

    return validate_experiment_config_runtime_contract(
        _materialize_video_action_sequence_contract(config)
    )


_VIDEO_ACTION_SEQUENCE_CONTRACT_MANAGED_OVERRIDE_KEYS = frozenset(
    {
        "policy_variant.proprio_context_mode",
        "policy_variant.context_condition_latent_source",
        "policy_variant.history_stream_visibility",
        "policy_variant.use_condition_latents",
        "policy_variant.require_condition_latents",
        "data.sample_construction.target_alignment",
        "data.sample_construction.rollout_context_policy",
        "data.sample_construction.condition_source_frame_offset",
        "data.sample_construction.start_padding_frames",
    }
)


def validate_video_action_sequence_contract_override_keys(
    overrides: Mapping[str, Any],
    *,
    contract_value: Any | None = None,
) -> None:
    """Reject ambiguous CLI overrides of fields owned by a sequence contract."""

    resolved_contract_value = overrides.get(
        "policy_variant.sequence_contract",
        contract_value,
    )
    if resolved_contract_value is None:
        return
    contract = coerce_enum(
        enums.VideoActionSequenceContract,
        resolved_contract_value,
    )
    if contract == enums.VideoActionSequenceContract.DEFAULT:
        return
    conflicting_keys = sorted(
        key
        for key in overrides
        if key in _VIDEO_ACTION_SEQUENCE_CONTRACT_MANAGED_OVERRIDE_KEYS
    )
    if conflicting_keys:
        joined = ", ".join(f"`{key}`" for key in conflicting_keys)
        raise ValueError(
            f"`policy_variant.sequence_contract={contract.value}` owns {joined}; "
            "drop the contract or drop the individual override(s)."
        )
