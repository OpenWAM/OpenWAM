"""Expansion and validation for policy sequence contracts."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import replace
from typing import Any

from . import enums
from .coercion import coerce_enum, coerce_strict_chunk_size, raw_enum_value
from .data_contracts import DataConfig
from .experiment import ExperimentConfig
from .policy_contracts import PolicyVariantConfig
from .policy_mot import MoTPolicyConfig
from .policy_parallel_stream import ParallelStreamPolicyConfig


__all__ = [
    "apply_parallel_sequence_contract",
    "expand_parallel_sequence_contract",
    "validate_experiment_config_runtime_contract",
    "validate_parallel_sequence_contract_override_keys",
    "validate_policy_data_sequence_contract",
]


_LEGACY_PREFIX_PARALLEL_RUNTIME_MODES = frozenset(
    {
        enums.ParallelRuntimeMode.LINGBOT_EXACT,
        enums.ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
    }
)


def _set_contract_default(
    mapping: dict[str, Any],
    *,
    key: str,
    value: Any,
    path: str,
    contract: enums.ParallelSequenceContract,
) -> None:
    existing = mapping.get(key)
    if key in mapping and raw_enum_value(existing) != raw_enum_value(value):
        raise ValueError(
            f"`policy_variant.parallel_sequence_contract={contract.value}` requires "
            f"`{path}={raw_enum_value(value)}`, got {existing!r}."
        )
    mapping[key] = value


def expand_parallel_sequence_contract(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand sequence contracts into raw defaults before typed parsing."""

    normalized = dict(raw)
    policy_variant_raw = normalized.get("policy_variant")
    if not isinstance(policy_variant_raw, dict):
        return normalized
    policy_variant_raw = dict(policy_variant_raw)
    normalized["policy_variant"] = policy_variant_raw

    contract = coerce_enum(
        enums.ParallelSequenceContract,
        policy_variant_raw.get(
            "parallel_sequence_contract",
            enums.ParallelSequenceContract.DEFAULT,
        ),
    )
    if contract == enums.ParallelSequenceContract.DEFAULT:
        return normalized

    policy_name = coerce_enum(
        enums.PolicyVariantName,
        policy_variant_raw.get("name", enums.PolicyVariantName.POST_LATENT),
    )
    if policy_name not in {
        enums.PolicyVariantName.PARALLEL_STREAM,
        enums.PolicyVariantName.MOT,
    }:
        raise ValueError(
            f"`policy_variant.parallel_sequence_contract={contract.value}` is only supported for "
            "`policy_variant.name` in {'parallel_stream', 'mot'}."
        )

    if contract == enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO:
        if policy_name == enums.PolicyVariantName.PARALLEL_STREAM:
            runtime_mode = coerce_enum(
                enums.ParallelRuntimeMode,
                policy_variant_raw.get(
                    "runtime_mode",
                    enums.ParallelRuntimeMode.LINGBOT_EXACT,
                ),
            )
            if runtime_mode not in _LEGACY_PREFIX_PARALLEL_RUNTIME_MODES:
                allowed = ", ".join(
                    f"'{mode.value}'"
                    for mode in sorted(_LEGACY_PREFIX_PARALLEL_RUNTIME_MODES)
                )
                raise ValueError(
                    "`policy_variant.parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` only "
                    f"supports `policy_variant.runtime_mode` in {{{allowed}}}, got {runtime_mode.value!r}."
                )
        else:
            runtime_mode = coerce_enum(
                enums.MoTRuntimeMode,
                policy_variant_raw.get(
                    "runtime_mode",
                    enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM,
                ),
            )
            if runtime_mode != enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM:
                raise ValueError(
                    "`policy_variant.parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` "
                    "requires `policy_variant.runtime_mode=non_joint_two_stream` for MoT/M5."
                )

    if contract not in {
        enums.ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
        enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    }:
        raise ValueError(
            f"Unsupported `policy_variant.parallel_sequence_contract={contract.value}`."
        )

    if (
        contract
        == enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
        and policy_name == enums.PolicyVariantName.MOT
        and "joint_timestep_coupling" not in policy_variant_raw
    ):
        policy_variant_raw["joint_timestep_coupling"] = (
            enums.JointTimestepCoupling.INDEPENDENT
        )

    for key, value in (
        ("proprio_context_mode", enums.ProprioContextMode.PER_CHUNK_ADDITIVE),
        (
            "context_condition_latent_source",
            enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        ),
        (
            "history_stream_visibility",
            enums.ParallelHistoryStreamVisibility.VIDEO_ONLY,
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
        == enums.ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
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
    if not isinstance(
        policy_variant_config,
        (ParallelStreamPolicyConfig, MoTPolicyConfig),
    ):
        return
    if (
        policy_variant_config.context_condition_latent_source
        != enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
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

    if (
        isinstance(config.policy_variant, MoTPolicyConfig)
        and config.policy_variant.mot_generalist_training_mode_probs is not None
        and (
            int(config.data.train_batch_size) != 1
            or int(config.data.val_batch_size) != 1
        )
    ):
        raise ValueError(
            "`policy_variant.mot_generalist_training_mode_probs` currently requires "
            "`data.train_batch_size = data.val_batch_size = 1` because M5 GJD samples one mode per "
            "segment/forward pass and forced per-sample metadata is only unambiguous for rank-local batch size 1."
        )

    if (
        getattr(config.policy_variant, "generalist_training_paradigm", None)
        == enums.GeneralistTrainingParadigm.MIXED_DYNAMICS
    ):
        if config.trainer.batch_adapter != enums.BatchAdapterName.LATENTS:
            raise ValueError(
                "`policy_variant.generalist_training_paradigm=mixed_dynamics` requires "
                "`trainer.batch_adapter=latents` because the mixed-dynamics source mixture wraps latent datasets."
            )
        sample_construction = config.data.sample_construction
        if sample_construction.sample_weight_mode != enums.SampleWeightMode.UNIFORM:
            raise ValueError(
                "`data.sample_construction.sample_weight_mode` must be `uniform` with "
                "`policy_variant.generalist_training_paradigm=mixed_dynamics` because the mixed-dynamics "
                "wrapper owns source sampling and only preserves parity for uniform replacement draws."
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


def apply_parallel_sequence_contract(
    config: ExperimentConfig,
    *,
    explicit_override_keys: Collection[str] | None = None,
) -> ExperimentConfig:
    """Apply typed sequence-contract defaults after config overrides."""

    policy_variant = config.policy_variant
    contract = coerce_enum(
        enums.ParallelSequenceContract,
        getattr(
            policy_variant,
            "parallel_sequence_contract",
            enums.ParallelSequenceContract.DEFAULT,
        ),
    )
    if contract == enums.ParallelSequenceContract.DEFAULT:
        return validate_experiment_config_runtime_contract(config)

    policy_name = coerce_enum(enums.PolicyVariantName, policy_variant.name)
    if policy_name not in {
        enums.PolicyVariantName.PARALLEL_STREAM,
        enums.PolicyVariantName.MOT,
    }:
        raise ValueError(
            f"`policy_variant.parallel_sequence_contract={contract.value}` is only supported for "
            "`policy_variant.name` in {'parallel_stream', 'mot'}."
        )

    if contract == enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO:
        if policy_name == enums.PolicyVariantName.PARALLEL_STREAM:
            runtime_mode = coerce_enum(
                enums.ParallelRuntimeMode,
                getattr(
                    policy_variant,
                    "runtime_mode",
                    enums.ParallelRuntimeMode.LINGBOT_EXACT,
                ),
            )
            if runtime_mode not in _LEGACY_PREFIX_PARALLEL_RUNTIME_MODES:
                allowed = ", ".join(
                    f"'{mode.value}'"
                    for mode in sorted(_LEGACY_PREFIX_PARALLEL_RUNTIME_MODES)
                )
                raise ValueError(
                    "`policy_variant.parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` only "
                    f"supports `policy_variant.runtime_mode` in {{{allowed}}}, got {runtime_mode.value!r}."
                )
        else:
            runtime_mode = coerce_enum(
                enums.MoTRuntimeMode,
                getattr(
                    policy_variant,
                    "runtime_mode",
                    enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM,
                ),
            )
            if runtime_mode != enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM:
                raise ValueError(
                    "`policy_variant.parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` "
                    "requires `policy_variant.runtime_mode=non_joint_two_stream` for MoT/M5."
                )

    if contract not in {
        enums.ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
        enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    }:
        raise ValueError(
            f"Unsupported `policy_variant.parallel_sequence_contract={contract.value}`."
        )

    policy_updates: dict[str, Any] = {
        "proprio_context_mode": enums.ProprioContextMode.PER_CHUNK_ADDITIVE,
        "context_condition_latent_source": (
            enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
        ),
        "history_stream_visibility": (
            enums.ParallelHistoryStreamVisibility.VIDEO_ONLY
        ),
    }
    if hasattr(policy_variant, "use_condition_latents"):
        policy_updates["use_condition_latents"] = True
    if hasattr(policy_variant, "require_condition_latents"):
        policy_updates["require_condition_latents"] = True
    if (
        contract
        == enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
        and hasattr(policy_variant, "joint_timestep_coupling")
    ):
        explicit_keys = set(explicit_override_keys or ())
        contract_set_by_cli = (
            "policy_variant.parallel_sequence_contract" in explicit_keys
        )
        if (
            contract_set_by_cli
            and "policy_variant.joint_timestep_coupling" not in explicit_keys
            and policy_variant.joint_timestep_coupling
            == enums.JointTimestepCoupling.MATCH_SIGMA
        ):
            policy_updates["joint_timestep_coupling"] = (
                enums.JointTimestepCoupling.INDEPENDENT
            )
    updated_policy_variant = replace(policy_variant, **policy_updates)

    sample_updates: dict[str, Any] = {
        "condition_source_frame_offset": -1,
        "start_padding_frames": 0,
    }
    if (
        contract
        == enums.ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
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
    return validate_experiment_config_runtime_contract(
        replace(
            config,
            data=updated_data,
            policy_variant=updated_policy_variant,
        )
    )


_PARALLEL_SEQUENCE_CONTRACT_MANAGED_OVERRIDE_KEYS = frozenset(
    {
        "policy_variant.proprio_context_mode",
        "policy_variant.context_condition_latent_source",
        "policy_variant.history_stream_visibility",
        "policy_variant.use_condition_latents",
        "policy_variant.require_condition_latents",
        "policy_variant.joint_timestep_coupling",
        "data.sample_construction.target_alignment",
        "data.sample_construction.rollout_context_policy",
        "data.sample_construction.condition_source_frame_offset",
        "data.sample_construction.start_padding_frames",
    }
)


def validate_parallel_sequence_contract_override_keys(
    overrides: Mapping[str, Any],
    *,
    contract_value: Any | None = None,
) -> None:
    """Reject ambiguous CLI overrides of fields owned by a sequence contract."""

    resolved_contract_value = overrides.get(
        "policy_variant.parallel_sequence_contract",
        contract_value,
    )
    if resolved_contract_value is None:
        return
    contract = coerce_enum(
        enums.ParallelSequenceContract,
        resolved_contract_value,
    )
    if contract == enums.ParallelSequenceContract.DEFAULT:
        return
    managed_keys = set(_PARALLEL_SEQUENCE_CONTRACT_MANAGED_OVERRIDE_KEYS)
    if (
        contract
        == enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    ):
        managed_keys.discard("policy_variant.joint_timestep_coupling")
    conflicting_keys = sorted(
        key for key in overrides if key in managed_keys
    )
    if conflicting_keys:
        joined = ", ".join(f"`{key}`" for key in conflicting_keys)
        raise ValueError(
            f"`policy_variant.parallel_sequence_contract={contract.value}` owns {joined}; "
            "drop the contract or drop the individual override(s)."
        )
