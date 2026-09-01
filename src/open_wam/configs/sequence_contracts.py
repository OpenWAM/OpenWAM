"""Expansion and validation for policy sequence contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from . import enums
from .coercion import coerce_enum, coerce_strict_chunk_size, raw_enum_value
from .data_contracts import DataConfig
from .experiment import ExperimentConfig
from .policy_contracts import (
    CausalVideoPredictionPolicyConfig,
    PolicyVariantConfig,
)
from .policy_video_action import (
    VideoActionPolicyConfig,
    fixed_conditioning_mode_for_program,
    supports_dynamics_routing,
)
from .sequence_contract_specs import (
    get_video_action_sequence_contract_spec,
    video_action_sequence_contract_managed_override_keys,
)
from .training import TrainingConfig

__all__ = [
    "apply_video_action_sequence_contract",
    "expand_video_action_sequence_contract",
    "materialize_video_action_sequence_contract",
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
    spec = get_video_action_sequence_contract_spec(contract)
    if spec is None:
        return normalized

    for key, value in spec.policy_variant_updates().items():
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

    for key, value in spec.sample_construction_updates().items():
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
    if isinstance(policy_variant_config, CausalVideoPredictionPolicyConfig):
        sample_config = data_config.sample_construction
        if policy_variant_config.program == enums.CausalVideoProgram.PREFIX_SUFFIX:
            if sample_config.mode != enums.WindowSamplingMode.CAUSAL_PREFIX_SUFFIX:
                raise ValueError(
                    "Causal video `program=prefix_suffix` requires "
                    "`data.sample_construction.mode=causal_prefix_suffix`."
                )
        else:
            if sample_config.mode != enums.WindowSamplingMode.UNIFORM_SEGMENT:
                raise ValueError(
                    "Causal video `program=chunked_conditioned_video` requires "
                    "`data.sample_construction.mode=uniform_segment`."
                )
            if int(sample_config.condition_source_frame_offset) != -1:
                raise ValueError(
                    "Chunked conditioned video requires "
                    "`data.sample_construction.condition_source_frame_offset=-1`."
                )
            if sample_config.target_alignment != enums.SampleTargetAlignment.LEGACY:
                raise ValueError(
                    "Chunked conditioned video is the M5 VTA legacy-prefix marginal "
                    "and requires `data.sample_construction.target_alignment=legacy`."
                )
            if int(sample_config.start_padding_frames) != 0:
                raise ValueError(
                    "Chunked conditioned video requires "
                    "`data.sample_construction.start_padding_frames=0`."
                )
            if sample_config.causal_prefix_suffix_buckets:
                raise ValueError(
                    "Chunked conditioned video does not accept causal prefix/suffix buckets."
                )
        return
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

    if isinstance(config.policy_variant, CausalVideoPredictionPolicyConfig):
        if config.action_decoder.name != enums.ActionDecoderName.VIDEO_ONLY:
            raise ValueError(
                "Causal video prediction requires `action_decoder.name=video_only_decoder`."
            )
        if (
            int(config.data.action_schema.action_horizon) != 0
            or int(config.data.action_schema.state_horizon) != 0
            or int(config.action_decoder.action_horizon) != 0
        ):
            raise ValueError(
                "Causal video prediction must not materialize action or state "
                "targets; data action/state horizons and the decoder action "
                "horizon must all be zero."
            )
        if config.policy_variant.program == enums.CausalVideoProgram.PREFIX_SUFFIX:
            default_training = TrainingConfig()
            for field_name in ("chunk_size", "window_size"):
                configured_value = getattr(config.training, field_name)
                default_value = getattr(default_training, field_name)
                if configured_value != default_value:
                    raise ValueError(
                        "Prefix/suffix video sample geometry is configured by "
                        "`data.sample_construction.causal_prefix_suffix_buckets`; "
                        f"`training.{field_name}` must remain at its unused default "
                        f"{default_value!r}, got {configured_value!r}."
                    )
        else:
            sample_config = config.data.sample_construction
            if int(config.backbone.patch_size_t) != 1:
                raise ValueError(
                    "The M5 VTA legacy-prefix video marginal requires "
                    "`backbone.patch_size_t=1`."
                )
            if config.training.enabled_objectives != (
                enums.TrainingObjective.LATENT,
            ):
                raise ValueError(
                    "Chunked conditioned-video prediction requires "
                    "`training.enabled_objectives=[latent]`."
                )
            if float(config.training.action_loss_weight) != 0.0:
                raise ValueError(
                    "Chunked conditioned-video prediction requires "
                    "`training.action_loss_weight=0.0`."
                )
            if int(config.data.train_batch_size) != 1:
                raise ValueError(
                    "Randomized chunked conditioned-video geometry requires "
                    "`data.train_batch_size=1`."
                )
            if sample_config.sample_order_mode != enums.SampleOrderMode.REPLACEMENT:
                raise ValueError(
                    "Chunked conditioned-video training requires replacement sampling."
                )
            if not sample_config.randomize_geometry:
                raise ValueError(
                    "Chunked conditioned-video training requires randomized VTA "
                    "chunk/window geometry."
                )
            if not sample_config.require_full_segment:
                raise ValueError(
                    "Chunked conditioned-video training requires full segments."
                )
            for field_name in ("chunk_size", "window_size"):
                data_value = int(getattr(sample_config, field_name))
                training_value = int(getattr(config.training, field_name))
                if training_value != data_value:
                    raise ValueError(
                        "Chunked conditioned-video VTA geometry must agree across "
                        f"data and training config: data.{field_name}={data_value}, "
                        f"training.{field_name}={training_value}."
                    )

        text_mode = config.policy_variant.text_conditioning_mode
        dropout_probability = float(config.training.text_condition_dropout_prob)
        guidance_scale = float(config.inference.guidance_scale)
        if text_mode == enums.TextConditioningMode.DISABLED:
            if dropout_probability != 0.0:
                raise ValueError(
                    "Causal video `text_conditioning_mode=disabled` requires "
                    "`training.text_condition_dropout_prob=0.0`; every sample "
                    "already uses the blank-text embedding."
                )
            if guidance_scale != 1.0:
                raise ValueError(
                    "Causal video `text_conditioning_mode=disabled` requires "
                    "`inference.guidance_scale=1.0`; conditioned and "
                    "unconditioned branches are identical."
                )
        elif dropout_probability >= 1.0:
            raise ValueError(
                "Causal video `text_conditioning_mode=task_prompt` requires "
                "`training.text_condition_dropout_prob < 1.0`; select "
                "`text_conditioning_mode=disabled` for unconditional training."
            )

    if (
        isinstance(config.policy_variant, CausalVideoPredictionPolicyConfig)
        and float(config.training.text_condition_dropout_prob) > 0.0
        and config.trainer.batch_adapter != enums.BatchAdapterName.LATENTS
    ):
        raise ValueError(
            "Causal video text-condition dropout requires "
            "`trainer.batch_adapter=latents`; the views adapter does not provide "
            "the encoded text context on which training dropout operates."
        )

    if (
        isinstance(config.policy_variant, CausalVideoPredictionPolicyConfig)
        and float(config.inference.guidance_scale) > 1.0
        and float(config.training.text_condition_dropout_prob) <= 0.0
    ):
        raise ValueError(
            "Causal video classifier-free guidance requires "
            "`training.text_condition_dropout_prob > 0`; use "
            "`inference.guidance_scale=1.0` for checkpoints trained without an "
            "unconditional branch."
        )

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


def materialize_video_action_sequence_contract(
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
    spec = get_video_action_sequence_contract_spec(contract)
    if spec is None:
        return config

    updated_policy_variant = replace(
        policy_variant,
        **spec.policy_variant_updates(),
    )
    updated_sample_construction = replace(
        config.data.sample_construction,
        **spec.sample_construction_updates(),
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
        materialize_video_action_sequence_contract(config)
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
        if key in video_action_sequence_contract_managed_override_keys(contract)
    )
    if conflicting_keys:
        joined = ", ".join(f"`{key}`" for key in conflicting_keys)
        raise ValueError(
            f"`policy_variant.sequence_contract={contract.value}` owns {joined}; "
            "drop the contract or drop the individual override(s)."
        )
