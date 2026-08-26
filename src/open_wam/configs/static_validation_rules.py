"""Top-level experiment, evaluation, and auxiliary-validation rules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from open_wam.contracts.paths import validate_model_component_path

from .enums import (
    ActionDecoderName,
    ActionMappingLossMaskMode,
    ActionMappingMode,
    ActionMappingSamplerMaskMode,
    ActionTargetReferenceSource,
    ActionTargetRepresentation,
    ActionTargetStateEncoding,
    AttachSite,
    AttentionMode,
    AuxiliaryValidationSource,
    BackboneImplementation,
    BatchAdapterName,
    ContextConditionLatentSource,
    DataSplit,
    DualExpertActionExpertInitMode,
    DualExpertConditionMode,
    DynamicsObjective,
    EvalMode,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    LatentTemporalLayout,
    PolicyVariantName,
    ProprioContextMode,
    ReplayStatusPolicy,
    SampleOrderMode,
    SampleWeightMode,
    TextConditioningMode,
    TrainerAccelerator,
    TrainerPrecision,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from .policy_video_action import fixed_conditioning_mode_for_program
from .runtime_backbone_components import validate_runtime_backbone_components
from .static_validation_contracts import _IssueBuilder
from .static_validation_data import (
    _validate_action_mapping,
    _validate_action_schema_compatibility,
    _validate_dynamics_routing,
    _validate_sample_construction,
)
from .static_validation_policy import (
    _active_dynamics_routes,
    _validate_action_horizons,
    _validate_dynamics_route_contract,
    _validate_fixed_conditional_program,
    _validate_program_timestep_contract,
    _validate_single_frame_condition_offset,
    _validate_video_action_sequence_contract_static,
    _warn_deprecated_text_proprio_context,
)
from .static_validation_primitives import (
    _mapping,
    _optional_int,
    _resolve_relative,
    _validate_enum,
    _validate_positive_ints,
)


def _validate_video_action_policy_contract(
    policy_variant: Mapping[str, Any],
    *,
    data: Mapping[str, Any],
    sample_construction: Mapping[str, Any] | None,
    issues: _IssueBuilder,
) -> None:
    """Validate fields shared by every video/action policy architecture."""

    _validate_enum(
        policy_variant,
        "program",
        VideoActionProgram,
        issues,
        "policy_variant",
    )
    if policy_variant.get("program") is None:
        issues.error(
            "policy_variant.program",
            "Video/action policies require an explicit program.",
        )
    for field_name, enum_type in (
        ("joint_timestep_coupling", JointTimestepCoupling),
        ("sequence_contract", VideoActionSequenceContract),
        ("proprio_context_mode", ProprioContextMode),
        ("context_condition_latent_source", ContextConditionLatentSource),
        ("history_stream_visibility", HistoryStreamVisibility),
    ):
        _validate_enum(
            policy_variant,
            field_name,
            enum_type,
            issues,
            "policy_variant",
        )
    _warn_deprecated_text_proprio_context(policy_variant, issues)
    _validate_single_frame_condition_offset(
        policy_variant,
        sample_construction,
        issues,
    )
    _validate_video_action_sequence_contract_static(
        policy_variant,
        sample_construction,
        issues,
    )
    _validate_dynamics_route_contract(policy_variant, data, issues)


def _validate_experiment_config(raw: Mapping[str, Any], issues: _IssueBuilder, *, relaxed: bool) -> None:
    required = ("data",) if relaxed else ("data", "backbone", "trainer")
    for key in required:
        if key not in raw:
            issues.error(key, "Missing required top-level section.")

    data = _mapping(raw.get("data"))
    if data is None:
        return
    if not data.get("dataset_type") and not data.get("dataset_name"):
        issues.error("data", "Expected `dataset_type` or `dataset_name`.")
    if "generalist_dynamics_mixture" in data:
        issues.error(
            "data.generalist_dynamics_mixture",
            "This field is retired; use `data.dynamics_routing`.",
        )
    if "adapter_options" in data and not isinstance(data["adapter_options"], Mapping):
        issues.error("data.adapter_options", "Expected a mapping of dataset-adapter options.")
    _validate_enum(data, "latent_temporal_layout", LatentTemporalLayout, issues, "data")
    if data.get("latent_temporal_layout") == LatentTemporalLayout.EQUAL_BUCKET_LEGACY.value:
        issues.error(
            "data.latent_temporal_layout",
            "`equal_bucket_legacy` is deprecated and unsupported. Equal-bucket latent/action alignment "
            "silently drops early actions for Wan/LingBot latents; use `wan_causal_stride4`.",
        )
    _validate_enum(data, "replay_status_policy", ReplayStatusPolicy, issues, "data")
    _validate_enum(data, "val_replay_status_policy", ReplayStatusPolicy, issues, "data")
    _validate_positive_ints(
        data,
        issues,
        "data",
        ("canonical_height", "canonical_width", "num_frames", "train_batch_size", "val_batch_size"),
    )
    action_schema = _mapping(data.get("action_schema"))
    if action_schema is not None:
        _validate_positive_ints(action_schema, issues, "data.action_schema", ("action_dim", "state_dim"))
    action_target = _mapping(data.get("action_target"))
    if action_target is not None:
        _validate_enum(action_target, "representation", ActionTargetRepresentation, issues, "data.action_target")
        _validate_enum(action_target, "state_encoding", ActionTargetStateEncoding, issues, "data.action_target")
        _validate_enum(action_target, "reference_source", ActionTargetReferenceSource, issues, "data.action_target")
    action_mapping = _mapping(data.get("action_mapping"))
    if action_mapping is not None:
        _validate_enum(action_mapping, "mode", ActionMappingMode, issues, "data.action_mapping")
        _validate_enum(action_mapping, "loss_mask_mode", ActionMappingLossMaskMode, issues, "data.action_mapping")
        _validate_enum(
            action_mapping,
            "sampler_mask_mode",
            ActionMappingSamplerMaskMode,
            issues,
            "data.action_mapping",
        )
        _validate_action_mapping(action_mapping, action_schema, issues)
    sample_construction = _mapping(data.get("sample_construction"))
    if sample_construction is not None:
        _validate_sample_construction(sample_construction, issues)
    dynamics_routing = _mapping(data.get("dynamics_routing"))
    if dynamics_routing is not None:
        _validate_dynamics_routing(dynamics_routing, issues)

    backbone = _mapping(raw.get("backbone"))
    if backbone is not None:
        _validate_enum(backbone, "implementation", BackboneImplementation, issues, "backbone")
        _validate_enum(backbone, "attn_mode", AttentionMode, issues, "backbone")
        _validate_enum(backbone, "train_attn_mode", AttentionMode, issues, "backbone")
        _validate_enum(backbone, "infer_attn_mode", AttentionMode, issues, "backbone")
        _validate_positive_ints(backbone, issues, "backbone", ("hidden_size", "num_layers", "num_heads"))
        if "transformer_subdir" in backbone:
            try:
                validate_model_component_path(
                    backbone["transformer_subdir"],
                    field_name="backbone.transformer_subdir",
                )
            except (TypeError, ValueError) as exc:
                issues.error("backbone.transformer_subdir", str(exc))

    trainer = _mapping(raw.get("trainer"))
    training = _mapping(raw.get("training"))
    inference = _mapping(raw.get("inference"))
    policy_variant = _mapping(raw.get("policy_variant"))
    action_decoder = _mapping(raw.get("action_decoder"))
    dropout_probability: float | None = 0.0
    if training is not None and "text_condition_dropout_prob" in training:
        raw_dropout_probability = training["text_condition_dropout_prob"]
        if isinstance(raw_dropout_probability, bool) or not isinstance(
            raw_dropout_probability, (int, float)
        ):
            issues.error(
                "training.text_condition_dropout_prob",
                "Expected a numeric probability within [0, 1].",
            )
            dropout_probability = None
        else:
            dropout_probability = float(raw_dropout_probability)
            if not math.isfinite(dropout_probability) or not (
                0.0 <= dropout_probability <= 1.0
            ):
                issues.error(
                    "training.text_condition_dropout_prob",
                    "Expected a finite probability within [0, 1].",
                )
                dropout_probability = None
    if not relaxed and policy_variant is None:
        issues.error("policy_variant", "Expected an explicit `policy_variant` section.")
    if "action_head" in raw:
        issues.error("action_head", "`action_head` was removed; configure `policy_variant` and `action_decoder`.")
    if policy_variant is not None:
        _validate_enum(policy_variant, "name", PolicyVariantName, issues, "policy_variant")
        _validate_enum(policy_variant, "attach_site", AttachSite, issues, "policy_variant")
        policy_name = policy_variant.get("name")
        if policy_name == PolicyVariantName.EXTENSION.value:
            _validate_extension_envelope(policy_variant, issues, "policy_variant")
        video_action_policy_names = {
            PolicyVariantName.PARALLEL_STREAM.value,
            PolicyVariantName.DUAL_EXPERT.value,
        }
        if policy_name in video_action_policy_names:
            _validate_video_action_policy_contract(
                policy_variant,
                data=data,
                sample_construction=sample_construction,
                issues=issues,
            )
        if policy_name == PolicyVariantName.PARALLEL_STREAM.value:
            for derived_field in (
                "runtime_mode",
                "current_block_coupling",
                "variant_profile",
                "video_condition_on_action",
            ):
                if derived_field in policy_variant:
                    issues.error(
                        f"policy_variant.{derived_field}",
                        "Parallel Stream derives this value from "
                        "`policy_variant.program`; remove this field.",
                    )
        if policy_name == PolicyVariantName.DUAL_EXPERT.value:
            for removed_field in (
                "runtime_mode",
                "current_block_coupling",
                "video_can_attend_action",
            ):
                if removed_field in policy_variant:
                    issues.error(
                        f"policy_variant.{removed_field}",
                        "Dual Expert derives execution and attention semantics from "
                        "`policy_variant.program`; remove this field.",
                    )
            _validate_enum(
                policy_variant,
                "condition_mode",
                DualExpertConditionMode,
                issues,
                "policy_variant",
            )
            _validate_enum(
                policy_variant,
                "action_expert_init_mode",
                DualExpertActionExpertInitMode,
                issues,
                "policy_variant",
            )
        if policy_name == PolicyVariantName.CAUSAL_VIDEO_PREDICTION.value:
            if "require_text_conditioning" in policy_variant:
                issues.error(
                    "policy_variant.require_text_conditioning",
                    "This field was removed; select "
                    "`policy_variant.text_conditioning_mode: task_prompt` or "
                    "`disabled`.",
                )
            _validate_enum(
                policy_variant,
                "text_conditioning_mode",
                TextConditioningMode,
                issues,
                "policy_variant",
            )
            for field_name in ("chunk_size", "window_size"):
                if training is not None and field_name in training:
                    issues.error(
                        f"training.{field_name}",
                        "Causal video sample geometry is configured by "
                        "`data.sample_construction.causal_prefix_suffix_buckets`; "
                        "remove this unused generic training field.",
                    )
            guidance_scale = (inference or {}).get("guidance_scale", 1.0)
            text_conditioning_mode = policy_variant.get(
                "text_conditioning_mode",
                TextConditioningMode.TASK_PROMPT.value,
            )
            if text_conditioning_mode == TextConditioningMode.DISABLED.value:
                if dropout_probability not in (None, 0.0):
                    issues.error(
                        "training.text_condition_dropout_prob",
                        "Disabled causal-video text conditioning requires 0.0; "
                        "every sample already uses the blank-text embedding.",
                    )
                if guidance_scale != 1.0:
                    issues.error(
                        "inference.guidance_scale",
                        "Disabled causal-video text conditioning requires 1.0; "
                        "conditioned and unconditioned branches are identical.",
                    )
            elif (
                text_conditioning_mode == TextConditioningMode.TASK_PROMPT.value
                and dropout_probability is not None
                and dropout_probability >= 1.0
            ):
                issues.error(
                    "training.text_condition_dropout_prob",
                    "Task-prompt causal-video conditioning requires a probability "
                    "below 1.0; select `text_conditioning_mode: disabled` for "
                    "unconditional training.",
                )
            if (
                isinstance(dropout_probability, (int, float))
                and not isinstance(dropout_probability, bool)
                and float(dropout_probability) > 0.0
                and (
                    trainer is None
                    or trainer.get("batch_adapter")
                    != BatchAdapterName.LATENTS.value
                )
            ):
                issues.error(
                    "trainer.batch_adapter",
                    "Causal video text-condition dropout requires "
                    "`trainer.batch_adapter=latents`; the views adapter does not "
                    "provide encoded text context.",
                )
            if (
                isinstance(guidance_scale, (int, float))
                and not isinstance(guidance_scale, bool)
                and float(guidance_scale) > 1.0
                and isinstance(dropout_probability, (int, float))
                and not isinstance(dropout_probability, bool)
                and float(dropout_probability) <= 0.0
            ):
                issues.error(
                    "inference.guidance_scale",
                    "Causal video classifier-free guidance requires "
                    "`training.text_condition_dropout_prob > 0`; use guidance 1.0 "
                    "when no unconditional branch was trained.",
                )
        _validate_fixed_conditional_program(policy_variant, data, issues)
        _validate_program_timestep_contract(policy_variant, issues)
        if _active_dynamics_routes(data):
            if trainer is None or trainer.get("batch_adapter") != BatchAdapterName.LATENTS.value:
                issues.error(
                    "trainer.batch_adapter",
                    "Active `data.dynamics_routing.routes` require `trainer.batch_adapter=latents`.",
                )
            effective_sample_order = (sample_construction or {}).get(
                "sample_order_mode",
                SampleOrderMode.REPLACEMENT.value,
            )
            if effective_sample_order != SampleOrderMode.REPLACEMENT.value:
                issues.error(
                    "data.sample_construction.sample_order_mode",
                    "`sample_order_mode` must be `replacement` with active dynamics "
                    "routes because route weights define replacement probabilities.",
                )
            if (
                sample_construction is not None
                and sample_construction.get("sample_weight_mode") not in (None, SampleWeightMode.UNIFORM.value)
            ):
                issues.error(
                    "data.sample_construction.sample_weight_mode",
                    "`sample_weight_mode` must be `uniform` with active dynamics routes "
                    "because the dynamics router only preserves parity for uniform replacement draws.",
                )
        _validate_positive_ints(policy_variant, issues, "policy_variant", ("hidden_size",))
    if action_decoder is not None:
        _validate_enum(action_decoder, "name", ActionDecoderName, issues, "action_decoder")
        if action_decoder.get("name") == ActionDecoderName.EXTENSION.value:
            _validate_extension_envelope(action_decoder, issues, "action_decoder")
        _validate_positive_ints(action_decoder, issues, "action_decoder", ("hidden_size", "action_dim"))
    _validate_action_horizons(action_schema, policy_variant, action_decoder, issues)
    _validate_action_schema_compatibility(action_schema, action_decoder, issues)

    if trainer is not None:
        _validate_enum(trainer, "accelerator", TrainerAccelerator, issues, "trainer")
        _validate_enum(trainer, "batch_adapter", BatchAdapterName, issues, "trainer")
        _validate_enum(trainer, "precision", TrainerPrecision, issues, "trainer")
        if "runtime_backbone_export_components" in trainer:
            try:
                validate_runtime_backbone_components(
                    trainer["runtime_backbone_export_components"],
                    scope="`trainer.runtime_backbone_export_components`",
                )
            except (TypeError, ValueError) as exc:
                issues.error(
                    "trainer.runtime_backbone_export_components",
                    str(exc),
                )
        _validate_positive_ints(
            trainer,
            issues,
            "trainer",
            ("max_epochs", "devices", "log_every_n_steps", "validation_interval"),
        )

    validation = _mapping(raw.get("validation"))
    if validation is not None:
        _validate_validation_config(
            validation,
            issues,
            data=data,
            policy_variant=policy_variant,
        )


def _validate_extension_envelope(
    section: Mapping[str, Any],
    issues: _IssueBuilder,
    path: str,
) -> None:
    _validate_enum(
        section,
        "proprio_context_mode",
        ProprioContextMode,
        issues,
        path,
    )
    _validate_enum(
        section,
        "text_conditioning_mode",
        TextConditioningMode,
        issues,
        path,
    )
    if "dynamics_mode_context_enabled" in section and not isinstance(
        section["dynamics_mode_context_enabled"], bool
    ):
        issues.error(
            f"{path}.dynamics_mode_context_enabled",
            "Expected a boolean.",
        )
    extension_type = section.get("extension_type")
    if (
        not isinstance(extension_type, str)
        or not extension_type.strip()
        or extension_type != extension_type.strip()
    ):
        issues.error(
            f"{path}.extension_type",
            "Expected a non-empty string without surrounding whitespace.",
        )
    if "options" in section:
        options = section["options"]
        if not isinstance(options, Mapping):
            issues.error(f"{path}.options", "Expected a mapping of extension-owned options.")
        elif not all(isinstance(key, str) for key in options):
            issues.error(f"{path}.options", "Expected extension option keys to be strings.")


def _validate_eval_config(raw: Mapping[str, Any], issues: _IssueBuilder) -> None:
    experiment_config = raw.get("experiment_config")
    if experiment_config is None:
        issues.error("experiment_config", "Eval configs must point at an experiment config.")
    elif not isinstance(experiment_config, str):
        issues.error("experiment_config", "Expected a string path.")
    else:
        target = _resolve_relative(issues.source_path, experiment_config)
        if not target.exists():
            issues.error("experiment_config", f"Referenced config does not exist: {experiment_config}")
    _validate_enum(raw, "mode", EvalMode, issues, "")
    _validate_enum(raw, "split", DataSplit, issues, "")
    _validate_positive_ints(
        raw,
        issues,
        "",
        ("max_batches", "max_trajectories", "max_steps_per_trajectory", "batch_size"),
    )


def _validate_validation_config(
    validation: Mapping[str, Any],
    issues: _IssueBuilder,
    *,
    data: Mapping[str, Any],
    policy_variant: Mapping[str, Any] | None,
) -> None:
    tasks = validation.get("auxiliary_tasks", ())
    if tasks is None:
        return
    if not isinstance(tasks, list):
        issues.error("validation.auxiliary_tasks", "Expected a list of auxiliary validation task mappings.")
        return
    seen_names: set[str] = set()
    seen_phases: set[str] = set()
    try:
        fixed_mode = (
            fixed_conditioning_mode_for_program(policy_variant.get("program"))
            if policy_variant is not None
            else None
        )
    except (TypeError, ValueError):
        fixed_mode = None
    for index, task in enumerate(tasks):
        task_path = f"validation.auxiliary_tasks[{index}]"
        if not isinstance(task, Mapping):
            issues.error(task_path, "Expected a mapping.")
            continue
        name = task.get("name")
        if not isinstance(name, str) or not name:
            issues.error(f"{task_path}.name", "Expected a non-empty string.")
        elif name in seen_names:
            issues.error(f"{task_path}.name", f"Duplicate auxiliary validation task name {name!r}.")
        else:
            seen_names.add(name)
        report_prefix = task.get("report_prefix", name)
        task_runs = task.get("enabled", True) is not False and task.get("max_batches", 16) != 0
        if report_prefix is not None:
            if not isinstance(report_prefix, str) or not report_prefix:
                issues.error(f"{task_path}.report_prefix", "Expected a non-empty string when set.")
            elif task_runs and report_prefix in seen_phases:
                issues.error(
                    f"{task_path}.report_prefix",
                    f"Duplicate auxiliary validation report prefix {report_prefix!r}.",
                )
            elif task_runs:
                seen_phases.add(report_prefix)
        _validate_enum(task, "mode_override", DynamicsObjective, issues, task_path)
        effective_mode = task.get("mode_override")
        if effective_mode is None and fixed_mode is not None:
            effective_mode = fixed_mode.value
        if (
            effective_mode
            in {
                DynamicsObjective.ACTION_CONDITIONED_VIDEO.value,
                DynamicsObjective.VIDEO_CONDITIONED_ACTION.value,
            }
            and task.get("drop_text_conditioning") is not None
        ):
            issues.error(
                f"{task_path}.drop_text_conditioning",
                "Conditional FDM/IDM validation always removes task text; "
                "remove this field.",
            )
        if (
            task_runs
            and fixed_mode is not None
            and task.get("mode_override") is not None
            and task.get("mode_override") != fixed_mode.value
        ):
            issues.error(
                f"{task_path}.mode_override",
                f"Policy program {policy_variant.get('program')!r} fixes auxiliary "
                f"validation mode to {fixed_mode.value!r}.",
            )
        _validate_enum(task, "dataset_split", DataSplit, issues, task_path)
        _validate_enum(task, "source", AuxiliaryValidationSource, issues, task_path)
        if (
            task_runs
            and effective_mode
            in {
                DynamicsObjective.ACTION_CONDITIONED_VIDEO.value,
                DynamicsObjective.VIDEO_CONDITIONED_ACTION.value,
            }
        ):
            if not _active_dynamics_routes(data):
                issues.error(
                    "data.dynamics_routing.routes",
                    f"Conditional auxiliary validation task {task.get('name')!r} "
                    "requires active routes for the target-only t0 projection.",
                )
            route_modes = {
                str(route.get("mode"))
                for route in _active_dynamics_routes(data)
            }
            if (
                task.get("source", AuxiliaryValidationSource.DATASET.value)
                == AuxiliaryValidationSource.DATASET.value
                and route_modes != {effective_mode}
            ):
                issues.error(
                    f"{task_path}.source",
                    "Conditional auxiliary validation may use the dataset view only "
                    "when every active route has the same objective.",
                )
        max_batches = task.get("max_batches", 16)
        if max_batches is not None:
            value = _optional_int(max_batches)
            if value is None or value < 0:
                issues.error(f"{task_path}.max_batches", "Expected a non-negative integer or null.")
        for bool_key in ("enabled", "drop_text_conditioning"):
            if bool_key in task and task[bool_key] is not None and not isinstance(task[bool_key], bool):
                issues.error(f"{task_path}.{bool_key}", "Expected a boolean or null.")
