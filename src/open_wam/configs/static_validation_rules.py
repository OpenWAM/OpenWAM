"""Top-level experiment, evaluation, and auxiliary-validation rules."""

from __future__ import annotations

from typing import Any, Mapping

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
    BatchAdapterName,
    BackboneImplementation,
    CurrentBlockCoupling,
    DataSplit,
    EvalMode,
    GeneralistTrainingParadigm,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    LatentTemporalLayout,
    MoTActionExpertInitMode,
    MoTConditionMode,
    MoTRuntimeMode,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelRuntimeMode,
    ParallelSequenceContract,
    ParallelStreamVariantProfile,
    PolicyVariantName,
    ProprioContextMode,
    ReplayStatusPolicy,
    SampleWeightMode,
    TrainerAccelerator,
    TrainerPrecision,
)
from .static_validation_contracts import _IssueBuilder
from .static_validation_data import (
    _validate_action_mapping,
    _validate_action_schema_compatibility,
    _validate_generalist_dynamics_mixture,
    _validate_sample_construction,
)
from .static_validation_policy import (
    _validate_action_horizons,
    _validate_joint_denoise_training_mode_probs,
    _validate_mot_generalist_training_mode_probs,
    _validate_parallel_sequence_contract_static,
    _validate_single_frame_condition_offset,
    _warn_deprecated_text_proprio_context,
)
from .static_validation_primitives import (
    _mapping,
    _optional_int,
    _resolve_relative,
    _validate_enum,
    _validate_positive_ints,
)


def _validate_experiment_config(raw: Mapping[str, Any], issues: "_IssueBuilder", *, relaxed: bool) -> None:
    required = ("data",) if relaxed else ("data", "backbone", "trainer")
    for key in required:
        if key not in raw:
            issues.error(key, "Missing required top-level section.")

    data = _mapping(raw.get("data"))
    if data is None:
        return
    if not data.get("dataset_type") and not data.get("dataset_name"):
        issues.error("data", "Expected `dataset_type` or `dataset_name`.")
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
    generalist_dynamics = _mapping(data.get("generalist_dynamics_mixture"))
    if generalist_dynamics is not None:
        _validate_generalist_dynamics_mixture(generalist_dynamics, issues)

    backbone = _mapping(raw.get("backbone"))
    if backbone is not None:
        _validate_enum(backbone, "implementation", BackboneImplementation, issues, "backbone")
        _validate_enum(backbone, "attn_mode", AttentionMode, issues, "backbone")
        _validate_enum(backbone, "train_attn_mode", AttentionMode, issues, "backbone")
        _validate_enum(backbone, "infer_attn_mode", AttentionMode, issues, "backbone")
        _validate_positive_ints(backbone, issues, "backbone", ("hidden_size", "num_layers", "num_heads"))

    trainer = _mapping(raw.get("trainer"))
    policy_variant = _mapping(raw.get("policy_variant"))
    action_decoder = _mapping(raw.get("action_decoder"))
    action_head = _mapping(raw.get("action_head"))
    if not relaxed and policy_variant is None and action_head is None:
        issues.error("policy_variant", "Expected `policy_variant` or legacy `action_head`.")
    if action_head is not None:
        issues.warning("action_head", "Legacy compatibility section; prefer `policy_variant` + `action_decoder`.")
        _validate_positive_ints(action_head, issues, "action_head", ("hidden_size", "action_dim", "action_horizon"))
    if policy_variant is not None:
        _validate_enum(policy_variant, "name", PolicyVariantName, issues, "policy_variant")
        _validate_enum(policy_variant, "attach_site", AttachSite, issues, "policy_variant")
        if policy_variant.get("name") == PolicyVariantName.EXTENSION.value:
            _validate_extension_envelope(policy_variant, issues, "policy_variant")
        if policy_variant.get("name") == PolicyVariantName.PARALLEL_STREAM.value:
            _validate_enum(policy_variant, "runtime_mode", ParallelRuntimeMode, issues, "policy_variant")
            _validate_enum(policy_variant, "variant_profile", ParallelStreamVariantProfile, issues, "policy_variant")
            _validate_enum(policy_variant, "current_block_coupling", CurrentBlockCoupling, issues, "policy_variant")
            _validate_enum(policy_variant, "joint_timestep_coupling", JointTimestepCoupling, issues, "policy_variant")
            _validate_enum(policy_variant, "parallel_sequence_contract", ParallelSequenceContract, issues, "policy_variant")
            _validate_enum(
                policy_variant,
                "generalist_training_paradigm",
                GeneralistTrainingParadigm,
                issues,
                "policy_variant",
            )
            _validate_enum(policy_variant, "proprio_context_mode", ProprioContextMode, issues, "policy_variant")
            _warn_deprecated_text_proprio_context(policy_variant, issues)
            _validate_enum(
                policy_variant,
                "context_condition_latent_source",
                ParallelContextConditionLatentSource,
                issues,
                "policy_variant",
            )
            _validate_enum(
                policy_variant,
                "history_stream_visibility",
                ParallelHistoryStreamVisibility,
                issues,
                "policy_variant",
            )
            _validate_single_frame_condition_offset(policy_variant, sample_construction, issues)
            _validate_parallel_sequence_contract_static(policy_variant, sample_construction, issues)
            _validate_joint_denoise_training_mode_probs(policy_variant, issues)
        if policy_variant.get("name") == PolicyVariantName.MOT.value:
            _validate_enum(policy_variant, "runtime_mode", MoTRuntimeMode, issues, "policy_variant")
            _validate_enum(policy_variant, "condition_mode", MoTConditionMode, issues, "policy_variant")
            _validate_enum(policy_variant, "action_expert_init_mode", MoTActionExpertInitMode, issues, "policy_variant")
            _validate_enum(policy_variant, "current_block_coupling", CurrentBlockCoupling, issues, "policy_variant")
            _validate_enum(policy_variant, "joint_timestep_coupling", JointTimestepCoupling, issues, "policy_variant")
            _validate_enum(policy_variant, "parallel_sequence_contract", ParallelSequenceContract, issues, "policy_variant")
            _validate_enum(policy_variant, "proprio_context_mode", ProprioContextMode, issues, "policy_variant")
            _warn_deprecated_text_proprio_context(policy_variant, issues)
            _validate_enum(
                policy_variant,
                "context_condition_latent_source",
                ParallelContextConditionLatentSource,
                issues,
                "policy_variant",
            )
            _validate_enum(
                policy_variant,
                "history_stream_visibility",
                ParallelHistoryStreamVisibility,
                issues,
                "policy_variant",
            )
            _validate_enum(
                policy_variant,
                "generalist_training_paradigm",
                GeneralistTrainingParadigm,
                issues,
                "policy_variant",
            )
            _validate_single_frame_condition_offset(policy_variant, sample_construction, issues)
            _validate_parallel_sequence_contract_static(policy_variant, sample_construction, issues)
            _validate_mot_generalist_training_mode_probs(policy_variant, data, issues)
        if policy_variant.get("generalist_training_paradigm") == GeneralistTrainingParadigm.MIXED_DYNAMICS.value:
            if trainer is None or trainer.get("batch_adapter") != BatchAdapterName.LATENTS.value:
                issues.error(
                    "trainer.batch_adapter",
                    "`generalist_training_paradigm=mixed_dynamics` requires `trainer.batch_adapter=latents`.",
                )
            if (
                sample_construction is not None
                and sample_construction.get("sample_weight_mode") not in (None, SampleWeightMode.UNIFORM.value)
            ):
                issues.error(
                    "data.sample_construction.sample_weight_mode",
                    "`sample_weight_mode` must be `uniform` with `generalist_training_paradigm=mixed_dynamics` "
                    "because the mixed-dynamics wrapper only preserves parity for uniform replacement draws.",
                )
        _validate_positive_ints(policy_variant, issues, "policy_variant", ("hidden_size",))
    if action_decoder is not None:
        _validate_enum(action_decoder, "name", ActionDecoderName, issues, "action_decoder")
        if action_decoder.get("name") == ActionDecoderName.EXTENSION.value:
            _validate_extension_envelope(action_decoder, issues, "action_decoder")
        _validate_positive_ints(action_decoder, issues, "action_decoder", ("hidden_size", "action_dim"))
    _validate_action_horizons(action_schema, policy_variant, action_decoder, issues)
    _validate_action_schema_compatibility(action_schema, action_decoder, action_head, issues)

    if trainer is not None:
        _validate_enum(trainer, "accelerator", TrainerAccelerator, issues, "trainer")
        _validate_enum(trainer, "batch_adapter", BatchAdapterName, issues, "trainer")
        _validate_enum(trainer, "precision", TrainerPrecision, issues, "trainer")
        _validate_positive_ints(
            trainer,
            issues,
            "trainer",
            ("max_epochs", "devices", "log_every_n_steps", "validation_interval"),
        )

    validation = _mapping(raw.get("validation"))
    if validation is not None:
        _validate_validation_config(validation, issues)


def _validate_extension_envelope(
    section: Mapping[str, Any],
    issues: "_IssueBuilder",
    path: str,
) -> None:
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


def _validate_eval_config(raw: Mapping[str, Any], issues: "_IssueBuilder") -> None:
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


def _validate_validation_config(validation: Mapping[str, Any], issues: "_IssueBuilder") -> None:
    tasks = validation.get("auxiliary_tasks", ())
    if tasks is None:
        return
    if not isinstance(tasks, list):
        issues.error("validation.auxiliary_tasks", "Expected a list of auxiliary validation task mappings.")
        return
    seen_names: set[str] = set()
    seen_phases: set[str] = set()
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
        _validate_enum(task, "mode_override", JointDenoiseTrainingMode, issues, task_path)
        _validate_enum(task, "dataset_split", DataSplit, issues, task_path)
        _validate_enum(task, "source", AuxiliaryValidationSource, issues, task_path)
        max_batches = task.get("max_batches", 16)
        if max_batches is not None:
            value = _optional_int(max_batches)
            if value is None or value < 0:
                issues.error(f"{task_path}.max_batches", "Expected a non-negative integer or null.")
        for bool_key in ("enabled", "drop_text_conditioning"):
            if bool_key in task and task[bool_key] is not None and not isinstance(task[bool_key], bool):
                issues.error(f"{task_path}.{bool_key}", "Expected a boolean or null.")
