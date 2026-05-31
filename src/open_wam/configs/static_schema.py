from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml

from open_wam.configs.enums import (
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
    MoTGeneralistTrainingMode,
    MoTRuntimeMode,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelRuntimeMode,
    ParallelSequenceContract,
    ParallelStreamVariantProfile,
    PaddedTargetPolicy,
    PolicyVariantName,
    ProprioContextMode,
    ReplayStatusPolicy,
    RolloutContextPolicy,
    SampleOrderMode,
    SampleStateAnchorMode,
    SampleTargetAlignment,
    SampleWeightMode,
    SegmentContextPolicy,
    StrEnum,
    TailPaddingPolicy,
    TrainerAccelerator,
    TrainerPrecision,
    WindowSamplingMode,
)
from open_wam.configs.variant_semantics import probability_map_static_issues


LOCAL_PATH_PATTERN = re.compile(r"\$\{paths\.([A-Za-z0-9_.-]+)\}")
ENUM_VALUE_ALIASES: dict[type[StrEnum], dict[str, str]] = {
    BackboneImplementation: {
        "lingbot_replica": BackboneImplementation.SHARED_TRANSFORMER.value,
    }
}


@dataclass(frozen=True)
class StaticConfigIssue:
    level: str
    path: str
    message: str


@dataclass(frozen=True)
class StaticConfigReport:
    source_path: Path
    errors: tuple[StaticConfigIssue, ...]
    warnings: tuple[StaticConfigIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_config_file(path: str | Path, *, repo_root: str | Path | None = None) -> StaticConfigReport:
    """Validate one Open-WAM YAML config without importing model/runtime code."""

    source_path = Path(path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _find_repo_root(source_path)
    raw = _read_yaml_mapping(source_path)
    builder = _IssueBuilder(source_path=source_path, repo_root=root)
    if "experiment_config" in raw:
        _validate_eval_config(raw, builder)
    else:
        _validate_experiment_config(raw, builder, relaxed=source_path.parent.name == "examples")
    _validate_local_path_placeholders(raw, builder)
    return StaticConfigReport(
        source_path=source_path,
        errors=tuple(builder.errors),
        warnings=tuple(builder.warnings),
    )


def validate_config_files(
    paths: Iterable[str | Path],
    *,
    repo_root: str | Path | None = None,
) -> tuple[StaticConfigReport, ...]:
    return tuple(validate_config_file(path, repo_root=repo_root) for path in paths)


def reports_to_exit_code(reports: Iterable[StaticConfigReport]) -> int:
    return 1 if any(not report.ok for report in reports) else 0


def format_report(report: StaticConfigReport, *, repo_root: str | Path | None = None) -> str:
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _find_repo_root(report.source_path)
    try:
        source = str(report.source_path.relative_to(root))
    except ValueError:
        source = str(report.source_path)
    lines = [f"{source}: {'ok' if report.ok else 'failed'}"]
    for issue in (*report.errors, *report.warnings):
        lines.append(f"  {issue.level}: {issue.path}: {issue.message}")
    return "\n".join(lines)


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
            if sample_construction is not None and sample_construction.get("sample_order_mode") == SampleOrderMode.REPLACEMENT.value:
                issues.error(
                    "data.sample_construction.sample_order_mode",
                    "`sample_order_mode=replacement` is not supported with "
                    "`generalist_training_paradigm=mixed_dynamics` because the mixed-dynamics wrapper owns sampling.",
                )
            if (
                sample_construction is not None
                and sample_construction.get("sample_weight_mode") not in (None, SampleWeightMode.UNIFORM.value)
            ):
                issues.error(
                    "data.sample_construction.sample_weight_mode",
                    "`sample_weight_mode` must be `uniform` with `generalist_training_paradigm=mixed_dynamics` "
                    "because the mixed-dynamics wrapper owns sampling.",
                )
        _validate_positive_ints(policy_variant, issues, "policy_variant", ("hidden_size",))
    if action_decoder is not None:
        _validate_enum(action_decoder, "name", ActionDecoderName, issues, "action_decoder")
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


def _validate_single_frame_condition_offset(
    policy_variant: Mapping[str, Any],
    sample_construction: Mapping[str, Any] | None,
    issues: "_IssueBuilder",
) -> None:
    if (
        policy_variant.get("context_condition_latent_source")
        != ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT.value
    ):
        return
    offset = None if sample_construction is None else _optional_int(sample_construction.get("condition_source_frame_offset"))
    if offset != -1:
        issues.error(
            "data.sample_construction.condition_source_frame_offset",
            "Expected -1 when "
            "`policy_variant.context_condition_latent_source=single_frame_condition_latent`; "
            "offset 0 can expose the first target raw frame.",
        )


def _warn_deprecated_text_proprio_context(policy_variant: Mapping[str, Any], issues: "_IssueBuilder") -> None:
    if policy_variant.get("proprio_context_mode") != ProprioContextMode.TEXT_CONTEXT_TOKEN.value:
        return
    issues.warning(
        "policy_variant.proprio_context_mode",
        "Deprecated text-space proprio token path; current proprio context is "
        "`per_chunk_additive` hidden-state conditioning.",
    )


def _validate_parallel_sequence_contract_static(
    policy_variant: Mapping[str, Any],
    sample_construction: Mapping[str, Any] | None,
    issues: "_IssueBuilder",
) -> None:
    raw_contract = policy_variant.get("parallel_sequence_contract")
    if raw_contract in (None, ParallelSequenceContract.DEFAULT.value):
        return
    try:
        contract = ParallelSequenceContract(str(raw_contract))
    except ValueError:
        return
    if contract not in {
        ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
        ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    }:
        return

    policy_name = policy_variant.get("name")
    if policy_name not in {PolicyVariantName.PARALLEL_STREAM.value, PolicyVariantName.MOT.value}:
        issues.error(
            "policy_variant.parallel_sequence_contract",
            f"`{contract.value}` is only supported for policy_variant.name parallel_stream or mot.",
        )
        return

    if contract == ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO:
        runtime_mode = policy_variant.get("runtime_mode")
        if policy_name == PolicyVariantName.PARALLEL_STREAM.value and runtime_mode not in (
            None,
            ParallelRuntimeMode.LINGBOT_EXACT.value,
            ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED.value,
        ):
            issues.error(
                "policy_variant.runtime_mode",
                "legacy_prefix_single_frame_perchunk_proprio requires runtime_mode "
                "lingbot_exact or lingbot_exact_action_conditioned for parallel_stream.",
            )
        if policy_name == PolicyVariantName.MOT.value and runtime_mode not in (
            None,
            MoTRuntimeMode.NON_JOINT_TWO_STREAM.value,
        ):
            issues.error(
                "policy_variant.runtime_mode",
                "legacy_prefix_single_frame_perchunk_proprio requires runtime_mode=non_joint_two_stream for mot.",
            )

    expected_policy = {
        "proprio_context_mode": ProprioContextMode.PER_CHUNK_ADDITIVE.value,
        "context_condition_latent_source": ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT.value,
        "history_stream_visibility": ParallelHistoryStreamVisibility.VIDEO_ONLY.value,
        "use_condition_latents": True,
        "require_condition_latents": True,
    }
    for key, expected_value in expected_policy.items():
        if key in policy_variant and policy_variant[key] != expected_value:
            issues.error(
                f"policy_variant.{key}",
                f"`parallel_sequence_contract={contract.value}` owns `{key}`; expected {expected_value!r}.",
            )

    if sample_construction is None:
        return
    expected_sample: dict[str, Any] = {
        "condition_source_frame_offset": -1,
        "start_padding_frames": 0,
    }
    if contract == ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO:
        expected_sample.update(
            {
                "target_alignment": SampleTargetAlignment.NEXT_AFTER_CONTEXT.value,
                "rollout_context_policy": RolloutContextPolicy.ONE_FRAME.value,
            }
        )
    else:
        expected_sample["target_alignment"] = SampleTargetAlignment.LEGACY.value
    for key, expected_value in expected_sample.items():
        if key not in sample_construction:
            continue
        actual_value = sample_construction[key]
        if key in {"condition_source_frame_offset", "start_padding_frames"}:
            actual_value = _optional_int(actual_value)
        if actual_value != expected_value:
            issues.error(
                f"data.sample_construction.{key}",
                f"`parallel_sequence_contract={contract.value}` owns `{key}`; expected {expected_value!r}.",
            )


def _validate_action_horizons(
    action_schema: Mapping[str, Any] | None,
    policy_variant: Mapping[str, Any] | None,
    action_decoder: Mapping[str, Any] | None,
    issues: "_IssueBuilder",
) -> None:
    video_only = False
    if action_decoder is not None and action_decoder.get("name") == ActionDecoderName.VIDEO_ONLY.value:
        video_only = True
    if policy_variant is not None and policy_variant.get("name") == PolicyVariantName.CAUSAL_VIDEO_PREDICTION.value:
        video_only = True
    if action_schema is not None:
        for key in ("action_horizon", "state_horizon"):
            value = _optional_int(action_schema.get(key))
            if value is None:
                continue
            if value < 0 or (value == 0 and not video_only):
                issues.error(
                    f"data.action_schema.{key}",
                    "Expected a positive integer except for video-only configs, where zero is allowed.",
                )
    if action_decoder is not None:
        value = _optional_int(action_decoder.get("action_horizon"))
        if value is not None and (value < 0 or (value == 0 and not video_only)):
            issues.error(
                "action_decoder.action_horizon",
                "Expected a positive integer except for video-only configs, where zero is allowed.",
            )


def _validate_joint_denoise_training_mode_probs(
    policy_variant: Mapping[str, Any],
    issues: "_IssueBuilder",
) -> None:
    _validate_probability_map(
        policy_variant,
        issues,
        field_name="joint_denoise_training_mode_probs",
        enum_cls=JointDenoiseTrainingMode,
    )


def _validate_mot_generalist_training_mode_probs(
    policy_variant: Mapping[str, Any],
    data: Mapping[str, Any],
    issues: "_IssueBuilder",
) -> None:
    raw_probs = policy_variant.get("mot_generalist_training_mode_probs")
    if bool(policy_variant.get("generalist_mode_text_token", False)) and raw_probs is None:
        issues.error(
            "policy_variant.generalist_mode_text_token",
            "`generalist_mode_text_token: true` for MoT requires "
            "`policy_variant.mot_generalist_training_mode_probs`.",
        )
    if raw_probs is None:
        return
    if policy_variant.get("current_block_coupling") != CurrentBlockCoupling.JOINT.value:
        issues.error(
            "policy_variant.mot_generalist_training_mode_probs",
            "Expected `current_block_coupling: joint` when MoT generalist sampling is enabled.",
        )
    for key in ("train_batch_size", "val_batch_size"):
        raw_batch_size = data.get(key, 2)
        try:
            batch_size = int(raw_batch_size)
        except (TypeError, ValueError):
            continue
        if batch_size != 1:
            issues.error(
                f"data.{key}",
                "`mot_generalist_training_mode_probs` requires `data.train_batch_size: 1` and "
                "`data.val_batch_size: 1` because M5 GJD samples one mode per segment/forward pass.",
            )
    _validate_probability_map(
        policy_variant,
        issues,
        field_name="mot_generalist_training_mode_probs",
        enum_cls=MoTGeneralistTrainingMode,
    )


def _validate_probability_map(
    policy_variant: Mapping[str, Any],
    issues: "_IssueBuilder",
    *,
    field_name: str,
    enum_cls: type[StrEnum],
) -> None:
    raw_probs = policy_variant.get(field_name)
    for issue in probability_map_static_issues(raw_probs, enum_cls=enum_cls):
        path = f"policy_variant.{field_name}"
        if issue.path_suffix is not None:
            path = f"{path}.{issue.path_suffix}"
        issues.error(path, issue.message)


def _validate_local_path_placeholders(value: Any, issues: "_IssueBuilder", *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = str(key) if not path else f"{path}.{key}"
            _validate_local_path_placeholders(item, issues, path=child_path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_local_path_placeholders(item, issues, path=f"{path}[{index}]")
        return
    if not isinstance(value, str) or "${paths." not in value:
        return
    matches = LOCAL_PATH_PATTERN.findall(value)
    if not matches:
        issues.error(path, "Malformed local path placeholder. Expected `${paths.alias}`.")
    for alias in matches:
        if ".." in alias or alias.startswith(".") or alias.endswith("."):
            issues.error(path, f"Invalid local path alias syntax: {alias!r}.")


def _validate_enum(
    mapping: Mapping[str, Any],
    key: str,
    enum_cls: type[StrEnum],
    issues: "_IssueBuilder",
    path_prefix: str,
) -> None:
    if key not in mapping or mapping[key] is None:
        return
    value = mapping[key]
    if isinstance(value, enum_cls):
        return
    if not isinstance(value, str):
        issues.error(_join_path(path_prefix, key), f"Expected a string enum value for {enum_cls.__name__}.")
        return
    value = ENUM_VALUE_ALIASES.get(enum_cls, {}).get(value, value)
    valid = {item.value for item in enum_cls}
    if value not in valid:
        issues.error(
            _join_path(path_prefix, key),
            f"Invalid {enum_cls.__name__} value {value!r}. Expected one of {sorted(valid)}.",
        )


def _validate_positive_ints(
    mapping: Mapping[str, Any],
    issues: "_IssueBuilder",
    path_prefix: str,
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key not in mapping or mapping[key] is None:
            continue
        value = _optional_int(mapping[key])
        if value is None or value <= 0:
            issues.error(_join_path(path_prefix, key), "Expected a positive integer.")


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _join_path(prefix: str, key: str) -> str:
    return key if not prefix else f"{prefix}.{key}"


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return raw


def _resolve_relative(source_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    local_candidate = (source_path.parent / candidate).resolve()
    if local_candidate.exists():
        return local_candidate
    return (_find_repo_root(source_path) / candidate).resolve()


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


class _IssueBuilder:
    def __init__(self, *, source_path: Path, repo_root: Path) -> None:
        self.source_path = source_path
        self.repo_root = repo_root
        self.errors: list[StaticConfigIssue] = []
        self.warnings: list[StaticConfigIssue] = []

    def error(self, path: str, message: str) -> None:
        self.errors.append(StaticConfigIssue(level="error", path=path or "<root>", message=message))

    def warning(self, path: str, message: str) -> None:
        self.warnings.append(StaticConfigIssue(level="warning", path=path or "<root>", message=message))
