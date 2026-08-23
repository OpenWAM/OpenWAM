"""Dependency-light public static configuration validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .config_paths import resolve_config_path_alias
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
    CurrentBlockCoupling,
    DataSplit,
    DualExpertActionExpertInitMode,
    DualExpertConditionMode,
    EvalMode,
    DynamicsObjective,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    LatentTemporalLayout,
    PaddedTargetPolicy,
    ParallelRuntimeMode,
    ParallelStreamVariantProfile,
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
    VideoActionProgram,
    VideoActionSequenceContract,
    WindowSamplingMode,
)
from .policy_compatibility import normalize_video_action_config_fields
from .static_validation_contracts import (
    StaticConfigIssue,
    StaticConfigReport,
    _IssueBuilder,
)
from .static_validation_primitives import (
    ENUM_VALUE_ALIASES,
    LOCAL_PATH_PATTERN,
    _find_repo_root,
    _read_yaml_mapping,
    _validate_local_path_placeholders,
)
from .static_validation_rules import _validate_eval_config, _validate_experiment_config


def validate_config_file(path: str | Path, *, repo_root: str | Path | None = None) -> StaticConfigReport:
    """Validate one Open-WAM YAML config without importing model/runtime code."""

    source_path = resolve_config_path_alias(path).resolve()
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _find_repo_root(source_path)
    raw = _read_yaml_mapping(source_path)
    builder = _IssueBuilder(source_path=source_path, repo_root=root)
    try:
        raw = normalize_video_action_config_fields(raw, warn=False)
    except (TypeError, ValueError) as exc:
        builder.error("policy_variant", str(exc))
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


_STATIC_SCHEMA_COMPATIBILITY_EXPORTS = (
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
    ENUM_VALUE_ALIASES,
    EvalMode,
    DynamicsObjective,
    JointTimestepCoupling,
    LOCAL_PATH_PATTERN,
    LatentTemporalLayout,
    DualExpertActionExpertInitMode,
    DualExpertConditionMode,
    ContextConditionLatentSource,
    HistoryStreamVisibility,
    ParallelRuntimeMode,
    VideoActionSequenceContract,
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
    VideoActionProgram,
    WindowSamplingMode,
)

__all__ = [
    "StaticConfigIssue",
    "StaticConfigReport",
    "format_report",
    "reports_to_exit_code",
    "validate_config_file",
    "validate_config_files",
]
