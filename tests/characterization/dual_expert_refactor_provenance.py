from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from open_wam.configs import (
    ExperimentConfig,
    current_block_coupling_for_program,
    load_experiment_config,
)

from .dual_expert_refactor_contract import (
    DualExpertMethodSpec,
    DualExpertTrainingProfile,
    apply_gjd_ablation,
    apply_ground_truth_training_profile,
)

_MISSING = object()
_MISSING_JSON = {"missing": True}
_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "experiments"

# Saved schema-v1 checkpoint configs predate the public architecture naming.
# Runtime contracts are canonical; this table is only a deserialization bridge
# for immutable checkpoint metadata.
_CHECKPOINT_FIELD_ALIASES = {
    "policy_variant.sequence_contract": (
        "policy_variant.parallel_sequence_contract",
    ),
}

_LEGACY_GJD_ROUTE_FIELDS = (
    ("real_joint_weight", "real_demo", "joint"),
    (
        "real_action_conditioned_video_weight",
        "real_demo",
        "action_conditioned_video",
    ),
    (
        "real_video_conditioned_action_weight",
        "real_demo",
        "video_conditioned_action",
    ),
    (
        "counterfactual_action_conditioned_video_weight",
        "counterfactual_dynamics",
        "action_conditioned_video",
    ),
    (
        "counterfactual_video_conditioned_action_weight",
        "counterfactual_dynamics",
        "video_conditioned_action",
    ),
)
_CHECKPOINT_DYNAMICS_ROUTING_PATHS = (
    "data.dynamics_routing",
    "data.generalist_dynamics_mixture",
)
_DYNAMICS_ROUTES_FIELD = "data.dynamics_routing.routes"

_COMMON_FIELDS = (
    "policy_variant.current_block_coupling",
    "policy_variant.sequence_contract",
    "policy_variant.noisy_video_condition_prob",
    "policy_variant.proprio_context_mode",
    "policy_variant.context_condition_latent_source",
    "policy_variant.use_condition_latents",
    "policy_variant.require_condition_latents",
    "policy_variant.joint_timestep_coupling",
    "data.replay_status_policy",
    "data.val_replay_status_policy",
    "data.require_replay_status",
    "data.val_require_replay_status",
    "data.sample_construction.mode",
    "data.sample_construction.sample_order_mode",
    "data.sample_construction.randomize_geometry",
    "data.sample_construction.segment_min_frames",
    "data.sample_construction.segment_max_frames",
    "data.sample_construction.segment_length_stride",
    "data.sample_construction.segment_locality_block_size",
    "data.sample_construction.window_size",
    "data.sample_construction.randomize_segment_length",
    "data.sample_construction.randomize_segment_start",
    "data.sample_construction.require_full_segment",
    "data.sample_construction.task_start_power",
    "data.sample_construction.demo_count_power",
    "data.sample_construction.trajectory_start_power",
    "data.sample_construction.sample_weight_mode",
    "data.sample_construction.condition_source_frame_offset",
    "data.sample_construction.start_padding_frames",
    "data.sample_construction.target_alignment",
    "training.chunk_size",
    "training.window_size",
    "training.sample_loss_weight_mode",
    "trainer.checkpoint_mode",
    "inference.frame_chunk_size",
    "data.action_schema.action_horizon",
)

_GJD_FIELDS = (
    "policy_variant.generalist_mode_text_token",
    "data.dynamics_routing.routes",
)


def checkpoint_provenance_report(
    *,
    method: DualExpertMethodSpec,
    expected_config: ExperimentConfig,
    resolved_config_path: Path,
) -> dict[str, Any]:
    """Compare a checkpoint's saved training contract with the exercised graph."""

    resolved_config_path = resolved_config_path.expanduser().resolve()
    with resolved_config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"Expected a YAML mapping in checkpoint config {resolved_config_path}."
        )

    expected = expected_checkpoint_contract(
        method=method,
        config=expected_config,
    )
    fields = tuple(expected)
    actual = {
        field: canonical_checkpoint_contract_value(
            field,
            _read_mapping_path(raw, field),
        )
        for field in fields
    }
    mismatches = [
        {
            "field": field,
            "expected": expected[field],
            "actual": actual[field],
        }
        for field in fields
        if expected[field] != actual[field]
    ]
    return {
        "resolved_config_path": str(resolved_config_path),
        "resolved_config_sha256": _sha256_file(resolved_config_path),
        "strict_match": not mismatches,
        "contract_fields": list(fields),
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
    }


def apply_checkpoint_provenance_policy(
    report: Mapping[str, Any],
    *,
    accepted_origin_mismatch_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Require an exact manifest allowlist for approved checkpoint origins."""

    mismatches = report.get("mismatches")
    if not isinstance(mismatches, list):
        raise TypeError("Checkpoint provenance report has no mismatch list.")
    actual_fields = tuple(str(item["field"]) for item in mismatches)
    accepted_fields = tuple(
        dict.fromkeys(
            _canonical_checkpoint_field(field)
            for field in accepted_origin_mismatch_fields
        )
    )
    actual_set = set(actual_fields)
    accepted_set = set(accepted_fields)
    return {
        **report,
        "accepted_origin_mismatch_fields": list(accepted_fields),
        "unaccepted_origin_mismatch_fields": sorted(actual_set - accepted_set),
        "unused_accepted_origin_mismatch_fields": sorted(accepted_set - actual_set),
        "accepted_for_characterization": actual_set == accepted_set,
    }


def build_checkpoint_source_contract_config(
    method: DualExpertMethodSpec,
) -> ExperimentConfig:
    """Resolve the source training contract independently of runtime overrides."""

    config = load_experiment_config(_CONFIG_ROOT / f"{method.config_name}.yaml")
    if method.is_gjd:
        return apply_gjd_ablation(config, method)
    return apply_ground_truth_training_profile(
        config,
        DualExpertTrainingProfile.FULL_SEGMENT_W64,
    )


def expected_checkpoint_contract(
    *,
    method: DualExpertMethodSpec,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Return only fields that define the characterized training semantics."""

    fields = _COMMON_FIELDS + (_GJD_FIELDS if method.is_gjd else ())
    if method.gjd_ablation == "pure_joint":
        # Conditional buckets are disabled in this ablation, so their source
        # weights and history geometry do not describe the trained graph.
        fields = tuple(
            field
            for field in fields
            if not field.startswith("data.dynamics_routing.")
        )
    return {
        field: canonical_checkpoint_contract_value(
            field,
            _read_attribute_path(config, field),
        )
        for field in fields
    }


def assert_checkpoint_provenance(
    report: Mapping[str, Any],
    *,
    asset_id: str,
) -> None:
    accepted = report.get("accepted_for_characterization")
    if accepted is True:
        return
    mismatches = report.get("mismatches")
    if not isinstance(mismatches, list):
        raise TypeError("Checkpoint provenance report has no mismatch list.")
    if not mismatches and accepted is not False:
        return
    unaccepted = report.get("unaccepted_origin_mismatch_fields") or ()
    unused = report.get("unused_accepted_origin_mismatch_fields") or ()
    preview = "; ".join(
        f"{item['field']}: expected {item['expected']!r}, got {item['actual']!r}"
        for item in mismatches[:8]
    )
    remainder = len(mismatches) - min(8, len(mismatches))
    suffix = "" if remainder <= 0 else f"; plus {remainder} more"
    policy_details = ""
    if unaccepted or unused:
        policy_details = (
            f"; unaccepted fields={list(unaccepted)!r}; "
            f"unused allowlist fields={list(unused)!r}"
        )
    raise AssertionError(
        f"Checkpoint {asset_id!r} was not trained under the strict "
        f"characterization contract: {preview}{suffix}{policy_details}"
    )


def _read_attribute_path(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return _MISSING
        current = getattr(current, part)
    return current


def _read_mapping_path(value: Mapping[str, Any], dotted_path: str) -> Any:
    resolved = _read_mapping_path_exact(value, dotted_path)
    if resolved is not _MISSING:
        return resolved
    if dotted_path == _DYNAMICS_ROUTES_FIELD:
        resolved = _checkpoint_dynamics_routes(value)
        if resolved is not _MISSING:
            return resolved
    if dotted_path == "policy_variant.current_block_coupling":
        program = _read_mapping_path_exact(value, "policy_variant.program")
        if program is not _MISSING:
            return current_block_coupling_for_program(program).value
    for alias in _CHECKPOINT_FIELD_ALIASES.get(dotted_path, ()):
        resolved = _read_mapping_path_exact(value, alias)
        if resolved is not _MISSING:
            return resolved
    return _MISSING


def _checkpoint_dynamics_routes(value: Mapping[str, Any]) -> Any:
    """Project immutable routing schemas into the canonical route contract."""

    for routing_path in _CHECKPOINT_DYNAMICS_ROUTING_PATHS:
        routing = _read_mapping_path_exact(value, routing_path)
        if not isinstance(routing, Mapping):
            continue
        if "routes" in routing:
            return _canonical_dynamics_routes(routing["routes"])
        if any(
            field_name in routing
            for field_name, _, _ in _LEGACY_GJD_ROUTE_FIELDS
        ):
            return [
                {"source": source, "mode": mode, "weight": weight}
                for field_name, source, mode in _LEGACY_GJD_ROUTE_FIELDS
                if (weight := routing.get(field_name)) is not None
                and float(weight) > 0.0
            ]
    return _MISSING


def _read_mapping_path_exact(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _canonical_checkpoint_field(field: str) -> str:
    for canonical, aliases in _CHECKPOINT_FIELD_ALIASES.items():
        if field == canonical or field in aliases:
            return canonical
    return field


def canonical_checkpoint_contract_value(field: str, value: Any) -> Any:
    """Normalize a persisted value through its typed config contract."""

    normalized = _normalize_contract_value(value)
    if field in {
        _DYNAMICS_ROUTES_FIELD,
        "data.generalist_dynamics_mixture.routes",
    }:
        return _canonical_dynamics_routes(normalized)
    if not isinstance(normalized, str):
        return normalized
    return normalized


def _canonical_dynamics_routes(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    routes: list[Any] = []
    for route in value:
        if not isinstance(route, Mapping):
            routes.append(route)
            continue
        canonical_route = dict(route)
        if canonical_route.get("source") == "counterfactual":
            canonical_route["source"] = "counterfactual_dynamics"
        routes.append(dict(sorted(canonical_route.items())))
    return routes


def _normalize_contract_value(value: Any) -> Any:
    if value is _MISSING:
        return _MISSING_JSON
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_contract_value(asdict(value))
    if isinstance(value, Enum):
        return _normalize_contract_value(value.value)
    if isinstance(value, Mapping):
        normalized = {
            str(_normalize_contract_value(key)): _normalize_contract_value(item)
            for key, item in value.items()
        }
        return dict(sorted(normalized.items()))
    if isinstance(value, (tuple, list)):
        return [_normalize_contract_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
