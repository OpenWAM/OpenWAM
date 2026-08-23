"""Compatibility migration for immutable checkpoint-era experiment configs.

Authored configs never pass through this module. It exists only so numerical
characterization and inference can load checkpoint metadata produced before
the current program and dynamics-routing contracts became authoritative.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from . import enums
from .coercion import raw_enum_value
from .policy_compatibility import migrate_checkpoint_video_action_config_fields
from .policy_parallel_stream import (
    parallel_runtime_mode_for_program,
    parallel_video_conditioning_for_program,
)
from .policy_video_action import (
    current_block_coupling_for_program,
    fixed_conditioning_mode_for_program,
    supports_dynamics_routing,
)

_CONDITIONAL_DYNAMICS_MODES = frozenset(
    {
        enums.DynamicsObjective.ACTION_CONDITIONED_VIDEO.value,
        enums.DynamicsObjective.VIDEO_CONDITIONED_ACTION.value,
    }
)

_LEGACY_DYNAMICS_ROUTE_FIELDS = (
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


def _has_positive_conditional_route(routes: Any) -> bool:
    if not isinstance(routes, (list, tuple)):
        return False
    return any(
        raw_enum_value(route.get("mode")) in _CONDITIONAL_DYNAMICS_MODES
        and float(route.get("weight", 0.0)) > 0.0
        for route in routes
        if isinstance(route, Mapping)
    )


def _has_positive_conditional_probability(
    probabilities: Mapping[str, Any] | None,
) -> bool:
    return any(
        raw_enum_value(mode_name) in _CONDITIONAL_DYNAMICS_MODES
        and float(weight) > 0.0
        for mode_name, weight in (probabilities or {}).items()
    )


def _checkpoint_selects_generalist_program(
    *,
    policy_name: str,
    variant_profile: Any,
    routing_marker: bool | None,
    has_conditional_routes: bool,
    legacy_probabilities: Mapping[str, Any] | None,
) -> bool:
    """Distinguish old GJD configs from ordinary joint-policy configs."""

    if routing_marker is True or has_conditional_routes:
        return True
    if _has_positive_conditional_probability(legacy_probabilities):
        return True
    if policy_name == enums.PolicyVariantName.PARALLEL_STREAM.value:
        return (
            raw_enum_value(variant_profile)
            == enums.ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING.value
        )
    # Dual Expert did not synthesize a joint-only probability map for ordinary
    # programs. Its presence is therefore sufficient to identify pure-joint GJD.
    return legacy_probabilities is not None


def _checkpoint_routing_marker(value: Any) -> bool:
    raw_value = raw_enum_value(value)
    markers = {
        "disabled": False,
        "demo_only": False,
        "required": True,
        "dynamics_routed": True,
        "mixed_dynamics": True,
    }
    try:
        return markers[raw_value]
    except KeyError as exc:
        raise ValueError(
            f"Invalid checkpoint dynamics-routing marker {raw_value!r}."
        ) from exc


def _normalize_routing_keys(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], bool | None]:
    normalized = dict(raw)
    routing_marker: bool | None = None
    policy_raw = normalized.get("policy_variant")
    if isinstance(policy_raw, Mapping):
        policy_raw = dict(policy_raw)
        normalized["policy_variant"] = policy_raw
        markers = tuple(
            _checkpoint_routing_marker(policy_raw.pop(field_name))
            for field_name in (
                "generalist_training_paradigm",
                "dynamics_routing_requirement",
            )
            if field_name in policy_raw
        )
        if len(set(markers)) > 1:
            raise ValueError(
                "Checkpoint contains conflicting retired dynamics-routing markers."
            )
        if markers:
            routing_marker = markers[0]

    data_raw = normalized.get("data")
    if not isinstance(data_raw, Mapping):
        return normalized, routing_marker
    data_raw = dict(data_raw)
    normalized["data"] = data_raw
    legacy_routing = data_raw.pop("generalist_dynamics_mixture", None)
    current_routing = data_raw.get("dynamics_routing")
    if (
        legacy_routing is not None
        and current_routing is not None
        and legacy_routing != current_routing
    ):
        raise ValueError(
            "Checkpoint contains conflicting `data.generalist_dynamics_mixture` "
            "and `data.dynamics_routing` mappings."
        )
    routing_raw = current_routing if current_routing is not None else legacy_routing
    if routing_raw is not None:
        data_raw["dynamics_routing"] = routing_raw
    if not isinstance(routing_raw, Mapping):
        return normalized, routing_marker
    routing_raw = dict(routing_raw)
    data_raw["dynamics_routing"] = routing_raw
    routes = routing_raw.get("routes")
    if isinstance(routes, (list, tuple)):
        normalized_routes: list[Any] = []
        for route in routes:
            if not isinstance(route, Mapping):
                normalized_routes.append(route)
                continue
            normalized_route = dict(route)
            if raw_enum_value(normalized_route.get("source")) == "counterfactual":
                normalized_route["source"] = "counterfactual_dynamics"
            normalized_routes.append(normalized_route)
        routing_raw["routes"] = normalized_routes
    return normalized, routing_marker


def _pop_legacy_mode_probabilities(
    policy_raw: dict[str, Any],
) -> Mapping[str, Any] | None:
    legacy_probabilities: Mapping[str, Any] | None = None
    for field_name in (
        "generalist_denoising_mode_probs",
        "joint_denoise_training_mode_probs",
        "mot_generalist_training_mode_probs",
    ):
        candidate = policy_raw.pop(field_name, None)
        if candidate is None:
            continue
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"Checkpoint field `policy_variant.{field_name}` must be a mapping."
            )
        if (
            legacy_probabilities is not None
            and dict(candidate) != dict(legacy_probabilities)
        ):
            raise ValueError(
                "Checkpoint contains conflicting legacy generalist denoising distributions."
            )
        legacy_probabilities = candidate
    return legacy_probabilities


def _migrate_policy_program(
    policy_raw: dict[str, Any],
    *,
    routing_marker: bool | None,
    checkpoint_has_conditional_routes: bool,
    legacy_probabilities: Mapping[str, Any] | None,
) -> str | None:
    policy_name = raw_enum_value(policy_raw.get("name"))
    if policy_name not in {
        enums.PolicyVariantName.DUAL_EXPERT.value,
        enums.PolicyVariantName.PARALLEL_STREAM.value,
        "mot",
    }:
        return None

    legacy_runtime_mode = policy_raw.pop("runtime_mode", None)
    legacy_variant_profile = policy_raw.pop("variant_profile", None)
    legacy_video_conditioning = policy_raw.pop("video_condition_on_action", None)
    legacy_coupling = policy_raw.pop("current_block_coupling", None)
    if policy_name in {enums.PolicyVariantName.DUAL_EXPERT.value, "mot"}:
        policy_raw.pop("video_can_attend_action", None)

    program = policy_raw.get("program")
    if program is None:
        if legacy_coupling is None:
            detail = (
                f" runtime_mode={legacy_runtime_mode!r}"
                if legacy_runtime_mode is not None
                else ""
            )
            raise ValueError(
                "Cannot migrate video/action checkpoint config without an explicit "
                "`program` or `current_block_coupling`; the old runtime mode does not "
                f"uniquely define attention semantics.{detail}"
            )
        coupling = enums.CurrentBlockCoupling(raw_enum_value(legacy_coupling))
        selects_generalist_program = _checkpoint_selects_generalist_program(
            policy_name=policy_name,
            variant_profile=legacy_variant_profile,
            routing_marker=routing_marker,
            has_conditional_routes=checkpoint_has_conditional_routes,
            legacy_probabilities=legacy_probabilities,
        )
        program = (
            enums.VideoActionProgram.GENERALIST_JOINT_DENOISING
            if coupling == enums.CurrentBlockCoupling.JOINT
            and selects_generalist_program
            else enums.VideoActionProgram(coupling.value)
        )
        policy_raw["program"] = program.value

    resolved_program = enums.VideoActionProgram(raw_enum_value(policy_raw["program"]))
    if legacy_coupling is not None:
        expected_coupling = current_block_coupling_for_program(resolved_program)
        resolved_legacy_coupling = enums.CurrentBlockCoupling(
            raw_enum_value(legacy_coupling)
        )
        if resolved_legacy_coupling != expected_coupling:
            raise ValueError(
                "Checkpoint `policy_variant.current_block_coupling` conflicts "
                "with its canonical program: "
                f"program={resolved_program.value!r} requires "
                f"current_block_coupling={expected_coupling.value!r}, got "
                f"{resolved_legacy_coupling.value!r}."
            )

    if policy_name == enums.PolicyVariantName.PARALLEL_STREAM.value:
        _validate_parallel_backend_metadata(
            program=resolved_program,
            runtime_mode=legacy_runtime_mode,
            variant_profile=legacy_variant_profile,
            video_conditioning=legacy_video_conditioning,
        )
    return resolved_program.value


def _validate_parallel_backend_metadata(
    *,
    program: enums.VideoActionProgram,
    runtime_mode: Any,
    variant_profile: Any,
    video_conditioning: Any,
) -> None:
    expected_runtime = parallel_runtime_mode_for_program(program)
    if (
        runtime_mode is not None
        and enums.ParallelRuntimeMode(raw_enum_value(runtime_mode))
        != expected_runtime
    ):
        raise ValueError(
            "Checkpoint `policy_variant.runtime_mode` conflicts with its "
            f"program {program.value!r}; expected {expected_runtime.value!r}."
        )
    expected_profile = (
        enums.ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
        if program == enums.VideoActionProgram.GENERALIST_JOINT_DENOISING
        else enums.ParallelStreamVariantProfile.STANDARD
    )
    if (
        variant_profile is not None
        and enums.ParallelStreamVariantProfile(raw_enum_value(variant_profile))
        != expected_profile
    ):
        raise ValueError(
            "Checkpoint `policy_variant.variant_profile` conflicts with its "
            f"program {program.value!r}; expected {expected_profile.value!r}."
        )
    expected_video_conditioning = parallel_video_conditioning_for_program(program)
    if video_conditioning is None:
        return
    if not isinstance(video_conditioning, bool):
        raise ValueError(
            "Checkpoint `policy_variant.video_condition_on_action` must be boolean."
        )
    if video_conditioning != expected_video_conditioning:
        raise ValueError(
            "Checkpoint `policy_variant.video_condition_on_action` conflicts "
            f"with its program {program.value!r}; expected "
            f"{expected_video_conditioning!r}."
        )


def _legacy_source_weight_routes(
    routing_raw: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Translate source-aware weights used by the old mixed-data adapter."""

    return [
        {"source": source, "mode": mode_name, "weight": routing_raw[field_name]}
        for field_name, source, mode_name in _LEGACY_DYNAMICS_ROUTE_FIELDS
        if field_name in routing_raw and float(routing_raw[field_name]) > 0.0
    ]


def _legacy_real_demo_routes(
    probabilities: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Translate old policy-side mode draws over ordinary demo samples."""

    return [
        {
            "source": enums.DynamicsSource.REAL_DEMO.value,
            "mode": raw_enum_value(mode_name),
            "weight": weight,
        }
        for mode_name, weight in (probabilities or {}).items()
        if float(weight) > 0.0
    ]


def _has_active_route(routes: Any) -> bool:
    return isinstance(routes, (list, tuple)) and any(
        isinstance(route, Mapping) and float(route.get("weight", 0.0)) > 0.0
        for route in routes
    )


def _migrate_dynamics_routes(
    routing_raw: dict[str, Any],
    *,
    program: str | None,
    routing_marker: bool | None,
    legacy_probabilities: Mapping[str, Any] | None,
) -> None:
    fixed_mode = fixed_conditioning_mode_for_program(program)
    legacy_probabilities_have_conditional_mode = (
        _has_positive_conditional_probability(legacy_probabilities)
    )
    legacy_route_fields_present = any(
        field_name in routing_raw
        for field_name, _, _ in _LEGACY_DYNAMICS_ROUTE_FIELDS
    )
    if fixed_mode is not None and (
        legacy_probabilities is not None or legacy_route_fields_present
    ):
        raise ValueError(
            "Strict forward- and inverse-dynamics checkpoint configs cannot be "
            "migrated from legacy GJD probability or source-weight fields. Use "
            "an authored config with one canonical `data.dynamics_routing.routes` authority."
        )

    routes_were_explicit = "routes" in routing_raw
    routes = routing_raw.get("routes")
    if not routes_were_explicit:
        if program != enums.VideoActionProgram.GENERALIST_JOINT_DENOISING.value:
            # The old data config serialized mixture defaults for every policy,
            # but only a mixed-dynamics GJD program ever consumed them.
            routes = []
        elif routing_marker is True:
            routes = _legacy_source_weight_routes(routing_raw)
            if not routes and not legacy_route_fields_present:
                routes = _legacy_real_demo_routes(legacy_probabilities)
        elif legacy_probabilities_have_conditional_mode:
            # demo_only still sampled FDM/IDM policy modes, but exclusively
            # over ordinary demonstrations. Its source-weight table was inert.
            routes = _legacy_real_demo_routes(legacy_probabilities)
        else:
            # Preserve pure-joint GJD's policy-side categorical RNG draw. An
            # empty route table is the canonical pure-joint compatibility path.
            routes = []
        routing_raw["routes"] = routes

    for field_name, _, _ in _LEGACY_DYNAMICS_ROUTE_FIELDS:
        routing_raw.pop(field_name, None)
    routing_raw.pop("conditional_history_frames", None)

    has_active_routes = _has_active_route(routes)
    if not supports_dynamics_routing(program) and (
        routing_marker is True
        or legacy_probabilities_have_conditional_mode
        or has_active_routes
    ):
        raise ValueError(
            "Checkpoint dynamics-routing metadata conflicts with its explicit "
            f"policy program {program!r}. Compatibility loading will not discard "
            "an active execution contract; use the checkpoint's matching authored "
            "program/config."
        )
    if routing_marker is True and not has_active_routes:
        raise ValueError(
            "Checkpoint declares routed dynamics training but contains no positive "
            "source/objective route that can be migrated."
        )
    if routing_marker is False and routes_were_explicit and has_active_routes:
        warnings.warn(
            "Checkpoint contains explicit dynamics routes alongside a stale disabled "
            "routing marker. Explicit routes are authoritative; the retired marker "
            "was ignored.",
            UserWarning,
            stacklevel=3,
        )


def _remove_obsolete_sampling_fields(data_raw: dict[str, Any]) -> None:
    sample_raw = data_raw.get("sample_construction")
    if not isinstance(sample_raw, dict):
        return
    sample_raw = dict(sample_raw)
    data_raw["sample_construction"] = sample_raw

    target_alignment = raw_enum_value(sample_raw.get("target_alignment"))
    if target_alignment == enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT.value:
        for key in ("context_prefix_policy", "context_prefix_frames"):
            sample_raw.pop(key, None)

    mode = raw_enum_value(sample_raw.get("mode"))
    if mode == enums.WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT.value:
        for key in (
            "segment_min_frames",
            "segment_max_frames",
            "randomize_segment_length",
            "randomize_segment_start",
            "require_full_segment",
            "sample_weight_mode",
            "sample_weight_length_power",
        ):
            sample_raw.pop(key, None)


def _normalize_routed_checkpoint_sample_order(
    data_raw: dict[str, Any],
    *,
    routes: Any,
) -> None:
    """Restore the sampler contract consumed by historical routed training."""

    if not _has_active_route(routes):
        return
    sample_raw = data_raw.get("sample_construction")
    if not isinstance(sample_raw, Mapping):
        return
    sample_raw = dict(sample_raw)
    data_raw["sample_construction"] = sample_raw
    sample_order = raw_enum_value(sample_raw.get("sample_order_mode"))
    if sample_order != enums.SampleOrderMode.EPOCH_ORDER.value:
        return
    sample_raw["sample_order_mode"] = enums.SampleOrderMode.REPLACEMENT.value
    warnings.warn(
        "Migrated routed checkpoint sampling from `epoch_order` to "
        "`replacement`; route weights are replacement probabilities and require "
        "the replacement sampler.",
        UserWarning,
        stacklevel=3,
    )


def _disable_unroutable_conditional_validation_tasks(
    raw: dict[str, Any],
    *,
    routes: Any,
) -> None:
    """Disable checkpoint-era probes that cannot obtain a routed source view."""

    if isinstance(routes, (list, tuple)) and any(
        isinstance(route, Mapping) and float(route.get("weight", 0.0)) > 0.0
        for route in routes
    ):
        return
    validation_raw = raw.get("validation")
    if not isinstance(validation_raw, Mapping):
        return
    tasks = validation_raw.get("auxiliary_tasks")
    if not isinstance(tasks, (list, tuple)):
        return

    normalized_tasks: list[Any] = []
    disabled_names: list[str] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            normalized_tasks.append(task)
            continue
        normalized_task = dict(task)
        mode = raw_enum_value(normalized_task.get("mode_override"))
        if (
            mode in _CONDITIONAL_DYNAMICS_MODES
            and normalized_task.get("enabled", True) is not False
            and normalized_task.get("max_batches", 16) != 0
        ):
            normalized_task["enabled"] = False
            disabled_names.append(str(normalized_task.get("name", mode)))
        normalized_tasks.append(normalized_task)

    if not disabled_names:
        return
    normalized_validation = dict(validation_raw)
    normalized_validation["auxiliary_tasks"] = normalized_tasks
    raw["validation"] = normalized_validation
    warnings.warn(
        "Disabled checkpoint-era conditional validation tasks because the "
        "checkpoint has no active dynamics routes: " + ", ".join(disabled_names),
        UserWarning,
        stacklevel=3,
    )


def apply_checkpoint_runtime_compat(raw: dict[str, Any]) -> dict[str, Any]:
    """Migrate immutable checkpoint metadata into the current typed contract."""

    normalized, routing_marker = _normalize_routing_keys(
        migrate_checkpoint_video_action_config_fields(raw)
    )
    data_raw = normalized.get("data")
    initial_routing = (
        data_raw.get("dynamics_routing")
        if isinstance(data_raw, Mapping)
        else None
    )
    checkpoint_has_conditional_routes = _has_positive_conditional_route(
        initial_routing.get("routes")
        if isinstance(initial_routing, Mapping)
        else ()
    )

    policy_raw = normalized.get("policy_variant")
    legacy_probabilities: Mapping[str, Any] | None = None
    resolved_program: str | None = None
    if isinstance(policy_raw, Mapping):
        policy_raw = dict(policy_raw)
        normalized["policy_variant"] = policy_raw
        legacy_probabilities = _pop_legacy_mode_probabilities(policy_raw)
        resolved_program = _migrate_policy_program(
            policy_raw,
            routing_marker=routing_marker,
            checkpoint_has_conditional_routes=checkpoint_has_conditional_routes,
            legacy_probabilities=legacy_probabilities,
        )

    data_raw = normalized.get("data")
    data_raw = dict(data_raw) if isinstance(data_raw, Mapping) else {}
    normalized["data"] = data_raw
    routing_raw = data_raw.get("dynamics_routing")
    routing_raw = dict(routing_raw) if isinstance(routing_raw, Mapping) else {}
    data_raw["dynamics_routing"] = routing_raw
    _migrate_dynamics_routes(
        routing_raw,
        program=resolved_program,
        routing_marker=routing_marker,
        legacy_probabilities=legacy_probabilities,
    )
    _normalize_routed_checkpoint_sample_order(
        data_raw,
        routes=routing_raw.get("routes"),
    )
    _disable_unroutable_conditional_validation_tasks(
        normalized,
        routes=routing_raw.get("routes"),
    )
    _remove_obsolete_sampling_fields(data_raw)
    return normalized


__all__ = ["apply_checkpoint_runtime_compat"]
