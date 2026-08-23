"""Static validation rules for policy, sequence, and GJD contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .enums import (
    ActionDecoderName,
    ContextConditionLatentSource,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    PolicyVariantName,
    ProprioContextMode,
    RolloutContextPolicy,
    SampleTargetAlignment,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from .policy_video_action import (
    fixed_conditioning_mode_for_program,
    requires_independent_timestep_clocks,
    supports_dynamics_routing,
)
from .static_validation_contracts import _IssueBuilder
from .static_validation_primitives import _optional_int


def _validate_single_frame_condition_offset(
    policy_variant: Mapping[str, Any],
    sample_construction: Mapping[str, Any] | None,
    issues: _IssueBuilder,
) -> None:
    if (
        policy_variant.get("context_condition_latent_source")
        != ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT.value
    ):
        return
    for field_name in ("use_condition_latents", "require_condition_latents"):
        if policy_variant.get(field_name) is False:
            issues.error(
                f"policy_variant.{field_name}",
                "`context_condition_latent_source=single_frame_condition_latent` "
                f"requires `{field_name}=true`.",
            )
    offset = (
        None
        if sample_construction is None
        else _optional_int(sample_construction.get("condition_source_frame_offset"))
    )
    if offset != -1:
        issues.error(
            "data.sample_construction.condition_source_frame_offset",
            "Expected -1 when "
            "`policy_variant.context_condition_latent_source=single_frame_condition_latent`; "
            "offset 0 can expose the first target raw frame.",
        )


def _warn_deprecated_text_proprio_context(
    policy_variant: Mapping[str, Any], issues: _IssueBuilder
) -> None:
    if (
        policy_variant.get("proprio_context_mode")
        != ProprioContextMode.TEXT_CONTEXT_TOKEN.value
    ):
        return
    issues.warning(
        "policy_variant.proprio_context_mode",
        "Deprecated text-space proprio token path; current proprio context is "
        "`per_chunk_additive` hidden-state conditioning.",
    )


def _validate_fixed_conditional_program(
    policy_variant: Mapping[str, Any],
    data: Mapping[str, Any],
    issues: _IssueBuilder,
) -> None:
    raw_program = policy_variant.get("program")
    try:
        program = VideoActionProgram(str(raw_program))
    except ValueError:
        return
    fixed_mode = fixed_conditioning_mode_for_program(program)
    if fixed_mode is None:
        return
    if bool(policy_variant.get("generalist_mode_text_token", False)):
        issues.error(
            "policy_variant.generalist_mode_text_token",
            f"`program: {program.value}` has one fixed mode and does not use a GJD mode token.",
        )
    routing = data.get("dynamics_routing")
    if isinstance(routing, Mapping):
        routes = routing.get("routes", ())
        active_modes = {
            str(route.get("mode"))
            for route in routes
            if isinstance(route, Mapping)
            and isinstance(route.get("weight"), (int, float))
            and not isinstance(route.get("weight"), bool)
            and float(route["weight"]) > 0.0
        }
        if active_modes and active_modes != {fixed_mode.value}:
            issues.error(
                "data.dynamics_routing.routes",
                f"`program: {program.value}` accepts only {fixed_mode.value!r} routes.",
            )


def _validate_program_timestep_contract(
    policy_variant: Mapping[str, Any],
    issues: _IssueBuilder,
) -> None:
    """Reject joint-clock settings for programs without a joint noise clock."""

    try:
        program = VideoActionProgram(str(policy_variant.get("program")))
    except ValueError:
        return
    raw_coupling = policy_variant.get("joint_timestep_coupling")
    if raw_coupling is None:
        return
    try:
        coupling = JointTimestepCoupling(str(raw_coupling))
    except ValueError:
        return
    if (
        requires_independent_timestep_clocks(program)
        and coupling != JointTimestepCoupling.INDEPENDENT
    ):
        issues.error(
            "policy_variant.joint_timestep_coupling",
            f"`program: {program.value}` does not define a jointly coupled "
            "video/action noise clock and requires `independent`.",
        )


def _validate_video_action_sequence_contract_static(
    policy_variant: Mapping[str, Any],
    sample_construction: Mapping[str, Any] | None,
    issues: _IssueBuilder,
) -> None:
    raw_contract = policy_variant.get("sequence_contract")
    if raw_contract in (None, VideoActionSequenceContract.DEFAULT.value):
        return
    try:
        contract = VideoActionSequenceContract(str(raw_contract))
    except ValueError:
        return
    if contract not in {
        VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    }:
        return

    expected_policy = {
        "proprio_context_mode": ProprioContextMode.PER_CHUNK_ADDITIVE.value,
        "context_condition_latent_source": ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT.value,
        "history_stream_visibility": HistoryStreamVisibility.VIDEO_ONLY.value,
        "use_condition_latents": True,
        "require_condition_latents": True,
    }
    for key, expected_value in expected_policy.items():
        if key in policy_variant and policy_variant[key] != expected_value:
            issues.error(
                f"policy_variant.{key}",
                f"`sequence_contract={contract.value}` owns `{key}`; expected {expected_value!r}.",
            )

    if sample_construction is None:
        return
    expected_sample: dict[str, Any] = {
        "condition_source_frame_offset": -1,
        "start_padding_frames": 0,
    }
    if (
        contract
        == VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
    ):
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
                f"`sequence_contract={contract.value}` owns `{key}`; expected {expected_value!r}.",
            )


def _validate_action_horizons(
    action_schema: Mapping[str, Any] | None,
    policy_variant: Mapping[str, Any] | None,
    action_decoder: Mapping[str, Any] | None,
    issues: _IssueBuilder,
) -> None:
    video_only = False
    if (
        action_decoder is not None
        and action_decoder.get("name") == ActionDecoderName.VIDEO_ONLY.value
    ):
        video_only = True
    if (
        policy_variant is not None
        and policy_variant.get("name")
        == PolicyVariantName.CAUSAL_VIDEO_PREDICTION.value
    ):
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


def _active_dynamics_routes(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    routing = data.get("dynamics_routing")
    routes = routing.get("routes", ()) if isinstance(routing, Mapping) else ()
    return tuple(
        route
        for route in routes
        if isinstance(route, Mapping)
        and isinstance(route.get("weight"), (int, float))
        and not isinstance(route.get("weight"), bool)
        and float(route["weight"]) > 0.0
    )


def _validate_dynamics_route_contract(
    policy_variant: Mapping[str, Any],
    data: Mapping[str, Any],
    issues: _IssueBuilder,
) -> None:
    for field_name in (
        "dynamics_routing_requirement",
        "generalist_training_paradigm",
        "generalist_denoising_mode_probs",
        "joint_denoise_training_mode_probs",
        "mot_generalist_training_mode_probs",
    ):
        if field_name in policy_variant:
            issues.error(
                f"policy_variant.{field_name}",
                "This field is retired; configure source and objective sampling "
                "once in `data.dynamics_routing.routes`.",
            )
    try:
        program = VideoActionProgram(str(policy_variant.get("program")))
    except ValueError:
        return
    if (
        bool(policy_variant.get("generalist_mode_text_token", False))
        and program != VideoActionProgram.GENERALIST_JOINT_DENOISING
    ):
        issues.error(
            "policy_variant.generalist_mode_text_token",
            "A generalist mode token requires `program: generalist_joint_denoising`.",
        )
    active_routes = _active_dynamics_routes(data)
    if active_routes:
        if not supports_dynamics_routing(program):
            issues.error(
                "data.dynamics_routing.routes",
                "Active routes require a generalist, forward-dynamics, or "
                "inverse-dynamics program.",
            )
    elif fixed_conditioning_mode_for_program(program) is not None:
        issues.error(
            "data.dynamics_routing.routes",
            f"`program: {program.value}` requires at least one positive route so "
            "every sample uses the target-only t0 contract.",
        )
    if supports_dynamics_routing(program):
        for key in ("train_batch_size", "val_batch_size"):
            raw_batch_size = data.get(key, 2)
            try:
                batch_size = int(raw_batch_size)
            except (TypeError, ValueError):
                continue
            if batch_size != 1:
                issues.error(
                    f"data.{key}",
                    "Generalist and conditional-dynamics programs require rank-local "
                    "batch size 1 because each rank executes one routed objective at a time.",
                )
