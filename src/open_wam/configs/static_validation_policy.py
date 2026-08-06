"""Static validation rules for policy, sequence, and GJD contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .enums import (
    ActionDecoderName,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DualExpertRuntimeMode,
    GeneralistDenoisingMode,
    HistoryStreamVisibility,
    ParallelRuntimeMode,
    PolicyVariantName,
    ProprioContextMode,
    RolloutContextPolicy,
    SampleTargetAlignment,
    StrEnum,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from .policy_video_action import (
    conditional_denoising_modes_enabled,
    fixed_conditioning_mode_for_program,
    is_dynamics_routed_paradigm,
)
from .static_validation_contracts import _IssueBuilder
from .static_validation_primitives import _optional_int
from .variant_semantics import probability_map_static_issues


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
    offset = None if sample_construction is None else _optional_int(sample_construction.get("condition_source_frame_offset"))
    if offset != -1:
        issues.error(
            "data.sample_construction.condition_source_frame_offset",
            "Expected -1 when "
            "`policy_variant.context_condition_latent_source=single_frame_condition_latent`; "
            "offset 0 can expose the first target raw frame.",
        )


def _warn_deprecated_text_proprio_context(policy_variant: Mapping[str, Any], issues: _IssueBuilder) -> None:
    if policy_variant.get("proprio_context_mode") != ProprioContextMode.TEXT_CONTEXT_TOKEN.value:
        return
    issues.warning(
        "policy_variant.proprio_context_mode",
        "Deprecated text-space proprio token path; current proprio context is "
        "`per_chunk_additive` hidden-state conditioning.",
    )


def _resolved_static_video_action_coupling(
    policy_variant: Mapping[str, Any],
) -> str | None:
    raw_coupling = policy_variant.get("current_block_coupling")
    if raw_coupling is not None:
        return str(raw_coupling)
    raw_program = policy_variant.get("program")
    try:
        program = VideoActionProgram(str(raw_program))
    except ValueError:
        return None
    if program in {
        VideoActionProgram.GENERALIST_JOINT_DENOISING,
        VideoActionProgram.FORWARD_DYNAMICS,
        VideoActionProgram.INVERSE_DYNAMICS,
    }:
        return CurrentBlockCoupling.JOINT.value
    return program.value


def _validate_video_action_program_coupling(
    policy_variant: Mapping[str, Any],
    issues: _IssueBuilder,
) -> None:
    raw_program = policy_variant.get("program")
    if raw_program is None:
        return
    try:
        program = VideoActionProgram(str(raw_program))
    except ValueError:
        return
    expected_coupling = _resolved_static_video_action_coupling(
        {"program": program.value}
    )
    assert expected_coupling is not None
    raw_coupling = policy_variant.get("current_block_coupling")
    if raw_coupling is not None and raw_coupling != expected_coupling:
        issues.error(
            "policy_variant.current_block_coupling",
            f"`program: {program.value}` requires "
            f"`current_block_coupling: {expected_coupling}` when both are provided.",
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
    if policy_variant.get("name") != PolicyVariantName.DUAL_EXPERT.value:
        issues.error(
            "policy_variant.program",
            f"`{program.value}` is currently supported only by `policy_variant.name: dual_expert`.",
        )
    if (
        not is_dynamics_routed_paradigm(
            policy_variant.get("generalist_training_paradigm")
        )
    ):
        issues.error(
            "policy_variant.generalist_training_paradigm",
            f"`program: {program.value}` requires `dynamics_routed` so real and optional "
            "counterfactual sources use the same target-only t0 layout.",
        )
    if bool(policy_variant.get("generalist_mode_text_token", False)):
        issues.error(
            "policy_variant.generalist_mode_text_token",
            f"`program: {program.value}` has one fixed mode and does not use a GJD mode token.",
        )
    raw_probs = policy_variant.get("generalist_denoising_mode_probs")
    if isinstance(raw_probs, Mapping):
        positive_modes = {
            str(mode)
            for mode, weight in raw_probs.items()
            if not isinstance(weight, bool)
            and isinstance(weight, (int, float))
            and float(weight) > 0.0
        }
        if positive_modes != {fixed_mode.value}:
            issues.error(
                "policy_variant.generalist_denoising_mode_probs",
                f"`program: {program.value}` permits only the fixed {fixed_mode.value!r} mode.",
            )
    mixture = data.get("generalist_dynamics_mixture")
    if isinstance(mixture, Mapping):
        source_keys = (
            (
                "real_action_conditioned_video_weight",
                "counterfactual_action_conditioned_video_weight",
            )
            if fixed_mode == GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO
            else (
                "real_video_conditioned_action_weight",
                "counterfactual_video_conditioned_action_weight",
            )
        )
        numeric_weights = [
            float(mixture[key])
            for key in source_keys
            if isinstance(mixture.get(key), (int, float))
            and not isinstance(mixture.get(key), bool)
        ]
        if len(numeric_weights) == len(source_keys) and sum(numeric_weights) <= 0.0:
            issues.error(
                "data.generalist_dynamics_mixture",
                f"`program: {program.value}` requires a positive real or counterfactual "
                f"{fixed_mode.value} source weight.",
            )
    for key in ("train_batch_size", "val_batch_size"):
        try:
            batch_size = int(data.get(key, 2))
        except (TypeError, ValueError):
            continue
        if batch_size != 1:
            issues.error(
                f"data.{key}",
                f"`program: {program.value}` requires rank-local train and validation batch size 1.",
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

    policy_name = policy_variant.get("name")
    if policy_name not in {PolicyVariantName.PARALLEL_STREAM.value, PolicyVariantName.DUAL_EXPERT.value}:
        issues.error(
            "policy_variant.sequence_contract",
            f"`{contract.value}` is only supported for policy_variant.name parallel_stream or dual_expert.",
        )
        return

    if contract == VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO:
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
        if policy_name == PolicyVariantName.DUAL_EXPERT.value and runtime_mode not in (
            None,
            DualExpertRuntimeMode.NON_JOINT_TWO_STREAM.value,
        ):
            issues.error(
                "policy_variant.runtime_mode",
                "legacy_prefix_single_frame_perchunk_proprio requires "
                "runtime_mode=non_joint_two_stream for dual_expert.",
            )

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
    if contract == VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO:
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


def _validate_generalist_denoising_mode_probs(
    policy_variant: Mapping[str, Any],
    data: Mapping[str, Any],
    issues: _IssueBuilder,
    *,
    require_joint_coupling: bool,
    require_batch_size_one: bool,
) -> None:
    field_name = "generalist_denoising_mode_probs"
    raw_probs = policy_variant.get(field_name)
    if bool(policy_variant.get("generalist_mode_text_token", False)) and raw_probs is None:
        issues.error(
            "policy_variant.generalist_mode_text_token",
            "`generalist_mode_text_token: true` requires "
            f"`policy_variant.{field_name}`.",
        )
    if raw_probs is None:
        return
    if (
        isinstance(raw_probs, Mapping)
        and conditional_denoising_modes_enabled(raw_probs)
        and not is_dynamics_routed_paradigm(
            policy_variant.get("generalist_training_paradigm")
        )
    ):
        issues.error(
            "policy_variant.generalist_training_paradigm",
            "Positive FDM/IDM `generalist_denoising_mode_probs` require `dynamics_routed` "
            "so real and optional counterfactual samples use the target-only t0 contract. "
            "Counterfactual source weights may be zero; pure joint may use `demo_only`.",
        )
    if (
        require_joint_coupling
        and _resolved_static_video_action_coupling(policy_variant)
        != CurrentBlockCoupling.JOINT.value
    ):
        issues.error(
            f"policy_variant.{field_name}",
            "Expected `program: generalist_joint_denoising` or "
            "`current_block_coupling: joint` when generalist denoising is enabled.",
        )
    if require_batch_size_one:
        for key in ("train_batch_size", "val_batch_size"):
            raw_batch_size = data.get(key, 2)
            try:
                batch_size = int(raw_batch_size)
            except (TypeError, ValueError):
                continue
            if batch_size != 1:
                issues.error(
                    f"data.{key}",
                    f"`{field_name}` requires `data.train_batch_size: 1` and "
                    "`data.val_batch_size: 1` for this policy architecture.",
                )
    _validate_probability_map(
        policy_variant,
        issues,
        field_name=field_name,
        enum_cls=GeneralistDenoisingMode,
    )


def _validate_probability_map(
    policy_variant: Mapping[str, Any],
    issues: _IssueBuilder,
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
