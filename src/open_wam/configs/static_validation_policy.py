"""Static validation rules for policy, sequence, and GJD contracts."""

from __future__ import annotations

from typing import Any, Mapping

from .enums import (
    ActionDecoderName,
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    MoTGeneralistTrainingMode,
    MoTRuntimeMode,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelRuntimeMode,
    ParallelSequenceContract,
    PolicyVariantName,
    ProprioContextMode,
    RolloutContextPolicy,
    SampleTargetAlignment,
    StrEnum,
)
from .static_validation_contracts import _IssueBuilder
from .static_validation_primitives import _optional_int
from .variant_semantics import probability_map_static_issues


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
