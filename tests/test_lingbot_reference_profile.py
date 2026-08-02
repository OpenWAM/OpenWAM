from __future__ import annotations

from dataclasses import replace

import pytest

from open_wam.models.policy_variants.parallel_stream.reference_profile import (
    LingbotReferenceRuntimeContract,
    load_reference_profile,
    validate_reference_profile,
)


def _libero_joint_runtime_contract() -> LingbotReferenceRuntimeContract:
    return LingbotReferenceRuntimeContract(
        max_text_tokens=512,
        action_dim=30,
        action_per_frame=4,
        policy_frame_chunk_size=4,
        inference_frame_chunk_size=4,
        attn_window=30,
        guidance_scale=5.0,
        require_guidance_scale_match=True,
        action_guidance_scale=1.0,
        video_num_inference_steps=20,
        action_num_inference_steps=20,
        video_exec_step=-1,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    (
        ("max_text_tokens", 511, "max_text_tokens"),
        ("action_dim", 29, "action_dim"),
        ("action_per_frame", 3, "action_per_frame"),
        ("policy_frame_chunk_size", 3, "policy config"),
        ("inference_frame_chunk_size", 3, "inference config"),
        ("attn_window", 29, "attn_window"),
        ("action_guidance_scale", 2.0, "action_guidance_scale"),
        ("video_num_inference_steps", 21, "video_num_inference_steps"),
        ("action_num_inference_steps", 21, "action_num_inference_steps"),
        ("video_exec_step", 0, "video_exec_step"),
        ("video_sigma_shift", 4.0, "video_sigma_shift"),
        ("action_sigma_shift", 2.0, "action_sigma_shift"),
    ),
)
def test_reference_profile_contract_rejects_each_mismatched_runtime_value(
    field_name: str,
    invalid_value: int | float,
    message: str,
) -> None:
    profile = load_reference_profile("libero_joint")
    runtime = replace(
        _libero_joint_runtime_contract(),
        **{field_name: invalid_value},
    )

    with pytest.raises(ValueError, match=message):
        validate_reference_profile(profile, runtime)


def test_reference_profile_guidance_match_is_generalist_only() -> None:
    profile = load_reference_profile("libero_joint")
    mismatched_runtime = replace(
        _libero_joint_runtime_contract(),
        guidance_scale=1.0,
        require_guidance_scale_match=False,
    )

    validate_reference_profile(profile, mismatched_runtime)

    with pytest.raises(ValueError, match="guidance_scale"):
        validate_reference_profile(
            profile,
            replace(mismatched_runtime, require_guidance_scale_match=True),
        )


def test_absent_reference_profile_has_no_runtime_requirements() -> None:
    validate_reference_profile(None, _libero_joint_runtime_contract())
