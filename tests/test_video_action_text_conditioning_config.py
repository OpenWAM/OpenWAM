"""Regression coverage for video action text conditioning config."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from open_wam.configs import (
    DualExpertPolicyConfig,
    ExtensionPolicyConfig,
    ParallelStreamPolicyConfig,
    TextConditioningMode,
    VideoActionProgram,
    load_experiment_config,
    validate_config_file,
)
from open_wam.configs.serialization import serialize_experiment_config
from open_wam.utils.config_overrides import apply_config_overrides


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_TYPES = (DualExpertPolicyConfig, ParallelStreamPolicyConfig)
RECIPES = ("dual_expert_robotwin_smoke", "parallel_stream_robotwin_smoke")
pytestmark = pytest.mark.unit


def _load_recipe(recipe: str):
    return load_experiment_config(REPO_ROOT / "configs/experiments" / f"{recipe}.yaml")


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
@pytest.mark.parametrize(
    "program", (VideoActionProgram.VIDEO_THEN_ACTION, VideoActionProgram.JOINT)
)
def test_video_action_text_mode_is_typed_and_defaults_to_task_prompt(policy_type, program):
    original = policy_type(program=program)
    explicit = policy_type(program=program, text_conditioning_mode="task_prompt")
    disabled = replace(original, text_conditioning_mode="disabled")

    assert original == explicit
    assert original.text_conditioning_mode is TextConditioningMode.TASK_PROMPT
    assert (
        original.conditioning_requirements.text_conditioning_mode
        is TextConditioningMode.TASK_PROMPT
    )
    assert disabled.text_conditioning_mode is TextConditioningMode.DISABLED
    assert (
        disabled.conditioning_requirements.text_conditioning_mode
        is TextConditioningMode.DISABLED
    )
    assert replace(disabled, text_conditioning_mode="task_prompt") == original


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
@pytest.mark.parametrize("invalid", ("drop_all", "", True, None))
def test_video_action_text_mode_rejects_untyped_values(policy_type, invalid):
    with pytest.raises((TypeError, ValueError)):
        policy_type(
            program=VideoActionProgram.VIDEO_THEN_ACTION, text_conditioning_mode=invalid
        )


@pytest.mark.parametrize("recipe", RECIPES)
def test_video_action_blank_mode_roundtrips_strict_schema_without_topology_changes(
    tmp_path, recipe
):
    original = _load_recipe(recipe)
    updated = apply_config_overrides(
        original,
        {
            "policy_variant.text_conditioning_mode": "disabled",
            "training.text_condition_dropout_prob": 0.0,
            "inference.guidance_scale": 1.0,
        },
    )
    path = tmp_path / "blank.yaml"
    path.write_text(yaml.safe_dump(serialize_experiment_config(updated)), encoding="utf-8")
    reloaded = load_experiment_config(path)

    assert serialize_experiment_config(reloaded) == serialize_experiment_config(updated)
    assert (
        reloaded.policy_variant.conditioning_requirements.text_conditioning_mode
        is TextConditioningMode.DISABLED
    )
    assert reloaded.backbone == original.backbone
    assert reloaded.data == original.data
    assert reloaded.action_decoder == original.action_decoder
    assert reloaded.trainer == original.trainer
    assert reloaded.policy_variant == replace(
        original.policy_variant, text_conditioning_mode=TextConditioningMode.DISABLED
    )
    assert validate_config_file(path, repo_root=REPO_ROOT).ok


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize(
    "field,value,message",
    (
        (
            "training.text_condition_dropout_prob", 0.1,
            "every sample already uses the blank-text embedding",
        ),
        (
            "inference.guidance_scale", 2.0,
            "conditioned and unconditioned branches are identical",
        ),
    ),
)
def test_video_action_blank_mode_rejects_conflicts_in_yaml_and_overrides(
    tmp_path, recipe, field, value, message
):
    original = _load_recipe(recipe)
    overrides = {
        "policy_variant.text_conditioning_mode": "disabled",
        "training.text_condition_dropout_prob": 0.0,
        "inference.guidance_scale": 1.0,
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        apply_config_overrides(original, overrides)
    raw = serialize_experiment_config(original)
    for key, setting in overrides.items():
        section, name = key.split(".")
        raw[section][name] = setting
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)
    report = validate_config_file(path, repo_root=REPO_ROOT)
    assert any(issue.path == field and message in issue.message for issue in report.errors)


@pytest.mark.parametrize("recipe", RECIPES)
def test_video_action_task_prompt_preserves_existing_full_dropout_contract(
    tmp_path, recipe
):
    original = _load_recipe(recipe)
    assert original.policy_variant.text_conditioning_mode is TextConditioningMode.TASK_PROMPT
    updated = apply_config_overrides(
        original, {"training.text_condition_dropout_prob": 1.0}
    )
    raw = serialize_experiment_config(updated)
    raw["policy_variant"].pop("text_conditioning_mode")
    path = tmp_path / "original_default.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    implicit = load_experiment_config(path)
    assert serialize_experiment_config(implicit) == serialize_experiment_config(updated)
    assert (
        implicit.policy_variant.conditioning_requirements.text_conditioning_mode
        is TextConditioningMode.TASK_PROMPT
    )
    assert validate_config_file(path, repo_root=REPO_ROOT).ok


@pytest.mark.parametrize("recipe", RECIPES)
def test_video_action_static_and_typed_loaders_reject_unknown_text_mode(tmp_path, recipe):
    raw = serialize_experiment_config(_load_recipe(recipe))
    raw["policy_variant"]["text_conditioning_mode"] = "drop_all"
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(path)
    assert any(
        issue.path == "policy_variant.text_conditioning_mode"
        for issue in validate_config_file(path, repo_root=REPO_ROOT).errors
    )


def test_new_video_action_validation_does_not_change_extension_policy_contract(tmp_path):
    original = _load_recipe(RECIPES[0])
    extension = replace(
        original,
        policy_variant=ExtensionPolicyConfig(
            extension_type="application_owned_policy",
            text_conditioning_mode=TextConditioningMode.DISABLED,
        ),
        training=replace(original.training, text_condition_dropout_prob=0.2),
        inference=replace(original.inference, guidance_scale=2.0),
    )
    path = tmp_path / "extension.yaml"
    path.write_text(
        yaml.safe_dump(serialize_experiment_config(extension)), encoding="utf-8"
    )

    reloaded = load_experiment_config(path)

    assert reloaded.policy_variant == extension.policy_variant
    assert reloaded.training.text_condition_dropout_prob == 0.2
    assert reloaded.inference.guidance_scale == 2.0
    assert validate_config_file(path, repo_root=REPO_ROOT).ok
