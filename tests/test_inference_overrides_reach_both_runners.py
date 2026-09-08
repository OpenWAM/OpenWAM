"""Regression coverage for inference overrides reach both runners."""

from __future__ import annotations

import types

import pytest

from open_wam.evals.libero_realtime_runtime import apply_inference_overrides


def _inference_config():
    return types.SimpleNamespace(
        video_num_inference_steps=50,
        action_num_inference_steps=10,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
    )


def _exact_shaped_runner():
    """LingbotExactRunner: the variant is directly on the runner."""
    return types.SimpleNamespace(
        policy_variant=types.SimpleNamespace(inference_config=_inference_config()))


def _variant_shaped_runner():
    """VariantRolloutRunner: the variant hangs off the pipeline."""
    return types.SimpleNamespace(
        pipeline=types.SimpleNamespace(
            policy_variant=types.SimpleNamespace(inference_config=_inference_config())))


def _config_of(runner):
    variant = getattr(runner, "policy_variant", None) or runner.pipeline.policy_variant
    return variant.inference_config


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_runner", [_exact_shaped_runner, _variant_shaped_runner],
    ids=["exact_runner", "variant_runner"])
def test_step_overrides_apply_to_either_runner_shape(make_runner) -> None:
    runner = make_runner()
    apply_inference_overrides(
        runner, video_num_inference_steps=8, action_num_inference_steps=4,
        guidance_scale=None, action_guidance_scale=None)
    cfg = _config_of(runner)
    assert cfg.video_num_inference_steps == 8
    assert cfg.action_num_inference_steps == 4
    assert cfg.guidance_scale == 1.0, "an override that was not asked for was applied"


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_runner", [_exact_shaped_runner, _variant_shaped_runner],
    ids=["exact_runner", "variant_runner"])
def test_no_overrides_changes_nothing(make_runner) -> None:
    runner = make_runner()
    apply_inference_overrides(
        runner, video_num_inference_steps=None, action_num_inference_steps=None,
        guidance_scale=None, action_guidance_scale=None)
    cfg = _config_of(runner)
    assert (cfg.video_num_inference_steps, cfg.action_num_inference_steps) == (50, 10)


@pytest.mark.unit
def test_a_runner_with_neither_shape_fails_by_name() -> None:
    """Only when an override is actually requested -- silence is fine otherwise."""

    runner = types.SimpleNamespace()
    apply_inference_overrides(
        runner, video_num_inference_steps=None, action_num_inference_steps=None,
        guidance_scale=None, action_guidance_scale=None)
    with pytest.raises(AttributeError, match="policy_variant"):
        apply_inference_overrides(
            runner, video_num_inference_steps=4, action_num_inference_steps=None,
            guidance_scale=None, action_guidance_scale=None)
