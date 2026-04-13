from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import pytest


def _load_sandbox_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_libero_realtime_sandbox.py"
    module_name = f"run_libero_realtime_sandbox_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
    return module


def test_frame_index_to_action_start_matches_exact_realtime_convention() -> None:
    sandbox = _load_sandbox_module()

    assert sandbox._frame_index_to_action_start(1, 4) == 0
    assert sandbox._frame_index_to_action_start(2, 4) == 4
    assert sandbox._frame_index_to_action_start(3, 4) == 8


def test_merge_future_step_actions_drops_stale_steps_and_prefers_newer_future() -> None:
    sandbox = _load_sandbox_module()
    step_cls = sandbox.PlannedControlStep

    existing = {
        0: step_cls(absolute_action_index=0, generation_action_start=0, source="old"),
        1: step_cls(absolute_action_index=1, generation_action_start=0, source="old"),
        2: step_cls(absolute_action_index=2, generation_action_start=0, source="old"),
    }
    incoming = [
        step_cls(absolute_action_index=1, generation_action_start=1, source="new"),
        step_cls(absolute_action_index=3, generation_action_start=1, source="new"),
    ]

    merged = sandbox._merge_future_step_actions(existing, incoming, next_action_to_execute=1)

    assert list(merged) == [1, 2, 3]
    assert merged[1].source == "new"
    assert merged[2].source == "old"
    assert merged[3].source == "new"


def test_mot_realtime_replan_resets_observation_conditioned_session() -> None:
    sandbox = _load_sandbox_module()
    calls = []

    class Runner:
        def reset(self, *, task_text=None, text_context=None, negative_text_context=None):
            calls.append(
                {
                    "task_text": task_text,
                    "text_context": text_context,
                    "negative_text_context": negative_text_context,
                }
            )
            return SimpleNamespace(
                task_text=task_text,
                text_context=text_context,
                negative_text_context=negative_text_context,
            )

    session = SimpleNamespace(
        task_text=("task",),
        text_context="text",
        negative_text_context="negative",
    )
    mot_config = SimpleNamespace(policy_variant=SimpleNamespace(name="mot"))
    method4_config = SimpleNamespace(policy_variant=SimpleNamespace(name="post_latent"))

    resolved = sandbox._resolve_observation_conditioned_replan_session(
        runner=Runner(),
        session=session,
        config=mot_config,
    )

    assert resolved is not session
    assert calls == [
        {
            "task_text": ("task",),
            "text_context": "text",
            "negative_text_context": "negative",
        }
    ]
    assert (
        sandbox._resolve_observation_conditioned_replan_session(
            runner=Runner(),
            session=session,
            config=method4_config,
        )
        is session
    )


def test_realtime_common_inference_overrides_preserve_config_values_by_default() -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        inference=SimpleNamespace(
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
        )
    )

    sandbox._apply_common_inference_overrides(
        config,
        video_num_inference_steps=None,
        action_num_inference_steps=None,
        guidance_scale=None,
        action_guidance_scale=None,
    )

    assert config.inference.video_num_inference_steps == 20
    assert config.inference.action_num_inference_steps == 50
    assert config.inference.guidance_scale == 5.0
    assert config.inference.action_guidance_scale == 1.0


def test_realtime_common_inference_overrides_apply_explicit_smoke_values() -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        inference=SimpleNamespace(
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
        )
    )

    sandbox._apply_common_inference_overrides(
        config,
        video_num_inference_steps=2,
        action_num_inference_steps=3,
        guidance_scale=1.5,
        action_guidance_scale=0.75,
    )

    assert config.inference.video_num_inference_steps == 2
    assert config.inference.action_num_inference_steps == 3
    assert config.inference.guidance_scale == 1.5
    assert config.inference.action_guidance_scale == 0.75


@pytest.mark.parametrize(
    ("video_steps", "action_steps"),
    [
        (0, None),
        (None, 0),
        (-1, None),
        (None, -1),
    ],
)
def test_realtime_common_inference_overrides_reject_nonpositive_step_values(
    video_steps: int | None,
    action_steps: int | None,
) -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        inference=SimpleNamespace(
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
        )
    )

    with pytest.raises(ValueError, match="must be positive"):
        sandbox._apply_common_inference_overrides(
            config,
            video_num_inference_steps=video_steps,
            action_num_inference_steps=action_steps,
            guidance_scale=None,
            action_guidance_scale=None,
        )

    assert config.inference.video_num_inference_steps == 20
    assert config.inference.action_num_inference_steps == 50


def test_exact_realtime_inference_overrides_preserve_config_values_by_default() -> None:
    sandbox = _load_sandbox_module()
    runner = SimpleNamespace(
        policy_variant=SimpleNamespace(
            inference_config=SimpleNamespace(
                video_num_inference_steps=20,
                action_num_inference_steps=50,
                guidance_scale=5.0,
                action_guidance_scale=1.0,
            )
        )
    )

    sandbox.exact_sandbox._apply_inference_overrides(
        runner,
        video_num_inference_steps=None,
        action_num_inference_steps=None,
        guidance_scale=None,
        action_guidance_scale=None,
    )

    assert runner.policy_variant.inference_config.video_num_inference_steps == 20
    assert runner.policy_variant.inference_config.action_num_inference_steps == 50
    assert runner.policy_variant.inference_config.guidance_scale == 5.0
    assert runner.policy_variant.inference_config.action_guidance_scale == 1.0


@pytest.mark.parametrize(
    ("video_steps", "action_steps"),
    [
        (0, None),
        (None, 0),
        (-1, None),
        (None, -1),
    ],
)
def test_exact_realtime_inference_overrides_reject_nonpositive_step_values(
    video_steps: int | None,
    action_steps: int | None,
) -> None:
    sandbox = _load_sandbox_module()
    runner = SimpleNamespace(
        policy_variant=SimpleNamespace(
            inference_config=SimpleNamespace(
                video_num_inference_steps=20,
                action_num_inference_steps=50,
                guidance_scale=5.0,
                action_guidance_scale=1.0,
            )
        )
    )

    with pytest.raises(ValueError, match="must be positive"):
        sandbox.exact_sandbox._apply_inference_overrides(
            runner,
            video_num_inference_steps=video_steps,
            action_num_inference_steps=action_steps,
            guidance_scale=None,
            action_guidance_scale=None,
        )

    assert runner.policy_variant.inference_config.video_num_inference_steps == 20
    assert runner.policy_variant.inference_config.action_num_inference_steps == 50
