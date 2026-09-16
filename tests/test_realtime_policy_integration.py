"""Real policy transitions with benchmark adapters restricted to representation I/O."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_wam.configs import (
    ActionMappingConfig,
    ActionNormalizationConfig,
    ActionTargetRepresentation,
    VideoActionProgram,
)
from open_wam.data.action_adapter import ConfiguredActionAdapter
from open_wam.integrations.libero_realtime import LiberoRolloutAdapter
from open_wam.models.policy_variants import (
    PolicyExecutionCommit,
    PolicyInferContext,
    PolicyObservedHistory,
    PolicyTemporalSpan,
)
from open_wam.pipelines import VariantRolloutRunner
from tests.test_rollout_history_ownership import assert_state_equal
from tests.test_unified_policy_inference import pipeline_for


def first_step(architecture, chunk_size=2):
    pipeline = pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    geometry = replace(
        pipeline.default_temporal_geometry,
        frame_chunk_size=chunk_size,
        attention_window_size=16,
    )
    pipeline.default_temporal_geometry = geometry
    pipeline.policy_variant.inference_config = replace(
        pipeline.policy_variant.inference_config,
        frame_chunk_size=chunk_size,
        attention_window_size=16,
    )
    runner = VariantRolloutRunner(pipeline)
    first = runner.infer_step(
        session=runner.reset(text_context=torch.ones(1, 3, 16)),
        video_latents=torch.ones(1, 48, 1, 4, 4),
        context=PolicyInferContext(state=torch.ones(1, 1, 4)),
    )
    return runner, first


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
@torch.no_grad()
def test_published_startup_span_contains_every_prediction(architecture, chunk_size):
    runner, first = first_step(architecture, chunk_size)
    state = first.session.policy_state
    snapshot = copy.deepcopy(state)
    assert first.action_plan.frame_span == PolicyTemporalSpan(1, chunk_size)
    history = state.variant_state
    density = runner.pipeline.policy_variant.rollout_contract.action_tokens_per_frame
    assert history.past_clean_latents.shape[2] == 1 + chunk_size
    assert not history.past_clean_action_mask[:, :density].any()
    assert history.past_clean_action_mask[:, density:].all()
    torch.testing.assert_close(
        history.past_clean_latents[:, :, 1:],
        first.infer_output.policy_output.generated_video.latents,
    )
    extension = runner.infer_step(
        session=first.session,
        video_latents=torch.ones(1, 48, 1, 4, 4),
        context=PolicyInferContext(state=torch.ones(1, 1, 4)),
    )
    assert extension.action_plan.frame_span == PolicyTemporalSpan(
        1 + chunk_size, chunk_size
    )
    assert_state_equal(state, snapshot)


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.ones(1, 4, 4),
        torch.full((1, 4, 1), float("nan")),
        torch.full((1, 4, 1), 0.5),
    ],
)
@torch.no_grad()
def test_invalid_action_mask_never_publishes_state(architecture, bad_mask):
    runner, first = first_step(architecture)
    state = first.session.policy_state
    snapshot = copy.deepcopy(state)
    with pytest.raises(ValueError, match="action_mask"):
        runner.pipeline.reconcile_observed_history(
            PolicyObservedHistory(
                video_latents=torch.ones(1, 48, 2, 4, 4),
                observation_frame_count=4,
                action_history=torch.ones(1, 4, 4),
                action_mask=bad_mask,
                execution_commit=PolicyExecutionCommit(PolicyTemporalSpan(1, 2), 2),
            ),
            state,
        )
    assert_state_equal(state, snapshot)


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("missing_controls", [False, True])
@torch.no_grad()
def test_libero_adapter_uses_one_transform_for_plans_and_observed_actions(
    architecture, missing_controls
):
    runner, first = first_step(architecture)
    runner.pipeline.action_adapter = ConfiguredActionAdapter(
        ActionMappingConfig(
            mode="pad_and_reorder",
            source_dim=4,
            target_dim=4,
            source_to_target_indices=(3, 2, 1, 0),
        ),
        ActionNormalizationConfig(mode="gaussian", mean=(0.25,) * 4, std=(2.0,) * 4),
        model_dim=4,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            action_schema=SimpleNamespace(state_horizon=0),
            action_target=SimpleNamespace(
                representation=ActionTargetRepresentation.RAW,
                rotation_representation="axis_angle",
            ),
        )
    )
    adapter = LiberoRolloutAdapter(
        runner, config, None, "task", torch.device("cpu"), torch.device("cpu")
    )
    visual = runner.pipeline.prepare_visual_outputs_from_latents(
        torch.full((1, 48, 3, 4, 4), 2.0),
        text_context=first.session.text_context,
    )
    controls = tuple(np.arange(4, dtype=np.float32) + i for i in range(4))
    snapshot = copy.deepcopy(first.session.policy_state)
    if missing_controls:
        with pytest.raises(ValueError, match="every actual executed control"):
            adapter.observed_history(
                tuple(range(5)), (), visual, PolicyTemporalSpan(1, 2), first.session
            )
        assert_state_equal(first.session.policy_state, snapshot)
        return
    history = adapter.observed_history(
        tuple(range(5)), controls, visual, PolicyTemporalSpan(1, 2), first.session
    )
    updated = runner.pipeline.reconcile_observed_history(
        history, first.session.policy_state
    )
    expected = ((torch.from_numpy(np.stack(controls)) - 0.25) / 2).flip(-1)
    torch.testing.assert_close(
        updated.next_state.variant_state.past_clean_actions[:, 2:],
        expected.unsqueeze(0),
    )
    assert not updated.next_state.variant_state.past_clean_action_mask[:, :2].any()
    assert_state_equal(first.session.policy_state, snapshot)
    steps = adapter.controls(
        first.action_plan, {}, 0, "test", 0.0,
        step_index=first.session.policy_state.step_index,
    )
    source = torch.from_numpy(np.stack([step.raw_action for step in steps]))
    torch.testing.assert_close(source, first.action_plan.actions.flip(-1) * 2 + 0.25)


def test_pose_history_fails_preflight_without_loading_a_model():
    config = SimpleNamespace(
        data=SimpleNamespace(
            action_target=SimpleNamespace(
                representation=ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE,
            )
        )
    )
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        LiberoRolloutAdapter.validate_config(config)


@pytest.mark.parametrize("model_dim", [7, 30])
def test_action_adapter_roundtrip_preserves_environment_controls(model_dim):
    adapter = ConfiguredActionAdapter(
        ActionMappingConfig(
            mode="pad_and_reorder",
            source_dim=7,
            target_dim=model_dim,
            source_to_target_indices=tuple(reversed(range(7))),
        ),
        ActionNormalizationConfig(mode="gaussian", mean=(0.25,) * 7, std=(2.0,) * 7),
        model_dim=model_dim,
    )
    controls = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    torch.testing.assert_close(
        adapter.to_source(adapter.to_model(controls)), controls, rtol=0, atol=0
    )
