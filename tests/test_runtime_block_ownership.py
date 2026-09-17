"""Execution views must not change checkpoint or optimizer ownership."""

import copy
import random
from dataclasses import replace

import pytest
import torch

from open_wam.configs import BatchingMode, VideoActionProgram
from open_wam.models.policy_variants.contracts import PolicyInferContext
from open_wam.models.policy_variants.dual_expert.module_topology import (
    build_dual_expert_module_topology,
)
from tests.test_variable_batch_pipeline import _collate, _samples, _tiny_pipeline


def test_module_topology_requires_assembled_ownership():
    pipeline, _ = _tiny_pipeline(VideoActionProgram.VIDEO_THEN_ACTION)
    with pytest.raises(RuntimeError, match="assembled paired block stack"):
        build_dual_expert_module_topology(
            visual_tower=pipeline.visual_tower,
            action_expert=pipeline.policy_variant.action_expert,
            packed_block_stack=None,
        )


@pytest.mark.parametrize(
    "program",
    (VideoActionProgram.VIDEO_THEN_ACTION, VideoActionProgram.DECOUPLED_SAME_STEP),
)
def test_rollout_preserves_parameter_ownership_and_subsequent_training(program):
    torch.manual_seed(89)
    pipeline, executor = _tiny_pipeline(program)
    variant = pipeline.policy_variant
    variant.inference_config = replace(
        variant.inference_config,
        video_num_inference_steps=2,
        action_num_inference_steps=2,
    )
    owner = variant.packed_block_stack
    keys = tuple(pipeline.state_dict())
    parameters = dict(pipeline.named_parameters())
    batch = _collate(_samples(program), BatchingMode.PADDED)

    def training_step():
        pipeline.train()
        pipeline.zero_grad(set_to_none=True)
        random.seed(91)
        torch.manual_seed(91)
        result = executor.forward_train(batch)
        result.loss.backward()
        return result.loss.detach().clone(), {
            name: parameter.grad.detach().clone()
            for name, parameter in pipeline.named_parameters()
            if parameter.grad is not None
        }

    before_loss, before_gradients = training_step()
    pipeline.eval()
    with torch.no_grad():
        pipeline.forward_infer_step_from_latents(
            torch.randn(1, 48, 1, 4, 4),
            PolicyInferContext(state=torch.randn(1, 1, 4)),
            text_context=torch.randn(1, 3, 16),
        )
    assert variant.packed_block_stack is owner
    assert tuple(pipeline.state_dict()) == keys
    assert all(
        dict(pipeline.named_parameters())[name] is parameter
        for name, parameter in parameters.items()
    )
    after_loss, after_gradients = training_step()
    torch.testing.assert_close(after_loss, before_loss, rtol=0, atol=0)
    assert after_gradients.keys() == before_gradients.keys()
    for name, gradient in before_gradients.items():
        torch.testing.assert_close(after_gradients[name], gradient, rtol=0, atol=0)


def test_execution_views_are_nonowning_and_follow_deepcopy():
    pipeline, _ = _tiny_pipeline(VideoActionProgram.VIDEO_THEN_ACTION)
    clone = copy.deepcopy(pipeline)
    for model in (pipeline, clone):
        owner = model.policy_variant.packed_block_stack
        core = model.visual_tower.core
        expert = model.policy_variant.action_expert
        assert len(core.blocks) == len(expert.blocks) == 0
        assert tuple(core.execution_blocks) == tuple(
            block.video_block for block in owner.packed_blocks
        )
        assert tuple(expert.execution_blocks) == tuple(
            block.action_block for block in owner.packed_blocks
        )
        assert not any("execution_blocks" in key for key in model.state_dict())
        named = tuple(model.named_parameters(remove_duplicate=False))
        assert len({id(parameter) for _, parameter in named}) == len(named)
    assert (
        clone.visual_tower.core.execution_blocks[0]
        is not pipeline.visual_tower.core.execution_blocks[0]
    )
