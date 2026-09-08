"""Regression coverage for video action text conditioning disabled."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch

from open_wam.configs import BatchingConfig, BatchingMode, TextConditioningMode, VideoActionProgram
from open_wam.data.latent_batching import LatentBatchCollator
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training.step_executor import LatentBatchAdapter, PipelineTrainStepExecutor


_SPEC = importlib.util.spec_from_file_location(
    "_text_conditioning_tiny_fixture", Path(__file__).with_name("test_token_budget_pipeline.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _pipeline(program, *, disabled=True, dropout=0.0):
    config = _FIXTURE._config(program, False)
    config = replace(
        config,
        policy_variant=replace(
            config.policy_variant,
            text_conditioning_mode=(TextConditioningMode.DISABLED if disabled else TextConditioningMode.TASK_PROMPT),
        ),
        training=replace(config.training, text_condition_dropout_prob=dropout),
        inference=replace(config.inference, guidance_scale=1.0, action_guidance_scale=1.0),
    )
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.policy_variant.initialize_for_training(pipeline.visual_tower)
    return pipeline, PipelineTrainStepExecutor(
        pipeline=pipeline, batch_adapter=LatentBatchAdapter(), training_config=config.training,
    )


def _samples(program):
    originals = _FIXTURE._samples(program)
    return [
        replace(
            originals[index % len(originals)],
            negative_text_context=torch.full_like(originals[index % len(originals)].text_context, 0.125),
            task_text=f"original task {index}",
            metadata={**originals[index % len(originals)].metadata, "sample_index": index},
        )
        for index in range(6)
    ]


def _collate(samples):
    return LatentBatchCollator(BatchingConfig(mode=BatchingMode.PACKED, pad_to_multiple_of=8))(samples)


def _run_with_gradients(pipeline, executor, samples):
    pipeline.zero_grad(set_to_none=True)
    torch.manual_seed(20260908)
    result = executor.forward_train(_collate(samples))
    result.loss.backward()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in pipeline.named_parameters()
        if parameter.requires_grad
    }
    assert torch.isfinite(result.loss)
    assert any(value is not None and bool(value.abs().max() > 0) for value in gradients.values())
    return result, gradients


@pytest.mark.parametrize("program", [VideoActionProgram.VIDEO_THEN_ACTION, VideoActionProgram.JOINT])
@pytest.mark.parametrize("training", [True, False], ids=["train", "validation_eval"])
def test_disabled_packed_six_samples_ignore_changed_task_embeddings_and_strings(
    program, training, monkeypatch
):
    pipeline, executor = _pipeline(program)
    pipeline.train(training)
    samples = _samples(program)
    changed = [
        replace(sample, text_context=sample.text_context * -17 + 29, task_text=f"different task {index}")
        for index, sample in enumerate(samples)
    ]
    observed = []
    original_prepare = pipeline.policy_variant.prepare_train_inputs

    def observe(visual_outputs, batch):
        observed.append(visual_outputs.frontend.conditioning.text_context.detach().clone())
        return original_prepare(visual_outputs, batch)

    monkeypatch.setattr(pipeline.policy_variant, "prepare_train_inputs", observe)
    first, first_gradients = _run_with_gradients(pipeline, executor, samples)
    second, second_gradients = _run_with_gradients(pipeline, executor, changed)
    assert len(observed) == 12
    for index, actual in enumerate(observed):
        torch.testing.assert_close(actual, samples[index % 6].negative_text_context[None], atol=0, rtol=0)
    torch.testing.assert_close(first.loss, second.loss, atol=0, rtol=0)
    torch.testing.assert_close(first.output.decoder_output.action_pred, second.output.decoder_output.action_pred, atol=0, rtol=0)
    assert first_gradients.keys() == second_gradients.keys()
    for name, first_gradient in first_gradients.items():
        other = second_gradients[name]
        if first_gradient is None:
            assert other is None, name
        else:
            torch.testing.assert_close(first_gradient, other, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("training", [True, False], ids=["train_replaced", "eval_not_replaced"])
def test_dropout_one_alone_only_replaces_conditioning_in_train_mode(training):
    pipeline, executor = _pipeline(VideoActionProgram.VIDEO_THEN_ACTION, disabled=False, dropout=1.0)
    pipeline.train(training)
    samples = _samples(VideoActionProgram.VIDEO_THEN_ACTION)
    result = executor.forward_train(_collate(samples))
    for sample, output in zip(samples, result.output.sample_outputs, strict=True):
        expected = sample.negative_text_context if training else sample.text_context
        torch.testing.assert_close(output.visual_outputs.frontend.conditioning.text_context, expected[None], atol=0, rtol=0)


def test_default_task_prompt_model_remains_sensitive_to_task_embeddings():
    program = VideoActionProgram.VIDEO_THEN_ACTION
    assert _FIXTURE._config(program, False).policy_variant.text_conditioning_mode is TextConditioningMode.TASK_PROMPT
    pipeline, executor = _pipeline(program, disabled=False, dropout=0.0)
    samples = _samples(program)
    changed = [replace(sample, text_context=sample.text_context * -17 + 29) for sample in samples]
    first, _ = _run_with_gradients(pipeline, executor, samples)
    second, _ = _run_with_gradients(pipeline, executor, changed)
    for original, altered, sample, changed_sample in zip(
        first.output.sample_outputs, second.output.sample_outputs, samples, changed, strict=True
    ):
        torch.testing.assert_close(original.visual_outputs.frontend.conditioning.text_context, sample.text_context[None], atol=0, rtol=0)
        torch.testing.assert_close(altered.visual_outputs.frontend.conditioning.text_context, changed_sample.text_context[None], atol=0, rtol=0)
    assert not torch.equal(first.output.decoder_output.action_pred, second.output.decoder_output.action_pred)
    assert not torch.equal(first.loss, second.loss)


@pytest.mark.parametrize("supplied_blank", [True, False], ids=["cached_blank", "encode_blank"])
def test_disabled_inference_frontend_ignores_tasks_and_never_encodes_task_prompt(
    supplied_blank, monkeypatch
):
    program = VideoActionProgram.VIDEO_THEN_ACTION
    pipeline, _ = _pipeline(program)
    pipeline.eval()
    sample = _samples(program)[0]
    blank = sample.negative_text_context[None]
    calls = []
    assets = pipeline.visual_tower.frontend.reference_assets

    def reject_task_encoding(*args, **kwargs):
        pytest.fail("Disabled text conditioning must not encode a task prompt")

    def encode_blank(*, batch_size, device, dtype):
        calls.append(batch_size)
        assert batch_size == 1
        return blank.to(device=device, dtype=dtype)

    monkeypatch.setattr(assets, "encode_text", reject_task_encoding)
    monkeypatch.setattr(assets, "encode_blank_text", encode_blank)
    for task, embedding in (("stack cube", sample.text_context[None]), ("unrelated instruction", None)):
        visual = pipeline.prepare_visual_outputs_from_latents(
            sample.video_latents[None], task_text=(task,), text_context=embedding,
            negative_text_context=blank if supplied_blank else None,
        )
        torch.testing.assert_close(visual.frontend.conditioning.text_context, blank, atol=0, rtol=0)
        torch.testing.assert_close(visual.frontend.conditioning.negative_text_context, blank, atol=0, rtol=0)
    assert calls == ([] if supplied_blank else [1, 1])


def test_disabled_mode_preserves_complete_checkpoint_parameter_schema():
    program = VideoActionProgram.VIDEO_THEN_ACTION
    conditioned, _ = _pipeline(program, disabled=False)
    disabled, _ = _pipeline(program, disabled=True)
    source, target = conditioned.state_dict(), disabled.state_dict()
    assert source.keys() == target.keys()
    # The current shared block names its text cross-attention module attn2.
    assert {
        f"policy_variant.packed_block_stack.packed_blocks.0.{expert}_block.attn2.to_{projection}.weight"
        for expert in ("video", "action") for projection in ("q", "k", "v")
    } <= source.keys()
    assert all(source[name].shape == target[name].shape and source[name].dtype == target[name].dtype for name in source)
    loaded = disabled.load_state_dict(source, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    for name, tensor in source.items():
        torch.testing.assert_close(disabled.state_dict()[name], tensor, atol=0, rtol=0)
