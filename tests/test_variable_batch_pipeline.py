"""End-to-end contracts for padded and isolated packed Dual Expert batches.

These use real tiny visual/policy/decoder modules on CPU. They deliberately
cover VTA's legacy prefix/proprio sequence and Joint's full history visibility.
"""

from dataclasses import replace

import pytest
import torch

from open_wam.configs import (
    ActionSchemaConfig,
    BatchingConfig,
    BatchingMode,
    DualExpertActionDecoderConfig,
    DualExpertPolicyConfig,
    ExperimentConfig,
    HistoryStreamVisibility,
    InferenceConfig,
    ProprioContextMode,
    RobotWinDataConfig,
    TrainingConfig,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.data.latent_batching import LatentBatchCollator
from open_wam.data.latent_contracts import LatentWAMSample, collate_latent_wam_samples
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training.step_executor import (
    LatentBatchAdapter,
    PipelineTrainStepExecutor,
)

MODES = (BatchingMode.BUCKET, BatchingMode.PADDED, BatchingMode.PACKED)
PROGRAMS = tuple(
    program
    for program in VideoActionProgram
    if program
    not in {
        VideoActionProgram.GENERALIST_JOINT_DENOISING,
        VideoActionProgram.FORWARD_DYNAMICS,
        VideoActionProgram.INVERSE_DYNAMICS,
    }
)


def _tiny_pipeline(
    program,
    *,
    activation_checkpointing=False,
    attention_head_dim=8,
    generalist_mode_text_token=False,
    sequence_contract=None,
):
    legacy = program is VideoActionProgram.VIDEO_THEN_ACTION
    hidden_size = 4 * attention_head_dim
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=hidden_size,
            num_layers=2,
            num_heads=4,
            attention_head_dim=attention_head_dim,
            ffn_dim=hidden_size * 2,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
            load_text_conditioning=False,
            load_wan_vae_frontend=False,
        ),
        policy_variant=DualExpertPolicyConfig(
            hidden_size=hidden_size,
            program=program,
            num_action_layers=2,
            generalist_mode_text_token=generalist_mode_text_token,
            use_activation_checkpointing=activation_checkpointing,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            history_stream_visibility=(
                HistoryStreamVisibility.VIDEO_ONLY
                if legacy
                else HistoryStreamVisibility.FULL
            ),
            sequence_contract=sequence_contract
            or (
                VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
                if legacy
                else VideoActionSequenceContract.DEFAULT
            ),
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=hidden_size, action_dim=4, action_horizon=4
        ),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            enabled_objectives=("action", "latent"),
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
        ),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.policy_variant.initialize_for_training(pipeline.visual_tower)
    return pipeline, PipelineTrainStepExecutor(
        pipeline=pipeline,
        batch_adapter=LatentBatchAdapter(),
        training_config=config.training,
    )


def _samples(program):
    torch.manual_seed(1234)
    return [
        LatentWAMSample(
            video_latents=torch.randn(48, frames, 4, 4),
            actions=torch.randn(2 * frames, 4),
            action_mask=torch.ones(2 * frames, 4),
            state=torch.randn(1, 4),
            state_mask=torch.ones(1, 4),
            condition_latents=torch.randn(
                48,
                1 if program is VideoActionProgram.VIDEO_THEN_ACTION else frames,
                4,
                4,
            ),
            proprio_context_frames=torch.randn(frames, 4),
            proprio_context_frames_mask=torch.ones(frames, 4),
            text_context=torch.randn(text_tokens, 16),
            negative_text_context=torch.zeros(text_tokens, 16),
            task_text=f"synthetic sample {index}",
            metadata={
                "sample_index": index,
                "sampled_chunk_size": 2,
                "sampled_window_size": 8,
                "history_frames": 2,
                "action_tokens_per_frame": 2,
            },
        )
        for index, (frames, text_tokens) in enumerate(((4, 5), (7, 7)))
    ]


def _collate(samples, mode):
    return LatentBatchCollator(BatchingConfig(mode=mode))(samples)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("program", PROGRAMS)
def test_variable_pipeline_prepares_real_extents_and_batches_model_once(
    mode, program, monkeypatch
):
    torch.manual_seed(71)
    pipeline, executor = _tiny_pipeline(program)
    batch = _collate(_samples(program), mode)
    prepared_calls = []
    batch_calls = []
    original_prepare = pipeline.policy_variant.prepare_train_inputs
    original_forward_batch = pipeline.policy_variant.forward_train_batch

    def prepare(visual_outputs, batch):
        prepared = original_prepare(visual_outputs, batch)
        prepared_calls.append((visual_outputs, prepared))
        return prepared

    def forward_batch(**kwargs):
        batch_calls.append(kwargs)
        return original_forward_batch(**kwargs)

    monkeypatch.setattr(pipeline.policy_variant, "prepare_train_inputs", prepare)
    monkeypatch.setattr(pipeline.policy_variant, "forward_train_batch", forward_batch)
    result = executor.forward_train(batch)

    assert len(prepared_calls) == 2
    assert len(batch_calls) == 1
    assert batch_calls[0]["batching_mode"] is mode
    assert len(result.output.sample_outputs) == 2
    assert result.output.visual_outputs is None
    assert result.output.decoder_output.action_pred.shape == (2, 14, 4)
    sample_losses = []
    for index, ((visual, prepared), sample_output) in enumerate(
        zip(prepared_calls, result.output.sample_outputs, strict=True)
    ):
        frames = batch.sequence_lengths[index]
        assert visual.frontend.video_latents.shape[2] == frames
        assert visual.frontend.conditioning.text_context.shape[1] == (5, 7)[index]
        assert prepared.batch.actions.shape == (1, frames * 2, 4)
        assert prepared.batch.extra["proprio_context_frames"].shape == (1, frames, 4)
        condition_frames = (
            1 if program is VideoActionProgram.VIDEO_THEN_ACTION else frames
        )
        assert prepared.batch.extra["condition_latents"].shape == (
            1,
            48,
            condition_frames,
            4,
            4,
        )
        metadata = prepared.batch.extra["metadata"][0]
        assert metadata["sample_index"] == index
        assert metadata["sampled_chunk_size"] == 2
        assert metadata["sampled_window_size"] == 8
        # The original decoder is the authority for per-sample loss reduction.
        decoded = pipeline.action_decoder.forward_train(
            sample_output.policy_output, prepared.batch
        )
        torch.testing.assert_close(decoded.loss, sample_output.decoder_output.loss)
        sample_losses.append(decoded.loss)
    torch.testing.assert_close(result.loss, torch.stack(sample_losses).mean())
    for name, value in result.output.decoder_output.metrics.items():
        expected = torch.stack(
            [
                sample.decoder_output.metrics[name]
                for sample in result.output.sample_outputs
            ]
        ).mean()
        torch.testing.assert_close(value, expected)

    result.loss.backward()
    gradients = [
        param.grad for param in pipeline.parameters() if param.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("program", PROGRAMS)
def test_padding_poison_does_not_change_valid_predictions_or_loss(mode, program):
    _pipeline, executor = _tiny_pipeline(program)
    batch = _collate(_samples(program), mode)
    poisoned_fields = {}
    for name, lengths in batch.tensor_lengths.items():
        value = getattr(batch, name)
        if value is None:
            continue
        poisoned = value.clone()
        axis = 2 if name in {"video_latents", "condition_latents"} else 1
        for index, length in enumerate(lengths):
            slices = [slice(None)] * value.ndim
            slices[0] = index
            slices[axis] = slice(length, None)
            poisoned[tuple(slices)] = float("nan")
        poisoned_fields[name] = poisoned
    poisoned = replace(batch, **poisoned_fields)

    torch.manual_seed(301)
    baseline = executor.forward_train(batch)
    torch.manual_seed(301)
    changed = executor.forward_train(poisoned)

    torch.testing.assert_close(baseline.loss, changed.loss)
    assert torch.isfinite(changed.loss)
    for reference, actual in zip(
        baseline.output.sample_outputs, changed.output.sample_outputs, strict=True
    ):
        torch.testing.assert_close(
            reference.decoder_output.action_pred, actual.decoder_output.action_pred
        )
        torch.testing.assert_close(
            reference.decoder_output.aux["predicted_latents"],
            actual.decoder_output.aux["predicted_latents"],
        )


@pytest.mark.parametrize("mode", [BatchingMode.PADDED, BatchingMode.PACKED])
@pytest.mark.parametrize("program", PROGRAMS)
def test_peer_sequence_cannot_change_first_samples_predictions(mode, program):
    _pipeline, executor = _tiny_pipeline(program)
    samples = _samples(program)
    peer = samples[1]
    changed_peer = replace(
        peer,
        video_latents=peer.video_latents * 17 + 9,
        actions=peer.actions * -11,
        state=peer.state + 40,
        condition_latents=peer.condition_latents - 23,
        proprio_context_frames=peer.proprio_context_frames + 31,
        text_context=peer.text_context * -7,
    )
    torch.manual_seed(502)
    baseline = executor.forward_train(_collate(samples, mode))
    torch.manual_seed(502)
    changed = executor.forward_train(_collate([samples[0], changed_peer], mode))
    first = baseline.output.sample_outputs[0].decoder_output
    first_after = changed.output.sample_outputs[0].decoder_output

    torch.testing.assert_close(first.action_pred, first_after.action_pred)
    torch.testing.assert_close(
        first.aux["predicted_latents"], first_after.aux["predicted_latents"]
    )
    torch.testing.assert_close(first.loss, first_after.loss)


@pytest.mark.parametrize("mode", [BatchingMode.PADDED, BatchingMode.PACKED])
@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_variable_batch_matches_independent_single_sample_training(
    mode, program, activation_checkpointing
):
    pipeline, executor = _tiny_pipeline(
        program, activation_checkpointing=activation_checkpointing
    )
    samples = _samples(program)
    batch = _collate(samples, mode)
    singles = [collate_latent_wam_samples([sample]) for sample in samples]
    torch.manual_seed(911)
    together = executor.forward_train(batch)
    torch.manual_seed(911)
    independent = [executor.forward_train(single) for single in singles]

    for batched, reference in zip(
        together.output.sample_outputs, independent, strict=True
    ):
        torch.testing.assert_close(
            batched.decoder_output.action_pred,
            reference.output.decoder_output.action_pred,
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            batched.decoder_output.loss, reference.loss, rtol=2e-5, atol=2e-6
        )
    torch.testing.assert_close(
        together.loss,
        torch.stack([item.loss for item in independent]).mean(),
        rtol=2e-5,
        atol=2e-6,
    )
    together.loss.backward()
    batch_gradients = {
        name: parameter.grad.clone()
        for name, parameter in pipeline.named_parameters()
        if parameter.grad is not None
    }
    pipeline.zero_grad(set_to_none=True)
    torch.stack([item.loss for item in independent]).mean().backward()
    independent_gradients = {
        name: parameter.grad
        for name, parameter in pipeline.named_parameters()
        if parameter.grad is not None
    }
    assert batch_gradients.keys() == independent_gradients.keys()
    for name, gradient in batch_gradients.items():
        torch.testing.assert_close(
            gradient,
            independent_gradients[name],
            rtol=3e-4,
            atol=3e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )


@pytest.mark.parametrize("mode", [BatchingMode.PADDED, BatchingMode.PACKED])
@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("stamped", [False, True])
def test_batching_preserves_each_samples_geometry_and_rng_order(mode, program, stamped):
    pipeline, executor = _tiny_pipeline(program)
    samples = _samples(program)
    if stamped:
        samples[1] = replace(
            samples[1],
            metadata={
                **samples[1].metadata,
                "sampled_chunk_size": 3,
                "sampled_window_size": 4,
                "history_frames": 1,
                "frame_shift": 7,
            },
        )
    else:
        samples = [replace(sample, metadata={}) for sample in samples]
    torch.manual_seed(419)
    together = executor.forward_train(_collate(samples, mode))
    rng_after = torch.random.get_rng_state()
    torch.manual_seed(419)
    independent = [
        executor.forward_train(collate_latent_wam_samples([sample]))
        for sample in samples
    ]
    assert torch.equal(rng_after, torch.random.get_rng_state())
    for actual, expected in zip(
        together.output.sample_outputs, independent, strict=True
    ):
        assert actual.policy_output.aux == expected.output.policy_output.aux
        torch.testing.assert_close(
            actual.decoder_output.action_pred,
            expected.output.decoder_output.action_pred,
        )
        torch.testing.assert_close(actual.decoder_output.loss, expected.loss)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("negative_only", [False, True])
def test_variable_collator_rejects_ambiguous_positive_negative_text_lengths(
    mode, negative_only
):
    samples = _samples(VideoActionProgram.VIDEO_THEN_ACTION)
    samples[0] = replace(
        samples[0],
        text_context=None if negative_only else samples[0].text_context,
        negative_text_context=torch.zeros(3, 16),
    )
    with pytest.raises(ValueError, match="text|Text"):
        _collate(samples, mode)


@pytest.mark.parametrize("mode", [BatchingMode.PADDED, BatchingMode.PACKED])
@pytest.mark.parametrize("program", PROGRAMS)
def test_variable_text_dropout_preserves_each_original_context_extent(
    mode, program, monkeypatch
):
    pipeline, executor = _tiny_pipeline(program)
    executor.training_config = replace(
        executor.training_config, text_condition_dropout_prob=1.0
    )
    samples = _samples(program)
    observed = []
    original_prepare = pipeline.policy_variant.prepare_train_inputs

    def prepare(visual_outputs, batch):
        observed.append((visual_outputs.frontend.conditioning.text_context, batch))
        return original_prepare(visual_outputs, batch)

    monkeypatch.setattr(pipeline.policy_variant, "prepare_train_inputs", prepare)

    result = executor.forward_train(_collate(samples, mode))

    assert torch.isfinite(result.loss)
    assert len(observed) == 2
    for index, (condition, batch) in enumerate(observed):
        torch.testing.assert_close(
            condition, samples[index].negative_text_context[None]
        )
        torch.testing.assert_close(
            batch.source_text_context, samples[index].text_context[None]
        )
