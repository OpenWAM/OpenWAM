"""Real VTA/Joint runtime subdivision parity against single-sample training."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
    TrainerConfig,
    TrainingConfig,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.data.latent_batching import LatentBatchCollator
from open_wam.data.latent_contracts import LatentWAMSample, collate_latent_wam_samples
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.pipelines.token_cost import build_training_token_cost_fn
from open_wam.training.launch import DistributedLaunchContext
from open_wam.training.runtime import TrainingRuntime
from open_wam.training.state import TrainState
from open_wam.training.step_executor import LatentBatchAdapter, PipelineTrainStepExecutor
from open_wam.training.strategies import SingleDeviceStrategy


def _config(program, activation_checkpointing):
    legacy = program is VideoActionProgram.VIDEO_THEN_ACTION
    return ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            train_batch_size=3,
            val_batch_size=1,
            batching=BatchingConfig(
                mode=BatchingMode.PACKED,
                pad_to_multiple_of=8,
                max_tokens=112 if legacy else 96,
            ),
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
            load_text_conditioning=False,
            load_wan_vae_frontend=False,
        ),
        policy_variant=DualExpertPolicyConfig(
            hidden_size=32,
            program=program,
            num_action_layers=2,
            use_activation_checkpointing=activation_checkpointing,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            history_stream_visibility=(
                HistoryStreamVisibility.VIDEO_ONLY
                if legacy
                else HistoryStreamVisibility.FULL
            ),
            sequence_contract=(
                VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
                if legacy
                else VideoActionSequenceContract.DEFAULT
            ),
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            gradient_accumulation_steps=1,
            text_condition_dropout_prob=0.0,
            enabled_objectives=("action", "latent"),
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
        ),
        inference=InferenceConfig(frame_chunk_size=2),
        trainer=TrainerConfig(batch_adapter="latents"),
    )


def _samples(program):
    generator = torch.Generator().manual_seed(1234)
    samples = []
    for index, (frames, text_tokens) in enumerate(((4, 5), (4, 5), (7, 7))):
        samples.append(
            LatentWAMSample(
                video_latents=torch.randn(48, frames, 4, 4, generator=generator),
                actions=torch.randn(2 * frames, 4, generator=generator),
                action_mask=torch.ones(2 * frames, 4),
                state=torch.randn(1, 4, generator=generator),
                state_mask=torch.ones(1, 4),
                condition_latents=torch.randn(
                    48,
                    1 if program is VideoActionProgram.VIDEO_THEN_ACTION else frames,
                    4,
                    4,
                    generator=generator,
                ),
                proprio_context_frames=torch.randn(frames, 4, generator=generator),
                proprio_context_frames_mask=torch.ones(frames, 4),
                text_context=torch.randn(text_tokens, 16, generator=generator),
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
        )
    return samples


@pytest.mark.integration
@pytest.mark.parametrize(
    "program", [VideoActionProgram.VIDEO_THEN_ACTION, VideoActionProgram.JOINT]
)
@pytest.mark.parametrize(
    ("mode", "sample_count"),
    [(BatchingMode.STRICT, 1)]
    + [
        (mode, count)
        for mode in (BatchingMode.BUCKET, BatchingMode.PADDED, BatchingMode.PACKED)
        for count in (1, 2)
    ],
)
def test_real_pipeline_condition_noise_synchronizes_only_strict_samples(
    monkeypatch, program, mode, sample_count
):
    import open_wam.models.policy_variants.dual_expert.packed_training as packed

    config = _config(program, False)
    config = replace(
        config,
        data=replace(config.data, batching=BatchingConfig(mode=mode)),
        policy_variant=replace(config.policy_variant, noisy_video_condition_prob=1.0),
    )
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.policy_variant.initialize_for_training(pipeline.visual_tower)
    executor = PipelineTrainStepExecutor(
        pipeline=pipeline,
        batch_adapter=LatentBatchAdapter(),
        training_config=config.training,
    )
    samples = _samples(program)[:sample_count]
    batch = (
        collate_latent_wam_samples(samples)
        if mode is BatchingMode.STRICT
        else LatentBatchCollator(config.data.batching)(samples)
    )
    observed = []
    broadcasts = []
    original = packed.build_video_flow_match_train_artifacts

    def build_artifacts(*args, **kwargs):
        observed.append(kwargs["synchronize_noisy_condition_decision"])
        assert kwargs["noisy_condition_prob"] == 1.0
        return original(*args, **kwargs)

    monkeypatch.setattr(packed, "build_video_flow_match_train_artifacts", build_artifacts)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, src: broadcasts.append(src)
    )
    result = executor.forward_train(batch)
    assert torch.isfinite(result.loss)
    assert observed == [mode is BatchingMode.STRICT] * sample_count
    assert broadcasts == ([0] if mode is BatchingMode.STRICT else [])


@pytest.mark.integration
@pytest.mark.parametrize(
    "program", [VideoActionProgram.VIDEO_THEN_ACTION, VideoActionProgram.JOINT]
)
@pytest.mark.parametrize("activation_checkpointing", [False, True])
@pytest.mark.parametrize("compact_action", [False, True])
def test_token_runtime_preserves_real_pipeline_predictions_loss_gradients_and_update(
    program, activation_checkpointing, compact_action, monkeypatch
):
    torch.manual_seed(57)
    config = _config(program, activation_checkpointing)
    if compact_action:
        # Scaled-width analogue of the 500M profile: preserve depth and Q/K/V
        # dimensions while shrinking residual/FFN width. The actual 30-layer
        # profile is counted on meta tensors in test_action_expert_size.py.
        config = replace(config, policy_variant=replace(
            config.policy_variant, action_hidden_size=6, action_ffn_dim=24,
            action_expert_init_mode="video_weight_interpolate",
        ))
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.policy_variant.initialize_for_training(pipeline.visual_tower)
    executor = PipelineTrainStepExecutor(
        pipeline=pipeline,
        batch_adapter=LatentBatchAdapter(),
        training_config=config.training,
    )
    samples = _samples(program)
    batch = LatentBatchCollator(config.data.batching)(samples)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in pipeline.named_parameters()
    }

    # Keep the logical batch's geometry in the B1 reference. Disable text
    # dropout so both arrangements consume diffusion randomness in sample order.
    reference_losses = []
    reference_predictions = []
    torch.manual_seed(911)
    for index, sample in enumerate(samples):
        single = collate_latent_wam_samples(
            [replace(sample, metadata=batch.metadata[index])]
        )
        result = executor.forward_train(single)
        reference_losses.append(result.loss.detach().clone())
        reference_predictions.append(result.output.decoder_output.action_pred.detach().clone())
        (result.loss / len(samples)).backward()
        del result
    reference_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in pipeline.named_parameters()
        if parameter.grad is not None
    }
    pipeline.zero_grad(set_to_none=True)

    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.config = config
    runtime.model = pipeline
    runtime.strategy = SingleDeviceStrategy(
        accelerator=config.trainer.accelerator,
        precision=config.trainer.precision,
        launch_context=DistributedLaunchContext.from_env({}),
    )
    runtime.token_cost_fn = build_training_token_cost_fn(config)
    runtime.train_state = TrainState()
    runtime._accumulated_train_metrics = {}
    runtime.last_token_batch_plan = None
    learning_rate = 0.001
    runtime.optimizer = torch.optim.SGD(pipeline.parameters(), lr=learning_rate)
    runtime.scheduler = torch.optim.lr_scheduler.StepLR(
        runtime.optimizer, step_size=1, gamma=0.5
    )
    logged = []
    runtime.log_sink = SimpleNamespace(log_metrics=lambda **payload: logged.append(payload))
    actual_predictions = []
    actual_sample_losses = []
    actual_metadata = []

    def forward(physical_batch):
        result = executor.forward_train(physical_batch)
        actual_metadata.extend(physical_batch.metadata)
        for sample_output in result.output.sample_outputs:
            actual_predictions.append(sample_output.decoder_output.action_pred.detach().clone())
            actual_sample_losses.append(sample_output.decoder_output.loss.detach().clone())
        return result

    runtime.step_executor = SimpleNamespace(
        batch_adapter=executor.batch_adapter, forward_train=forward
    )
    actual_gradients = {}
    optimizer_calls = []
    original_optimizer_step = runtime.strategy.optimizer_step

    def optimizer_step(optimizer):
        optimizer_calls.append(1)
        actual_gradients.update(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in pipeline.named_parameters()
                if parameter.grad is not None
            }
        )
        original_optimizer_step(optimizer)

    monkeypatch.setattr(runtime.strategy, "optimizer_step", optimizer_step)
    torch.manual_seed(911)
    runtime._train_micro_step(batch)

    expected_costs = (
        (56, 56, 92)
        if program is VideoActionProgram.VIDEO_THEN_ACTION
        else (48, 48, 84)
    )
    assert runtime.last_token_batch_plan["token_costs"] == expected_costs
    assert runtime.last_token_batch_plan["groups"] == ((0, 1), (2,))
    assert actual_metadata == list(batch.metadata)
    assert runtime.train_state.global_step == runtime.train_state.seen_batches == 1
    assert runtime.train_state.optimizer_step == runtime.scheduler.last_epoch == 1
    assert optimizer_calls == [1]
    assert actual_gradients.keys() == reference_gradients.keys()
    assert len(actual_gradients) == (140 if compact_action else 138)
    for actual, expected in zip(actual_predictions, reference_predictions, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(actual_sample_losses, reference_losses, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        torch.tensor(logged[0]["metrics"]["loss"]),
        torch.stack(reference_losses).mean(),
        atol=2e-6,
        rtol=2e-5,
    )
    for name, expected in reference_gradients.items():
        torch.testing.assert_close(
            actual_gradients[name], expected, atol=3e-6, rtol=3e-4,
            msg=lambda detail, name=name: f"{name}: {detail}",
        )
    for name, parameter in pipeline.named_parameters():
        expected = initial_parameters[name]
        if name in reference_gradients:
            expected = expected - learning_rate * reference_gradients[name]
        torch.testing.assert_close(
            parameter, expected, atol=2e-6, rtol=2e-5,
            msg=lambda detail, name=name: f"{name}: {detail}",
        )
