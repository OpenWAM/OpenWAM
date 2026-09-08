"""Opt-in two-rank semantic smokes plus long-sequence scaling cases.

Run with torchrun --standalone --nproc-per-node=2 -m pytest -q
tests/test_sequence_batch_fsdp.py after allocating the GPUs explicitly.
The transformer is small; this is not a pretrained-model benchmark.
"""

import os

import pytest
import torch

from open_wam.configs import (
    BatchingConfig,
    BatchingMode,
    DynamicsObjective,
    StrategyName,
    TrainerAccelerator,
    TrainerPrecision,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.data.latent_batching import LatentBatchCollator
from open_wam.data.latent_contracts import LatentWAMSample
from open_wam.training.strategies import DistributedStrategy
from tests.test_variable_batch_pipeline import _tiny_pipeline
from tests.test_sequence_batch_dynamics import dynamics_samples


@pytest.fixture(scope="module")
def fsdp_strategy():
    if (
        os.getenv("OPEN_WAM_RUN_GPU_SANITY") != "1"
        or int(os.getenv("WORLD_SIZE", "1")) < 2
    ):
        pytest.skip(
            "Requires explicitly allocated GPUs and a multi-rank torchrun launch."
        )
    strategy = DistributedStrategy(
        accelerator=TrainerAccelerator.GPU,
        precision=TrainerPrecision.BF16,
        kind=StrategyName.FSDP,
    )
    try:
        yield strategy
    finally:
        strategy.close()


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("program", list(VideoActionProgram))
@pytest.mark.parametrize("mode", [BatchingMode.PACKED, BatchingMode.PADDED])
@pytest.mark.parametrize(
    "frame_counts", [(9, 13), pytest.param((33, 65), marks=pytest.mark.slow)]
)
def test_sequence_batch_fsdp_training(program, mode, frame_counts, fsdp_strategy):
    strategy = fsdp_strategy
    torch.compiler.reset()
    torch.manual_seed(71)
    pipeline, executor = _tiny_pipeline(
        program,
        activation_checkpointing=True,
        attention_head_dim=32,
        generalist_mode_text_token=program
        is VideoActionProgram.GENERALIST_JOINT_DENOISING,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    )
    executor.pipeline = strategy.prepare_model(pipeline)
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=1e-4)
    for step in range(2):
        torch.manual_seed(701 + strategy.rank + step * strategy.world_size)
        samples = [
            LatentWAMSample(
                video_latents=torch.randn(48, frames, 8, 16),
                actions=torch.randn(4 * frames, 4),
                state=torch.randn(1, 4),
                state_mask=torch.ones(1, 4),
                condition_latents=torch.randn(
                    48,
                    1 if program is VideoActionProgram.VIDEO_THEN_ACTION else frames,
                    8,
                    16,
                ),
                proprio_context_frames=torch.randn(frames, 4),
                proprio_context_frames_mask=torch.ones(frames, 4),
                text_context=torch.randn(5, 16),
                negative_text_context=torch.zeros(5, 16),
                metadata={
                    "sampled_chunk_size": 1 + step + index,
                    "sampled_window_size": 64 - index * 16,
                    "action_tokens_per_frame": 4,
                },
            )
            for index, frames in enumerate(
                tuple(frames + 4 * strategy.rank for frames in frame_counts)
            )
        ]
        if program is VideoActionProgram.GENERALIST_JOINT_DENOISING:
            samples = dynamics_samples(tuple(DynamicsObjective), base=samples)
        elif program is VideoActionProgram.FORWARD_DYNAMICS:
            samples = dynamics_samples(
                (DynamicsObjective.ACTION_CONDITIONED_VIDEO,) * 2, base=samples
            )
        elif program is VideoActionProgram.INVERSE_DYNAMICS:
            samples = dynamics_samples(
                (DynamicsObjective.VIDEO_CONDITIONED_ACTION,) * 2, base=samples
            )
        batch = LatentBatchCollator(BatchingConfig(mode=mode))(samples)
        batch = executor.batch_adapter.move_to_device(batch, strategy.device)
        optimizer.zero_grad(set_to_none=True)
        with strategy.autocast_context():
            result = executor.forward_train(batch)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        gradients = [
            parameter.grad.to_local()
            for parameter in pipeline.parameters()
            if parameter.grad is not None
        ]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        assert any(value.abs().sum() > 0 for value in gradients)
        optimizer.step()
