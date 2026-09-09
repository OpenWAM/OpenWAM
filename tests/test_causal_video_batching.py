"""Video-only batching preserves attention isolation and frame-weighted loss."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import BatchingConfig, BatchingMode
from open_wam.data.latent_batching import LatentBatchCollator, LengthBucketSampler
from open_wam.data.latent_contracts import LatentWAMSample
from open_wam.data.mixed_video_planning import MixedVideoWindowPlanner
from open_wam.training.step_executor import LatentBatchAdapter, PipelineTrainStepExecutor

from tests.test_causal_video_prediction import _tiny_causal_video_pipeline, _deterministic_cpu_math


def _samples():
    torch.manual_seed(61)
    return [
        LatentWAMSample(
            video_latents=torch.randn(48, n, 2, 4),
            actions=torch.zeros(0, 7), action_mask=torch.zeros(0, 7),
            state=torch.zeros(0, 8), state_mask=torch.zeros(0, 8),
            text_context=torch.randn(3, 8),
            task_text="move object",
            metadata={
                "dataset_type": "mixed_video", "observed_prefix_frames": 1,
                "future_suffix_frames": n - 1, "valid_video_frames": n,
            },
        ) for n in (3, 6)
    ]


@pytest.mark.parametrize("mode", [BatchingMode.PADDED, BatchingMode.BUCKET, BatchingMode.PACKED])
def test_empty_actions_video_batch_trains_and_keeps_checkpoint_keys(mode):
    with _deterministic_cpu_math():
        config, pipeline, _ = _tiny_causal_video_pipeline()
        before = tuple(pipeline.state_dict())
        batch = LatentBatchCollator(BatchingConfig(mode=mode))(_samples())
        assert batch.video_latents.shape == (2, 48, 6, 2, 4)
        assert batch.actions.shape == (2, 0, 7)
        assert batch.sequence_lengths == (3, 6)
        assert batch.metadata[0]["valid_video_frames"] == 3
        executor = PipelineTrainStepExecutor(
            pipeline=pipeline, batch_adapter=LatentBatchAdapter(), training_config=config.training
        )
        result = executor.forward_train(batch)
        result.loss.backward()
        assert torch.isfinite(result.loss)
        assert tuple(pipeline.state_dict()) == before
        assert all(torch.isfinite(p.grad).all() for p in pipeline.parameters() if p.grad is not None)


def test_packed_loss_and_gradients_match_padded_frame_weighting():
    with _deterministic_cpu_math():
        samples = _samples()
        results = []
        for mode in (BatchingMode.PADDED, BatchingMode.PACKED):
            config, pipeline, _ = _tiny_causal_video_pipeline()
            executor = PipelineTrainStepExecutor(
                pipeline=pipeline, batch_adapter=LatentBatchAdapter(), training_config=config.training
            )
            batch = LatentBatchCollator(BatchingConfig(mode=mode))(samples)
            torch.manual_seed(79)
            result = executor.forward_train(batch)
            result.loss.backward()
            results.append((result.loss.detach(), {
                name: p.grad.clone() for name, p in pipeline.named_parameters() if p.grad is not None
            }))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=2e-5, atol=2e-6)
        assert results[0][1].keys() == results[1][1].keys()
        for name in results[0][1]:
            torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=2e-4, atol=3e-6, msg=name)


def test_packed_video_and_text_are_isolated_and_positions_reset():
    with _deterministic_cpu_math(), torch.no_grad():
        _, pipeline, _ = _tiny_causal_video_pipeline()
        tower = pipeline.visual_tower
        torch.manual_seed(25)
        video = torch.randn(2, 48, 6, 2, 4)
        times = torch.rand(2, 6) * 1000
        text = torch.randn(2, 3, 8)
        kwargs = dict(noisy_latents=video, timesteps=times, text_context=text)
        calls = []
        handles = [b.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1])) for b in tower.core.blocks]
        packed = tower.predict_video_flow(**kwargs, sequence_lengths=(3, 6))
        for h in handles:
            h.remove()
        assert len(calls) == len(tower.core.blocks)
        assert set(calls) == {18}  # two spatial tokens per frame, nine real frames
        for i, length in enumerate((3, 6)):
            single = tower.predict_video_flow(
                noisy_latents=video[i:i+1, :, :length], timesteps=times[i:i+1, :length],
                text_context=text[i:i+1],
            )
            torch.testing.assert_close(packed[i:i+1, :, :length], single, rtol=2e-5, atol=2e-6)
        changed_video, changed_text = video.clone(), text.clone()
        changed_video[1] += 19
        changed_text[1] -= 11
        changed = tower.predict_video_flow(
            noisy_latents=changed_video, timesteps=times, text_context=changed_text,
            sequence_lengths=(3, 6),
        )
        torch.testing.assert_close(packed[0], changed[0], rtol=0, atol=0)
        assert torch.count_nonzero(packed[0, :, 3:]) == 0


def test_spatial_then_length_bucketing_preserves_rank_draws_and_duplicates():
    planner = MixedVideoWindowPlanner(SimpleNamespace(sampling_seed=31))
    original = tuple(range(29)) + (2, 5, 2, 17)
    planner.build_source_balanced_epoch_order = lambda **kwargs: original
    shapes = [i % 3 for i in range(29)]
    order = planner.build_shape_bucketed_epoch_order(
        sample_index=list(range(29)), episode_records={}, shape_ids=shapes,
        world_size=2, batch_size=4, epoch=7,
    )
    assert not (Counter(original) - Counter(order))
    for rank in range(2):
        retained = order[rank::2]
        sampler = LengthBucketSampler(
            retained, length_for_index=lambda i: 64 - i, shape_for_index=lambda i: shapes[i],
            batch_size=4, pool_size=16, drop_last=True,
        )
        result = list(sampler)
        assert Counter(result) == Counter(retained)
        assert result == list(sampler)
        for start in range(0, len(result), 4):
            assert len({shapes[i] for i in result[start:start+4]}) == 1
