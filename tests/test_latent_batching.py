from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Sampler

from open_wam.configs import (
    BatchAdapterName,
    BatchingConfig,
    BatchingMode,
    GenericDataConfig,
    parse_data_config,
)
from open_wam.data.distributed_sampling import WeightedReplacementDistributedSampler
from open_wam.data.latent_batching import LatentBatchCollator, LengthBucketSampler
from open_wam.data.latent_contracts import (
    LatentWAMSample,
    collate_latent_wam_samples,
    move_latent_wam_batch_to_device,
)
from open_wam.training.data_loading import (
    _build_variable_length_loader,
)
from open_wam.configs.sequence_contracts import validate_variable_batching_source


def _sample(
    length: int, *, chunk: int = 2, window: int = 4, text_length: int = 3
) -> LatentWAMSample:
    return LatentWAMSample(
        video_latents=torch.arange(2 * length * 4, dtype=torch.float32).reshape(
            2, length, 2, 2
        ),
        actions=torch.ones(length * 2, 3),
        action_mask=None,
        state=torch.ones(1, 3),
        state_mask=torch.ones(1, 3),
        text_context=torch.ones(text_length, 5),
        negative_text_context=torch.zeros(text_length, 5),
        condition_latents=torch.ones(2, 1, 2, 2),
        proprio_context_frames=torch.ones(length, 3),
        proprio_context_frames_mask=torch.ones(length, 3),
        metadata={
            "sampled_chunk_size": chunk,
            "sampled_window_size": window,
            "history_frames": max(1, min(chunk * ((window + 1) // 2), length - chunk)),
            "latent_loss_frame_start": 0,
            "latent_loss_frame_end": length,
            "frame_shift": 19,
        },
    )


@pytest.mark.parametrize("mode", list(BatchingMode))
def test_batching_config_parses_and_round_trips(mode):
    raw = {
        "dataset_type": "lerobot_v2_latent_local",
        "batching": {"mode": mode.value, "pad_to_multiple_of": 4},
    }
    config = parse_data_config(raw)
    assert config.batching.mode is mode
    assert parse_data_config(asdict(config)).batching == config.batching
    assert GenericDataConfig(batching={"mode": mode.value}).batching.mode is mode


def test_default_and_invalid_batching_config():
    assert GenericDataConfig().batching.mode is BatchingMode.STRICT
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            BatchingConfig(bucket_pool_size=invalid)
        with pytest.raises(ValueError, match="positive integer"):
            BatchingConfig(pad_to_multiple_of=invalid)
    with pytest.raises(ValueError):
        BatchingConfig(mode="auto")
    with pytest.raises(TypeError, match="boolean"):
        BatchingConfig(drop_last_train="false")
    with pytest.raises(TypeError, match="mapping"):
        parse_data_config({"batching": []})


def test_strict_collation_is_unchanged():
    samples = [_sample(4), _sample(4)]
    original = collate_latent_wam_samples(samples)
    batch = LatentBatchCollator(BatchingConfig())(samples)
    torch.testing.assert_close(batch.video_latents, original.video_latents)
    assert batch.action_mask is original.action_mask is None
    assert batch.metadata == original.metadata
    assert batch.sequence_lengths == ()
    assert batch.tensor_lengths == {}
    with pytest.raises(RuntimeError, match="stack"):
        LatentBatchCollator(BatchingConfig())([_sample(4), _sample(6)])


@pytest.mark.parametrize(
    "mode", [BatchingMode.PADDED, BatchingMode.PACKED, BatchingMode.BUCKET]
)
def test_padding_preserves_content_lengths_and_masks(mode):
    first, second = (
        _sample(4, text_length=3),
        _sample(6, chunk=3, window=8, text_length=5),
    )
    batch = LatentBatchCollator(BatchingConfig(mode=mode, pad_to_multiple_of=4))(
        [first, second]
    )
    assert batch.video_latents.shape == (2, 2, 8, 2, 2)
    assert batch.actions.shape == batch.action_mask.shape == (2, 16, 3)
    assert batch.sequence_lengths == (4, 6)
    assert batch.tensor_lengths["actions"] == (8, 12)
    assert batch.tensor_lengths["text_context"] == (3, 5)
    assert batch.tensor_lengths["condition_latents"] == (1, 1)
    assert batch.condition_latents.shape == (2, 2, 1, 2, 2)
    assert batch.state.shape == (2, 1, 3)
    for index, sample in enumerate((first, second)):
        length = sample.video_latents.shape[1]
        torch.testing.assert_close(
            batch.video_latents[index, :, :length], sample.video_latents
        )
        assert torch.count_nonzero(batch.video_latents[index, :, length:]) == 0
        assert torch.all(batch.action_mask[index, : length * 2] == 1)
        assert torch.count_nonzero(batch.action_mask[index, length * 2 :]) == 0
    assert batch.metadata[1]["sampled_chunk_size"] == 3
    assert batch.metadata[1]["sampled_window_size"] == 8
    assert batch.metadata[1]["history_frames"] == second.metadata["history_frames"]
    assert "batching_original_geometry" not in batch.metadata[1]
    assert batch.metadata[1]["latent_loss_frame_end"] == 6
    assert batch.metadata[1]["frame_shift"] == 19
    assert second.metadata["sampled_chunk_size"] == 3  # no caller mutation
    moved = move_latent_wam_batch_to_device(batch, "cpu")
    assert moved.batching_mode is mode
    assert moved.sequence_lengths == batch.sequence_lengths
    assert moved.tensor_lengths == batch.tensor_lengths


def test_existing_invalid_action_tail_is_preserved():
    sample = _sample(4)
    mask = torch.ones_like(sample.actions)
    mask[-2:] = 0
    sample = replace(sample, action_mask=mask)
    batch = LatentBatchCollator(BatchingConfig(mode="padded"))([sample, _sample(6)])
    assert torch.count_nonzero(batch.action_mask[0, 6:]) == 0
    assert batch.metadata[0]["valid_action_steps"] == 6


def test_collation_rejects_incompatible_fields():
    collate = LatentBatchCollator(BatchingConfig(mode="padded"))
    mixed = collate([_sample(4), replace(_sample(6), condition_latents=None)])
    assert mixed.tensor_lengths["condition_latents"] == (1, None)
    rgb = collate([replace(_sample(4), canonical_video=torch.zeros(3, 4, 2, 2))])
    assert rgb.tensor_lengths["canonical_video"] == (4,)
    # Transport does not impose a consumer's alignment or geometry rules.
    batch = collate([_sample(4), replace(_sample(6), actions=torch.ones(13, 3), metadata={})])
    assert batch.tensor_lengths["actions"] == (8, 13)


class _EpochSampler(Sampler[int]):
    def __init__(self, indices):
        self.indices = indices
        self.epoch = 0

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices[self.epoch :] + self.indices[: self.epoch])

    def set_epoch(self, epoch):
        self.epoch = epoch


def test_bucket_preserves_replacement_multiset_and_drops_original_tail():
    original = [4, 0, 1, 1, 2, 3, 0]
    sampler = _EpochSampler(original)
    bucket = LengthBucketSampler(
        sampler,
        length_for_index=lambda index: [100, 2, 9, 8, 7][index],
        batch_size=2,
        pool_size=5,
        drop_last=True,
    )
    ordered = list(bucket)
    assert len(bucket) == 6
    assert Counter(ordered) == Counter(original[:6])
    assert ordered == [1, 1, 4, 0, 3, 2]
    # The longest draw (index 0) is retained; only original tail is dropped.
    assert ordered.count(0) == 1
    assert list(bucket) == ordered
    bucket.set_epoch(1)
    expected = list(sampler)[:6]
    assert Counter(bucket) == Counter(expected)


def test_bucket_keeps_validation_tail_and_fixed_rank_counts():
    for rank in range(3):
        source = _EpochSampler([(rank + i) % 5 for i in range(7)])
        bucket = LengthBucketSampler(
            source,
            length_for_index=lambda i: i + 1,
            batch_size=3,
            pool_size=4,
            drop_last=False,
        )
        assert len(list(bucket)) == len(bucket) == 7
        assert Counter(bucket) == Counter(source)
        # Cursor continuation replays the same epoch ordering and skips batches.
        batches = [
            list(bucket)[start : start + 3] for start in range(0, len(bucket), 3)
        ]
        assert batches[1:] == [
            list(bucket)[start : start + 3] for start in range(3, len(bucket), 3)
        ]


def test_variable_loader_drop_tail_before_bucketing_and_retain_val_tail():
    class Dataset:
        def __len__(self):
            return 5

        def __getitem__(self, index):
            return _sample([9, 4, 7, 6, 3][index])

        def batching_length_hint(self, index):
            return [9, 4, 7, 6, 3][index]

    dataset = Dataset()
    config = SimpleNamespace(
        data=GenericDataConfig(
            batching=BatchingConfig(mode="bucket", bucket_pool_size=8)
        )
    )
    sampler = _EpochSampler(list(range(5)))
    train = _build_variable_length_loader(
        config, dataset, sampler, batch_size=2, shuffle=False, train=True
    )
    val = _build_variable_length_loader(
        config, dataset, sampler, batch_size=2, shuffle=False, train=False
    )
    assert [len(batch.sequence_lengths) for batch in train] == [2, 2]
    assert [len(batch.sequence_lengths) for batch in val] == [2, 2, 1]
    assert train.sampler is not sampler
    train.sampler.set_epoch(1)
    assert sampler.epoch == 1


def test_non_strict_loader_accepts_latent_contract_not_dataset_name():
    data = GenericDataConfig(batching={"mode": "padded"})
    config = SimpleNamespace(
        data=data, trainer=SimpleNamespace(batch_adapter=BatchAdapterName.LATENTS)
    )
    validate_variable_batching_source(config)
    config.trainer = SimpleNamespace(batch_adapter=BatchAdapterName.VIEWS)
    with pytest.raises(ValueError, match="batch_adapter=latents"):
        validate_variable_batching_source(config)


def test_bucket_preserves_distributed_replacement_epochs_and_rng():
    dataset = list(range(31))
    rank_lengths = []
    for rank in range(4):
        source = WeightedReplacementDistributedSampler(
            dataset,
            weights=[float(index + 1) for index in dataset],
            base_seed=71,
            world_size=4,
            rank=rank,
        )
        sampler = LengthBucketSampler(
            source,
            length_for_index=lambda index: 100 - index,
            batch_size=3,
            pool_size=7,
            drop_last=True,
        )
        sampler.set_epoch(2)
        expected = list(source)[: len(sampler)]
        python_state, torch_state = random.getstate(), torch.random.get_rng_state()
        actual = list(sampler)
        assert Counter(actual) == Counter(expected)
        assert random.getstate() == python_state
        assert torch.equal(torch.random.get_rng_state(), torch_state)
        # Reconstructing the same epoch after a saved loader cursor is stable.
        sampler.set_epoch(2)
        assert list(sampler)[3:] == actual[3:]
        rank_lengths.append(len(actual))
    assert rank_lengths == [6, 6, 6, 6]


def test_order_sensitive_sampler_cannot_be_bucketed():
    sampler = _EpochSampler([0, 1])
    sampler.supports_reordering = False
    config = SimpleNamespace(data=GenericDataConfig(batching={"mode": "bucket"}))
    with pytest.raises(ValueError, match="cannot reorder"):
        _build_variable_length_loader(
            config, [_sample(3), _sample(4)], sampler,
            batch_size=2, shuffle=False, train=True,
        )


def test_transport_does_not_consume_rng_or_modify_geometry_and_loss_boundaries():
    samples = [_sample(4), _sample(8, chunk=4, window=16)]
    python_state, torch_state = random.getstate(), torch.random.get_rng_state()
    batch = LatentBatchCollator(BatchingConfig(mode="padded"))(samples)
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    for sample, metadata in zip(samples, batch.metadata, strict=True):
        for key, value in sample.metadata.items():
            assert metadata[key] == value
        assert (
            metadata["latent_loss_frame_start"]
            == sample.metadata["latent_loss_frame_start"]
        )
        assert (
            metadata["latent_loss_frame_end"]
            == sample.metadata["latent_loss_frame_end"]
        )
