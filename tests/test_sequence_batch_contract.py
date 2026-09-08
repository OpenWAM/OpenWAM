"""Consumer-neutral batching contracts and pre-allocation validation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import (
    BatchingConfig,
    BatchingMode,
    load_experiment_config,
    validate_experiment_config_runtime_contract,
)
from open_wam.data.latent_batching import LatentBatchCollator
from open_wam.data.latent_contracts import LatentWAMSample
from open_wam.models.action_decoders import ActionDecoderTrainOutput
from open_wam.models.common.sequence_batch_attention import (
    build_sequence_batch_self_attention,
    build_sequence_batch_cross_attention,
    build_sequence_self_attention,
    combine_sequence_layouts,
)
from open_wam.models.policy_variants import PolicyPreparedInputs, PolicyTrainOutput
from open_wam.models.policy_variants.base import PolicyVariant
from open_wam.pipelines.variable_batching import forward_variable_latent_batch
from open_wam.training.step_executor import LatentBatchAdapter


@pytest.mark.parametrize(
    "mode", [BatchingMode.PADDED, BatchingMode.PACKED, BatchingMode.BUCKET]
)
def test_action_free_consumer_uses_shared_transport_and_loss_reduction(mode):
    samples = [
        LatentWAMSample(
            video_latents=torch.full((2, frames, 2, 2), value),
            actions=torch.empty(0, 0),
            metadata={"sampled_chunk_size": index + 1, "sampled_window_size": 4},
        )
        for index, (frames, value) in enumerate(((1, 1.0), (3, 2.0)))
    ]
    batch = LatentBatchCollator(BatchingConfig(mode=mode))(samples)
    assert batch.actions.shape == (2, 0, 0)
    adapter = LatentBatchAdapter()
    policy_batch = adapter.prepare(batch).policy_batch
    projection = torch.nn.Linear(2, 2, bias=False)
    calls = []

    def execute_batch(*, visual_tower, visual_outputs, prepared_inputs, batching_mode):
        del visual_tower
        calls.append(batching_mode)
        # One shared operation; this test consumer has no action stream.
        features = projection(
            torch.stack([visual.mean((0, 2, 3, 4)) for visual in visual_outputs])
        )
        return tuple(
            PolicyTrainOutput(features[index : index + 1, None], metrics={})
            for index in range(len(prepared_inputs))
        )

    consumer = SimpleNamespace(
        prepare_train_inputs=lambda visual, item: PolicyPreparedInputs(batch=item),
        forward_train_batch=execute_batch,
    )

    def decode(output, item):
        loss = output.policy_features.square().mean()
        return ActionDecoderTrainOutput(
            action_pred=item.actions, loss=loss, metrics={"loss": loss}
        )

    pipeline = SimpleNamespace(
        policy_variant=consumer,
        visual_tower=None,
        prepare_visual_outputs_from_latents=lambda video, **kwargs: video,
        resolve_train_decoder_output=decode,
    )
    result = forward_variable_latent_batch(
        pipeline, batch.video_latents, policy_batch, batching_mode=mode
    )
    assert calls == [mode]
    assert result.decoder_output.action_pred.shape == (2, 0, 0)
    torch.testing.assert_close(
        result.decoder_output.loss,
        torch.stack(
            [item.decoder_output.loss for item in result.sample_outputs]
        ).mean(),
    )
    result.decoder_output.loss.backward()
    assert torch.isfinite(projection.weight.grad).all()
    assert projection.weight.grad.abs().sum() > 0
    assert batch.metadata[1]["sampled_chunk_size"] == 2


def test_unimplemented_batch_hook_fails_instead_of_serial_full_forwards():
    with pytest.raises(NotImplementedError, match="does not support"):
        PolicyVariant.forward_train_batch(
            SimpleNamespace(), None, (), (), batching_mode=BatchingMode.PACKED
        )


@pytest.mark.parametrize(
    "mode", [BatchingMode.BUCKET, BatchingMode.PADDED, BatchingMode.PACKED]
)
@pytest.mark.parametrize("program", ["action_then_video", "joint", "video_then_action"])
def test_batching_capability_validation_precedes_model_loading(mode, program):
    config = load_experiment_config(
        f"configs/experiments/dual_expert_libero_{program}.yaml"
    )
    config = replace(
        config, data=replace(config.data, batching=BatchingConfig(mode=mode))
    )
    assert validate_experiment_config_runtime_contract(config) is config


def test_grouping_and_execution_are_independent_choices():
    assert BatchingMode.BUCKET.groups_by_length
    assert BatchingMode.BUCKET.execution_mode is BatchingMode.PADDED
    assert not BatchingMode.PACKED.groups_by_length
    assert BatchingMode.PACKED.execution_mode is BatchingMode.PACKED


def test_optional_conditioning_round_trips_without_synthesizing_presence():
    from open_wam.pipelines.variable_batching import split_latent_train_batch

    samples = [
        LatentWAMSample(video_latents=torch.ones(2, 3, 2, 2), actions=torch.empty(0, 0)),
        LatentWAMSample(
            video_latents=torch.ones(2, 5, 2, 2), actions=torch.empty(0, 0),
            state=torch.ones(1, 4), condition_latents=torch.ones(2, 1, 2, 2),
            text_context=torch.ones(3, 4), negative_text_context=torch.zeros(3, 4),
            canonical_video=torch.ones(3, 17, 4, 4),
        ),
    ]
    batch = LatentBatchCollator(BatchingConfig(mode="packed"))(samples)
    prepared = LatentBatchAdapter().prepare(batch)
    items = split_latent_train_batch(
        prepared.video_latents, prepared.policy_batch,
        text_context=prepared.text_context, negative_text_context=prepared.negative_text_context,
        canonical_video=prepared.canonical_video,
    )
    assert items[0].batch.state is None
    assert items[0].batch.extra["condition_latents"] is None
    assert items[0].text_context is None and items[0].canonical_video is None
    torch.testing.assert_close(items[1].batch.state, samples[1].state[None])
    torch.testing.assert_close(items[1].text_context, samples[1].text_context[None])
    torch.testing.assert_close(items[1].canonical_video, samples[1].canonical_video[None])


def test_registered_latent_source_needs_no_batching_whitelist(monkeypatch):
    from open_wam.data.registries import DATASET_ADAPTERS, register_latent_dataset_builder
    from open_wam.training.data_loading import build_runtime_dataloaders
    from tests.test_variable_batch_pipeline import _samples
    from open_wam.configs import VideoActionProgram

    monkeypatch.setattr(DATASET_ADAPTERS, "_entries", dict(DATASET_ADAPTERS._entries))
    samples = _samples(VideoActionProgram.ACTION_THEN_VIDEO)
    register_latent_dataset_builder("external_sequence_test", lambda config: (samples, samples))
    config = load_experiment_config("configs/experiments/dual_expert_libero_action_then_video.yaml")
    config = replace(config, data=replace(
        config.data, dataset_type="external_sequence_test", batching=BatchingConfig(mode="packed"),
        train_batch_size=2, val_batch_size=2, num_workers=0,
    ))
    validate_experiment_config_runtime_contract(config)
    train, val = build_runtime_dataloaders(
        config, SimpleNamespace(world_size=1, rank=0, distributed=False),
    )
    assert sorted(next(iter(train)).sequence_lengths) == [4, 7]
    assert next(iter(val)).sequence_lengths == (4, 7)


@pytest.mark.parametrize(
    "lengths,slots,context,masks",
    [
        ([2, 3], [2], [3, 3], [None, None]),
        ([3], [2], [3], [None]),
        ([2], [2], [0], [None]),
    ],
)
def test_cross_attention_rejects_inconsistent_sequence_extents(
    lengths, slots, context, masks
):
    with pytest.raises(ValueError, match="Cross-attention"):
        build_sequence_batch_cross_attention(
            lengths, slots, context, masks, device=torch.device("cpu")
        )


def test_common_layout_isolation_accepts_video_only_and_an_arbitrary_local_rule():
    from open_wam.models.common.packed_token_layout import (
        build_exact_conditioned_video_token_layout,
    )

    layouts = [
        build_exact_conditioned_video_token_layout(
            batch_size=1,
            latent_frames=frames,
            latent_height=2,
            latent_width=2,
            patch_size=(1, 2, 2),
            chunk_size=1,
            device=torch.device("cpu"),
            chunk_origin_frame=0,
            current_block_coupling="video_then_action",
        )
        for frames in (2, 4)
    ]
    lengths = [layout.token_count for layout in layouts]
    layout, _ = combine_sequence_layouts(layouts, [lengths], [[max(lengths)] * 2])
    mask, sparse = build_sequence_self_attention(
        layout,
        lambda q, k: layout.frame_id[k] <= layout.frame_id[q],
        device=torch.device("cpu"),
    )
    assert sparse is None
    for index in range(2):
        ids = torch.where(layout.seq_id == index)[0]
        assert not mask[ids][:, layout.seq_id != index].any()
        torch.testing.assert_close(
            mask[ids[:, None], ids[None]],
            layout.frame_id[ids][None] <= layout.frame_id[ids][:, None],
        )
    dummy = torch.where(layout.seq_id < 0)[0]
    assert (mask[dummy].sum(1) == 1).all()


def test_prepared_predicates_compose_without_method_metadata():
    from open_wam.models.common.attention_contracts import PreparedAttentionProfile, AttentionProfileSpec
    from open_wam.models.common.packed_token_layout import build_exact_conditioned_video_token_layout

    profiles = []
    for index, frames in enumerate((3, 5)):
        layout = build_exact_conditioned_video_token_layout(
            batch_size=1, latent_frames=frames, latent_height=1, latent_width=1,
            patch_size=(1, 1, 1), chunk_size=1, device=torch.device("cpu"),
            chunk_origin_frame=0, current_block_coupling="joint",
        )
        # Deliberately unrelated to the built-in chunked visibility law.
        rule = (lambda q, k: k <= q) if index == 0 else (lambda q, k: (q - k).abs() <= 2)
        profiles.append(PreparedAttentionProfile(
            spec=AttentionProfileSpec(name="external", family="external", backend="flex_or_sdpa"),
            token_layout=layout, self_attention_visibility=rule,
        ))
    lengths = [item.token_layout.token_count for item in profiles]
    mask, _, combined = build_sequence_batch_self_attention(
        profiles, [lengths], [[max(lengths)] * 2], device=torch.device("cpu"),
    )
    for index, profile in enumerate(profiles):
        ids = torch.where(combined.seq_id == index)[0]
        local = torch.arange(len(ids))
        assert torch.equal(mask[ids[:, None], ids[None]], profile.self_attention_visibility(local[:, None], local[None]))
        assert not mask[ids][:, combined.seq_id != index].any()
