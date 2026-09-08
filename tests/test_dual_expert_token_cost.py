"""Exact admission counts use materialized shapes, never sampling estimates."""

import ast
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import get_type_hints

import pytest
import torch

from open_wam.configs import (
    ActionSchemaConfig,
    BatchingConfig,
    BatchingMode,
    DualExpertActionDecoderConfig,
    DualExpertPolicyConfig,
    DynamicsRouteConfig,
    DynamicsRoutingConfig,
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
from open_wam.data.latent_contracts import LatentWAMSample
from open_wam.models.policy_variants.dual_expert.token_cost import (
    LatentTokenAdmissionBatch,
    dual_expert_token_costs,
)
from open_wam.models.policy_variants.dual_expert.packed_block import DualExpertPackedBlock
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training.step_executor import LatentBatchAdapter, PipelineTrainStepExecutor


def _config(*, program=VideoActionProgram.JOINT, legacy=False):
    config = ExperimentConfig()
    return replace(
        config,
        data=replace(config.data, batching=BatchingConfig(mode=BatchingMode.PACKED)),
        backbone=replace(config.backbone, latent_channels=2),
        policy_variant=DualExpertPolicyConfig(
            program=program,
            sequence_contract=(
                VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
                if legacy
                else VideoActionSequenceContract.DEFAULT
            ),
        ),
    )


def _batch(*, lengths=(3, 5), spatial=(4, 6), alignment=1):
    samples = [
        LatentWAMSample(
            video_latents=torch.zeros(2, frames, *spatial),
            actions=torch.zeros(frames * 4, 3),
            action_mask=torch.zeros(frames * 4, 3),
            condition_latents=torch.zeros(2, 1, *spatial),
            metadata={
                "sampled_chunk_size": 1,
                "sampled_window_size": 4,
                "history_frames": 1,
                # These are deliberately NOT the materialized temporal extent.
                "segment_valid_latent_frames": 1,
                "batching_length_hint": 10000,
            },
        )
        for frames in lengths
    ]
    return LatentBatchCollator(
        BatchingConfig(mode=BatchingMode.PACKED, pad_to_multiple_of=alignment)
    )(samples)


@dataclass(frozen=True)
class _IndependentAdmissionBatch:
    """A producer with no data-batch inheritance, sharing tensors without copies."""

    video_latents: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor | None
    state: torch.Tensor | None
    state_mask: torch.Tensor | None
    text_context: torch.Tensor | None
    negative_text_context: torch.Tensor | None
    canonical_video: torch.Tensor | None
    condition_latents: torch.Tensor | None
    proprio_context_state: torch.Tensor | None
    proprio_context_state_mask: torch.Tensor | None
    proprio_context_frames: torch.Tensor | None
    proprio_context_frames_mask: torch.Tensor | None
    batching_mode: BatchingMode
    sequence_lengths: tuple[int, ...] | list[int]
    tensor_lengths: Mapping[str, tuple[int | None, ...] | list[int | None]]
    metadata: tuple[Mapping[str, object], ...] | list[Mapping[str, object]]


@pytest.mark.parametrize("program", [VideoActionProgram.JOINT, VideoActionProgram.VIDEO_THEN_ACTION])
def test_optional_absent_and_empty_context_preserves_self_token_count(program):
    batch = _batch()
    lengths = dict(batch.tensor_lengths)
    lengths.update(state=(0, None), text_context=(2, None), condition_latents=(1, None))
    mixed = replace(
        batch,
        state=torch.empty(2, 0, 3),
        text_context=torch.zeros(2, 2, 8),
        tensor_lengths=lengths,
    )
    config = _config(program=program)
    assert dual_expert_token_costs(config=config, batch=mixed) == dual_expert_token_costs(config=config, batch=batch)


def test_legacy_prefix_rejects_missing_condition_in_one_sample():
    batch = _batch()
    lengths = dict(batch.tensor_lengths, condition_latents=(1, None))
    with pytest.raises(ValueError, match="matching condition latents"):
        dual_expert_token_costs(config=_config(legacy=True), batch=replace(batch, tensor_lengths=lengths))


@pytest.mark.parametrize("extent", [-1, 1.5, True, 3])
def test_optional_context_rejects_invalid_original_extents(extent):
    batch = _batch()
    lengths = dict(batch.tensor_lengths, text_context=(extent, None))
    with pytest.raises(ValueError, match="Original text_context lengths"):
        dual_expert_token_costs(
            config=_config(),
            batch=replace(batch, text_context=torch.zeros(2, 2, 8), tensor_lengths=lengths),
        )


def _independent_admission_batch(batch):
    return _IndependentAdmissionBatch(
        **{field.name: getattr(batch, field.name) for field in fields(_IndependentAdmissionBatch)}
    )


def test_token_admission_protocol_declares_every_consumed_field_read_only():
    import open_wam.models.policy_variants.dual_expert.token_cost as token_cost

    properties = {
        name: value for name, value in vars(LatentTokenAdmissionBatch).items()
        if isinstance(value, property)
    }
    expected = get_type_hints(_IndependentAdmissionBatch)
    assert properties.keys() == expected.keys()
    for name, prop in properties.items():
        assert prop.fset is None
        assert get_type_hints(prop.fget) == {"return": expected[name]}
    assert get_type_hints(dual_expert_token_costs)["batch"] is LatentTokenAdmissionBatch

    # The structural interface must include direct accesses AND optional tensor
    # accesses performed through the counter's fixed time-axis table.
    tree = ast.parse(inspect.getsource(dual_expert_token_costs))
    direct_fields = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id == "batch"
    }
    assert direct_fields | set(token_cost._TENSOR_TIME_AXES) == properties.keys()


@pytest.mark.parametrize("program", [VideoActionProgram.VIDEO_THEN_ACTION, VideoActionProgram.JOINT])
@pytest.mark.parametrize("legacy", [False, True])
def test_token_admission_accepts_an_independent_typed_producer(program, legacy):
    batch = _batch(alignment=8)
    view: LatentTokenAdmissionBatch = _independent_admission_batch(batch)
    assert not isinstance(view, type(batch))
    for field in fields(_IndependentAdmissionBatch):
        assert getattr(view, field.name) is getattr(batch, field.name)
    config = _config(program=program, legacy=legacy)
    assert dual_expert_token_costs(config=config, batch=view) == dual_expert_token_costs(
        config=config, batch=batch
    )


@pytest.mark.parametrize("invalid", ["sequence", "extent", "cpu", "rgb", "geometry", "mode"])
def test_token_admission_checks_are_identical_for_independent_producers(invalid):
    batch = _batch()
    if invalid == "sequence":
        batch = replace(batch, sequence_lengths=(True, 5))
    elif invalid == "extent":
        batch = replace(batch, tensor_lengths={**batch.tensor_lengths, "actions": (12, 21)})
    elif invalid == "cpu":
        batch = replace(batch, actions=batch.actions.to("meta"))
    elif invalid == "rgb":
        batch = replace(batch, canonical_video=torch.zeros(2, 3, 5, 4, 6))
    elif invalid == "geometry":
        batch = replace(batch, metadata=({**batch.metadata[0], "sampled_chunk_size": 0}, batch.metadata[1]))
    else:
        batch = replace(batch, batching_mode=BatchingMode.STRICT)
    with pytest.raises(ValueError) as expected:
        dual_expert_token_costs(config=_config(), batch=batch)
    with pytest.raises(ValueError) as observed:
        dual_expert_token_costs(config=_config(), batch=_independent_admission_batch(batch))
    assert str(observed.value) == str(expected.value)


@pytest.mark.parametrize("program", [VideoActionProgram.VIDEO_THEN_ACTION, VideoActionProgram.JOINT])
@pytest.mark.parametrize("legacy", [False, True])
def test_counts_both_stream_copies_and_only_the_contract_video_prefix(program, legacy):
    # P=2*3=6, A=4*T; there is no extra action prefix in either program.
    result = dual_expert_token_costs(
        config=_config(program=program, legacy=legacy), batch=_batch()
    )
    assert result == tuple(2 * (t + int(legacy)) * 6 + 2 * (4 * t) for t in (3, 5))


@pytest.mark.parametrize("alignment", [1, 8, 16])
def test_ignores_transport_padding_but_counts_intrinsic_padding_and_masked_actions(alignment):
    batch = _batch(alignment=alignment)
    assert batch.video_latents.shape[2] >= 5
    assert not batch.action_mask.any()
    assert dual_expert_token_costs(config=_config(), batch=batch) == (60, 100)


def test_history_visibility_and_chunk_geometry_do_not_change_token_slots():
    config, batch = _config(), _batch()
    metadata = tuple(
        {**item, "sampled_chunk_size": 2, "sampled_window_size": 64, "history_frames": 2}
        for item in batch.metadata
    )
    updated = replace(config, policy_variant=replace(config.policy_variant, history_stream_visibility="full"))
    assert dual_expert_token_costs(config=updated, batch=replace(batch, metadata=metadata)) == (60, 100)


def test_uses_real_spatial_patch_geometry():
    config = _config()
    config = replace(config, backbone=replace(config.backbone, patch_size_h=1, patch_size_w=3))
    assert dual_expert_token_costs(config=config, batch=_batch()) == (72, 120)


@pytest.mark.parametrize("legacy, expected", [(False, 136000), (True, 136128)])
def test_full_1000_frame_20d_action_upper_bound(legacy, expected):
    batch = _batch(lengths=(1000,), spatial=(16, 16))
    assert dual_expert_token_costs(config=_config(legacy=legacy), batch=batch) == (expected,)


@pytest.mark.parametrize("bad_lengths", [(), (3,), (3, 6), (0, 5), (-1, 5), (True, 5), (3.0, 5)])
def test_rejects_unbounded_or_noninteger_sequence_lengths(bad_lengths):
    with pytest.raises(ValueError, match="length|frames"):
        dual_expert_token_costs(config=_config(), batch=replace(_batch(), sequence_lengths=bad_lengths))


@pytest.mark.parametrize("name, lengths", [
    ("video_latents", (4, 5)),
    ("actions", (12, 21)),
    ("actions", (11, 20)),
    ("actions", (True, 20)),
    ("condition_latents", (1, 2)),
    ("action_mask", (12, 21)),
])
def test_checks_each_original_tensor_extent_against_physical_capacity(name, lengths):
    batch = _batch()
    batch = replace(batch, tensor_lengths={**batch.tensor_lengths, name: lengths})
    with pytest.raises(ValueError, match="length|ratio"):
        dual_expert_token_costs(config=_config(), batch=batch)


@pytest.mark.parametrize("field", ["video_latents", "actions", "condition_latents"])
def test_rejects_missing_original_tensor_extents(field):
    batch = _batch()
    lengths = dict(batch.tensor_lengths)
    del lengths[field]
    with pytest.raises(ValueError, match="length"):
        dual_expert_token_costs(config=_config(), batch=replace(batch, tensor_lengths=lengths))


@pytest.mark.parametrize("mode", [BatchingMode.STRICT, BatchingMode.PADDED, BatchingMode.BUCKET])
def test_rejects_nonpacked_config_or_batch(mode):
    config, batch = _config(), _batch()
    with pytest.raises(ValueError, match="packed"):
        dual_expert_token_costs(config=config, batch=replace(batch, batching_mode=mode))
    with pytest.raises(ValueError, match="packed"):
        dual_expert_token_costs(
            config=replace(config, data=replace(config.data, batching=BatchingConfig(mode=mode))),
            batch=batch,
        )


@pytest.mark.parametrize("program", ["action_then_video", "generalist_joint_denoising"])
def test_rejects_other_programs(program):
    with pytest.raises(ValueError, match="VTA/Joint"):
        dual_expert_token_costs(config=_config(program=program), batch=_batch())


def test_rejects_other_policy_architectures():
    from open_wam.configs import ParallelStreamPolicyConfig

    config = replace(_config(), policy_variant=ParallelStreamPolicyConfig(program="joint"))
    with pytest.raises(ValueError, match="DualExpert"):
        dual_expert_token_costs(config=config, batch=_batch())


@pytest.mark.parametrize("key", [
    "generalist_training_mode_override",
    "generalist_training_source",
    "generalist_conditional_contract",
    "generalist_gjd_chunk_contract",
    "generalist_conditional_history_policy",
])
def test_rejects_routed_or_conditional_sample_metadata(key):
    batch = _batch()
    metadata = ({**batch.metadata[0], key: "unsupported"}, batch.metadata[1])
    with pytest.raises(ValueError, match="dynamics"):
        dual_expert_token_costs(config=_config(), batch=replace(batch, metadata=metadata))


def test_rejects_configured_routes():
    config = _config()
    routing = DynamicsRoutingConfig(routes=(DynamicsRouteConfig(source="real_demo", mode="joint", weight=1.0),))
    config = replace(config, data=replace(config.data, dynamics_routing=routing))
    with pytest.raises(ValueError, match="dynamics"):
        dual_expert_token_costs(config=config, batch=_batch())


def test_rejects_other_sequence_contracts():
    config = _config()
    config = replace(config, policy_variant=replace(config.policy_variant, sequence_contract="rollout_parity_single_frame_perchunk_proprio"))
    with pytest.raises(ValueError, match="prefix contract"):
        dual_expert_token_costs(config=config, batch=_batch())


@pytest.mark.parametrize("patch_t, patch_h, patch_w", [(2, 2, 2), (1, 3, 2)])
def test_rejects_temporal_patching_and_incompatible_spatial_shapes(patch_t, patch_h, patch_w):
    config = _config()
    config = replace(config, backbone=replace(config.backbone, patch_size_t=patch_t, patch_size_h=patch_h, patch_size_w=patch_w))
    with pytest.raises(ValueError, match="patch"):
        dual_expert_token_costs(config=config, batch=_batch())


def test_legacy_requires_a_real_compatible_condition_frame():
    config, batch = _config(legacy=True), _batch()
    for condition in (None, torch.zeros(2, 3, 1, 4, 6)):
        with pytest.raises(ValueError, match="condition latents"):
            dual_expert_token_costs(config=config, batch=replace(batch, condition_latents=condition))


def test_requires_cpu_admission_and_rejects_online_rgb():
    batch = _batch()
    with pytest.raises(ValueError, match="CPU"):
        dual_expert_token_costs(config=_config(), batch=replace(batch, actions=batch.actions.to("meta")))
    with pytest.raises(ValueError, match="RGB"):
        dual_expert_token_costs(config=_config(), batch=replace(batch, canonical_video=torch.zeros(2, 3, 5, 4, 6)))


@pytest.mark.parametrize("field", ["sampled_chunk_size", "sampled_window_size"])
@pytest.mark.parametrize("invalid", [None, False, 0, -1, 1.0, "1"])
def test_requires_materialized_positive_integer_geometry_before_subdivision(field, invalid):
    batch = _batch()
    row = dict(batch.metadata[0])
    if invalid is None:
        del row[field]
    else:
        row[field] = invalid
    with pytest.raises(ValueError, match="fallback geometry"):
        dual_expert_token_costs(
            config=_config(), batch=replace(batch, metadata=(row, batch.metadata[1]))
        )


@pytest.mark.parametrize("field", ["sampled_chunk_size", "sampled_window_size"])
def test_preserves_distinct_sample_geometry_without_changing_token_cost(field):
    batch = _batch()
    metadata = ({**batch.metadata[0], field: 2}, batch.metadata[1])
    changed = replace(batch, metadata=metadata)
    assert dual_expert_token_costs(config=_config(), batch=changed) == dual_expert_token_costs(
        config=_config(), batch=batch
    )
    assert changed.metadata == metadata


def test_rejects_chunk_larger_than_any_original_sequence():
    batch = _batch()
    metadata = tuple({**row, "sampled_chunk_size": 4} for row in batch.metadata)
    with pytest.raises(ValueError, match="chunk_size <= sample frames"):
        dual_expert_token_costs(config=_config(), batch=replace(batch, metadata=metadata))


def test_allows_different_history_lengths_with_fixed_chunk_window():
    batch = _batch()
    metadata = (batch.metadata[0], {**batch.metadata[1], "history_frames": 3})
    assert dual_expert_token_costs(
        config=_config(), batch=replace(batch, metadata=metadata)
    ) == (60, 100)


@pytest.mark.parametrize("program", [VideoActionProgram.VIDEO_THEN_ACTION, VideoActionProgram.JOINT])
def test_cost_matches_actual_packed_transformer_slots_with_proprio_and_text(program, monkeypatch):
    legacy = program is VideoActionProgram.VIDEO_THEN_ACTION
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            batching=BatchingConfig(mode=BatchingMode.PACKED),
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=1,
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
            num_action_layers=1,
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
            enabled_objectives=("action", "latent"),
        ),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    torch.manual_seed(17)
    samples = [
        LatentWAMSample(
            video_latents=torch.randn(48, frames, 4, 4),
            actions=torch.randn(2 * frames, 4),
            action_mask=torch.ones(2 * frames, 4),
            state=torch.randn(1, 4),
            state_mask=torch.ones(1, 4),
            condition_latents=torch.randn(48, 1 if legacy else frames, 4, 4),
            proprio_context_frames=torch.randn(frames, 4),
            proprio_context_frames_mask=torch.ones(frames, 4),
            text_context=torch.randn(text_tokens, 16),
            negative_text_context=torch.zeros(text_tokens, 16),
            metadata={
                "sampled_chunk_size": 2,
                "sampled_window_size": 8,
                "history_frames": 2,
                "action_tokens_per_frame": 2,
            },
        )
        for frames, text_tokens in ((4, 5), (7, 7))
    ]
    batch = LatentBatchCollator(config.data.batching)(samples)
    costs = dual_expert_token_costs(config=config, batch=batch)
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.policy_variant.initialize_for_training(pipeline.visual_tower)
    executor = PipelineTrainStepExecutor(
        pipeline=pipeline,
        batch_adapter=LatentBatchAdapter(),
        training_config=config.training,
    )
    observed = []
    original_forward = DualExpertPackedBlock.forward

    def capture(self, video_hidden, action_hidden, **kwargs):
        observed.append((video_hidden.shape[1], action_hidden.shape[1]))
        return original_forward(self, video_hidden, action_hidden, **kwargs)

    monkeypatch.setattr(DualExpertPackedBlock, "forward", capture)
    result = executor.forward_train(batch)

    assert torch.isfinite(result.loss)
    assert costs == tuple(2 * (frames + int(legacy)) * 4 + 4 * frames for frames in (4, 7))
    assert observed == [(sum(2 * (frames + int(legacy)) * 4 for frames in (4, 7)), 44)]
    assert sum(observed[0]) == sum(costs)
