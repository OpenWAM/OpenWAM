"""Regression coverage for pose block supervision."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import (
    ActionMappingConfig, ActionSchemaConfig, ActionTargetConfig, GenericDataConfig,
    InferenceConfig, SampleConstructionConfig, TrainingConfig,
)
from open_wam.configs.data_parsing import parse_data_config
from open_wam.data.action_pose import continuous_6d_to_rotation_matrix, pose_blocks_relative_to_anchor
from open_wam.data.action_transforms import pose_blocks_relative_to_anchor as facade_transform
from open_wam.data.lerobot_v2_latent_segment import LocalLatentSegmentAssembler
from open_wam.data.lerobot_v2_latent_storage import LocalEpisodeWindow
from open_wam.data.lerobot_v2_latent_supervision import LocalLatentSupervisionAssembler
from open_wam.data.lerobot_video import LeRobotV2VideoWindowDataset
from open_wam.models.action_decoders.dual_expert_decoder import (
    DualExpertActionDecoder, _masked_action_flow_match_loss, _masked_action_mse_by_group,
)
from open_wam.models.policy_variants.contracts import (
    DecoderArtifactEnvelope, PolicyTrainBatch, PolicyTrainOutput,
)
from open_wam.models.policy_variants.dual_expert.decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT, DualExpertActionTrainArtifacts, DualExpertTrainArtifacts,
)


def _pose(x: float, grip: float, rotation: torch.Tensor | None = None) -> torch.Tensor:
    rotation = torch.eye(3) if rotation is None else rotation
    return torch.cat((torch.tensor([x, 0.0, 0.0]), rotation[:, 0], rotation[:, 1], torch.tensor([grip])))


def _config(**target_overrides) -> GenericDataConfig:
    return GenericDataConfig(
        num_frames=4,
        action_schema=ActionSchemaConfig(action_dim=20, state_dim=20, action_horizon=8),
        action_target=ActionTargetConfig(**target_overrides),
        sample_construction=SampleConstructionConfig(chunk_size=2),
    )


def _rows(count=48):
    return [
        {
            "state": torch.cat((_pose(10 + i, i / 10), _pose(50 + 2 * i, i / 20))).tolist(),
            "actions": torch.cat((_pose(11 + i, 0.3), _pose(52 + 2 * i, 0.7))).tolist(),
            "frame_index": i,
        }
        for i in range(count)
    ]


def test_pose_config_fields_survive_typed_parse_and_roundtrip():
    values = dict(relative_pose_block_dims=10, proprio_history_lag=4,
                  proprio_history_lag_pose=2, proprio_history_lag_gripper=3)
    parsed = parse_data_config({"dataset_name": "custom", "action_target": values})
    reparsed = parse_data_config(asdict(parsed))
    for field, value in values.items():
        assert getattr(parsed.action_target, field) == value
        assert getattr(reparsed.action_target, field) == value
    defaults = parse_data_config({"dataset_name": "custom"}).action_target
    assert defaults.relative_pose_block_dims is None
    assert defaults.proprio_history_lag == 0
    assert defaults.proprio_history_lag_pose is defaults.proprio_history_lag_gripper is None


@pytest.mark.parametrize("field,value", [
    ("relative_pose_block_dims", 8), ("relative_pose_block_dims", 10.5),
    ("relative_pose_block_dims", True), ("proprio_history_lag", -1),
    ("proprio_history_lag", 1.5), ("proprio_history_lag", "1"),
    ("proprio_history_lag_pose", True), ("proprio_history_lag_gripper", -2),
])
def test_pose_config_rejects_lossy_or_invalid_integer_fields(field, value):
    with pytest.raises(ValueError, match=field):
        parse_data_config({"action_target": {field: value}})


def test_multiblock_relative_pose_rotates_translation_and_preserves_passthrough():
    rz = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    anchor = torch.cat((_pose(3, 0.0, rz), _pose(7, 0.0, rx)))
    actions = torch.stack((torch.cat((_pose(4, 0.2, rx), _pose(9, 0.8, rz))), anchor))
    original = actions.clone()
    result = pose_blocks_relative_to_anchor(actions, anchor=anchor, block_dims=10)
    assert facade_transform is pose_blocks_relative_to_anchor
    for start, rotation in ((0, rz), (10, rx)):
        expected_pos = torch.einsum("ij,tj->ti", rotation.T, actions[:, start:start + 3] - anchor[start:start + 3])
        expected_rotation = rotation.T @ continuous_6d_to_rotation_matrix(actions[:, start + 3:start + 9])
        torch.testing.assert_close(result[:, start:start + 3], expected_pos)
        torch.testing.assert_close(continuous_6d_to_rotation_matrix(result[:, start + 3:start + 9]), expected_rotation)
        assert torch.equal(result[:, start + 9], actions[:, start + 9])
    assert torch.equal(actions, original)


@pytest.mark.parametrize("actions,anchor,block,match", [
    (torch.zeros(1, 8), torch.zeros(8), 8, "at least 9"),
    (torch.zeros(20), torch.zeros(20), 10, "shape"),
    (torch.zeros(1, 19), torch.zeros(19), 10, "whole number"),
    (torch.zeros(1, 20), torch.zeros(10), 10, "must match"),
])
def test_relative_pose_rejects_invalid_layout(actions, anchor, block, match):
    with pytest.raises(ValueError, match=match):
        pose_blocks_relative_to_anchor(actions, anchor=anchor, block_dims=block)


@pytest.mark.parametrize("relative_block", [None, 10])
def test_distinct_pose_and_gripper_lags_keep_absolute_anchor(relative_block):
    assembler = LocalLatentSupervisionAssembler(_config(
        relative_pose_block_dims=relative_block, proprio_history_lag=9,
        proprio_history_lag_pose=2, proprio_history_lag_gripper=3,
    ))
    rows = _rows()
    state, mask = assembler.extract_state_at_frame(rows=rows, frame_index=5)
    absolute, _ = assembler.extract_state_at_frame(rows=rows, frame_index=5, apply_history_lag=False)
    torch.testing.assert_close(state[[0, 10]], torch.tensor([-2.0, -4.0]))
    torch.testing.assert_close(state[[9, 19]], torch.tensor([0.2, 0.1]))
    torch.testing.assert_close(absolute, torch.tensor(rows[5]["state"]))
    assert torch.equal(mask, torch.ones(20))
    early, _ = assembler.extract_state_at_frame(rows=rows, frame_index=1)
    torch.testing.assert_close(early[[0, 10]], torch.tensor([-1.0, -2.0]))
    assert torch.count_nonzero(early[[9, 19]]) == 0


def test_default_and_equal_lag_proprio_behaviors():
    rows = _rows()
    baseline = LocalLatentSupervisionAssembler(_config())
    state, _ = baseline.extract_state_at_frame(rows=rows, frame_index=5)
    assert torch.equal(state, torch.tensor(rows[5]["state"]))
    lagged = LocalLatentSupervisionAssembler(_config(proprio_history_lag=2))
    result, _ = lagged.extract_state_at_frame(rows=rows, frame_index=5)
    torch.testing.assert_close(result[[0, 10, 9, 19]], torch.tensor([-2.0, -4.0, 0.3, 0.15]))
    invalid = replace(_config(proprio_history_lag=2), action_schema=ActionSchemaConfig(7, 8, 7))
    with pytest.raises(ValueError, match="pose-block width"):
        LocalLatentSupervisionAssembler(invalid).extract_state_at_frame(
            rows=[{"state": [0.0] * 7}], frame_index=0
        )


@pytest.mark.parametrize("chunk_size,origin,leading", [(1, 0, 1), (2, 1, 1), (3, 2, 2)])
def test_relative_actions_use_each_chunks_absolute_anchor_even_with_lag(chunk_size, origin, leading):
    rows = _rows()
    observed = [4, 8, 12, 16, 20, 24]
    assembler = LocalLatentSupervisionAssembler(_config(relative_pose_block_dims=10, proprio_history_lag=2))
    actions, mask, metadata = assembler.build_lingbot_window_action_targets(
        rows=rows, window=LocalEpisodeWindow(Path("."), 0, 4, 30),
        observed_frame_ids=observed, latent_num_frames=6,
        leading_zero_action_frames=leading, leading_zero_action_mask=0.0,
        proprio_chunk_size=chunk_size, proprio_loss_frame_start=origin,
    )
    for step in range(leading * 2, 12):
        local_frame = step // 2
        chunk_index = max(0, (local_frame - origin) // chunk_size)
        anchor_index = observed[max(0, min(len(observed) - 1, origin + chunk_index * chunk_size - 1))]
        source = torch.tensor(rows[4 + step - leading * 2]["actions"])
        anchor = torch.tensor(rows[anchor_index]["state"])
        expected = pose_blocks_relative_to_anchor(source[None], anchor=anchor, block_dims=10)[0]
        torch.testing.assert_close(actions[step], expected)
    assert torch.count_nonzero(actions[:leading * 2]) == 0
    assert torch.count_nonzero(mask[:leading * 2]) == 0
    assert metadata["relative_pose_block_dims"] == 10


@pytest.mark.parametrize("compact", [False, True])
def test_segment_threads_identical_chunk_geometry_to_actions_and_proprio(compact):
    class RecordingAssembler(LocalLatentSupervisionAssembler):
        def build_lingbot_window_action_targets(self, **kwargs):
            self.action_geometry = (kwargs["proprio_chunk_size"], kwargs["proprio_loss_frame_start"])
            return super().build_lingbot_window_action_targets(**kwargs)

        def extract_proprio_context_state_sequence(self, **kwargs):
            if kwargs.get("apply_history_lag", True):
                self.context_geometry = (kwargs["chunk_size"], kwargs["loss_frame_start"])
            return super().extract_proprio_context_state_sequence(**kwargs)

    config = _config(relative_pose_block_dims=10, proprio_history_lag=2)
    owner = RecordingAssembler(config)
    segment = LocalLatentSegmentAssembler(config, supervision_assembler=owner).build(
        video_latents=torch.zeros(2, 12, 2, 2), condition_latents=None,
        rows=_rows(), raw_frame_ids=list(range(45)),
        window=LocalEpisodeWindow(Path("."), 0, 0, 45),
        latent_start=5, segment_length=5, start_padding_frames=0,
        compact_boundary_padding=compact, compact_boundary_chunk_size=3,
        compact_boundary_context_prefix_frames=1 if compact else 0,
        rollout_parity_target_alignment=compact,
    )
    assert owner.action_geometry == owner.context_geometry
    assert owner.action_geometry == (3 if compact else 2, segment.loss_frame_start)
    assert segment.action_target_metadata["relative_pose_block_dims"] == 10


def test_raw_video_relative_transform_precedes_sparse_mapping_and_keeps_default():
    dataset = object.__new__(LeRobotV2VideoWindowDataset)
    rows = _rows(3)
    dataset.data_config = _config()
    baseline, baseline_mask, metadata = dataset._build_action_targets(rows)
    assert torch.equal(baseline[:3], torch.tensor([row["actions"] for row in rows]))
    assert metadata["relative_pose_block_dims"] is None
    dataset.data_config = replace(
        _config(relative_pose_block_dims=10), action_schema=ActionSchemaConfig(22, 8, 20),
        action_mapping=ActionMappingConfig(mode="sparse_canvas", source_dim=20, target_dim=22,
                                          source_to_target_indices=tuple(range(1, 21))),
    )
    with pytest.raises(ValueError, match="anchor state row"):
        dataset._build_action_targets(rows)
    actual, mask, metadata = dataset._build_action_targets(rows, anchor_state_row=rows[1])
    expected = pose_blocks_relative_to_anchor(baseline, anchor=torch.tensor(rows[1]["state"]), block_dims=10)
    # Sparse mapping pins masked padding to its inactive value after rotation.
    torch.testing.assert_close(actual[:, 1:21], expected * baseline_mask)
    assert torch.equal(mask[:, 1:21], baseline_mask)
    assert torch.count_nonzero(mask[:, [0, 21]]) == 0
    assert metadata["relative_pose_block_dims"] == 10


def test_raw_video_sample_anchors_to_last_observed_state():
    dataset = object.__new__(LeRobotV2VideoWindowDataset)
    dataset.data_config = _config(relative_pose_block_dims=10)
    rows = _rows()
    dataset.sample_index = [SimpleNamespace(episode_index=0, observation_start=2)]
    dataset.metadata = SimpleNamespace(repo_root=Path("."), tasks_by_index={})
    dataset._load_episode_rows = lambda index: rows
    dataset._decode_video_sequence = lambda observations, **kwargs: torch.zeros(len(observations), 2, 2, 3)
    sample = dataset[0]
    anchor_index = 2 + dataset.data_config.num_frames - 1
    expected = pose_blocks_relative_to_anchor(
        torch.tensor([row["actions"] for row in rows[anchor_index:anchor_index + 8]]),
        anchor=torch.tensor(rows[anchor_index]["state"]), block_dims=10,
    )
    torch.testing.assert_close(sample.actions, expected)
    assert sample.metadata["anchor_frame_index"] == anchor_index


@pytest.mark.parametrize("dims", [10, 20])
@pytest.mark.parametrize("mask_kind", ["none", "full", "broadcast", "zero"])
def test_group_metrics_match_explicit_masked_channels(dims, mask_kind):
    predicted = torch.arange(1, 1 + 4 * dims, dtype=torch.float32).reshape(2, 2, dims)
    target = torch.zeros_like(predicted)
    mask = None if mask_kind == "none" else torch.ones_like(predicted)
    if mask_kind == "full":
        mask[0, :, 1::2] = 0
    elif mask_kind == "broadcast":
        mask = torch.tensor([[[1.0], [0.0]]])
    elif mask_kind == "zero":
        mask.zero_()
    actual = _masked_action_mse_by_group(action_pred=predicted, target_actions=target,
                                         action_mask=mask, action_dim=dims)
    expanded = torch.ones_like(predicted) if mask is None else mask.expand_as(predicted)
    for name, lo, hi in (("xyz", 0, 3), ("rot6", 3, 9), ("grip", 9, 10), ("pose", 0, 9)):
        columns = [b + c for b in range(0, dims, 10) for c in range(lo, hi)]
        expected = (predicted.square() * expanded)[..., columns].sum() / expanded[..., columns].sum().clamp_min(1)
        torch.testing.assert_close(actual[f"action_mse_{name}"], expected)
    assert _masked_action_mse_by_group(action_pred=predicted[..., :7], target_actions=target[..., :7],
                                       action_mask=None, action_dim=7) == {}


def test_group_metrics_do_not_change_decoder_total_loss_or_gradients():
    training = TrainingConfig()
    decoder = DualExpertActionDecoder(hidden_size=8, action_dim=20, action_horizon=2,
                                      training_config=training, inference_config=InferenceConfig())
    flow = torch.linspace(-1, 1, 80).reshape(2, 2, 20).requires_grad_()
    denoised = (flow * 0.5).clone()
    target = torch.zeros_like(flow)
    mask = torch.ones_like(flow)
    mask[..., 9] = 0
    steps = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scheduler = SimpleNamespace(training_weight=lambda value: value + 1)
    artifacts = DualExpertTrainArtifacts(
        action=DualExpertActionTrainArtifacts(flow, target, steps, scheduler, denoised, mask),
        video=None, condition_mode="observed", program="video_then_action", history_frames=1,
    )
    output = decoder.forward_train(
        PolicyTrainOutput(policy_features=torch.empty(0), metrics={},
                          decoder_artifacts=DecoderArtifactEnvelope(DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT, artifacts)),
        PolicyTrainBatch(actions=target),
    )
    baseline = _masked_action_flow_match_loss(flow_pred=flow, targets=target, timesteps=steps,
                                             scheduler=scheduler, action_mask=mask, action_dim=20)
    baseline = baseline * training.objective_weight("action")
    torch.testing.assert_close(output.loss, baseline, rtol=0, atol=0)
    expected_gradient = torch.autograd.grad(baseline, flow, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(output.loss, flow)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    for key in ("xyz", "rot6", "grip", "pose"):
        assert f"action_mse_{key}" in output.metrics
        assert not output.metrics[f"action_mse_{key}"].requires_grad
