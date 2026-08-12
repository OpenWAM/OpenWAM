from __future__ import annotations

import torch
import pytest

from open_wam.configs import (
    ActionMappingConfig,
    ActionMappingLossMaskMode,
    ActionMappingSamplerMaskMode,
    ActionNormalizationConfig,
)
from open_wam.data.action_mapping import (
    apply_action_mapping,
    build_action_sampler_mask,
    inverse_action_mapping,
    validate_action_mapping_preflight,
)
from open_wam.data.action_transforms import (
    build_absolute_joint_position_targets,
    build_relative_pose_targets,
    denormalize_action_targets,
    denormalize_joint_positions,
    expected_joint_position_target_dim,
    expected_pose_target_dim,
    normalize_action_targets,
    reconstruct_absolute_pose_targets,
)
from open_wam.models.action_decoders import ActionDecoder


class _MaskProbeDecoder(ActionDecoder):
    def forward_train(self, policy_output, batch):
        raise NotImplementedError

    def forward_infer(self, policy_output, previous_state=None):
        raise NotImplementedError


def test_calvin_7d_sparse_30d_mapping_round_trips_active_channels() -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=7,
        target_dim=30,
        source_to_target_indices=(0, 1, 2, 3, 4, 5, 28),
        active_target_indices=(0, 1, 2, 3, 4, 5, 28),
        loss_mask_mode=ActionMappingLossMaskMode.ACTIVE_TARGET_INDICES,
        sampler_mask_mode=ActionMappingSamplerMaskMode.PIN_INACTIVE_CHANNELS,
    )
    source = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    source_mask = torch.ones_like(source)

    mapped = apply_action_mapping(source, source_mask, mapping, target_dim=30)

    assert mapped.actions.shape == (2, 30)
    assert torch.equal(mapped.actions[:, 0:6], source[:, 0:6])
    assert torch.equal(mapped.actions[:, 28], source[:, 6])
    assert mapped.actions[:, 6:28].abs().sum().item() == 0.0
    assert mapped.actions[:, 29].abs().sum().item() == 0.0
    assert mapped.action_mask[:, 0:6].sum().item() == 12.0
    assert mapped.action_mask[:, 28].sum().item() == 2.0
    assert mapped.action_mask[:, 6:28].sum().item() == 0.0
    assert mapped.action_mask[:, 29].sum().item() == 0.0
    assert torch.equal(inverse_action_mapping(mapped.actions, mapping), source)

    report = validate_action_mapping_preflight(mapping, action_schema_dim=30)
    assert report["source_dim"] == 7
    assert report["target_dim"] == 30
    assert report["inactive_channel_count"] == 23


def test_robotwin_16d_sparse_30d_mapping_uses_lingbot_channel_order() -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=16,
        target_dim=30,
        source_to_target_indices=(0, 1, 2, 3, 4, 5, 6, 28, 7, 8, 9, 10, 11, 12, 13, 29),
        active_target_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 28, 29),
        loss_mask_mode="active_target_indices",
    )
    source = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    source_mask = torch.ones_like(source)

    mapped = apply_action_mapping(source, source_mask, mapping, target_dim=30)

    assert torch.equal(mapped.actions[0, 0:7], source[0, 0:7])
    assert mapped.actions[0, 28].item() == source[0, 7].item()
    assert torch.equal(mapped.actions[0, 7:14], source[0, 8:15])
    assert mapped.actions[0, 29].item() == source[0, 15].item()
    assert torch.equal(inverse_action_mapping(mapped.actions, mapping), source)


def test_sparse_mapping_quantile_normalization_supports_source_quantiles() -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=2,
        target_dim=4,
        source_to_target_indices=(1, 3),
        active_target_indices=(1, 3),
        normalization=ActionNormalizationConfig(
            mode="quantiles",
            q01=(0.0, -1.0),
            q99=(2.0, 1.0),
            clip_min=-1.5,
            clip_max=1.5,
        ),
    )
    source = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    mapped = apply_action_mapping(source, torch.ones_like(source), mapping, target_dim=4)

    assert mapped.actions[0, 1].item() == 0.0
    assert mapped.actions[0, 3].item() == 1.0
    assert torch.allclose(inverse_action_mapping(mapped.actions, mapping), source)


def test_sparse_mapping_quantile_normalization_supports_target_quantiles() -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=2,
        target_dim=4,
        source_to_target_indices=(1, 3),
        active_target_indices=(1, 3),
        normalization=ActionNormalizationConfig(
            mode="quantiles",
            q01=(-1.0, 0.0, -1.0, -2.0),
            q99=(1.0, 2.0, 1.0, 2.0),
        ),
    )
    source = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    mapped = apply_action_mapping(source, torch.ones_like(source), mapping, target_dim=4)

    assert mapped.actions[0, 1].item() == 0.0
    assert mapped.actions[0, 3].item() == 0.0
    assert torch.allclose(inverse_action_mapping(mapped.actions, mapping), source)
    report = validate_action_mapping_preflight(mapping, action_schema_dim=4)
    assert report["target_dim"] == 4


def test_sparse_mapping_rejects_mismatched_normalization_stats_length() -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=2,
        target_dim=4,
        source_to_target_indices=(1, 3),
        active_target_indices=(1, 3),
        normalization=ActionNormalizationConfig(
            mode="gaussian",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        ),
    )

    with pytest.raises(ValueError, match="normalization stats length"):
        validate_action_mapping_preflight(mapping, action_schema_dim=4)
    with pytest.raises(ValueError, match="normalization stats length"):
        apply_action_mapping(torch.zeros(1, 2), torch.ones(1, 2), mapping, target_dim=4)


def test_sparse_mapping_rejects_normalized_same_dim_mapping() -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=4,
        target_dim=4,
        source_to_target_indices=(0, 1, 2, 3),
        active_target_indices=(0, 1, 2, 3),
        normalization=ActionNormalizationConfig(
            mode="quantiles",
            q01=(0.0, 0.0, 0.0, 0.0),
            q99=(1.0, 1.0, 1.0, 1.0),
        ),
    )

    with pytest.raises(ValueError, match="source_dim == target_dim"):
        validate_action_mapping_preflight(mapping, action_schema_dim=4)


def test_joint_limit_normalization_round_trips_active_channels() -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=2,
        target_dim=4,
        source_to_target_indices=(0, 2),
        active_target_indices=(0, 2),
        normalization=ActionNormalizationConfig(
            mode="joint_limits",
            lower=(-2.0, 0.0),
            upper=(2.0, 4.0),
            clip_min=-1.0,
            clip_max=1.0,
        ),
    )
    source = torch.tensor([[0.0, 4.0]], dtype=torch.float32)
    mapped = apply_action_mapping(source, torch.ones_like(source), mapping, target_dim=4)

    assert torch.equal(mapped.actions[0, [0, 2]], torch.tensor([0.0, 1.0]))
    assert torch.allclose(inverse_action_mapping(mapped.actions, mapping), source)


def test_gaussian_action_target_normalization_round_trips() -> None:
    normalization = ActionNormalizationConfig(
        mode="gaussian",
        mean=(1.0, -2.0, 0.5),
        std=(2.0, 4.0, 0.25),
    )
    actions = torch.tensor([[3.0, -6.0, 1.0], [1.0, 2.0, 0.0]], dtype=torch.float32)

    normalized = normalize_action_targets(actions, normalization=normalization)
    recovered = denormalize_action_targets(normalized, normalization=normalization)

    assert torch.allclose(normalized, torch.tensor([[1.0, -1.0, 2.0], [0.0, 1.0, -2.0]]))
    assert torch.allclose(recovered, actions)


def test_absolute_joint_position_targets_append_action_gripper_and_normalize() -> None:
    normalization = ActionNormalizationConfig(
        mode="joint_limits",
        lower=(-2.0, -1.0),
        upper=(2.0, 3.0),
    )
    joint_positions = torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=torch.float32)
    raw_actions = torch.tensor([[0.1, 0.9], [0.2, -0.8]], dtype=torch.float32)

    targets, mask, metadata = build_absolute_joint_position_targets(
        joint_positions,
        include_gripper=True,
        gripper_representation="action_command",
        raw_action_sequence=raw_actions,
        gripper_action_index=-1,
        normalization=normalization,
    )

    assert torch.allclose(targets, torch.tensor([[0.0, 0.0, 0.9], [1.0, -1.0, -0.8]]))
    assert torch.equal(mask, torch.ones_like(targets))
    assert metadata["action_target_family"] == "absolute_joint_position"
    assert torch.allclose(denormalize_joint_positions(targets[:, :2], normalization=normalization), joint_positions)


def test_absolute_joint_position_targets_can_append_measured_gripper_qpos() -> None:
    joint_positions = torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=torch.float32)
    gripper_positions = torch.tensor([[0.04, -0.04], [0.01, -0.01]], dtype=torch.float32)

    targets, mask, metadata = build_absolute_joint_position_targets(
        joint_positions,
        include_gripper=True,
        gripper_representation="first_channel",
        gripper_position_sequence=gripper_positions,
        normalization=ActionNormalizationConfig(mode="none"),
    )

    assert torch.allclose(targets, torch.tensor([[0.0, 1.0, 0.04], [2.0, -1.0, 0.01]]))
    assert torch.equal(mask, torch.ones_like(targets))
    assert metadata["gripper_representation"] == "first_channel"
    assert expected_joint_position_target_dim(
        joint_dim=2,
        include_gripper=True,
        gripper_representation="first_channel",
    ) == 3


def test_continuous_6d_pose_targets_round_trip_absolute_pose() -> None:
    state = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, -0.04],
            [0.1, -0.2, 0.3, 0.0, 0.0, 0.5, 0.03, -0.03],
        ],
        dtype=torch.float32,
    )
    raw_action = torch.tensor([[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    targets, mask, metadata = build_relative_pose_targets(
        state,
        state_encoding="eef_pos_axisangle_gripper_2d",
        rotation_representation="continuous_6d",
        include_gripper=True,
        gripper_representation="action_command",
        raw_action_sequence=raw_action,
        gripper_action_index=-1,
    )
    reconstructed = reconstruct_absolute_pose_targets(
        reference_position=state[0, 0:3],
        reference_quaternion=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
        relative_pose_targets=targets,
        rotation_representation="continuous_6d",
    )

    assert targets.shape == (2, 10)
    assert torch.equal(mask, torch.ones_like(targets))
    assert metadata["rotation_representation"] == "continuous_6d"
    assert expected_pose_target_dim(
        rotation_representation="continuous_6d",
        include_gripper=True,
        gripper_representation="action_command",
    ) == 10
    assert torch.allclose(reconstructed.position, state[:, 0:3], atol=1e-5)
    assert torch.allclose(reconstructed.gripper, raw_action[:, -1:], atol=1e-6)


def test_sparse_mapping_preserves_inactive_fill_and_builds_sampler_mask() -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=2,
        target_dim=4,
        source_to_target_indices=(0, 2),
        active_target_indices=(0, 2),
        inactive_value=-0.25,
        sampler_mask_mode=ActionMappingSamplerMaskMode.PIN_INACTIVE_CHANNELS,
    )
    source = torch.tensor([[1.0, 2.0], [9.0, 9.0]], dtype=torch.float32)
    source_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    mapped = apply_action_mapping(source, source_mask, mapping, target_dim=4)

    assert torch.equal(mapped.actions[0, [0, 2]], torch.tensor([1.0, 2.0]))
    assert torch.allclose(mapped.actions[0, [1, 3]], torch.full((2,), -0.25))
    assert torch.allclose(mapped.actions[1], torch.full((4,), -0.25))
    assert mapped.sampler_mask is not None
    assert torch.equal(
        mapped.sampler_mask,
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]),
    )
    assert torch.equal(
        build_action_sampler_mask(mapping, action_horizon=1, target_dim=4),
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
    )


def test_action_decoder_sampler_mask_pins_inactive_channels_during_inference() -> None:
    decoder = _MaskProbeDecoder()
    decoder.configure_action_sampler_mask(
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]),
        inactive_value=-0.5,
    )
    action_pred = decoder._apply_action_sampler_mask(torch.randn(1, 2, 4))

    assert torch.allclose(action_pred[:, :, [1, 3]], torch.full((1, 2, 2), -0.5))
