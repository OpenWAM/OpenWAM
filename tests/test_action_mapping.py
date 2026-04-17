from __future__ import annotations

import torch

from open_wam.configs import (
    ActionMappingConfig,
    ActionMappingLossMaskMode,
    ActionMappingSamplerMaskMode,
    ActionNormalizationConfig,
    InferenceConfig,
    TrainingConfig,
)
from open_wam.data.action_mapping import (
    apply_action_mapping,
    build_action_sampler_mask,
    inverse_action_mapping,
    validate_action_mapping_preflight,
)
from open_wam.models.action_decoders import MLPActionDecoder
from open_wam.models.policy_variants import PolicyInferOutput, PolicyInferState


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
    decoder = MLPActionDecoder(
        hidden_size=8,
        action_dim=4,
        action_horizon=2,
        training_config=TrainingConfig(action_num_train_timesteps=8),
        inference_config=InferenceConfig(action_num_inference_steps=2),
    )
    decoder.configure_action_sampler_mask(
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]),
        inactive_value=-0.5,
    )
    output = decoder.forward_infer(
        PolicyInferOutput(
            policy_features=torch.zeros(1, 2, 8),
            next_state=PolicyInferState(),
        )
    )

    assert torch.allclose(output.action_pred[:, :, [1, 3]], torch.full((1, 2, 2), -0.5))
