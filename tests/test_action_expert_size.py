"""Action-only size selection, exact parameter accounting and shape contracts."""

from pathlib import Path

import pytest
import torch
import yaml

from open_wam.configs import (
    DualExpertActionExpertInitMode,
    DualExpertActionExpertSize,
    DualExpertPolicyConfig,
    InferenceConfig,
    TrainingConfig,
    VideoActionProgram,
    load_experiment_config,
)
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.serialization import serialize_experiment_config
from open_wam.models.policy_variants.dual_expert.variant import DualExpertPolicyVariant
from open_wam.utils.config_overrides import apply_config_overrides


REPO_ROOT = Path(__file__).resolve().parents[1]


def _small(**overrides):
    values = dict(
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        action_expert_size="small_500m",
        action_expert_init_mode="video_weight_interpolate",
    )
    return DualExpertPolicyConfig(**(values | overrides))


def _backbone(**overrides):
    values = dict(
        hidden_size=3072, num_layers=30, num_heads=24, attention_head_dim=128,
        ffn_dim=14336, text_dim=4096, freq_dim=256,
    )
    return SharedVideoTransformerConfig(**(values | overrides))


def test_default_size_preserves_existing_explicit_topology():
    config = DualExpertPolicyConfig(
        program="video_then_action", action_hidden_size=2048, action_ffn_dim=8192
    )
    assert config.action_expert_size is DualExpertActionExpertSize.CONFIGURED
    assert (config.action_hidden_size, config.action_ffn_dim, config.num_action_layers) == (2048, 8192, 30)
    assert config.action_expert_init_mode is DualExpertActionExpertInitMode.VIDEO_WEIGHT_COPY


@pytest.mark.parametrize("mode", ["random", "video_weight_interpolate"])
def test_small_profile_materializes_width_without_dropping_layers(mode):
    config = _small(action_expert_init_mode=mode)
    assert config.action_expert_size is DualExpertActionExpertSize.SMALL_500M
    assert (config.action_hidden_size, config.action_ffn_dim, config.num_action_layers) == (576, 2048, 30)
    config.validate_backbone_config(_backbone())


@pytest.mark.parametrize("overrides", [
    {"action_hidden_size": 2048}, {"action_ffn_dim": 8192},
    {"num_action_layers": 15}, {"action_expert_init_mode": "video_weight_copy"},
    {"action_expert_size": "unknown"},
])
def test_small_profile_rejects_conflicting_topology_or_initialization(overrides):
    with pytest.raises(ValueError):
        _small(**overrides)


@pytest.mark.parametrize("overrides", [
    {"num_layers": 29}, {"num_layers": 40}, {"hidden_size": 2048},
    {"num_heads": 16}, {"attention_head_dim": 64},
])
def test_small_profile_rejects_unmeasured_backbone_geometry_before_allocation(overrides):
    with pytest.raises(ValueError, match="one action layer per video layer"):
        DualExpertPolicyVariant(
            _small(), _backbone(**overrides), TrainingConfig(), InferenceConfig(),
            action_dim=20, action_horizon=16,
        )


@pytest.mark.parametrize("action_dim,expected", [(20, 503254484), (7, 503239495)])
def test_real_meta_model_has_approximately_500m_parameters_and_shared_attention(action_dim, expected):
    # Meta allocation counts every real Parameter without loading billions of weights.
    with torch.device("meta"):
        variant = DualExpertPolicyVariant(
            _small(), _backbone(), TrainingConfig(), InferenceConfig(),
            action_dim=action_dim, action_horizon=16,
        )
    expert = variant.action_expert
    assert len(expert.blocks) == 30
    assert sum(p.numel() for p in expert.parameters()) == expected
    assert expert.hidden_context_proj.weight.shape == (576, 3072)
    for block in expert.blocks:
        assert block.attn1.to_q.weight.shape == (3072, 576)
        assert block.attn1.to_k.weight.shape == (3072, 576)
        assert block.attn1.to_v.weight.shape == (3072, 576)
        assert block.attn1.to_out[0].weight.shape == (576, 3072)
        assert block.attn1.heads == 24
        assert block.ffn.net[0].proj.weight.shape == (2048, 576)


def test_configured_wide_geometry_parameter_count_is_unchanged():
    with torch.device("meta"):
        variant = DualExpertPolicyVariant(
            DualExpertPolicyConfig(program="video_then_action", action_hidden_size=2048, action_ffn_dim=8192),
            _backbone(), TrainingConfig(), InferenceConfig(), action_dim=20, action_horizon=16,
        )
    assert sum(p.numel() for p in variant.action_expert.parameters()) == 2563098644


def _recipe():
    return load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_libero_video_then_action.yaml")


def _select_small(config):
    return apply_config_overrides(config, {
        "policy_variant.action_expert_size": "small_500m",
        "policy_variant.action_hidden_size": None,
        "policy_variant.action_ffn_dim": None,
        "policy_variant.action_expert_init_mode": "video_weight_interpolate",
    })


def test_real_recipe_override_and_checkpoint_yaml_roundtrip(tmp_path):
    original = _recipe()
    config = _select_small(original)
    assert config.backbone == original.backbone
    assert config.data == original.data
    assert config.training == original.training
    assert config.inference == original.inference
    assert config.policy_variant.num_action_layers == config.backbone.num_layers == 30
    payload = serialize_experiment_config(config)
    assert payload["policy_variant"]["action_expert_size"] == "small_500m"
    assert payload["policy_variant"]["action_hidden_size"] == 576
    path = tmp_path / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    reloaded = load_experiment_config(path)
    assert serialize_experiment_config(reloaded) == payload


def test_cli_backbone_override_cannot_break_layer_parity():
    config = _select_small(_recipe())
    with pytest.raises(ValueError, match="one action layer per video layer"):
        apply_config_overrides(config, {"backbone.num_layers": 29})


def test_selection_does_not_silently_discard_old_explicit_widths():
    with pytest.raises(ValueError, match="conflicting override"):
        apply_config_overrides(_recipe(), {"policy_variant.action_expert_size": "small_500m"})


def test_small_weights_fail_to_load_wider_expert_state():
    # The existing strict tensor-shape guard must not silently accept an old A expert.
    from open_wam.models.action_decoders.video_conditioned_expert import VideoConditionedActionExpert

    common = dict(action_dim=4, num_layers=1, num_heads=2, attention_head_dim=8,
                  ffn_dim=16, freq_dim=8, context_dim=16, hidden_context_dim=16)
    old = VideoConditionedActionExpert(hidden_size=16, **common)
    small = VideoConditionedActionExpert(hidden_size=8, **common)
    with pytest.raises(RuntimeError, match="size mismatch"):
        small.load_state_dict(old.state_dict(), strict=True)
