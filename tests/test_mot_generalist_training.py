"""Tests for the M5 generalist joint-denoise variant (A1, strict PR #95 parity).

Covers:
- Config validation: opt-in dict requires JOINT coupling; rejects
  non-finite / negative probs; default opt-out keeps existing 6-mode path.
- Sampling helper: respects the categorical and degenerate-prob shortcuts.
- 4-piece kit application: ACTION_CONDITIONED_VIDEO and
  VIDEO_CONDITIONED_ACTION rewrite the right tensors; JOINT is a no-op.
- Variant integration: when generalist probs are None, the existing 6-mode
  path is unchanged; per-mode metrics show up only when the segment ran in
  generalist mode.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace as _dataclass_replace

import math

import pytest
import torch

from open_wam.configs import TrainingConfig
from open_wam.configs.enums import (
    AttachSite,
    CurrentBlockCoupling,
    MoTGeneralistTrainingMode,
    MoTRuntimeMode,
    PolicyVariantName,
)
from open_wam.configs.policy_variant import (
    MoTPolicyConfig,
    _coerce_mot_generalist_training_mode_probs,
)
from open_wam.models.common.flow_matching import (
    VideoFlowMatchTrainArtifacts,
    build_frame_aligned_action_flow_match_train_artifacts,
    build_video_flow_match_train_artifacts,
)
from open_wam.models.policy_variants.mot.variant import (
    _apply_mot_generalist_training_mode,
    _sample_mot_generalist_training_mode,
)


def _make_mot_policy_config(**overrides) -> MoTPolicyConfig:
    base = dict(
        name=PolicyVariantName.MOT,
        hidden_size=256,
        attach_site=AttachSite.POST_VISUAL_CORE,
    )
    base.update(overrides)
    return MoTPolicyConfig(**base)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_default_opt_out_keeps_existing_six_mode_path() -> None:
    cfg = _make_mot_policy_config()
    assert cfg.mot_generalist_training_mode_probs is None


def test_opt_in_dict_normalizes_and_keeps_joint_coupling() -> None:
    cfg = _make_mot_policy_config(
        current_block_coupling=CurrentBlockCoupling.JOINT,
        mot_generalist_training_mode_probs={
            "joint": 6.0,
            "action_conditioned_video": 2.0,
            "video_conditioned_action": 2.0,
        },
    )
    probs = cfg.mot_generalist_training_mode_probs
    assert probs is not None
    assert math.isclose(sum(probs.values()), 1.0)
    assert math.isclose(probs[MoTGeneralistTrainingMode.JOINT], 0.6)
    assert math.isclose(probs[MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO], 0.2)
    assert math.isclose(probs[MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION], 0.2)


def test_opt_in_without_explicit_joint_coupling_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"current_block_coupling"):
        _make_mot_policy_config(
            mot_generalist_training_mode_probs={"joint": 1.0},
        )

def test_opt_in_with_directional_coupling_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"current_block_coupling"):
        _make_mot_policy_config(
            current_block_coupling=CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
            mot_generalist_training_mode_probs={"joint": 1.0},
        )
    with pytest.raises(ValueError, match=r"current_block_coupling"):
        _make_mot_policy_config(
            current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
            mot_generalist_training_mode_probs={"joint": 1.0},
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        {"joint": float("nan")},
        {"joint": float("inf")},
        {"joint": -0.1},
        {"joint": True},
        {"joint": 0.0, "action_conditioned_video": 0.0, "video_conditioned_action": 0.0},
    ],
)
def test_invalid_probs_rejected(bad_value: dict) -> None:
    with pytest.raises(ValueError, match=r"mot_generalist_training_mode_probs"):
        _coerce_mot_generalist_training_mode_probs(bad_value)


def test_unknown_mode_key_rejected() -> None:
    with pytest.raises(ValueError):
        _coerce_mot_generalist_training_mode_probs({"not_a_mode": 1.0})


def test_existing_six_mode_yamls_are_not_disturbed() -> None:
    """Sanity: any of the 6 fixed couplings keeps loading without generalist probs."""

    for coupling in CurrentBlockCoupling:
        cfg = _make_mot_policy_config(current_block_coupling=coupling)
        assert cfg.mot_generalist_training_mode_probs is None
        assert cfg.current_block_coupling == coupling


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_respects_categorical_distribution() -> None:
    torch.manual_seed(0)
    probs = {
        MoTGeneralistTrainingMode.JOINT: 0.6,
        MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO: 0.2,
        MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION: 0.2,
    }
    counts: Counter[MoTGeneralistTrainingMode] = Counter()
    for _ in range(2000):
        mode = _sample_mot_generalist_training_mode(probs, device=torch.device("cpu"))
        counts[mode] += 1
    total = sum(counts.values())
    assert total == 2000
    # Wide tolerance — just confirm none of the modes is missing and the
    # ordering matches the expected weights.
    joint_freq = counts[MoTGeneralistTrainingMode.JOINT] / total
    acv_freq = counts[MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO] / total
    vca_freq = counts[MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION] / total
    assert 0.55 <= joint_freq <= 0.65
    assert 0.15 <= acv_freq <= 0.25
    assert 0.15 <= vca_freq <= 0.25


def test_sample_degenerate_to_single_mode() -> None:
    torch.manual_seed(0)
    probs = {
        MoTGeneralistTrainingMode.JOINT: 0.0,
        MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO: 1.0,
        MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION: 0.0,
    }
    for _ in range(50):
        assert (
            _sample_mot_generalist_training_mode(probs, device=torch.device("cpu"))
            == MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO
        )


def test_joint_generalist_can_share_video_action_sigma_values() -> None:
    torch.manual_seed(0)
    training_config = TrainingConfig(video_sigma_shift=3.0, action_sigma_shift=5.0)
    video_latents = torch.randn(2, 4, 3, 2, 2)
    actions = torch.randn(2, 6, 7)

    video_artifacts = build_video_flow_match_train_artifacts(
        video_latents,
        training_config=training_config,
        noisy_condition_prob=0.0,
    )
    video_sigma_values = video_artifacts.scheduler.sigma_for_timesteps(video_artifacts.timesteps)
    action_artifacts = build_frame_aligned_action_flow_match_train_artifacts(
        actions,
        None,
        training_config=training_config,
        num_frames=3,
        action_per_frame=2,
        frame_sigma_values=video_sigma_values,
    )

    action_sigma_values = action_artifacts.scheduler.sigma_for_timesteps(action_artifacts.frame_timesteps)
    assert torch.allclose(action_sigma_values, video_sigma_values, atol=2e-3, rtol=2e-3)


# ---------------------------------------------------------------------------
# 4-piece kit application
# ---------------------------------------------------------------------------


def _make_video_artifacts(*, B: int = 1, F: int = 4, H: int = 4, W: int = 4) -> VideoFlowMatchTrainArtifacts:
    torch.manual_seed(1)
    return VideoFlowMatchTrainArtifacts(
        timesteps=torch.full((B, F), 0.7),
        noisy_latents=torch.randn(B, 16, F, H, W),
        targets=torch.randn(B, 16, F, H, W),
        condition_latents=torch.randn(B, 16, F, H, W),
        condition_timesteps=torch.full((B, F), 0.05),
        scheduler=None,  # sched is irrelevant for the kit application logic.
    )


def test_joint_mode_zeros_condition_slots() -> None:
    video_artifacts = _make_video_artifacts()
    noisy_actions = torch.randn(1, 64, 7)
    clean_actions = torch.randn(1, 64, 7)
    noisy_slot_timesteps = torch.full((1, 64), 0.5)
    future_loss_mask = torch.ones(1, 1, 4, 1, 1)
    effective_action_mask = torch.ones_like(noisy_actions)

    out = _apply_mot_generalist_training_mode(
        sampled_mode=MoTGeneralistTrainingMode.JOINT,
        video_artifacts=video_artifacts,
        noisy_actions=noisy_actions,
        clean_actions=clean_actions,
        noisy_slot_timesteps=noisy_slot_timesteps,
        future_loss_mask=future_loss_mask,
        effective_action_mask=effective_action_mask,
    )

    (out_video, out_noisy_actions, out_clean_actions,
     out_noisy_ts, out_future_mask, out_action_mask) = out
    assert torch.equal(out_video.noisy_latents, video_artifacts.noisy_latents)
    assert torch.equal(out_video.timesteps, video_artifacts.timesteps)
    assert torch.all(out_video.condition_latents == 0)
    assert torch.all(out_video.condition_timesteps == 0)
    assert out_noisy_actions is noisy_actions
    assert torch.all(out_clean_actions == 0)
    assert out_noisy_ts is noisy_slot_timesteps
    assert out_future_mask is future_loss_mask
    assert out_action_mask is effective_action_mask


def test_action_conditioned_video_replaces_action_slots() -> None:
    video_artifacts = _make_video_artifacts()
    noisy_actions = torch.randn(1, 64, 7)
    clean_actions = torch.randn(1, 64, 7)
    noisy_slot_timesteps = torch.full((1, 64), 0.5)
    future_loss_mask = torch.ones(1, 1, 4, 1, 1)
    effective_action_mask = torch.ones_like(noisy_actions)

    out = _apply_mot_generalist_training_mode(
        sampled_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_artifacts=video_artifacts,
        noisy_actions=noisy_actions,
        clean_actions=clean_actions,
        noisy_slot_timesteps=noisy_slot_timesteps,
        future_loss_mask=future_loss_mask,
        effective_action_mask=effective_action_mask,
    )

    (out_video, out_noisy_actions, out_clean_actions,
     out_noisy_ts, out_future_mask, out_action_mask) = out

    # Video noisy slot remains active; unused clean condition slot is zeroed.
    assert torch.equal(out_video.noisy_latents, video_artifacts.noisy_latents)
    assert torch.equal(out_video.timesteps, video_artifacts.timesteps)
    assert torch.all(out_video.condition_latents == 0)
    assert torch.all(out_video.condition_timesteps == 0)
    assert out_future_mask is future_loss_mask
    # A_noisy slot now holds the clean values.
    assert torch.equal(out_noisy_actions, clean_actions)
    # A_clean condition slot zeroed.
    assert torch.all(out_clean_actions == 0)
    # Action timesteps forced to 0.
    assert torch.all(out_noisy_ts == 0)
    # Action loss masked off.
    assert out_action_mask is not None
    assert torch.all(out_action_mask == 0)


def test_action_conditioned_video_masks_clean_action_conditioning() -> None:
    video_artifacts = _make_video_artifacts()
    noisy_actions = torch.randn(1, 4, 3)
    clean_actions = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
    noisy_slot_timesteps = torch.full((1, 4), 0.5)
    future_loss_mask = torch.ones(1, 1, 4, 1, 1)
    effective_action_mask = torch.tensor(
        [[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]]
    )

    out = _apply_mot_generalist_training_mode(
        sampled_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_artifacts=video_artifacts,
        noisy_actions=noisy_actions,
        clean_actions=clean_actions,
        noisy_slot_timesteps=noisy_slot_timesteps,
        future_loss_mask=future_loss_mask,
        effective_action_mask=effective_action_mask,
    )

    out_noisy_actions = out[1]
    out_action_mask = out[5]
    assert torch.equal(out_noisy_actions, clean_actions * effective_action_mask)
    assert out_action_mask is not None
    assert torch.all(out_action_mask == 0)


def test_video_conditioned_action_replaces_video_slots() -> None:
    video_artifacts = _make_video_artifacts()
    original_condition = video_artifacts.condition_latents.clone()
    noisy_actions = torch.randn(1, 64, 7)
    clean_actions = torch.randn(1, 64, 7)
    noisy_slot_timesteps = torch.full((1, 64), 0.5)
    future_loss_mask = torch.ones(1, 1, 4, 1, 1)
    effective_action_mask = torch.ones_like(noisy_actions)

    out = _apply_mot_generalist_training_mode(
        sampled_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
        video_artifacts=video_artifacts,
        noisy_actions=noisy_actions,
        clean_actions=clean_actions,
        noisy_slot_timesteps=noisy_slot_timesteps,
        future_loss_mask=future_loss_mask,
        effective_action_mask=effective_action_mask,
    )

    (out_video, out_noisy_actions, out_clean_actions,
     out_noisy_ts, out_future_mask, out_action_mask) = out

    # V_noisy slot now holds clean condition values.
    assert torch.equal(out_video.noisy_latents, original_condition)
    # V_clean condition slot zeroed.
    assert torch.all(out_video.condition_latents == 0)
    # Both video timestep tracks forced to 0 (cancels noisy_video_condition_prob).
    assert torch.all(out_video.timesteps == 0)
    assert torch.all(out_video.condition_timesteps == 0)
    # Future video loss mask zeroed.
    assert torch.all(out_future_mask == 0)
    # Action noisy slot remains active; unused clean condition slot is zeroed.
    assert out_noisy_actions is noisy_actions
    assert torch.all(out_clean_actions == 0)
    assert out_noisy_ts is noisy_slot_timesteps
    assert out_action_mask is effective_action_mask


def test_action_conditioned_video_with_no_action_mask_starts_from_zeros() -> None:
    video_artifacts = _make_video_artifacts()
    noisy_actions = torch.randn(1, 64, 7)
    clean_actions = torch.randn(1, 64, 7)
    noisy_slot_timesteps = torch.full((1, 64), 0.5)
    future_loss_mask = torch.ones(1, 1, 4, 1, 1)

    out = _apply_mot_generalist_training_mode(
        sampled_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_artifacts=video_artifacts,
        noisy_actions=noisy_actions,
        clean_actions=clean_actions,
        noisy_slot_timesteps=noisy_slot_timesteps,
        future_loss_mask=future_loss_mask,
        effective_action_mask=None,
    )

    out_action_mask = out[5]
    assert out_action_mask is not None
    assert out_action_mask.shape == noisy_actions.shape
    assert torch.all(out_action_mask == 0)


# ---------------------------------------------------------------------------
# Forced-mode integration (end-to-end forward_train through the variant +
# the MoT decoder, with the categorical pinned to a single mode so we can
# pattern-match on the loss/active flags deterministically).
# ---------------------------------------------------------------------------


def _build_tiny_generalist_pipeline(
    forced_mode: MoTGeneralistTrainingMode,
):
    """Construct a tiny CPU pipeline pinned to one generalist mode."""

    from open_wam.configs import (
        ActionSchemaConfig,
        ExperimentConfig,
        InferenceConfig,
        MoTActionDecoderConfig,
        MoTPolicyConfig as TopLevelMoTPolicyConfig,
        MoTRuntimeMode,
        RobotWinDataConfig,
        TrainingConfig,
    )
    from open_wam.models.policy_variants.contracts import PolicyTrainBatch
    from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
    from open_wam.pipelines import build_variant_pipeline_from_config

    forced_probs = {mode: 0.0 for mode in MoTGeneralistTrainingMode}
    forced_probs[forced_mode] = 1.0

    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
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
        policy_variant=TopLevelMoTPolicyConfig(
            hidden_size=32,
            runtime_mode=MoTRuntimeMode.NON_JOINT_TWO_STREAM,
            current_block_coupling=CurrentBlockCoupling.JOINT,
            video_prefix_frames=1,
            num_action_layers=1,
            mot_generalist_training_mode_probs=forced_probs,
        ),
        action_decoder=MoTActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            enabled_objectives=("action", "latent"),
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
        ),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = PolicyTrainBatch(actions=torch.randn(1, 4, 4))
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.randn(1, 5, 16)
    return pipeline, batch, video_latents, text_context


def test_forced_joint_keeps_both_losses_active() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.JOINT
    )
    output = pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    metrics = output.decoder_output.metrics
    assert metrics["mot_generalist/joint/count"].item() == 1.0
    assert metrics["mot_generalist/action_conditioned_video/count"].item() == 0.0
    assert metrics["mot_generalist/video_conditioned_action/count"].item() == 0.0
    assert metrics["mot_generalist/action_loss_active"].item() == 1.0
    assert metrics["mot_generalist/latent_loss_active"].item() == 1.0
    assert "mot_generalist/joint/action_denoised_mse_sum" in metrics
    assert "mot_generalist/joint/action_mse_sum" in metrics
    assert torch.equal(
        metrics["mot_generalist/joint/action_mse_sum"],
        metrics["mot_generalist/joint/action_denoised_mse_sum"],
    )
    assert metrics["weighted_action_diffusion_loss"].item() > 0.0
    assert metrics["weighted_video_diffusion_loss"].item() > 0.0


def test_forced_action_conditioned_video_zeros_action_loss() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO
    )
    output = pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    metrics = output.decoder_output.metrics
    assert metrics["mot_generalist/action_conditioned_video/count"].item() == 1.0
    assert metrics["mot_generalist/joint/count"].item() == 0.0
    assert metrics["mot_generalist/video_conditioned_action/count"].item() == 0.0
    # Action loss is fully masked off; video loss carries the gradient.
    assert metrics["mot_generalist/action_loss_active"].item() == 0.0
    assert metrics["mot_generalist/latent_loss_active"].item() == 1.0
    assert metrics["weighted_action_diffusion_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["weighted_video_diffusion_loss"].item() > 0.0


def test_forced_video_conditioned_action_zeros_video_loss() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION
    )
    output = pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    metrics = output.decoder_output.metrics
    assert metrics["mot_generalist/video_conditioned_action/count"].item() == 1.0
    assert metrics["mot_generalist/joint/count"].item() == 0.0
    assert metrics["mot_generalist/action_conditioned_video/count"].item() == 0.0
    # Video loss is fully masked off; action loss carries the gradient.
    assert metrics["mot_generalist/latent_loss_active"].item() == 0.0
    assert metrics["mot_generalist/action_loss_active"].item() == 1.0
    assert metrics["weighted_video_diffusion_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["weighted_action_diffusion_loss"].item() > 0.0


def test_no_generalist_metrics_when_probs_unset() -> None:
    """Sanity: existing 6-mode path emits no mot_generalist/* metrics."""

    from open_wam.configs import (
        ActionSchemaConfig,
        ExperimentConfig,
        InferenceConfig,
        MoTActionDecoderConfig,
        MoTPolicyConfig as TopLevelMoTPolicyConfig,
        MoTRuntimeMode,
        RobotWinDataConfig,
        TrainingConfig,
    )
    from open_wam.models.policy_variants.contracts import PolicyTrainBatch
    from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
    from open_wam.pipelines import build_variant_pipeline_from_config

    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
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
        policy_variant=TopLevelMoTPolicyConfig(
            hidden_size=32,
            runtime_mode=MoTRuntimeMode.NON_JOINT_TWO_STREAM,
            current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
            video_prefix_frames=1,
            num_action_layers=1,
            # mot_generalist_training_mode_probs left as default None
        ),
        action_decoder=MoTActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            enabled_objectives=("action", "latent"),
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
        ),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = PolicyTrainBatch(actions=torch.randn(1, 4, 4))
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.randn(1, 5, 16)

    output = pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    metrics = output.decoder_output.metrics
    for key in metrics:
        assert not key.startswith("mot_generalist/"), (
            f"mot_generalist metrics should not appear when probs are unset, got {key}"
        )
    # And the aux key is None (not the string).
    assert output.policy_output.aux.get("mot_generalist_training_mode") is None
