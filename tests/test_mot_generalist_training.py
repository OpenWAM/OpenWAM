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
    JointTimestepCoupling,
    MoTGeneralistTrainingMode,
    MoTRuntimeMode,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelSequenceContract,
    ProprioContextMode,
    PolicyVariantName,
)
from open_wam.configs.variant_semantics import GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY
from open_wam.configs.policy_variant import (
    MoTPolicyConfig,
    _coerce_mot_generalist_training_mode_probs,
)
from open_wam.models.common.flow_matching import (
    VideoFlowMatchTrainArtifacts,
    build_frame_aligned_action_flow_match_train_artifacts,
    build_video_flow_match_train_artifacts,
)
from open_wam.models.common.attention_profiles import build_chunked_text_context_cross_attention_mask
from open_wam.models.policy_variants.mot.variant import (
    _apply_mot_generalist_training_mode,
    _sample_mot_generalist_training_mode,
    _should_couple_mot_action_to_video_sigmas,
)
from open_wam.models.policy_variants.mot.runtime import build_mot_packed_coupling_attention_profile


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
    assert cfg.joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA


def test_generalist_sigma_coupling_is_explicitly_configurable() -> None:
    cfg = _make_mot_policy_config(
        current_block_coupling=CurrentBlockCoupling.JOINT,
        mot_generalist_training_mode_probs={"joint": 1.0},
    )
    assert _should_couple_mot_action_to_video_sigmas(cfg, CurrentBlockCoupling.JOINT) is True

    cfg = _make_mot_policy_config(
        current_block_coupling=CurrentBlockCoupling.JOINT,
        mot_generalist_training_mode_probs={"joint": 1.0},
        joint_timestep_coupling=JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    )
    assert _should_couple_mot_action_to_video_sigmas(cfg, CurrentBlockCoupling.JOINT) is True

    cfg = _make_mot_policy_config(
        current_block_coupling=CurrentBlockCoupling.JOINT,
        mot_generalist_training_mode_probs={"joint": 1.0},
        joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
    )
    assert _should_couple_mot_action_to_video_sigmas(cfg, CurrentBlockCoupling.JOINT) is False

    cfg = _make_mot_policy_config(
        current_block_coupling=CurrentBlockCoupling.JOINT,
        mot_generalist_training_mode_probs={"joint": 1.0},
        joint_timestep_coupling=JointTimestepCoupling.MATCH_INDEX,
    )
    assert _should_couple_mot_action_to_video_sigmas(cfg, CurrentBlockCoupling.JOINT) is False

    cfg = _make_mot_policy_config(current_block_coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP)
    assert _should_couple_mot_action_to_video_sigmas(cfg, CurrentBlockCoupling.DECOUPLED_SAME_STEP) is False


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


def test_m5_generalist_mode_text_token_requires_gjd_probs() -> None:
    cfg = _make_mot_policy_config(
        current_block_coupling=CurrentBlockCoupling.JOINT,
        mot_generalist_training_mode_probs={"joint": 1.0},
        generalist_mode_text_token=True,
    )
    assert cfg.generalist_mode_text_token is True

    with pytest.raises(ValueError, match="generalist_mode_text_token"):
        _make_mot_policy_config(
            current_block_coupling=CurrentBlockCoupling.JOINT,
            generalist_mode_text_token=True,
        )


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


def test_chunked_text_mask_keeps_mode_suffix_global() -> None:
    # Legacy text-mask layout: 3 task-text tokens, 2 deprecated chunk-local
    # proprio text tokens, 1 global mode token.
    mask = build_chunked_text_context_cross_attention_mask(
        query_chunk_ids=torch.tensor([0, 0, 1, 1]),
        batch_size=1,
        text_token_count=6,
        base_text_token_count=3,
        proprio_context_token_count=2,
        global_suffix_token_count=1,
        device=torch.device("cpu"),
    )[0]

    assert torch.all(mask[:, :3])
    assert torch.equal(mask[:, 3], torch.tensor([True, True, False, False]))
    assert torch.equal(mask[:, 4], torch.tensor([False, False, True, True]))
    assert torch.all(mask[:, 5])


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


def test_joint_mode_preserves_plain_joint_condition_slots() -> None:
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
    assert out_video is video_artifacts
    assert out_noisy_actions is noisy_actions
    assert out_clean_actions is clean_actions
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

    # Video branch keeps clean condition slots as history context.
    assert out_video is video_artifacts
    assert out_future_mask is future_loss_mask
    # A_noisy slot now holds the clean values.
    assert torch.equal(out_noisy_actions, clean_actions)
    # A_clean remains real clean context; visibility is controlled by masks.
    assert out_clean_actions is clean_actions
    # Action timesteps forced to 0.
    assert torch.all(out_noisy_ts == 0)
    # Action loss masked off.
    assert out_action_mask is not None
    assert torch.all(out_action_mask == 0)


def test_action_conditioned_video_uses_valid_mask_not_loss_mask_for_clean_action_conditioning() -> None:
    video_artifacts = _make_video_artifacts()
    noisy_actions = torch.randn(1, 4, 3)
    clean_actions = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
    noisy_slot_timesteps = torch.full((1, 4), 0.5)
    future_loss_mask = torch.ones(1, 1, 4, 1, 1)
    action_loss_mask = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]]
    )
    clean_action_condition_mask = torch.tensor(
        [[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]]
    )

    out = _apply_mot_generalist_training_mode(
        sampled_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_artifacts=video_artifacts,
        noisy_actions=noisy_actions,
        clean_actions=clean_actions,
        noisy_slot_timesteps=noisy_slot_timesteps,
        future_loss_mask=future_loss_mask,
        effective_action_mask=action_loss_mask,
        clean_action_condition_mask=clean_action_condition_mask,
    )

    out_noisy_actions = out[1]
    out_clean_actions = out[2]
    out_action_mask = out[5]
    assert torch.equal(out_noisy_actions, clean_actions * clean_action_condition_mask)
    assert out_clean_actions is clean_actions
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
    # V_clean remains available as clean history context.
    assert torch.equal(out_video.condition_latents, original_condition)
    # V_noisy timestep track is forced to 0; condition timesteps stay as supplied.
    assert torch.all(out_video.timesteps == 0)
    assert torch.equal(out_video.condition_timesteps, video_artifacts.condition_timesteps)
    # Future video loss mask zeroed.
    assert torch.all(out_future_mask == 0)
    # Action noisy slot remains active; clean actions remain available as past context.
    assert out_noisy_actions is noisy_actions
    assert out_clean_actions is clean_actions
    assert out_noisy_ts is noisy_slot_timesteps
    assert out_action_mask is effective_action_mask


def test_action_conditioned_video_with_no_clean_action_mask_uses_full_clean_actions() -> None:
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

    out_noisy_actions = out[1]
    out_action_mask = out[5]
    assert torch.equal(out_noisy_actions, clean_actions)
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
    *,
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA,
    action_hidden_size: int | None = None,
    generalist_mode_text_token: bool = False,
    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE,
):
    """Construct a tiny CPU pipeline pinned to one generalist mode."""

    from open_wam.configs import (
        ActionSchemaConfig,
        ExperimentConfig,
        InferenceConfig,
        MoTActionDecoderConfig,
        MoTActionExpertInitMode,
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
            action_hidden_size=action_hidden_size,
            action_expert_init_mode=(
                MoTActionExpertInitMode.VIDEO_WEIGHT_INTERPOLATE
                if action_hidden_size is not None
                else MoTActionExpertInitMode.VIDEO_WEIGHT_COPY
            ),
            mot_generalist_training_mode_probs=forced_probs,
            generalist_mode_text_token=generalist_mode_text_token,
            proprio_context_mode=proprio_context_mode,
            joint_timestep_coupling=joint_timestep_coupling,
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


def test_forced_joint_training_respects_timestep_coupling_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.mot.variant as mot_variant_module

    original_build_action_artifacts = mot_variant_module.build_frame_aligned_action_flow_match_train_artifacts
    saw_action_coupling_inputs: list[tuple[bool, bool, bool]] = []

    def spy_build_action_artifacts(*args, **kwargs):
        saw_action_coupling_inputs.append(
            (
                kwargs.get("frame_sigma_values") is not None,
                kwargs.get("frame_timestep_ids") is not None,
                kwargs.get("scheduler_override") is not None,
            )
        )
        return original_build_action_artifacts(*args, **kwargs)

    monkeypatch.setattr(
        mot_variant_module,
        "build_frame_aligned_action_flow_match_train_artifacts",
        spy_build_action_artifacts,
    )

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.JOINT,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.JOINT,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_INDEX,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.JOINT,
        joint_timestep_coupling=JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.JOINT,
        joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert saw_action_coupling_inputs == [
        (True, False, False),
        (False, True, False),
        (False, True, True),
        (False, False, False),
    ]


def test_m5_generalist_mode_token_is_appended_in_train_path() -> None:
    torch.manual_seed(0)
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.JOINT,
        generalist_mode_text_token=True,
    )

    assert pipeline.visual_tower.core.generalist_mode_context_encoder is not None
    output = pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert output.policy_output.aux["mot_generalist_training_mode"] == MoTGeneralistTrainingMode.JOINT.value
    assert output.policy_output.aux["mot_generalist_mode_text_token"] == MoTGeneralistTrainingMode.JOINT.value
    assert output.policy_output.aux["mot_generalist_mode_text_token_count"] == 1


def test_generalist_match_sigma_uses_video_clock_for_all_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.mot.variant as mot_variant_module

    original_build_action_artifacts = mot_variant_module.build_frame_aligned_action_flow_match_train_artifacts
    saw_action_coupling_inputs: list[tuple[bool, bool]] = []

    def spy_build_action_artifacts(*args, **kwargs):
        saw_action_coupling_inputs.append(
            (
                kwargs.get("frame_sigma_values") is not None,
                kwargs.get("frame_timestep_ids") is not None,
            )
        )
        return original_build_action_artifacts(*args, **kwargs)

    monkeypatch.setattr(
        mot_variant_module,
        "build_frame_aligned_action_flow_match_train_artifacts",
        spy_build_action_artifacts,
    )

    for mode in (
        MoTGeneralistTrainingMode.JOINT,
        MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    ):
        pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
            mode,
            joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
        )
        pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert saw_action_coupling_inputs == [(True, False), (True, False), (True, False)]


def test_forced_joint_preserves_configured_noisy_video_condition_prob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.mot.variant as mot_variant_module

    original_build_video_artifacts = mot_variant_module.build_video_flow_match_train_artifacts
    observed_probs: list[float] = []

    def spy_build_video_artifacts(*args, **kwargs):
        observed_probs.append(float(kwargs.get("noisy_condition_prob", 0.0)))
        return original_build_video_artifacts(*args, **kwargs)

    monkeypatch.setattr(
        mot_variant_module,
        "build_video_flow_match_train_artifacts",
        spy_build_video_artifacts,
    )

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.JOINT,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert observed_probs == [pytest.approx(0.5)]


def test_conditional_generalist_modes_force_clean_video_condition_prob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.mot.variant as mot_variant_module

    original_build_video_artifacts = mot_variant_module.build_video_flow_match_train_artifacts
    observed_probs: list[float] = []

    def spy_build_video_artifacts(*args, **kwargs):
        observed_probs.append(float(kwargs.get("noisy_condition_prob", 0.0)))
        return original_build_video_artifacts(*args, **kwargs)

    monkeypatch.setattr(
        mot_variant_module,
        "build_video_flow_match_train_artifacts",
        spy_build_video_artifacts,
    )

    for mode in (
        MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    ):
        pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(mode)
        pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert observed_probs == [pytest.approx(0.0), pytest.approx(0.0)]


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
    assert output.policy_output.aux["mot_generalist_text_dropped"] is False
    assert "mot_generalist/joint/action_denoised_mse_sum" in metrics
    assert "mot_generalist/joint/action_mse_sum" in metrics
    assert torch.equal(
        metrics["mot_generalist/joint/action_mse_sum"],
        metrics["mot_generalist/joint/action_denoised_mse_sum"],
    )
    assert metrics["weighted_action_diffusion_loss"].item() > 0.0
    assert metrics["weighted_video_diffusion_loss"].item() > 0.0
    assert output.policy_output.aux["sampled_window_size"] >= 4


def test_generalist_training_rejects_multi_sample_batches() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.JOINT
    )
    multi_batch = _dataclass_replace(batch, actions=batch.actions.repeat(2, 1, 1))

    with pytest.raises(ValueError, match="rank-local train_batch_size=1"):
        pipeline.forward_train_from_latents(
            video_latents.repeat(2, 1, 1, 1, 1),
            multi_batch,
            text_context=text_context.repeat(2, 1, 1),
        )


def test_m5_generalist_conditional_local_window_covers_full_previous_video_action_chunk() -> None:
    profile = build_mot_packed_coupling_attention_profile(
        num_video_frames=8,
        video_tokens_per_frame=1,
        num_action_frames=8,
        action_tokens_per_frame=1,
        chunk_size_frames=4,
        attention_window_size=3,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        device=torch.device("cpu"),
        build_dense_masks=True,
    )
    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    latent_tokens = 8
    action_tokens = 8
    current_video_noisy_frame4 = 4
    current_video_clean_frame4 = latent_tokens + 4
    current_action_noisy_frame4 = 2 * latent_tokens + 4
    previous_video_clean_frame0 = latent_tokens + 0
    previous_action_clean_frame0 = 2 * latent_tokens + action_tokens + 0
    current_action_clean_frame4 = 2 * latent_tokens + action_tokens + 4

    assert mask[current_action_noisy_frame4, previous_video_clean_frame0]
    assert mask[current_action_noisy_frame4, previous_action_clean_frame0]
    assert not mask[current_video_noisy_frame4, current_video_clean_frame4]
    assert not mask[current_action_noisy_frame4, current_action_clean_frame4]


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
    assert output.policy_output.aux["mot_generalist_text_dropped"] is True
    assert output.policy_output.aux["sampled_window_size"] == 3
    assert metrics["weighted_action_diffusion_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["weighted_video_diffusion_loss"].item() > 0.0


def test_forced_action_conditioned_video_drops_text_even_with_false_override() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO
    )
    batch.extra["metadata"] = {GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY: False}
    output = pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert output.policy_output.aux["mot_generalist_text_dropped"] is True


@pytest.mark.parametrize(
    ("metadata", "expected_text_dropped"),
    [
        (None, True),
        ({GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY: False}, True),
    ],
)
def test_forced_action_conditioned_video_threads_resolved_text_to_m5_runtime(
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, bool] | None,
    expected_text_dropped: bool,
) -> None:
    import open_wam.models.policy_variants.mot.variant as mot_variant_module
    from open_wam.models.policy_variants.mot.modules import MoTActionExpert

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO
    )
    if metadata is not None:
        batch.extra["metadata"] = metadata
    assert torch.count_nonzero(text_context) > 0

    action_pre_dit_contexts: list[torch.Tensor] = []
    packed_runtime_contexts: list[torch.Tensor] = []
    original_pre_dit = MoTActionExpert.pre_dit

    def spy_pre_dit(self, *args, **kwargs):
        action_pre_dit_contexts.append(kwargs["context"].detach().clone())
        return original_pre_dit(self, *args, **kwargs)

    def fake_forward_mot_packed_coupling_denoise(**kwargs):
        packed_runtime_contexts.append(kwargs["text_context"].detach().clone())
        return torch.zeros_like(kwargs["noisy_video_latents"]), torch.zeros_like(kwargs["packed_action_pre"].tokens)

    monkeypatch.setattr(MoTActionExpert, "pre_dit", spy_pre_dit)
    monkeypatch.setattr(
        mot_variant_module,
        "forward_mot_packed_coupling_denoise",
        fake_forward_mot_packed_coupling_denoise,
    )

    output = pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    expected_text = torch.zeros_like(text_context) if expected_text_dropped else text_context
    assert output.policy_output.aux["mot_generalist_text_dropped"] is expected_text_dropped
    assert len(action_pre_dit_contexts) == 1
    assert len(packed_runtime_contexts) == 1
    assert torch.equal(action_pre_dit_contexts[0], expected_text)
    assert torch.equal(packed_runtime_contexts[0], expected_text)


def test_m5_per_chunk_additive_proprio_threads_hidden_context_to_packed_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.mot.variant as mot_variant_module
    from open_wam.models.policy_variants.mot.modules import MoTActionExpert

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        MoTGeneralistTrainingMode.JOINT,
        action_hidden_size=16,
    )
    object.__setattr__(
        pipeline.policy_variant.config,
        "proprio_context_mode",
        ProprioContextMode.PER_CHUNK_ADDITIVE,
    )
    pipeline.policy_variant.attach_visual_tower(pipeline.visual_tower)
    batch.extra["proprio_context_state"] = torch.randn(1, 4, 4)
    batch.extra["proprio_context_state_mask"] = torch.ones(1, 4, 4)

    action_hidden_contexts: list[torch.Tensor | None] = []
    video_hidden_contexts: list[torch.Tensor | None] = []
    original_pre_dit = MoTActionExpert.pre_dit

    def spy_pre_dit(self, *args, **kwargs):
        hidden_context = kwargs.get("hidden_context")
        action_hidden_contexts.append(None if hidden_context is None else hidden_context.detach().clone())
        return original_pre_dit(self, *args, **kwargs)

    def fake_forward_mot_packed_coupling_denoise(**kwargs):
        video_hidden_context = kwargs.get("video_hidden_context")
        video_hidden_contexts.append(
            None if video_hidden_context is None else video_hidden_context.detach().clone()
        )
        return torch.zeros_like(kwargs["noisy_video_latents"]), torch.zeros_like(kwargs["packed_action_pre"].tokens)

    monkeypatch.setattr(MoTActionExpert, "pre_dit", spy_pre_dit)
    monkeypatch.setattr(
        mot_variant_module,
        "forward_mot_packed_coupling_denoise",
        fake_forward_mot_packed_coupling_denoise,
    )

    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert len(action_hidden_contexts) == 1
    assert action_hidden_contexts[0] is not None
    assert action_hidden_contexts[0].shape == (1, 8, 32)
    assert pipeline.policy_variant.action_expert.hidden_context_dim == 32
    assert pipeline.policy_variant.action_expert.hidden_size == 16
    assert len(video_hidden_contexts) == 1
    assert video_hidden_contexts[0] is not None
    assert video_hidden_contexts[0].shape == (1, 128, 32)


def test_m5_legacy_prefix_contract_prepends_video_only_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.mot.variant as mot_variant_module

    from open_wam.configs import (
        ActionSchemaConfig,
        ExperimentConfig,
        InferenceConfig,
        MoTActionDecoderConfig,
        MoTPolicyConfig as TopLevelMoTPolicyConfig,
        RobotWinDataConfig,
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
            parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
            history_stream_visibility=ParallelHistoryStreamVisibility.VIDEO_ONLY,
            use_condition_latents=True,
            require_condition_latents=True,
            noisy_video_condition_prob=0.0,
            joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
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
    video_latents = torch.randn(1, 48, 4, 8, 8)
    condition_latents = torch.full_like(video_latents, 3.0)
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 4, 4),
        state=torch.randn(1, 4),
        extra={
            "condition_latents": condition_latents,
            "proprio_context_frames": torch.randn(1, 4, 4),
            "proprio_context_frames_mask": torch.ones(1, 4, 4),
        },
    )
    observed: dict[str, object] = {}

    def fake_forward_mot_packed_coupling_denoise(**kwargs):
        observed["noisy_video_shape"] = tuple(kwargs["noisy_video_latents"].shape)
        observed["clean_video_shape"] = tuple(kwargs["clean_video_latents"].shape)
        observed["packed_action_shape"] = tuple(kwargs["packed_action_pre"].tokens.shape)
        observed["prefix_condition_frames"] = kwargs["attention_profile"].metadata["prefix_condition_frames"]
        observed["video_hidden_context"] = kwargs["video_hidden_context"]
        observed["frame_start"] = kwargs["frame_start"]
        return torch.zeros_like(kwargs["noisy_video_latents"]), torch.zeros_like(kwargs["packed_action_pre"].tokens)

    monkeypatch.setattr(
        mot_variant_module,
        "forward_mot_packed_coupling_denoise",
        fake_forward_mot_packed_coupling_denoise,
    )

    output = pipeline.forward_train_from_latents(
        video_latents,
        batch,
        text_context=torch.randn(1, 5, 16),
    )

    assert torch.isfinite(output.decoder_output.loss)
    assert observed["noisy_video_shape"] == (1, 48, 5, 8, 8)
    assert observed["clean_video_shape"] == (1, 48, 5, 8, 8)
    assert observed["packed_action_shape"] == (1, 8, 32)
    assert observed["prefix_condition_frames"] == 1
    assert observed["video_hidden_context"] is None
    assert observed["frame_start"] == -1
    assert output.policy_output.aux["video_condition_source"] == "condition_latents_prefix"
    assert output.decoder_output.aux["predicted_latents"].shape == (1, 48, 5, 8, 8)


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
    assert output.policy_output.aux["mot_generalist_text_dropped"] is True
    assert output.policy_output.aux["sampled_window_size"] == 3
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
