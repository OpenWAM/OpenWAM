from __future__ import annotations

import numpy as np
import pytest
import torch

from open_wam.configs import JointDenoiseTrainingMode, MoTGeneralistTrainingMode
from open_wam.configs.variant_semantics import (
    coerce_probability_map,
    default_video_action_conditioning_mode_probs,
    probability_map_static_issues,
)
from open_wam.models.common.flow_matching import FlowMatchScheduler
from open_wam.models.common.flow_noise_plan import sample_coupled_timestep_values, sample_timestep_values
from open_wam.models.common.joint_conditioning import (
    generalist_joint_conditioning_window_size,
    resolve_generalist_joint_conditioning_semantics,
    sample_conditioning_mode,
)
from open_wam.models.common.modality_slots import force_clean_noisy_slot, zero_condition_slot, zero_loss_mask_like
from open_wam.models.common.rollout_startup import (
    build_strict_action_context_mask,
    require_strict_startup_generation_frame,
    resolve_strict_startup_plan,
    strict_startup_conditioning_frame_index,
)
from open_wam.models.common.rollout_history import build_executed_action_history_tensor
from open_wam.models.common.metric_rollups import add_joint_conditioning_mode_metrics
from open_wam.models.common.video_geometry import slice_token_grid_frames, video_token_grid_from_latent_shape
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.frontend import SharedVideoFrontend


@pytest.mark.unit
def test_shared_probability_helpers_cover_m1_and_m5_mode_enums() -> None:
    m1_probs = default_video_action_conditioning_mode_probs(JointDenoiseTrainingMode, generalist=True)
    m5_probs = coerce_probability_map(
        {
            "joint": 6,
            "action_conditioned_video": 2,
            "video_conditioned_action": 2,
        },
        enum_cls=MoTGeneralistTrainingMode,
        field_name="mot_generalist_training_mode_probs",
    )

    assert m1_probs[JointDenoiseTrainingMode.JOINT] == pytest.approx(0.6)
    assert m5_probs[MoTGeneralistTrainingMode.JOINT] == pytest.approx(0.6)
    assert sum(m5_probs.values()) == pytest.approx(1.0)

    issues = probability_map_static_issues(
        {"typo": 1.0, "joint": True},
        enum_cls=MoTGeneralistTrainingMode,
    )
    assert any("Invalid MoTGeneralistTrainingMode" in issue.message for issue in issues)
    assert any("numeric probability" in issue.message for issue in issues)

    with pytest.raises(ValueError, match="mot_generalist_training_mode_probs.*invalid mode"):
        coerce_probability_map(
            {"typo": 1.0},
            enum_cls=MoTGeneralistTrainingMode,
            field_name="mot_generalist_training_mode_probs",
        )


@pytest.mark.unit
def test_shared_conditioning_mode_sampling_broadcasts_rank_zero_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    def fail_if_rank_one_samples(*args, **kwargs):
        raise AssertionError("nonzero ranks must not independently sample GJD mode")

    def fake_broadcast(tensor: torch.Tensor, *, src: int) -> None:
        assert src == 0
        tensor.fill_(2)

    monkeypatch.setattr(torch, "multinomial", fail_if_rank_one_samples)
    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    mode = sample_conditioning_mode(
        {mode: 1.0 for mode in MoTGeneralistTrainingMode},
        enum_cls=MoTGeneralistTrainingMode,
        device=torch.device("cpu"),
        error_label="test mode",
    )

    assert mode == tuple(MoTGeneralistTrainingMode)[2]


@pytest.mark.unit
def test_shared_generalist_joint_conditioning_semantics_cover_m1_and_m5() -> None:
    m1_joint = resolve_generalist_joint_conditioning_semantics(
        JointDenoiseTrainingMode.JOINT,
        joint_mode=JointDenoiseTrainingMode.JOINT,
        action_conditioned_video_mode=JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
    )
    m5_joint = resolve_generalist_joint_conditioning_semantics(
        MoTGeneralistTrainingMode.JOINT,
        joint_mode=MoTGeneralistTrainingMode.JOINT,
        action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    )
    assert m1_joint == m5_joint
    assert m1_joint.force_clean_video_condition is False
    assert m1_joint.action_loss_active is True
    assert m1_joint.video_loss_active is True
    assert m1_joint.drop_text_conditioning is False

    m1_fdm = resolve_generalist_joint_conditioning_semantics(
        JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        joint_mode=JointDenoiseTrainingMode.JOINT,
        action_conditioned_video_mode=JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
    )
    m5_fdm = resolve_generalist_joint_conditioning_semantics(
        MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        joint_mode=MoTGeneralistTrainingMode.JOINT,
        action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    )
    assert m1_fdm == m5_fdm
    assert m1_fdm.clean_action_noisy_slot is True
    assert m1_fdm.action_loss_active is False
    assert m1_fdm.video_loss_active is True
    assert m1_fdm.drop_text_conditioning is True
    assert m1_fdm.force_clean_video_condition is True
    assert m1_fdm.attention_window_size(fallback_window_size=30) == 3

    m1_idm = resolve_generalist_joint_conditioning_semantics(
        JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
        joint_mode=JointDenoiseTrainingMode.JOINT,
        action_conditioned_video_mode=JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
    )
    assert m1_idm.clean_video_noisy_slot is True
    assert m1_idm.action_loss_active is True
    assert m1_idm.video_loss_active is False
    assert (
        generalist_joint_conditioning_window_size(
            JointDenoiseTrainingMode.JOINT,
            joint_mode=JointDenoiseTrainingMode.JOINT,
            action_conditioned_video_mode=JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
            video_conditioned_action_mode=JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
            fallback_window_size=30,
        )
        == 30
    )


@pytest.mark.unit
def test_shared_coupled_noise_plan_matches_sigmas_across_schedulers() -> None:
    torch.manual_seed(0)
    video_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
    action_scheduler = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=500)
    video_scheduler.set_timesteps(1000, training=True)
    action_scheduler.set_timesteps(500, training=True)

    coupled = sample_coupled_timestep_values(
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        num_frames=4,
        device=torch.device("cpu"),
    )

    assert coupled.video_timesteps.shape == (4,)
    assert coupled.action_timesteps.shape == (4,)
    assert torch.allclose(video_scheduler.sigma_for_timesteps(coupled.video_timesteps), coupled.sigma_values)
    assert torch.allclose(
        action_scheduler.sigma_for_timesteps(coupled.action_timesteps),
        coupled.sigma_values,
        atol=2e-3,
        rtol=0.0,
    )


@pytest.mark.unit
def test_shared_noise_plan_accepts_timestep_grid_scheduler_protocol() -> None:
    class TimestepGridOnlyScheduler:
        num_train_timesteps = 1000

        def __init__(self) -> None:
            self.timesteps = torch.tensor([4.0, 3.0, 2.0, 1.0])
            self.sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25])

    torch.manual_seed(0)
    scheduler = TimestepGridOnlyScheduler()

    timestep_values = sample_timestep_values(
        scheduler,
        num_frames=3,
        device=torch.device("cpu"),
    )
    coupled = sample_coupled_timestep_values(
        video_scheduler=scheduler,
        action_scheduler=scheduler,
        num_frames=3,
        device=torch.device("cpu"),
    )

    assert timestep_values.shape == (3,)
    assert coupled.video_timesteps.shape == (3,)
    assert torch.equal(coupled.video_timesteps, coupled.action_timesteps)


@pytest.mark.unit
def test_shared_noise_plan_rejects_mismatched_grid_lengths() -> None:
    class BadScheduler:
        num_train_timesteps = 1000
        timesteps = torch.tensor([4.0, 3.0])
        sigmas = torch.tensor([1.0])

    with pytest.raises(ValueError, match="matching lengths"):
        sample_timestep_values(
            BadScheduler(),
            num_frames=1,
            device=torch.device("cpu"),
        )


@pytest.mark.unit
def test_shared_modality_slot_helpers_preserve_conditional_semantics() -> None:
    clean = torch.arange(6, dtype=torch.float32).view(1, 2, 3)
    mask = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
    artifact = {
        "noisy_latents": torch.ones_like(clean),
        "targets": torch.ones_like(clean),
        "timesteps": torch.ones(1, 2),
        "latent": clean.clone(),
        "cond_timesteps": torch.ones(1, 2),
    }

    force_clean_noisy_slot(artifact, clean, action_mask=mask)

    assert torch.equal(artifact["noisy_latents"], clean * mask)
    assert torch.equal(artifact["targets"], torch.zeros_like(clean))
    assert torch.equal(artifact["timesteps"], torch.zeros(1, 2))

    zero_condition_slot(artifact)
    assert torch.equal(artifact["latent"], torch.zeros_like(clean))
    assert torch.equal(artifact["cond_timesteps"], torch.zeros(1, 2))
    assert torch.equal(zero_loss_mask_like(mask, fallback_like=clean), torch.zeros_like(mask))
    assert torch.equal(zero_loss_mask_like(None, fallback_like=clean), torch.zeros_like(clean))

    half_clean = clean.to(dtype=torch.float16)
    half_mask = mask.to(dtype=torch.float32)
    force_clean_noisy_slot(artifact, half_clean, action_mask=half_mask)
    assert artifact["noisy_latents"].dtype == torch.float16


@pytest.mark.unit
def test_shared_metric_rollup_matches_m1_m5_generalist_shape() -> None:
    metrics: dict[str, torch.Tensor] = {}
    action_loss = torch.tensor(2.0)
    latent_loss = torch.tensor(3.0)

    add_joint_conditioning_mode_metrics(
        metrics,
        namespace="joint_denoise",
        mode_value="action_conditioned_video",
        modes=JointDenoiseTrainingMode,
        action_loss=action_loss,
        latent_loss=latent_loss,
        action_loss_active=torch.tensor(0.0),
        latent_loss_active=torch.tensor(1.0),
        action_metric_name="action_flow_loss_sum",
        latent_metric_name="latent_flow_loss_sum",
        action_metric_aliases=("action_mse_sum",),
        latent_metric_aliases=("latent_mse_sum",),
    )

    assert metrics["joint_denoise/action_conditioned_video/count"].item() == 1.0
    assert metrics["joint_denoise/joint/count"].item() == 0.0
    assert metrics["joint_denoise/action_conditioned_video/action_flow_loss_sum"].item() == 2.0
    assert metrics["joint_denoise/action_conditioned_video/latent_flow_loss_sum"].item() == 3.0
    assert metrics["joint_denoise/action_conditioned_video/action_mse_sum"].item() == 2.0
    assert metrics["joint_denoise/action_conditioned_video/latent_mse_sum"].item() == 3.0
    assert metrics["joint_denoise/action_loss_active"].item() == 0.0
    assert metrics["joint_denoise/latent_loss_active"].item() == 1.0


@pytest.mark.unit
def test_shared_rollout_history_rejects_bootstrap_zero_actions() -> None:
    executed = [
        np.array([1.0, -1.0], dtype=np.float32),
        np.array([0.5, -0.5], dtype=np.float32),
    ]

    with pytest.raises(ValueError, match="deprecated"):
        build_executed_action_history_tensor(
            executed,
            start_frame_group=1,
            action_per_frame=2,
            action_dim=2,
        )

    history = build_executed_action_history_tensor(
        executed,
        start_frame_group=0,
        action_per_frame=2,
        action_dim=2,
    )
    assert history is not None
    assert torch.equal(history[0], torch.from_numpy(np.stack(executed, axis=0)))


@pytest.mark.unit
def test_shared_strict_startup_plan_matches_rollout_contract() -> None:
    startup = resolve_strict_startup_plan(
        step_index=0,
        current_start_frame=0,
        frame_chunk_size=4,
        action_tokens_per_frame=4,
        action_horizon=16,
    )

    assert startup.is_startup is True
    assert startup.video_prefix_frames == 1
    assert startup.generation_frame_start == 1
    assert startup.action_prefix_tokens == 4
    assert startup.current_action_sequence_tokens == 20
    assert startup.chunk_origin_frame(history_frames=8) == 9

    next_chunk = resolve_strict_startup_plan(
        step_index=1,
        current_start_frame=5,
        frame_chunk_size=4,
        action_tokens_per_frame=4,
        action_horizon=16,
    )

    assert next_chunk.is_startup is False
    assert next_chunk.video_prefix_frames == 0
    assert next_chunk.generation_frame_start == 5
    assert next_chunk.action_prefix_tokens == 0
    assert next_chunk.current_action_sequence_tokens == 16
    assert next_chunk.chunk_origin_frame(history_frames=8) == 8


@pytest.mark.unit
def test_shared_strict_action_context_mask_hides_only_startup_prefix() -> None:
    mask = build_strict_action_context_mask(
        batch_size=2,
        history_action_tokens=8,
        current_action_sequence_tokens=20,
        invalid_current_prefix_tokens=4,
        device=torch.device("cpu"),
    )

    assert mask.shape == (2, 28, 1)
    assert torch.all(mask[:, :8] == 1.0)
    assert torch.all(mask[:, 8:12] == 0.0)
    assert torch.all(mask[:, 12:] == 1.0)


@pytest.mark.unit
def test_shared_strict_startup_generation_frame_guard() -> None:
    assert strict_startup_conditioning_frame_index(1) == 0
    assert strict_startup_conditioning_frame_index(5) == 4
    require_strict_startup_generation_frame(1)

    with pytest.raises(ValueError, match="generation_frame_start < 1"):
        require_strict_startup_generation_frame(0)


@pytest.mark.unit
def test_shape_only_video_token_grid_matches_frontend_tokenizer_metadata() -> None:
    config = SharedVideoTransformerConfig(
        input_channels=3,
        latent_channels=4,
        patch_size_t=2,
        patch_size_h=2,
        patch_size_w=2,
        hidden_size=8,
        load_reference_core_weights=False,
        load_text_conditioning=False,
        load_wan_vae_frontend=False,
    )
    frontend = SharedVideoFrontend(config)
    video_latents = torch.randn(1, 4, 4, 6, 8)

    _, token_grid = frontend.tokenize_video_latents(video_latents)
    shape_only_grid = video_token_grid_from_latent_shape(
        video_latents,
        patch_size=(config.patch_size_t, config.patch_size_h, config.patch_size_w),
    )

    assert shape_only_grid == token_grid


@pytest.mark.unit
def test_slice_token_grid_frames_respects_temporal_patch_size() -> None:
    video_latents = torch.randn(1, 4, 4, 6, 8)
    token_grid = video_token_grid_from_latent_shape(
        video_latents,
        patch_size=(2, 2, 2),
    )

    sliced = slice_token_grid_frames(token_grid, num_frames=2)

    assert sliced.num_frames == 2
    assert sliced.sequence_length == token_grid.tokens_per_frame
    with pytest.raises(ValueError, match="temporal patch size"):
        slice_token_grid_frames(token_grid, num_frames=3)
