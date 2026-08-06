from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import (
    CurrentBlockCoupling,
    GeneralistDenoisingMode,
    GeneralistTrainingParadigm,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    ParallelRuntimeMode,
    ParallelStreamVariantProfile,
)
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.models.common.flow_matching import FlowMatchScheduler
from open_wam.models.common.joint_conditioning import sample_conditioning_mode
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.generalist_training import (
    apply_generalist_joint_denoise_training_mode,
    apply_generalist_legacy_prefix_joint_training_mode,
    sample_generalist_joint_denoise_training_mode,
)


def _scheduler(*, shift: float, steps: int = 16) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=steps,
    )
    scheduler.set_timesteps(steps, training=True)
    return scheduler


def _policy(
    *,
    mode: GeneralistDenoisingMode | None,
    coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA,
    paradigm: GeneralistTrainingParadigm = GeneralistTrainingParadigm.DYNAMICS_ROUTED,
) -> ParallelStreamPolicyConfig:
    probabilities = None if mode is None else {mode: 1.0}
    return ParallelStreamPolicyConfig(
        hidden_size=16,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        variant_profile=ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=coupling,
        generalist_training_paradigm=paradigm,
        generalist_denoising_mode_probs=probabilities,
    )


def test_parallel_stream_conditional_gjd_rejects_demo_only_data_contract() -> None:
    with pytest.raises(ValueError, match="require.*dynamics_routed"):
        _policy(
            mode=GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
            paradigm=GeneralistTrainingParadigm.DEMO_ONLY,
        )


def _artifacts(
    *,
    video_latents: torch.Tensor,
    action_latents: torch.Tensor,
    text_emb: torch.Tensor,
    action_mask_latents: torch.Tensor | None = None,
) -> SimpleNamespace:
    latent_scheduler = _scheduler(shift=3.0)
    action_scheduler = _scheduler(shift=5.0)
    latent_timesteps = latent_scheduler.timesteps[
        torch.arange(int(video_latents.shape[2]))
    ][None]
    if action_mask_latents is None:
        action_mask_latents = torch.ones_like(action_latents)
    return SimpleNamespace(
        input_dict={
            "latent_dict": {
                "text_emb": text_emb,
                "noisy_latents": video_latents,
                "timesteps": latent_timesteps,
                "loss_mask": torch.ones_like(video_latents),
            },
            "action_dict": {
                "text_emb": text_emb,
                "loss_mask": torch.ones_like(action_latents),
                "actions_mask": action_mask_latents,
            },
            "window_size": 8,
        },
        latent_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
    )


def _backbone() -> SharedVideoTransformerConfig:
    return SharedVideoTransformerConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        text_dim=8,
        freq_dim=4,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )


def test_reference_runtime_generalist_training_names_alias_canonical_contract() -> None:
    assert (
        reference_runtime._apply_generalist_joint_denoise_training_mode
        is apply_generalist_joint_denoise_training_mode
    )
    assert (
        reference_runtime._apply_generalist_legacy_prefix_joint_training_mode
        is apply_generalist_legacy_prefix_joint_training_mode
    )
    assert (
        reference_runtime._sample_joint_denoise_training_mode
        is sample_generalist_joint_denoise_training_mode
    )


def test_generalist_mode_sampling_default_preserves_categorical_rng_draw() -> None:
    policy_config = ParallelStreamPolicyConfig(hidden_size=16)
    probabilities = policy_config.generalist_denoising_mode_probs
    assert probabilities is not None

    torch.manual_seed(31)
    expected = sample_conditioning_mode(
        probabilities,
        enum_cls=GeneralistDenoisingMode,
        device=torch.device("cpu"),
        error_label="Generalist joint-denoise training mode",
    )
    expected_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(31)
    mode = sample_generalist_joint_denoise_training_mode(
        policy_config,
        device=torch.device("cpu"),
    )

    assert mode == expected == GeneralistDenoisingMode.JOINT
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


def test_generalist_mode_sampling_preserves_categorical_rng_sequence() -> None:
    policy_config = _policy(mode=GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO)
    probabilities = policy_config.generalist_denoising_mode_probs
    assert probabilities is not None

    torch.manual_seed(37)
    expected = sample_conditioning_mode(
        probabilities,
        enum_cls=GeneralistDenoisingMode,
        device=torch.device("cpu"),
        error_label="Generalist joint-denoise training mode",
    )
    expected_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(37)
    actual = sample_generalist_joint_denoise_training_mode(
        policy_config,
        device=torch.device("cpu"),
    )

    assert actual == expected
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


def test_joint_mode_preserves_artifacts_and_metadata_order() -> None:
    video_latents = torch.zeros(1, 3, 3, 2, 2)
    condition_latents = torch.ones_like(video_latents)
    action_latents = torch.zeros(1, 5, 3, 2, 1)
    text_emb = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=text_emb,
    )
    latent_dict = artifacts.input_dict["latent_dict"]
    action_dict = artifacts.input_dict["action_dict"]
    torch.manual_seed(41)
    rng_before = torch.random.get_rng_state().clone()

    apply_generalist_joint_denoise_training_mode(
        artifacts=artifacts,
        policy_config=_policy(mode=GeneralistDenoisingMode.JOINT),
        backbone_config=_backbone(),
        video_latents=video_latents,
        condition_latents=condition_latents,
        action_latents=action_latents,
        action_mask_latents=None,
        frame_shift=0,
        training_mode_override=GeneralistDenoisingMode.JOINT,
        drop_text_conditioning=True,
        training_source="real_demo",
    )

    assert artifacts.input_dict["latent_dict"] is latent_dict
    assert artifacts.input_dict["action_dict"] is action_dict
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.count_nonzero(latent_dict["text_emb"]) == 0
    assert latent_dict["text_emb"] is action_dict["text_emb"]
    assert artifacts.input_dict["video_condition_source"] == "condition_latents"
    assert artifacts.input_dict["generalist_training_source"] == "real_demo"
    assert artifacts.input_dict["joint_denoise_training_mode"] == "joint"
    assert artifacts.input_dict["joint_denoise_training_mode_override"] == "joint"
    assert artifacts.input_dict["joint_denoise_text_dropped"] is True

    expected_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        latent_dict["timesteps"][0]
    )
    torch.testing.assert_close(
        artifacts.input_dict["joint_denoise_shared_sigmas"],
        expected_sigmas,
        rtol=0.0,
        atol=0.0,
    )
    assert list(artifacts.input_dict)[-10:] == [
        "variant_profile",
        "generalist_training_paradigm",
        "generalist_training_source",
        "joint_denoise_training_mode",
        "joint_timestep_coupling",
        "joint_denoise_training_mode_override",
        "joint_denoise_text_dropped",
        "generalist_denoising_mode_probs",
        "video_condition_source",
        "joint_denoise_shared_sigmas",
    ]


def test_fdm_mode_uses_clean_masked_action_slot_and_exact_gradient() -> None:
    video_latents = (
        torch.arange(36, dtype=torch.float64).reshape(1, 3, 3, 2, 2) / 10
    ).requires_grad_()
    condition_latents = (video_latents.detach() + 2).requires_grad_()
    action_latents = (
        torch.arange(30, dtype=torch.float64).reshape(1, 5, 3, 2, 1) / 10
    ).requires_grad_()
    action_mask = torch.tensor(
        [1, 0, 1, 0, 1],
        dtype=torch.float64,
    ).reshape(1, 5, 1, 1, 1).expand_as(action_latents)
    text_emb = torch.ones(1, 4, 8, dtype=torch.float64)
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=text_emb,
        action_mask_latents=action_mask,
    )
    original_latent_loss_mask = artifacts.input_dict["latent_dict"]["loss_mask"]

    torch.manual_seed(43)
    apply_generalist_joint_denoise_training_mode(
        artifacts=artifacts,
        policy_config=_policy(
            mode=GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO
        ),
        backbone_config=_backbone(),
        video_latents=video_latents,
        condition_latents=condition_latents,
        action_latents=action_latents,
        action_mask_latents=action_mask,
        frame_shift=2,
        training_mode_override=GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
        training_source="counterfactual_dynamics",
    )

    input_dict = artifacts.input_dict
    latent_dict = input_dict["latent_dict"]
    action_dict = input_dict["action_dict"]
    torch.testing.assert_close(
        action_dict["noisy_latents"],
        action_latents * action_mask,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(action_dict["timesteps"]) == 0
    assert torch.count_nonzero(action_dict["targets"]) == 0
    assert torch.count_nonzero(action_dict["loss_mask"]) == 0
    assert latent_dict["loss_mask"] is original_latent_loss_mask
    assert input_dict["window_size"] == 3
    assert (
        input_dict["history_stream_visibility"]
        == HistoryStreamVisibility.VIDEO_ONLY.value
    )
    assert input_dict["conditional_history_policy"] == "previous_boundary_video_only"
    assert input_dict["generalist_conditional_history_chunks"] == 1
    assert input_dict["video_condition_source"] == "condition_latents"
    assert input_dict["generalist_training_source"] == "counterfactual_dynamics"
    assert torch.count_nonzero(latent_dict["text_emb"]) == 0

    action_gradient = torch.autograd.grad(
        action_dict["noisy_latents"].sum(),
        action_latents,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(
        action_gradient,
        action_mask,
        rtol=0.0,
        atol=0.0,
    )
    video_gradient, condition_gradient = torch.autograd.grad(
        latent_dict["noisy_latents"].square().sum()
        + latent_dict["targets"].square().sum()
        + latent_dict["latent"].square().sum(),
        (video_latents, condition_latents),
    )
    assert torch.isfinite(video_gradient).all()
    assert torch.isfinite(condition_gradient).all()
    assert torch.count_nonzero(video_gradient) > 0
    assert torch.count_nonzero(condition_gradient) > 0


def test_idm_mode_uses_explicit_clean_video_slot_and_routes_gradient() -> None:
    video_latents = torch.zeros(
        1,
        3,
        3,
        2,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )
    condition_latents = torch.full_like(
        video_latents,
        7,
        requires_grad=True,
    )
    action_latents = torch.ones(
        1,
        5,
        3,
        2,
        1,
        dtype=torch.float64,
        requires_grad=True,
    )
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=torch.ones(1, 4, 8, dtype=torch.float64),
    )
    original_action_loss_mask = artifacts.input_dict["action_dict"]["loss_mask"]

    torch.manual_seed(47)
    apply_generalist_joint_denoise_training_mode(
        artifacts=artifacts,
        policy_config=_policy(
            mode=GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION
        ),
        backbone_config=_backbone(),
        video_latents=video_latents,
        condition_latents=condition_latents,
        action_latents=action_latents,
        action_mask_latents=None,
        frame_shift=0,
        training_mode_override=GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
    )

    latent_dict = artifacts.input_dict["latent_dict"]
    action_dict = artifacts.input_dict["action_dict"]
    torch.testing.assert_close(
        latent_dict["noisy_latents"],
        condition_latents,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(latent_dict["timesteps"]) == 0
    assert torch.count_nonzero(latent_dict["targets"]) == 0
    assert torch.count_nonzero(latent_dict["loss_mask"]) == 0
    assert action_dict["loss_mask"] is original_action_loss_mask

    condition_gradient = torch.autograd.grad(
        latent_dict["noisy_latents"].sum(),
        condition_latents,
    )[0]
    torch.testing.assert_close(
        condition_gradient,
        torch.ones_like(condition_latents),
        rtol=0.0,
        atol=0.0,
    )


def test_generalist_training_rejects_multi_sample_runtime_batch() -> None:
    video_latents = torch.zeros(2, 3, 3, 2, 2)
    action_latents = torch.zeros(2, 5, 3, 2, 1)
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=torch.ones(2, 4, 8),
    )

    with pytest.raises(ValueError, match="train_batch_size=1"):
        apply_generalist_joint_denoise_training_mode(
            artifacts=artifacts,
            policy_config=_policy(mode=GeneralistDenoisingMode.JOINT),
            backbone_config=_backbone(),
            video_latents=video_latents,
            condition_latents=None,
            action_latents=action_latents,
            action_mask_latents=None,
            frame_shift=0,
        )


def test_legacy_prefix_joint_mode_excludes_prefix_from_shared_sigmas() -> None:
    video_latents = torch.zeros(1, 3, 4, 2, 2)
    action_latents = torch.zeros(1, 5, 3, 2, 1)
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=torch.ones(1, 4, 8),
    )
    artifacts.input_dict["prefix_condition_frames"] = 1
    artifacts.input_dict["video_condition_source"] = "condition_latents_prefix"
    latent_timesteps = artifacts.input_dict["latent_dict"]["timesteps"][0]

    apply_generalist_legacy_prefix_joint_training_mode(
        artifacts=artifacts,
        policy_config=_policy(mode=GeneralistDenoisingMode.JOINT),
        drop_text_conditioning=True,
        training_source="real_demo",
    )

    assert artifacts.input_dict["joint_denoise_training_mode"] == "joint"
    assert artifacts.input_dict["video_condition_source"] == "condition_latents_prefix"
    assert artifacts.input_dict["generalist_training_source"] == "real_demo"
    assert torch.count_nonzero(
        artifacts.input_dict["latent_dict"]["text_emb"]
    ) == 0
    expected_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        latent_timesteps[1:]
    )
    torch.testing.assert_close(
        artifacts.input_dict["joint_denoise_shared_sigmas"],
        expected_sigmas,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "mode",
    [
        GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
        GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
    ],
)
def test_legacy_prefix_rejects_forced_conditional_modes(
    mode: GeneralistDenoisingMode,
) -> None:
    video_latents = torch.zeros(1, 3, 4, 2, 2)
    action_latents = torch.zeros(1, 5, 3, 2, 1)
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=torch.ones(1, 4, 8),
    )

    with pytest.raises(ValueError, match="only `joint` is parity-compatible"):
        apply_generalist_legacy_prefix_joint_training_mode(
            artifacts=artifacts,
            policy_config=_policy(mode=GeneralistDenoisingMode.JOINT),
            training_mode_override=mode,
        )
