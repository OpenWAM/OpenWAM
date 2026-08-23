from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs.enums import (
    DynamicsObjective,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    VideoActionProgram,
)
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.contracts import (
    DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_METADATA_KEY,
    DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_T0_SINGLETON,
    DYNAMICS_CONDITIONAL_HISTORY_POLICY_METADATA_KEY,
    DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
    DYNAMICS_CONDITIONAL_LAYOUT_METADATA_KEY,
    DYNAMICS_CONDITIONAL_LAYOUT_TARGET_ONLY_T0_PLUS_FUTURE,
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY,
    DYNAMICS_ROUTING_MODE_METADATA_KEY,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
    SampleConstructionMetadata,
)
from open_wam.models.common.dynamics_objectives import (
    compile_dynamics_training_plan,
    resolve_dynamics_training_plan,
)
from open_wam.models.common.flow_matching import FlowMatchScheduler
from open_wam.models.policy_variants.parallel_stream.dynamics_training import (
    apply_parallel_dynamics_training_plan,
    apply_parallel_prefix_dynamics_training_plan,
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
    coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA,
    program: VideoActionProgram = VideoActionProgram.GENERALIST_JOINT_DENOISING,
) -> ParallelStreamPolicyConfig:
    return ParallelStreamPolicyConfig(
        hidden_size=16,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=coupling,
        program=program,
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
    action_timesteps = action_scheduler.timesteps[
        torch.arange(int(action_latents.shape[2]))
    ][None]
    return SimpleNamespace(
        input_dict={
            "latent_dict": {
                "text_emb": text_emb,
                "noisy_latents": video_latents + 1,
                "targets": video_latents + 2,
                "latent": video_latents,
                "timesteps": latent_timesteps,
                "loss_mask": torch.ones_like(video_latents),
            },
            "action_dict": {
                "text_emb": text_emb,
                "noisy_latents": action_latents + 1,
                "targets": action_latents + 2,
                "timesteps": action_timesteps,
                "loss_mask": torch.ones_like(action_latents),
                "actions_mask": action_mask_latents,
            },
            "window_size": 8,
        },
        latent_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
    )


def _sample_metadata(
    mode: DynamicsObjective,
    *,
    source: str = "real_demo",
    drop_text: bool | None = None,
) -> SampleConstructionMetadata:
    metadata: dict[str, object] = {
        DYNAMICS_ROUTING_MODE_METADATA_KEY: mode.value,
        DYNAMICS_ROUTING_SOURCE_METADATA_KEY: source,
    }
    if drop_text is not None:
        metadata[DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY] = drop_text
    if mode.is_conditional:
        metadata.update(
            {
                DYNAMICS_CONDITIONAL_LAYOUT_METADATA_KEY: (
                    DYNAMICS_CONDITIONAL_LAYOUT_TARGET_ONLY_T0_PLUS_FUTURE
                ),
                DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_METADATA_KEY: (
                    DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_T0_SINGLETON
                ),
                DYNAMICS_CONDITIONAL_HISTORY_POLICY_METADATA_KEY: (
                    DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY
                ),
                "history_frames": 1,
                "loss_frame_start": 1,
                "latent_loss_frame_start": 1,
                "action_loss_frame_start": 1,
                "chunk_origin_frame": 1,
                "target_observation_frame_in_sample": 0,
                "singleton_chunk_frame": 0,
                "context_prefix_frames_in_sample": 1,
            }
        )
    result = SampleConstructionMetadata.from_mapping(metadata)
    assert result is not None
    return result


def _training_plan(
    policy_config: ParallelStreamPolicyConfig,
    sample_metadata: SampleConstructionMetadata | None,
    *,
    device: torch.device,
):
    plan = resolve_dynamics_training_plan(
        program=policy_config.program,
        sample_metadata=sample_metadata,
        device=device,
    )
    assert plan is not None
    return plan


def test_joint_mode_preserves_artifacts_and_metadata_order() -> None:
    video_latents = torch.zeros(1, 3, 3, 2, 2)
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

    policy_config = _policy()
    sample_metadata = _sample_metadata(
        DynamicsObjective.JOINT,
        drop_text=False,
    )
    apply_parallel_dynamics_training_plan(
        artifacts=artifacts,
        policy_config=policy_config,
        video_latents=video_latents,
        action_latents=action_latents,
        action_mask_latents=None,
        plan=_training_plan(
            policy_config,
            sample_metadata,
            device=video_latents.device,
        ),
    )

    assert artifacts.input_dict["latent_dict"] is latent_dict
    assert artifacts.input_dict["action_dict"] is action_dict
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    torch.testing.assert_close(latent_dict["text_emb"], text_emb)
    assert latent_dict["text_emb"] is action_dict["text_emb"]
    assert artifacts.input_dict["video_condition_source"] == "video_latents"
    assert artifacts.input_dict["generalist_training_source"] == "real_demo"
    assert artifacts.input_dict["joint_denoise_training_mode"] == "joint"
    assert artifacts.input_dict["joint_denoise_training_mode_override"] == "joint"
    assert artifacts.input_dict["joint_denoise_text_dropped"] is False

    expected_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        latent_dict["timesteps"][0]
    )
    torch.testing.assert_close(
        artifacts.input_dict["joint_denoise_shared_sigmas"],
        expected_sigmas,
        rtol=0.0,
        atol=0.0,
    )


def test_fdm_mode_uses_clean_masked_action_slot_and_exact_gradient() -> None:
    video_latents = (
        torch.arange(36, dtype=torch.float64).reshape(1, 3, 3, 2, 2) / 10
    ).requires_grad_()
    action_latents = (
        torch.arange(30, dtype=torch.float64).reshape(1, 5, 3, 2, 1) / 10
    ).requires_grad_()
    action_mask = (
        torch.tensor(
            [1, 0, 1, 0, 1],
            dtype=torch.float64,
        )
        .reshape(1, 5, 1, 1, 1)
        .expand_as(action_latents)
    )
    text_emb = torch.ones(1, 4, 8, dtype=torch.float64)
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=text_emb,
        action_mask_latents=action_mask,
    )
    original_latent_loss_mask = artifacts.input_dict["latent_dict"]["loss_mask"]

    torch.manual_seed(43)
    policy_config = _policy()
    sample_metadata = _sample_metadata(
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        source="counterfactual_dynamics",
    )
    apply_parallel_dynamics_training_plan(
        artifacts=artifacts,
        policy_config=policy_config,
        video_latents=video_latents,
        action_latents=action_latents,
        action_mask_latents=action_mask,
        plan=_training_plan(
            policy_config,
            sample_metadata,
            device=video_latents.device,
        ),
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
    assert input_dict["video_condition_source"] == "video_latents_target_only"
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
    video_gradient = torch.autograd.grad(
        latent_dict["noisy_latents"].square().sum()
        + latent_dict["targets"].square().sum()
        + latent_dict["latent"].square().sum(),
        video_latents,
    )[0]
    assert torch.isfinite(video_gradient).all()
    assert torch.count_nonzero(video_gradient) > 0


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
    policy_config = _policy()
    sample_metadata = _sample_metadata(
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
    )
    apply_parallel_dynamics_training_plan(
        artifacts=artifacts,
        policy_config=policy_config,
        video_latents=video_latents,
        action_latents=action_latents,
        action_mask_latents=None,
        plan=_training_plan(
            policy_config,
            sample_metadata,
            device=video_latents.device,
        ),
    )

    latent_dict = artifacts.input_dict["latent_dict"]
    action_dict = artifacts.input_dict["action_dict"]
    torch.testing.assert_close(
        latent_dict["noisy_latents"],
        video_latents,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(latent_dict["timesteps"]) == 0
    assert torch.count_nonzero(latent_dict["targets"]) == 0
    assert torch.count_nonzero(latent_dict["loss_mask"]) == 0
    assert action_dict["loss_mask"] is original_action_loss_mask

    video_gradient = torch.autograd.grad(
        latent_dict["noisy_latents"].sum(),
        video_latents,
    )[0]
    torch.testing.assert_close(
        video_gradient,
        torch.ones_like(video_latents),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("objective", "fixed_program"),
    [
        (
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            VideoActionProgram.FORWARD_DYNAMICS,
        ),
        (
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            VideoActionProgram.INVERSE_DYNAMICS,
        ),
    ],
)
def test_fixed_and_gjd_routes_produce_identical_parallel_training_artifacts(
    objective: DynamicsObjective,
    fixed_program: VideoActionProgram,
) -> None:
    source_video = torch.randn(1, 3, 3, 2, 2, dtype=torch.float64)
    source_action = torch.randn(1, 5, 3, 2, 1, dtype=torch.float64)
    metadata = _sample_metadata(objective)
    outputs: list[tuple[SimpleNamespace, torch.Tensor, torch.Tensor]] = []

    for program in (
        VideoActionProgram.GENERALIST_JOINT_DENOISING,
        fixed_program,
    ):
        video_latents = source_video.clone().requires_grad_()
        action_latents = source_action.clone().requires_grad_()
        artifacts = _artifacts(
            video_latents=video_latents,
            action_latents=action_latents,
            text_emb=torch.randn(1, 4, 8, dtype=torch.float64),
        )
        policy_config = _policy(
            coupling=JointTimestepCoupling.INDEPENDENT,
            program=program,
        )
        apply_parallel_dynamics_training_plan(
            artifacts=artifacts,
            policy_config=policy_config,
            video_latents=video_latents,
            action_latents=action_latents,
            action_mask_latents=None,
            plan=_training_plan(
                policy_config,
                metadata,
                device=video_latents.device,
            ),
        )
        loss = (
            artifacts.input_dict["latent_dict"]["noisy_latents"].sum()
            + artifacts.input_dict["action_dict"]["noisy_latents"].sum()
        )
        gradients = torch.autograd.grad(loss, (video_latents, action_latents))
        outputs.append((artifacts, gradients[0], gradients[1]))

    gjd, gjd_video_grad, gjd_action_grad = outputs[0]
    fixed, fixed_video_grad, fixed_action_grad = outputs[1]
    for stream in ("latent_dict", "action_dict"):
        for key in (
            "noisy_latents",
            "targets",
            "timesteps",
            "loss_mask",
            "text_emb",
        ):
            torch.testing.assert_close(
                gjd.input_dict[stream][key],
                fixed.input_dict[stream][key],
                rtol=0.0,
                atol=0.0,
            )
    torch.testing.assert_close(gjd_video_grad, fixed_video_grad, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gjd_action_grad, fixed_action_grad, rtol=0.0, atol=0.0)
    for key in (
        "joint_denoise_training_mode",
        "joint_timestep_coupling",
        "joint_denoise_text_dropped",
        "video_condition_source",
        "window_size",
        "history_stream_visibility",
        "conditional_history_policy",
    ):
        assert gjd.input_dict[key] == fixed.input_dict[key]


def test_parallel_dynamics_adapter_rejects_multi_sample_runtime_batch() -> None:
    video_latents = torch.zeros(2, 3, 3, 2, 2)
    action_latents = torch.zeros(2, 5, 3, 2, 1)
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=torch.ones(2, 4, 8),
    )

    with pytest.raises(ValueError, match="train_batch_size=1"):
        apply_parallel_dynamics_training_plan(
            artifacts=artifacts,
            policy_config=_policy(),
            video_latents=video_latents,
            action_latents=action_latents,
            action_mask_latents=None,
            plan=compile_dynamics_training_plan(
                objective=DynamicsObjective.JOINT,
            ),
        )


def test_external_prefix_joint_mode_excludes_prefix_from_shared_sigmas() -> None:
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

    policy_config = _policy()
    apply_parallel_prefix_dynamics_training_plan(
        artifacts=artifacts,
        policy_config=policy_config,
        plan=_training_plan(
            policy_config,
            _sample_metadata(
                DynamicsObjective.JOINT,
                drop_text=False,
            ),
            device=video_latents.device,
        ),
    )

    assert artifacts.input_dict["joint_denoise_training_mode"] == "joint"
    assert artifacts.input_dict["video_condition_source"] == "condition_latents_prefix"
    assert artifacts.input_dict["generalist_training_source"] == "real_demo"
    assert torch.count_nonzero(artifacts.input_dict["latent_dict"]["text_emb"]) > 0
    expected_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        latent_timesteps[1:]
    )
    torch.testing.assert_close(
        artifacts.input_dict["joint_denoise_shared_sigmas"],
        expected_sigmas,
        rtol=0.0,
        atol=0.0,
    )


def test_external_prefix_joint_plan_preserves_rng_state() -> None:
    video_latents = torch.zeros(1, 3, 4, 2, 2)
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=torch.zeros(1, 5, 3, 2, 1),
        text_emb=torch.ones(1, 4, 8),
    )
    artifacts.input_dict["prefix_condition_frames"] = 1

    torch.manual_seed(53)
    rng_before = torch.random.get_rng_state().clone()
    apply_parallel_prefix_dynamics_training_plan(
        artifacts=artifacts,
        policy_config=_policy(),
        plan=compile_dynamics_training_plan(
            objective=DynamicsObjective.JOINT,
        ),
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert artifacts.input_dict["joint_denoise_training_mode"] == "joint"
    assert artifacts.input_dict["joint_denoise_training_mode_override"] is None


@pytest.mark.parametrize(
    "mode",
    [
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
    ],
)
def test_external_prefix_rejects_conditional_modes(
    mode: DynamicsObjective,
) -> None:
    video_latents = torch.zeros(1, 3, 4, 2, 2)
    action_latents = torch.zeros(1, 5, 3, 2, 1)
    artifacts = _artifacts(
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=torch.ones(1, 4, 8),
    )

    with pytest.raises(ValueError, match="cannot use external-prefix"):
        policy_config = _policy()
        apply_parallel_prefix_dynamics_training_plan(
            artifacts=artifacts,
            policy_config=policy_config,
            plan=_training_plan(
                policy_config,
                _sample_metadata(mode),
                device=video_latents.device,
            ),
        )
