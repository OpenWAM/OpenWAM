"""Dual Expert coverage for shared joint, FDM, and IDM semantics."""

from __future__ import annotations

import random
from dataclasses import replace as _dataclass_replace

import pytest
import torch

from open_wam.configs import TrainingConfig
from open_wam.configs.enums import (
    AttachSite,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DynamicsObjective,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    PolicyVariantName,
    ProprioContextMode,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
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
from open_wam.models.common.attention_profiles import (
    build_chunked_text_context_cross_attention_mask,
)
from open_wam.models.common.dynamics_objectives import (
    DynamicsRolloutRequest,
    resolve_dynamics_rollout_objective,
    resolve_dynamics_training_plan,
)
from open_wam.models.common.flow_matching import (
    build_frame_aligned_action_flow_match_train_artifacts,
    build_video_flow_match_train_artifacts,
)
from open_wam.models.policy_variants.contracts import PolicyInferContext
from open_wam.models.policy_variants.dual_expert.attention import (
    build_dual_expert_packed_coupling_attention_profile,
)
from open_wam.models.policy_variants.dual_expert.coupling_semantics import (
    should_couple_dual_expert_action_to_video_sigmas,
)
from open_wam.models.policy_variants.dual_expert.decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
    DualExpertTrainArtifacts,
)


def _make_dual_expert_policy_config(**overrides) -> DualExpertPolicyConfig:
    base = dict(
        name=PolicyVariantName.DUAL_EXPERT,
        hidden_size=256,
        attach_site=AttachSite.POST_VISUAL_CORE,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
    )
    base.update(overrides)
    return DualExpertPolicyConfig(**base)


def _dynamics_sample_metadata(
    mode: DynamicsObjective,
    *,
    frame_count: int = 4,
    source: str = "real_demo",
    drop_text: bool | None = None,
) -> dict[str, object]:
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
                "loss_frame_end": int(frame_count),
                "latent_loss_frame_start": 1,
                "latent_loss_frame_end": int(frame_count),
                "action_loss_frame_start": 1,
                "action_loss_frame_end": int(frame_count),
                "chunk_origin_frame": 1,
                "target_observation_frame_in_sample": 0,
                "singleton_chunk_frame": 0,
                "context_prefix_frames_in_sample": 1,
            }
        )
    return metadata


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_default_opt_out_keeps_existing_six_mode_path() -> None:
    cfg = _make_dual_expert_policy_config()
    assert cfg.program is VideoActionProgram.VIDEO_THEN_ACTION


def test_generalist_program_derives_independent_joint_coupling() -> None:
    cfg = _make_dual_expert_policy_config(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
    )
    assert cfg.current_block_coupling is CurrentBlockCoupling.JOINT
    assert cfg.joint_timestep_coupling is JointTimestepCoupling.INDEPENDENT


@pytest.mark.parametrize(
    ("program", "expected_mode"),
    [
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
    ],
)
def test_standalone_conditional_program_owns_fixed_mode(
    program: VideoActionProgram,
    expected_mode: DynamicsObjective,
) -> None:
    cfg = _make_dual_expert_policy_config(
        program=program,
    )

    assert cfg.current_block_coupling == CurrentBlockCoupling.JOINT
    assert resolve_dynamics_rollout_objective(program=cfg.program) is expected_mode


def test_standalone_conditional_program_rejects_conflicting_sample_mode() -> None:
    config = _make_dual_expert_policy_config(
        program=VideoActionProgram.FORWARD_DYNAMICS,
    )
    metadata = SampleConstructionMetadata.from_mapping(
        {
            DYNAMICS_ROUTING_MODE_METADATA_KEY: (
                DynamicsObjective.VIDEO_CONDITIONED_ACTION.value
            )
        }
    )

    with pytest.raises(ValueError, match="forward_dynamics.*video_conditioned_action"):
        resolve_dynamics_training_plan(
            program=config.program,
            sample_metadata=metadata,
            device=torch.device("cpu"),
        )


def test_gjd_accepts_dataset_selected_sample_mode() -> None:
    config = _make_dual_expert_policy_config(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
    )
    metadata = SampleConstructionMetadata.from_mapping(
        _dynamics_sample_metadata(
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            drop_text=True,
        )
    )

    plan = resolve_dynamics_training_plan(
        program=config.program,
        sample_metadata=metadata,
        device=torch.device("cpu"),
    )
    assert plan is not None
    assert plan.objective is DynamicsObjective.VIDEO_CONDITIONED_ACTION
    assert plan.routed_objective is plan.objective


def test_generalist_sigma_coupling_is_explicitly_configurable() -> None:
    cfg = _make_dual_expert_policy_config(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
    )
    assert should_couple_dual_expert_action_to_video_sigmas(cfg) is False

    cfg = _make_dual_expert_policy_config(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    assert should_couple_dual_expert_action_to_video_sigmas(cfg) is True

    cfg = _make_dual_expert_policy_config(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        joint_timestep_coupling=JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    )
    assert should_couple_dual_expert_action_to_video_sigmas(cfg) is True

    cfg = _make_dual_expert_policy_config(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
    )
    assert should_couple_dual_expert_action_to_video_sigmas(cfg) is False

    cfg = _make_dual_expert_policy_config(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_INDEX,
    )
    assert should_couple_dual_expert_action_to_video_sigmas(cfg) is False

    cfg = _make_dual_expert_policy_config(
        program=VideoActionProgram.DECOUPLED_SAME_STEP
    )
    assert should_couple_dual_expert_action_to_video_sigmas(cfg) is False


def test_existing_six_mode_yamls_are_not_disturbed() -> None:
    """Sanity: any of the six planning programs keeps its coupling."""

    for coupling in CurrentBlockCoupling:
        cfg = _make_dual_expert_policy_config(
            program=VideoActionProgram(coupling.value)
        )
        assert cfg.current_block_coupling == coupling


def test_dual_expert_generalist_mode_text_token_requires_gjd_program() -> None:
    cfg = _make_dual_expert_policy_config(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        generalist_mode_text_token=True,
    )
    assert cfg.generalist_mode_text_token is True

    with pytest.raises(
        ValueError,
        match=r"mode_text_token.*generalist_joint_denoising",
    ):
        _make_dual_expert_policy_config(
            program=VideoActionProgram.JOINT,
            generalist_mode_text_token=True,
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
    video_sigma_values = video_artifacts.scheduler.sigma_for_timesteps(
        video_artifacts.timesteps
    )
    action_artifacts = build_frame_aligned_action_flow_match_train_artifacts(
        actions,
        None,
        training_config=training_config,
        num_frames=3,
        action_per_frame=2,
        frame_sigma_values=video_sigma_values,
    )

    action_sigma_values = action_artifacts.scheduler.sigma_for_timesteps(
        action_artifacts.frame_timesteps
    )
    assert torch.allclose(action_sigma_values, video_sigma_values, atol=2e-3, rtol=2e-3)


# ---------------------------------------------------------------------------
# Routed-mode integration (end-to-end forward_train through the variant +
# the DualExpert decoder, with sample metadata selecting one mode so we can
# pattern-match on the loss/active flags deterministically).
# ---------------------------------------------------------------------------


def _build_tiny_generalist_pipeline(
    forced_mode: DynamicsObjective,
    *,
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.INDEPENDENT,
    action_hidden_size: int | None = None,
    generalist_mode_text_token: bool = False,
    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE,
    program: VideoActionProgram | None = None,
):
    """Construct a tiny CPU pipeline pinned to one generalist mode."""

    from open_wam.configs import (
        ActionSchemaConfig,
        DualExpertActionDecoderConfig,
        DualExpertActionExpertInitMode,
        DynamicsRouteConfig,
        DynamicsRoutingConfig,
        ExperimentConfig,
        InferenceConfig,
        RobotWinDataConfig,
        TrainingConfig,
    )
    from open_wam.configs import (
        DualExpertPolicyConfig as TopLevelDualExpertPolicyConfig,
    )
    from open_wam.models.policy_variants.contracts import PolicyTrainBatch
    from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
    from open_wam.pipelines import build_variant_pipeline_from_config

    conditional_mode = forced_mode in {
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
    }
    routed = program is not None or conditional_mode

    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
            dynamics_routing=DynamicsRoutingConfig(
                routes=(
                    DynamicsRouteConfig(
                        source="real_demo",
                        mode=forced_mode,
                        weight=1.0,
                    ),
                )
                if routed
                else ()
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
        policy_variant=TopLevelDualExpertPolicyConfig(
            hidden_size=32,
            program=program or VideoActionProgram.GENERALIST_JOINT_DENOISING,
            video_prefix_frames=1,
            num_action_layers=1,
            action_hidden_size=action_hidden_size,
            action_expert_init_mode=(
                DualExpertActionExpertInitMode.VIDEO_WEIGHT_INTERPOLATE
                if action_hidden_size is not None
                else DualExpertActionExpertInitMode.VIDEO_WEIGHT_COPY
            ),
            generalist_mode_text_token=generalist_mode_text_token,
            proprio_context_mode=proprio_context_mode,
            joint_timestep_coupling=joint_timestep_coupling,
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
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
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 4, 4),
        extra={
            "metadata": _dynamics_sample_metadata(
                forced_mode,
                drop_text=conditional_mode,
            )
        }
        if routed
        else {},
    )
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.randn(1, 5, 16)
    return pipeline, batch, video_latents, text_context


@pytest.mark.parametrize(
    ("mode", "standalone_program"),
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
def test_standalone_conditional_training_matches_single_route_gjd(
    mode: DynamicsObjective,
    standalone_program: VideoActionProgram,
) -> None:
    from open_wam.models.policy_variants.contracts import PolicyTrainBatch

    torch.manual_seed(101)
    gjd_pipeline, gjd_batch, video_latents, text_context = (
        _build_tiny_generalist_pipeline(
            mode,
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
            joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
        )
    )
    torch.manual_seed(202)
    standalone_pipeline, _, _, _ = _build_tiny_generalist_pipeline(
        mode,
        program=standalone_program,
        joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
    )
    standalone_pipeline.load_state_dict(gjd_pipeline.state_dict(), strict=True)

    actions = gjd_batch.actions.detach().clone()
    metadata = _dynamics_sample_metadata(mode, drop_text=True)

    def run_once(pipeline):
        pipeline.zero_grad(set_to_none=True)
        batch = PolicyTrainBatch(
            actions=actions.clone(),
            extra={"metadata": dict(metadata)},
        )
        random.seed(303)
        torch.manual_seed(303)
        output = pipeline.forward_train_from_latents(
            video_latents.clone(),
            batch,
            text_context=text_context.clone(),
        )
        output.decoder_output.loss.backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in pipeline.named_parameters()
        }
        return output, gradients

    gjd_output, gjd_gradients = run_once(gjd_pipeline)
    standalone_output, standalone_gradients = run_once(standalone_pipeline)

    torch.testing.assert_close(
        standalone_output.decoder_output.loss,
        gjd_output.decoder_output.loss,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        standalone_output.policy_output.aux["dual_expert_generalist_training_mode"]
        == mode.value
    )
    assert (
        standalone_output.policy_output.aux["dual_expert_generalist_text_dropped"]
        is True
    )
    assert standalone_output.policy_output.aux["sampled_window_size"] == 3
    assert standalone_output.policy_output.aux["conditional_history_policy"] == (
        "previous_boundary_video_only"
    )
    for key in (
        "weighted_action_diffusion_loss",
        "weighted_video_diffusion_loss",
        "joint_loss",
    ):
        torch.testing.assert_close(
            standalone_output.decoder_output.metrics[key],
            gjd_output.decoder_output.metrics[key],
            rtol=0.0,
            atol=0.0,
        )
    for key in ("flow_pred", "predicted_latents", "future_video_flow_pred"):
        torch.testing.assert_close(
            standalone_output.decoder_output.aux[key],
            gjd_output.decoder_output.aux[key],
            rtol=0.0,
            atol=0.0,
        )
    assert standalone_gradients.keys() == gjd_gradients.keys()
    for name, standalone_gradient in standalone_gradients.items():
        gjd_gradient = gjd_gradients[name]
        assert (standalone_gradient is None) is (gjd_gradient is None), name
        if standalone_gradient is not None:
            torch.testing.assert_close(
                standalone_gradient,
                gjd_gradient,
                rtol=0.0,
                atol=0.0,
                msg=lambda message, parameter=name: f"{parameter}: {message}",
            )


def test_forced_joint_training_respects_timestep_coupling_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_training as dual_expert_packed_training_module

    original_build_action_artifacts = dual_expert_packed_training_module.build_frame_aligned_action_flow_match_train_artifacts
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
        dual_expert_packed_training_module,
        "build_frame_aligned_action_flow_match_train_artifacts",
        spy_build_action_artifacts,
    )

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.JOINT,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.JOINT,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_INDEX,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.JOINT,
        joint_timestep_coupling=JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.JOINT,
        joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert saw_action_coupling_inputs == [
        (True, False, False),
        (False, True, False),
        (False, True, True),
        (False, False, False),
    ]


def test_dual_expert_generalist_mode_token_is_appended_in_train_path() -> None:
    torch.manual_seed(0)
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.JOINT,
        generalist_mode_text_token=True,
    )

    assert pipeline.visual_tower.core.generalist_mode_context_encoder is not None
    output = pipeline.forward_train_from_latents(
        video_latents, batch, text_context=text_context
    )

    assert (
        output.policy_output.aux["dual_expert_generalist_training_mode"]
        == DynamicsObjective.JOINT.value
    )
    assert (
        output.policy_output.aux["dual_expert_generalist_mode_text_token"]
        == DynamicsObjective.JOINT.value
    )
    assert output.policy_output.aux["dual_expert_generalist_mode_text_token_count"] == 1


def test_generalist_match_sigma_uses_video_clock_for_all_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_training as dual_expert_packed_training_module

    original_build_action_artifacts = dual_expert_packed_training_module.build_frame_aligned_action_flow_match_train_artifacts
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
        dual_expert_packed_training_module,
        "build_frame_aligned_action_flow_match_train_artifacts",
        spy_build_action_artifacts,
    )

    for mode in (
        DynamicsObjective.JOINT,
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
    ):
        pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
            mode,
            joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
        )
        pipeline.forward_train_from_latents(
            video_latents, batch, text_context=text_context
        )

    assert saw_action_coupling_inputs == [(True, False), (True, False), (True, False)]


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        ("joint", DynamicsObjective.JOINT),
        ("action_conditioned_video", DynamicsObjective.ACTION_CONDITIONED_VIDEO),
        ("video_conditioned_action", DynamicsObjective.VIDEO_CONDITIONED_ACTION),
    ],
)
def test_dual_expert_gjd_rollout_objectives_match_training_modes(
    raw_mode: str,
    expected: DynamicsObjective,
) -> None:
    assert (
        resolve_dynamics_rollout_objective(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
            requested_objective=raw_mode,
        )
        is expected
    )


@pytest.mark.parametrize(
    "raw_mode",
    [
        "vanilla_joint_rollout",
        "clean_action_feedback",
        "forced_action_joint_fdm",
        "fdm",
        "idm",
        "not_a_mode",
    ],
)
def test_dual_expert_gjd_rollout_rejects_noncanonical_objectives(
    raw_mode: str,
) -> None:
    with pytest.raises(ValueError, match="Unsupported dynamics objective"):
        resolve_dynamics_rollout_objective(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
            requested_objective=raw_mode,
        )


@pytest.mark.parametrize(
    ("program", "expected_mode"),
    [
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
    ],
)
def test_standalone_conditional_program_is_default_offline_inference_mode(
    program: VideoActionProgram,
    expected_mode: DynamicsObjective,
) -> None:
    config = _make_dual_expert_policy_config(
        program=program,
    )

    assert resolve_dynamics_rollout_objective(program=config.program) is expected_mode


def test_gjd_defaults_to_joint_offline_inference_mode() -> None:
    config = _make_dual_expert_policy_config(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
    )

    assert (
        resolve_dynamics_rollout_objective(program=config.program)
        is DynamicsObjective.JOINT
    )


@pytest.mark.parametrize(
    "program",
    [VideoActionProgram.FORWARD_DYNAMICS, VideoActionProgram.INVERSE_DYNAMICS],
)
def test_standalone_conditional_program_rejects_conflicting_inference_mode(
    program: VideoActionProgram,
) -> None:
    config = _make_dual_expert_policy_config(
        program=program,
    )

    with pytest.raises(ValueError, match="requires rollout objective"):
        resolve_dynamics_rollout_objective(
            program=config.program,
            requested_objective=DynamicsObjective.JOINT,
        )


@pytest.mark.parametrize(
    ("mode", "standalone_program"),
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
def test_standalone_conditional_offline_inference_matches_explicit_gjd_mode(
    mode: DynamicsObjective,
    standalone_program: VideoActionProgram,
) -> None:
    from open_wam.models.common import RolloutCursor
    from open_wam.models.policy_variants.contracts import PolicyInferState
    from open_wam.models.policy_variants.dual_expert.contracts import (
        DualExpertRuntimeState,
    )

    torch.manual_seed(401)
    gjd_pipeline, _, _, _ = _build_tiny_generalist_pipeline(
        mode,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
    )
    torch.manual_seed(402)
    standalone_pipeline, _, _, _ = _build_tiny_generalist_pipeline(
        mode,
        program=standalone_program,
        joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
    )
    standalone_pipeline.load_state_dict(gjd_pipeline.state_dict(), strict=True)
    for pipeline in (gjd_pipeline, standalone_pipeline):
        pipeline.policy_variant.inference_config = _dataclass_replace(
            pipeline.policy_variant.inference_config,
            video_num_inference_steps=2,
            action_num_inference_steps=2,
        )

    history_video = torch.randn(1, 48, 2, 8, 8)
    history_actions = torch.randn(1, 4, 4)
    current_video = torch.randn(1, 48, 1, 8, 8)
    forced_actions = torch.randn(1, 4, 4)
    commit_actions = torch.randn(1, 4, 4)
    text_context = torch.randn(1, 5, 16)

    def make_state() -> PolicyInferState:
        return PolicyInferState(
            step_index=1,
            cursor=RolloutCursor(current_start_frame=2, block_index=0, chunk_size=2),
            variant_state=DualExpertRuntimeState(
                past_clean_latents=history_video.clone(),
                past_clean_actions=history_actions.clone(),
                next_condition_frame_start=2,
                chunk_advance_frames=2,
            ),
        )

    def run_once(pipeline, *, explicit_mode: bool):
        dynamics = DynamicsRolloutRequest(
            objective=mode if explicit_mode else None,
            clean_action=(
                forced_actions.clone()
                if mode == DynamicsObjective.ACTION_CONDITIONED_VIDEO
                else None
            ),
            clean_video=(
                current_video.clone()
                if mode == DynamicsObjective.VIDEO_CONDITIONED_ACTION
                else None
            ),
            history_action=(
                commit_actions.clone()
                if mode == DynamicsObjective.VIDEO_CONDITIONED_ACTION
                else None
            ),
        )
        random.seed(403)
        torch.manual_seed(403)
        return pipeline.forward_infer_step_from_latents(
            current_video.clone(),
            PolicyInferContext(dynamics=dynamics),
            infer_state=make_state(),
            text_context=text_context.clone(),
        )

    gjd_output = run_once(gjd_pipeline, explicit_mode=True)
    standalone_output = run_once(standalone_pipeline, explicit_mode=False)

    assert standalone_output.policy_output.aux["action_conditioning_mode"] == mode.value
    torch.testing.assert_close(
        standalone_output.decoder_output.action_pred,
        gjd_output.decoder_output.action_pred,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        standalone_output.decoder_output.aux["predicted_latents"],
        gjd_output.decoder_output.aux["predicted_latents"],
        rtol=0.0,
        atol=0.0,
    )


def test_dual_expert_gjd_fdm_inference_matches_conditional_training_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_inference as dual_expert_packed_inference_module
    from open_wam.models.common import RolloutCursor
    from open_wam.models.policy_variants.contracts import PolicyInferState
    from open_wam.models.policy_variants.dual_expert.contracts import (
        DualExpertRuntimeState,
    )
    from open_wam.models.policy_variants.dual_expert.modules import (
        DualExpertActionExpert,
    )

    pipeline, _, _, _ = _build_tiny_generalist_pipeline(DynamicsObjective.JOINT)
    pipeline.policy_variant.inference_config = _dataclass_replace(
        pipeline.policy_variant.inference_config,
        action_num_inference_steps=pipeline.policy_variant.inference_config.video_num_inference_steps,
    )
    history_video = torch.randn(1, 48, 2, 8, 8)
    history_actions = torch.randn(1, 4, 4)
    forced_actions = torch.randn(1, 4, 4)
    text_context = torch.randn(1, 5, 16)
    infer_state = PolicyInferState(
        step_index=1,
        cursor=RolloutCursor(current_start_frame=2, block_index=0, chunk_size=2),
        variant_state=DualExpertRuntimeState(
            past_clean_latents=history_video,
            past_clean_actions=history_actions,
            next_condition_frame_start=2,
            chunk_advance_frames=2,
        ),
    )
    observed_pre: list[dict[str, torch.Tensor]] = []
    observed_profiles: list[dict[str, object]] = []
    original_pre_dit = DualExpertActionExpert.pre_dit

    def spy_pre_dit(self, *args, **kwargs):
        observed_pre.append(
            {
                "action_tokens": kwargs["action_tokens"].detach().clone(),
                "timestep": kwargs["timestep"].detach().clone(),
                "context": kwargs["context"].detach().clone(),
            }
        )
        return original_pre_dit(self, *args, **kwargs)

    def fake_forward_dual_expert_packed_coupling_denoise(**kwargs):
        observed_profiles.append(dict(kwargs["attention_profile"].metadata))
        return torch.zeros_like(kwargs["noisy_video_latents"]), torch.zeros_like(
            kwargs["packed_action_pre"].tokens
        )

    monkeypatch.setattr(DualExpertActionExpert, "pre_dit", spy_pre_dit)
    monkeypatch.setattr(
        dual_expert_packed_inference_module,
        "forward_dual_expert_packed_coupling_denoise",
        fake_forward_dual_expert_packed_coupling_denoise,
    )

    output = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 2, 8, 8),
        PolicyInferContext(
            dynamics=DynamicsRolloutRequest(
                objective=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
                clean_action=forced_actions,
            ),
        ),
        infer_state=infer_state,
        text_context=text_context,
    )

    assert observed_pre
    first_pre = observed_pre[0]
    # Conditional rollout keeps one history frame and predicts one frame.
    # Packed order is [A_noisy(history,current), A_clean(history,current)].
    torch.testing.assert_close(
        first_pre["action_tokens"][:, :2], history_actions[:, -2:]
    )
    torch.testing.assert_close(
        first_pre["action_tokens"][:, 2:4], forced_actions[:, :2]
    )
    torch.testing.assert_close(
        first_pre["action_tokens"][:, 4:6], history_actions[:, -2:]
    )
    torch.testing.assert_close(
        first_pre["action_tokens"][:, 6:8], forced_actions[:, :2]
    )
    torch.testing.assert_close(
        first_pre["timestep"], torch.zeros_like(first_pre["timestep"])
    )
    torch.testing.assert_close(
        first_pre["context"], torch.zeros_like(first_pre["context"])
    )
    assert observed_profiles[0]["window_size"] == 3
    assert observed_profiles[0]["history_stream_visibility"] == "video_only"
    assert (
        observed_profiles[0]["conditional_history_policy"]
        == "previous_boundary_video_only"
    )
    assert (
        output.policy_output.aux["action_conditioning_mode"]
        == "action_conditioned_video"
    )
    assert output.policy_output.aux["cache_action_source"] == "commit_action_override"
    torch.testing.assert_close(
        output.policy_output.next_state.variant_state.past_clean_actions[:, -2:],
        forced_actions[:, :2],
    )


def test_dual_expert_gjd_idm_inference_matches_conditional_training_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_inference as dual_expert_packed_inference_module
    from open_wam.models.common import RolloutCursor
    from open_wam.models.policy_variants.contracts import PolicyInferState
    from open_wam.models.policy_variants.dual_expert.contracts import (
        DualExpertRuntimeState,
    )
    from open_wam.models.policy_variants.dual_expert.modules import (
        DualExpertActionExpert,
    )

    pipeline, _, _, _ = _build_tiny_generalist_pipeline(DynamicsObjective.JOINT)
    pipeline.policy_variant.inference_config = _dataclass_replace(
        pipeline.policy_variant.inference_config,
        action_num_inference_steps=pipeline.policy_variant.inference_config.video_num_inference_steps,
    )
    history_video = torch.randn(1, 48, 2, 8, 8)
    history_actions = torch.randn(1, 4, 4)
    clean_video = torch.randn(1, 48, 1, 8, 8)
    commit_actions = torch.randn(1, 4, 4)
    text_context = torch.randn(1, 5, 16)
    infer_state = PolicyInferState(
        step_index=1,
        cursor=RolloutCursor(current_start_frame=2, block_index=0, chunk_size=2),
        variant_state=DualExpertRuntimeState(
            past_clean_latents=history_video,
            past_clean_actions=history_actions,
            next_condition_frame_start=2,
            chunk_advance_frames=2,
        ),
    )
    observed_pre: list[dict[str, torch.Tensor]] = []
    observed_runtime: list[dict[str, torch.Tensor | dict[str, object]]] = []
    original_pre_dit = DualExpertActionExpert.pre_dit

    def spy_pre_dit(self, *args, **kwargs):
        observed_pre.append(
            {
                "context": kwargs["context"].detach().clone(),
            }
        )
        return original_pre_dit(self, *args, **kwargs)

    def fake_forward_dual_expert_packed_coupling_denoise(**kwargs):
        observed_runtime.append(
            {
                "noisy_video_latents": kwargs["noisy_video_latents"].detach().clone(),
                "clean_video_latents": kwargs["clean_video_latents"].detach().clone(),
                "noisy_video_timesteps": kwargs["noisy_video_timesteps"]
                .detach()
                .clone(),
                "metadata": dict(kwargs["attention_profile"].metadata),
            }
        )
        return torch.zeros_like(kwargs["noisy_video_latents"]), torch.zeros_like(
            kwargs["packed_action_pre"].tokens
        )

    monkeypatch.setattr(DualExpertActionExpert, "pre_dit", spy_pre_dit)
    monkeypatch.setattr(
        dual_expert_packed_inference_module,
        "forward_dual_expert_packed_coupling_denoise",
        fake_forward_dual_expert_packed_coupling_denoise,
    )

    output = pipeline.forward_infer_step_from_latents(
        clean_video,
        PolicyInferContext(
            dynamics=DynamicsRolloutRequest(
                objective=DynamicsObjective.VIDEO_CONDITIONED_ACTION,
                clean_video=clean_video,
                history_action=commit_actions,
            ),
        ),
        infer_state=infer_state,
        text_context=text_context,
    )

    assert observed_pre
    torch.testing.assert_close(
        observed_pre[0]["context"], torch.zeros_like(observed_pre[0]["context"])
    )
    first_runtime = observed_runtime[0]
    torch.testing.assert_close(
        first_runtime["noisy_video_latents"][:, :, :1], history_video[:, :, -1:]
    )
    torch.testing.assert_close(
        first_runtime["noisy_video_latents"][:, :, 1:], clean_video
    )
    torch.testing.assert_close(
        first_runtime["clean_video_latents"][:, :, :1], history_video[:, :, -1:]
    )
    torch.testing.assert_close(
        first_runtime["clean_video_latents"][:, :, 1:], clean_video
    )
    torch.testing.assert_close(
        first_runtime["noisy_video_timesteps"],
        torch.zeros_like(first_runtime["noisy_video_timesteps"]),
    )
    metadata = first_runtime["metadata"]
    assert metadata["window_size"] == 3
    assert metadata["history_stream_visibility"] == "video_only"
    assert metadata["conditional_history_policy"] == "previous_boundary_video_only"
    assert (
        output.policy_output.aux["action_conditioning_mode"]
        == "video_conditioned_action"
    )
    assert output.policy_output.aux["cache_action_source"] == "commit_action_override"
    torch.testing.assert_close(
        output.policy_output.next_state.variant_state.past_clean_actions[:, -2:],
        commit_actions[:, :2],
    )


def test_forced_joint_preserves_configured_noisy_video_condition_prob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_training as dual_expert_packed_training_module

    original_build_video_artifacts = (
        dual_expert_packed_training_module.build_video_flow_match_train_artifacts
    )
    observed_probs: list[float] = []

    def spy_build_video_artifacts(*args, **kwargs):
        observed_probs.append(float(kwargs.get("noisy_condition_prob", 0.0)))
        return original_build_video_artifacts(*args, **kwargs)

    monkeypatch.setattr(
        dual_expert_packed_training_module,
        "build_video_flow_match_train_artifacts",
        spy_build_video_artifacts,
    )

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.JOINT,
    )
    pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert observed_probs == [pytest.approx(0.5)]


def test_conditional_generalist_modes_force_clean_video_condition_prob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_training as dual_expert_packed_training_module

    original_build_video_artifacts = (
        dual_expert_packed_training_module.build_video_flow_match_train_artifacts
    )
    observed_probs: list[float] = []

    def spy_build_video_artifacts(*args, **kwargs):
        observed_probs.append(float(kwargs.get("noisy_condition_prob", 0.0)))
        return original_build_video_artifacts(*args, **kwargs)

    monkeypatch.setattr(
        dual_expert_packed_training_module,
        "build_video_flow_match_train_artifacts",
        spy_build_video_artifacts,
    )

    for mode in (
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
    ):
        pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
            mode
        )
        pipeline.forward_train_from_latents(
            video_latents, batch, text_context=text_context
        )

    assert observed_probs == [pytest.approx(0.0), pytest.approx(0.0)]


def test_forced_joint_keeps_both_losses_active() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.JOINT
    )
    output = pipeline.forward_train_from_latents(
        video_latents, batch, text_context=text_context
    )

    metrics = output.decoder_output.metrics
    assert metrics["dual_expert_generalist/joint/count"].item() == 1.0
    assert (
        metrics["dual_expert_generalist/action_conditioned_video/count"].item() == 0.0
    )
    assert (
        metrics["dual_expert_generalist/video_conditioned_action/count"].item() == 0.0
    )
    assert metrics["dual_expert_generalist/action_loss_active"].item() == 1.0
    assert metrics["dual_expert_generalist/latent_loss_active"].item() == 1.0
    assert output.policy_output.aux["dual_expert_generalist_text_dropped"] is False
    assert "dual_expert_generalist/joint/action_denoised_mse_sum" in metrics
    assert "dual_expert_generalist/joint/action_mse_sum" in metrics
    assert torch.equal(
        metrics["dual_expert_generalist/joint/action_mse_sum"],
        metrics["dual_expert_generalist/joint/action_denoised_mse_sum"],
    )
    assert metrics["weighted_action_diffusion_loss"].item() > 0.0
    assert metrics["weighted_video_diffusion_loss"].item() > 0.0
    assert output.policy_output.aux["sampled_window_size"] >= 4


def test_generalist_training_rejects_multi_sample_batches() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.JOINT
    )
    multi_batch = _dataclass_replace(batch, actions=batch.actions.repeat(2, 1, 1))

    with pytest.raises(ValueError, match="rank-local train_batch_size=1"):
        pipeline.forward_train_from_latents(
            video_latents.repeat(2, 1, 1, 1, 1),
            multi_batch,
            text_context=text_context.repeat(2, 1, 1),
        )


def test_dual_expert_generalist_conditional_local_window_sees_one_previous_video_frame_only() -> (
    None
):
    profile = build_dual_expert_packed_coupling_attention_profile(
        num_video_frames=8,
        video_tokens_per_frame=1,
        num_action_frames=8,
        action_tokens_per_frame=1,
        chunk_size_frames=1,
        attention_window_size=3,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        device=torch.device("cpu"),
        build_dense_masks=True,
        history_stream_visibility=HistoryStreamVisibility.VIDEO_ONLY,
    )
    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    latent_tokens = 8
    action_tokens = 8
    current_video_noisy_frame4 = 4
    current_video_clean_frame4 = latent_tokens + 4
    current_action_noisy_frame4 = 2 * latent_tokens + 4
    previous_video_clean_frame3 = latent_tokens + 3
    older_video_clean_frame2 = latent_tokens + 2
    previous_action_clean_frame3 = 2 * latent_tokens + action_tokens + 3
    current_action_clean_frame4 = 2 * latent_tokens + action_tokens + 4

    assert mask[current_video_noisy_frame4, previous_video_clean_frame3]
    assert not mask[current_video_noisy_frame4, older_video_clean_frame2]
    assert not mask[current_video_noisy_frame4, previous_action_clean_frame3]
    assert mask[current_action_noisy_frame4, previous_video_clean_frame3]
    assert not mask[current_action_noisy_frame4, older_video_clean_frame2]
    assert not mask[current_action_noisy_frame4, previous_action_clean_frame3]
    assert not mask[current_video_noisy_frame4, current_video_clean_frame4]
    assert not mask[current_action_noisy_frame4, current_action_clean_frame4]


def test_forced_action_conditioned_video_zeros_action_loss() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.ACTION_CONDITIONED_VIDEO
    )
    output = pipeline.forward_train_from_latents(
        video_latents, batch, text_context=text_context
    )

    metrics = output.decoder_output.metrics
    assert (
        metrics["dual_expert_generalist/action_conditioned_video/count"].item() == 1.0
    )
    assert metrics["dual_expert_generalist/joint/count"].item() == 0.0
    assert (
        metrics["dual_expert_generalist/video_conditioned_action/count"].item() == 0.0
    )
    # Action loss is fully masked off; video loss carries the gradient.
    assert metrics["dual_expert_generalist/action_loss_active"].item() == 0.0
    assert metrics["dual_expert_generalist/latent_loss_active"].item() == 1.0
    assert output.policy_output.aux["dual_expert_generalist_text_dropped"] is True
    assert 1 <= output.policy_output.aux["sampled_chunk_size"] <= 2
    assert output.policy_output.aux["sampled_window_size"] == 3
    assert metrics["weighted_action_diffusion_loss"].item() == pytest.approx(
        0.0, abs=1e-6
    )
    assert metrics["weighted_video_diffusion_loss"].item() > 0.0


def test_conditional_dynamics_rejects_false_drop_text_override() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.ACTION_CONDITIONED_VIDEO
    )
    batch.extra["metadata"][DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY] = False
    with pytest.raises(ValueError, match="always removes task text"):
        pipeline.forward_train_from_latents(
            video_latents,
            batch,
            text_context=text_context,
        )


def test_action_conditioned_video_threads_dropped_text_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_training as dual_expert_packed_training_module
    from open_wam.models.policy_variants.dual_expert.modules import (
        DualExpertActionExpert,
    )

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.ACTION_CONDITIONED_VIDEO
    )
    assert torch.count_nonzero(text_context) > 0

    action_pre_dit_contexts: list[torch.Tensor] = []
    packed_runtime_contexts: list[torch.Tensor] = []
    original_pre_dit = DualExpertActionExpert.pre_dit

    def spy_pre_dit(self, *args, **kwargs):
        action_pre_dit_contexts.append(kwargs["context"].detach().clone())
        return original_pre_dit(self, *args, **kwargs)

    def fake_forward_dual_expert_packed_coupling_denoise(**kwargs):
        packed_runtime_contexts.append(kwargs["text_context"].detach().clone())
        return torch.zeros_like(kwargs["noisy_video_latents"]), torch.zeros_like(
            kwargs["packed_action_pre"].tokens
        )

    monkeypatch.setattr(DualExpertActionExpert, "pre_dit", spy_pre_dit)
    monkeypatch.setattr(
        dual_expert_packed_training_module,
        "forward_dual_expert_packed_coupling_denoise",
        fake_forward_dual_expert_packed_coupling_denoise,
    )

    output = pipeline.forward_train_from_latents(
        video_latents, batch, text_context=text_context
    )

    expected_text = torch.zeros_like(text_context)
    assert output.policy_output.aux["dual_expert_generalist_text_dropped"] is True
    assert len(action_pre_dit_contexts) == 1
    assert len(packed_runtime_contexts) == 1
    assert torch.equal(action_pre_dit_contexts[0], expected_text)
    assert torch.equal(packed_runtime_contexts[0], expected_text)


def test_dual_expert_per_chunk_additive_proprio_threads_hidden_context_to_packed_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_training as dual_expert_packed_training_module
    from open_wam.models.policy_variants.dual_expert.modules import (
        DualExpertActionExpert,
    )

    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.JOINT,
        action_hidden_size=16,
    )
    object.__setattr__(
        pipeline.policy_variant.config,
        "proprio_context_mode",
        ProprioContextMode.PER_CHUNK_ADDITIVE,
    )
    pipeline.visual_tower.configure_policy_conditioning(
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        dynamics_mode_context_enabled=False,
    )
    batch.extra["proprio_context_state"] = torch.randn(1, 4, 4)
    batch.extra["proprio_context_state_mask"] = torch.ones(1, 4, 4)

    action_hidden_contexts: list[torch.Tensor | None] = []
    video_hidden_contexts: list[torch.Tensor | None] = []
    original_pre_dit = DualExpertActionExpert.pre_dit

    def spy_pre_dit(self, *args, **kwargs):
        hidden_context = kwargs.get("hidden_context")
        action_hidden_contexts.append(
            None if hidden_context is None else hidden_context.detach().clone()
        )
        return original_pre_dit(self, *args, **kwargs)

    def fake_forward_dual_expert_packed_coupling_denoise(**kwargs):
        video_hidden_context = kwargs.get("video_hidden_context")
        video_hidden_contexts.append(
            None
            if video_hidden_context is None
            else video_hidden_context.detach().clone()
        )
        return torch.zeros_like(kwargs["noisy_video_latents"]), torch.zeros_like(
            kwargs["packed_action_pre"].tokens
        )

    monkeypatch.setattr(DualExpertActionExpert, "pre_dit", spy_pre_dit)
    monkeypatch.setattr(
        dual_expert_packed_training_module,
        "forward_dual_expert_packed_coupling_denoise",
        fake_forward_dual_expert_packed_coupling_denoise,
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


def test_dual_expert_legacy_prefix_contract_prepends_video_only_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_training as dual_expert_packed_training_module
    from open_wam.configs import (
        ActionSchemaConfig,
        DualExpertActionDecoderConfig,
        ExperimentConfig,
        InferenceConfig,
        RobotWinDataConfig,
    )
    from open_wam.configs import (
        DualExpertPolicyConfig as TopLevelDualExpertPolicyConfig,
    )
    from open_wam.models.policy_variants.contracts import PolicyTrainBatch
    from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
    from open_wam.pipelines import build_variant_pipeline_from_config

    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
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
        policy_variant=TopLevelDualExpertPolicyConfig(
            hidden_size=32,
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            video_prefix_frames=1,
            num_action_layers=1,
            sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
            history_stream_visibility=HistoryStreamVisibility.VIDEO_ONLY,
            use_condition_latents=True,
            require_condition_latents=True,
            noisy_video_condition_prob=0.0,
            joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
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

    def fake_forward_dual_expert_packed_coupling_denoise(**kwargs):
        observed["noisy_video_shape"] = tuple(kwargs["noisy_video_latents"].shape)
        observed["clean_video_shape"] = tuple(kwargs["clean_video_latents"].shape)
        observed["packed_action_shape"] = tuple(
            kwargs["packed_action_pre"].tokens.shape
        )
        observed["prefix_condition_frames"] = kwargs["attention_profile"].metadata[
            "prefix_condition_frames"
        ]
        observed["conditional_history_policy"] = kwargs["attention_profile"].metadata[
            "conditional_history_policy"
        ]
        observed["video_hidden_context"] = kwargs["video_hidden_context"]
        observed["frame_start"] = kwargs["frame_start"]
        return torch.zeros_like(kwargs["noisy_video_latents"]), torch.zeros_like(
            kwargs["packed_action_pre"].tokens
        )

    monkeypatch.setattr(
        dual_expert_packed_training_module,
        "forward_dual_expert_packed_coupling_denoise",
        fake_forward_dual_expert_packed_coupling_denoise,
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
    assert observed["conditional_history_policy"] == "none"
    assert observed["video_hidden_context"] is None
    assert observed["frame_start"] == -1
    assert (
        output.policy_output.aux["video_condition_source"] == "condition_latents_prefix"
    )
    assert output.decoder_output.aux["predicted_latents"].shape == (1, 48, 5, 8, 8)


def test_dual_expert_target_only_fdm_keeps_t0_inside_the_model_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import open_wam.models.policy_variants.dual_expert.packed_training as dual_expert_packed_training_module
    from open_wam.configs import (
        ActionSchemaConfig,
        DualExpertActionDecoderConfig,
        DualExpertActionExpertInitMode,
        DynamicsRouteConfig,
        DynamicsRoutingConfig,
        ExperimentConfig,
        InferenceConfig,
        RobotWinDataConfig,
    )
    from open_wam.configs import (
        DualExpertPolicyConfig as TopLevelDualExpertPolicyConfig,
    )
    from open_wam.models.policy_variants.contracts import PolicyTrainBatch
    from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
    from open_wam.pipelines import build_variant_pipeline_from_config

    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=6,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=6, state_dim=4, state_horizon=1
            ),
            dynamics_routing=DynamicsRoutingConfig(
                routes=(
                    DynamicsRouteConfig(
                        source="real_demo",
                        mode=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
                        weight=1.0,
                    ),
                )
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
        policy_variant=TopLevelDualExpertPolicyConfig(
            hidden_size=32,
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
            video_prefix_frames=1,
            num_action_layers=1,
            action_expert_init_mode=DualExpertActionExpertInitMode.VIDEO_WEIGHT_COPY,
            sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
            history_stream_visibility=HistoryStreamVisibility.VIDEO_ONLY,
            use_condition_latents=True,
            require_condition_latents=True,
            noisy_video_condition_prob=0.0,
            joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=6
        ),
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
    video_latents = torch.randn(1, 48, 6, 4, 4)
    metadata = _dynamics_sample_metadata(
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        frame_count=6,
        drop_text=True,
    )
    metadata["sampled_chunk_size"] = 2
    proprio_frames = (
        torch.arange(6, dtype=torch.float32)
        .reshape(1, 6, 1)
        .expand(-1, -1, 4)
        .contiguous()
    )
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 6, 4),
        state=torch.randn(1, 4),
        extra={
            "metadata": metadata,
            "proprio_context_frames": proprio_frames,
            "proprio_context_frames_mask": torch.ones(1, 6, 4),
        },
    )
    observed: dict[str, object] = {}

    original_project = (
        dual_expert_packed_training_module.project_hidden_proprio_context_to_frames
    )

    def capture_aligned_proprio(*args, **kwargs):
        aligned = original_project(*args, **kwargs)
        observed["aligned_proprio"] = aligned.detach().clone()
        return aligned

    monkeypatch.setattr(
        dual_expert_packed_training_module,
        "project_hidden_proprio_context_to_frames",
        capture_aligned_proprio,
    )

    def fake_forward_dual_expert_packed_coupling_denoise(**kwargs):
        observed["prefix_condition_frames"] = kwargs["attention_profile"].metadata[
            "prefix_condition_frames"
        ]
        return torch.zeros_like(kwargs["noisy_video_latents"]), torch.zeros_like(
            kwargs["packed_action_pre"].tokens
        )

    monkeypatch.setattr(
        dual_expert_packed_training_module,
        "forward_dual_expert_packed_coupling_denoise",
        fake_forward_dual_expert_packed_coupling_denoise,
    )

    output = pipeline.forward_train_from_latents(
        video_latents,
        batch,
        text_context=torch.randn(1, 5, 16),
    )

    assert output.policy_output.decoder_artifacts is not None
    train_artifacts = output.policy_output.decoder_artifacts.require(
        contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
        payload_type=DualExpertTrainArtifacts,
    )
    assert "dual_expert_train_artifacts" not in output.policy_output.aux
    mask = train_artifacts.video.future_loss_mask.flatten()
    expected = torch.tensor([0, 1, 1, 1, 1, 1], device=mask.device, dtype=mask.dtype)

    torch.testing.assert_close(mask, expected)
    torch.testing.assert_close(
        observed["aligned_proprio"],
        torch.tensor([0, 0, 0, 2, 2, 4], dtype=torch.float32)
        .reshape(1, 6, 1)
        .expand(-1, -1, 4),
    )
    assert observed["prefix_condition_frames"] == 0
    assert (
        output.policy_output.aux["video_condition_source"]
        == "video_latents_target_only"
    )
    assert (
        output.policy_output.aux["dual_expert_generalist_training_mode"]
        == "action_conditioned_video"
    )
    assert (
        output.policy_output.aux["conditional_history_policy"]
        == "previous_boundary_video_only"
    )
    assert (
        output.decoder_output.metrics[
            "dual_expert_generalist/action_loss_active"
        ].item()
        == 0.0
    )


def test_forced_video_conditioned_action_zeros_video_loss() -> None:
    pipeline, batch, video_latents, text_context = _build_tiny_generalist_pipeline(
        DynamicsObjective.VIDEO_CONDITIONED_ACTION
    )
    output = pipeline.forward_train_from_latents(
        video_latents, batch, text_context=text_context
    )

    metrics = output.decoder_output.metrics
    assert (
        metrics["dual_expert_generalist/video_conditioned_action/count"].item() == 1.0
    )
    assert metrics["dual_expert_generalist/joint/count"].item() == 0.0
    assert (
        metrics["dual_expert_generalist/action_conditioned_video/count"].item() == 0.0
    )
    # Video loss is fully masked off; action loss carries the gradient.
    assert metrics["dual_expert_generalist/latent_loss_active"].item() == 0.0
    assert metrics["dual_expert_generalist/action_loss_active"].item() == 1.0
    assert output.policy_output.aux["dual_expert_generalist_text_dropped"] is True
    assert (
        output.policy_output.aux["conditional_history_policy"]
        == "previous_boundary_video_only"
    )
    assert 1 <= output.policy_output.aux["sampled_chunk_size"] <= 2
    assert output.policy_output.aux["sampled_window_size"] == 3
    assert metrics["weighted_video_diffusion_loss"].item() == pytest.approx(
        0.0, abs=1e-6
    )
    assert metrics["weighted_action_diffusion_loss"].item() > 0.0


def test_no_generalist_metrics_when_probs_unset() -> None:
    """Sanity: existing 6-mode path emits no dual_expert_generalist/* metrics."""

    from open_wam.configs import (
        ActionSchemaConfig,
        DualExpertActionDecoderConfig,
        ExperimentConfig,
        InferenceConfig,
        RobotWinDataConfig,
        TrainingConfig,
    )
    from open_wam.configs import (
        DualExpertPolicyConfig as TopLevelDualExpertPolicyConfig,
    )
    from open_wam.models.policy_variants.contracts import PolicyTrainBatch
    from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
    from open_wam.pipelines import build_variant_pipeline_from_config

    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
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
        policy_variant=TopLevelDualExpertPolicyConfig(
            hidden_size=32,
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
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

    output = pipeline.forward_train_from_latents(
        video_latents, batch, text_context=text_context
    )

    metrics = output.decoder_output.metrics
    for key in metrics:
        assert not key.startswith("dual_expert_generalist/"), (
            f"dual_expert_generalist metrics should not appear when probs are unset, got {key}"
        )
    # And the aux key is None (not the string).
    assert output.policy_output.aux.get("dual_expert_generalist_training_mode") is None
