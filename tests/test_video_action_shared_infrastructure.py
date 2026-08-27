from __future__ import annotations

import numpy as np
import pytest
import torch

from open_wam.configs import (
    ActionDecoderName,
    ActionSchemaConfig,
    CausalVideoProgram,
    DualExpertActionDecoderConfig,
    DynamicsObjective,
    DynamicsRouteConfig,
    DynamicsRoutingConfig,
    DynamicsSource,
    ExperimentConfig,
    InferenceConfig,
    ParallelStreamActionDecoderConfig,
    ProprioContextMode,
    RobotWinDataConfig,
    TrainingConfig,
    VideoActionProgram,
    VideoActionSequenceContract,
    parse_action_decoder_config,
)
from open_wam.configs.policy_contracts import (
    CausalVideoPredictionPolicyConfig,
    PolicyVariantConfig,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
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
    apply_dynamics_training_plan,
    dynamics_objective_attention_window_size,
    dynamics_objective_rollout_chunk_size,
    resolve_dynamics_objective_semantics,
    resolve_dynamics_rollout_geometry,
    resolve_dynamics_rollout_plan,
    resolve_dynamics_training_plan,
    sample_conditioning_mode,
)
from open_wam.models.common.flow_matching import FlowMatchScheduler
from open_wam.models.common.flow_noise_plan import (
    sample_coupled_timestep_values,
    sample_timestep_values,
)
from open_wam.models.common.metric_rollups import add_dynamics_objective_metrics
from open_wam.models.common.modality_slots import (
    force_clean_noisy_slot,
    zero_condition_slot,
    zero_loss_mask_like,
)
from open_wam.models.common.proprio_conditioning import (
    HiddenProprioContext,
    ProprioContextGranularity,
    prepend_hidden_proprio_context,
    project_hidden_proprio_context_to_frames,
    resolve_hidden_proprio_context,
)
from open_wam.models.common.rollout_history import build_executed_action_history_tensor
from open_wam.models.common.rollout_startup import (
    build_strict_action_context_mask,
    require_strict_startup_generation_frame,
    resolve_strict_startup_plan,
    strict_startup_conditioning_frame_index,
)
from open_wam.models.common.video_geometry import (
    slice_token_grid_frames,
    video_token_grid_from_latent_shape,
)
from open_wam.models.policy_variants import (
    DynamicsRolloutRequest,
    PolicyPipelineRequirements,
    PolicyTrainBatch,
)
from open_wam.models.policy_variants.dual_expert.conditioning import (
    DualExpertConditioning,
)
from open_wam.models.policy_variants.dual_expert.variant import (
    DualExpertPolicyVariant,
)
from open_wam.models.policy_variants.parallel_stream.conditioning import (
    ParallelStreamConditioning,
)
from open_wam.models.policy_variants.parallel_stream.variant import (
    ParallelStreamPolicyVariant,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.frontend import SharedVideoFrontend
from open_wam.pipelines import build_variant_pipeline_from_config


@pytest.mark.unit
@pytest.mark.parametrize(
    ("policy_config", "expected_decoder"),
    [
        (
            DualExpertPolicyConfig(program=VideoActionProgram.JOINT),
            ActionDecoderName.DUAL_EXPERT,
        ),
        (
            ParallelStreamPolicyConfig(program=VideoActionProgram.JOINT),
            ActionDecoderName.PARALLEL_STREAM,
        ),
        (
            CausalVideoPredictionPolicyConfig(program=CausalVideoProgram.PREFIX_SUFFIX),
            ActionDecoderName.VIDEO_ONLY,
        ),
    ],
)
def test_policy_configs_declare_their_default_decoder_without_parser_branching(
    policy_config: PolicyVariantConfig,
    expected_decoder: ActionDecoderName,
) -> None:
    data = RobotWinDataConfig()

    decoder = parse_action_decoder_config({}, policy_config, data)

    assert decoder.name is expected_decoder


@pytest.mark.unit
@pytest.mark.parametrize("program", tuple(VideoActionProgram))
def test_video_action_backends_expose_equivalent_pipeline_requirements(
    program: VideoActionProgram,
) -> None:
    """Topology substitution must preserve every shared assembly requirement."""

    backbone = SharedVideoTransformerConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        load_reference_core_weights=False,
    )
    training = TrainingConfig(chunk_size=2, window_size=8)
    inference = InferenceConfig(frame_chunk_size=2)
    shared_policy_fields = {
        "program": program,
        "hidden_size": 32,
        "proprio_context_mode": ProprioContextMode.PER_CHUNK_ADDITIVE,
        "generalist_mode_text_token": (
            program is VideoActionProgram.GENERALIST_JOINT_DENOISING
        ),
    }
    dual = DualExpertPolicyVariant(
        config=DualExpertPolicyConfig(
            **shared_policy_fields,
            num_action_layers=1,
        ),
        backbone_config=backbone,
        training_config=training,
        inference_config=inference,
        action_dim=4,
        action_horizon=4,
    )
    parallel = ParallelStreamPolicyVariant(
        config=ParallelStreamPolicyConfig(
            **shared_policy_fields,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        backbone_config=backbone,
        training_config=training,
        inference_config=inference,
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )

    kwargs = {
        "default_action_dim": 4,
        "default_action_horizon": 4,
        "default_state_dim": 3,
    }
    assert dual.pipeline_requirements(**kwargs) == parallel.pipeline_requirements(
        **kwargs
    )


@pytest.mark.unit
def test_pipeline_requirements_validate_source_action_shapes_without_backend_names() -> (
    None
):
    requirements = PolicyPipelineRequirements(
        action_dim=30,
        action_horizon=16,
        state_dim=8,
        accepted_source_action_shapes=((7, 16), (30, 16)),
    )

    requirements.validate_source_action_shape(action_dim=7, action_horizon=16)
    requirements.validate_source_action_shape(action_dim=30, action_horizon=16)
    with pytest.raises(ValueError, match="Dataset action geometry is not accepted"):
        requirements.validate_source_action_shape(action_dim=10, action_horizon=16)


@pytest.mark.unit
@pytest.mark.parametrize("program", tuple(VideoActionProgram))
def test_public_factory_preserves_requirements_when_video_action_backend_changes(
    program: VideoActionProgram,
) -> None:
    """Backend substitution must preserve the public pipeline contract."""

    data = RobotWinDataConfig(
        num_frames=2,
        action_schema=ActionSchemaConfig(
            action_dim=4,
            action_horizon=4,
            state_dim=3,
            state_horizon=1,
        ),
    )
    backbone = SharedVideoTransformerConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        load_reference_core_weights=False,
    )
    training = TrainingConfig(chunk_size=2, window_size=8)
    inference = InferenceConfig(frame_chunk_size=2)
    shared_policy_fields = {
        "program": program,
        "hidden_size": 32,
        "proprio_context_mode": ProprioContextMode.PER_CHUNK_ADDITIVE,
        "generalist_mode_text_token": (
            program is VideoActionProgram.GENERALIST_JOINT_DENOISING
        ),
    }
    configs = (
        ExperimentConfig(
            data=data,
            backbone=backbone,
            policy_variant=DualExpertPolicyConfig(
                **shared_policy_fields,
                num_action_layers=1,
            ),
            action_decoder=DualExpertActionDecoderConfig(
                hidden_size=32,
                action_dim=4,
                action_horizon=4,
            ),
            training=training,
            inference=inference,
        ),
        ExperimentConfig(
            data=data,
            backbone=backbone,
            policy_variant=ParallelStreamPolicyConfig(
                **shared_policy_fields,
                frame_chunk_size=2,
                action_per_frame=2,
                attn_window=8,
            ),
            action_decoder=ParallelStreamActionDecoderConfig(
                hidden_size=32,
                action_dim=4,
                action_horizon=4,
            ),
            training=training,
            inference=inference,
        ),
    )

    pipelines = tuple(build_variant_pipeline_from_config(config) for config in configs)
    requirements = tuple(
        pipeline.policy_variant.pipeline_requirements(
            default_action_dim=4,
            default_action_horizon=4,
            default_state_dim=3,
        )
        for pipeline in pipelines
    )

    assert requirements[0] == requirements[1]
    tower_state_shapes = tuple(
        {
            key: tuple(value.shape)
            for key, value in pipeline.visual_tower.state_dict().items()
            if not key.startswith("core.blocks.")
        }
        for pipeline in pipelines
    )
    assert tower_state_shapes[0] == tower_state_shapes[1]
    for pipeline in pipelines:
        requirements[0].validate_visual_tower(
            action_dim=pipeline.visual_tower.action_dim,
            state_dim=pipeline.visual_tower.state_dim,
        )


@pytest.mark.unit
def test_shared_route_contract_aggregates_architecture_independent_modes() -> None:
    mixture = DynamicsRoutingConfig(
        routes=(
            DynamicsRouteConfig(source="real_demo", mode="joint", weight=6),
            DynamicsRouteConfig(
                source="real_demo", mode="action_conditioned_video", weight=1
            ),
            DynamicsRouteConfig(
                source="counterfactual_dynamics",
                mode="action_conditioned_video",
                weight=1,
            ),
            DynamicsRouteConfig(
                source="real_demo", mode="video_conditioned_action", weight=2
            ),
        )
    )
    probabilities = mixture.mode_probabilities()

    assert probabilities[DynamicsObjective.JOINT] == pytest.approx(0.6)
    assert probabilities[DynamicsObjective.ACTION_CONDITIONED_VIDEO] == pytest.approx(
        0.2
    )
    assert probabilities[DynamicsObjective.VIDEO_CONDITIONED_ACTION] == pytest.approx(
        0.2
    )
    assert mixture.routes[0].source is DynamicsSource.REAL_DEMO

    with pytest.raises(ValueError, match="must not repeat"):
        DynamicsRoutingConfig(
            routes=(
                DynamicsRouteConfig(source="real_demo", mode="joint", weight=1),
                DynamicsRouteConfig(source="real_demo", mode="joint", weight=1),
            )
        )


@pytest.mark.unit
def test_shared_conditioning_mode_sampling_broadcasts_rank_zero_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        {mode: 1.0 for mode in DynamicsObjective},
        enum_cls=DynamicsObjective,
        device=torch.device("cpu"),
        error_label="test mode",
    )

    assert mode == tuple(DynamicsObjective)[2]


@pytest.mark.unit
def test_shared_dynamics_objective_semantics_define_all_three_modes() -> None:
    joint = resolve_dynamics_objective_semantics(DynamicsObjective.JOINT)
    assert joint.force_clean_video_condition is False
    assert joint.action_loss_active is True
    assert joint.video_loss_active is True
    assert joint.drop_text_conditioning is False

    fdm = resolve_dynamics_objective_semantics(
        DynamicsObjective.ACTION_CONDITIONED_VIDEO
    )
    assert fdm.clean_action_noisy_slot is True
    assert fdm.action_loss_active is False
    assert fdm.video_loss_active is True
    assert fdm.drop_text_conditioning is True
    assert fdm.force_clean_video_condition is True
    assert fdm.history_frame_count == 1
    assert fdm.rollout_chunk_size_frames(fallback_chunk_size=4) == 1
    assert fdm.attention_window_size(fallback_window_size=30) == 3

    idm = resolve_dynamics_objective_semantics(
        DynamicsObjective.VIDEO_CONDITIONED_ACTION
    )
    assert idm.clean_video_noisy_slot is True
    assert idm.action_loss_active is True
    assert idm.video_loss_active is False
    assert (
        dynamics_objective_rollout_chunk_size(
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            fallback_chunk_size=4,
        )
        == 1
    )
    assert (
        dynamics_objective_attention_window_size(
            DynamicsObjective.JOINT,
            fallback_window_size=30,
        )
        == 30
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("objective", "chunk_size", "window_size", "history_visibility", "history_policy"),
    [
        (
            DynamicsObjective.JOINT,
            4,
            30,
            "video_queries_video_only",
            None,
        ),
        (
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            1,
            3,
            "video_only",
            DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
        ),
        (
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            1,
            3,
            "video_only",
            DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
        ),
    ],
)
def test_shared_rollout_geometry_is_complete_and_architecture_neutral(
    objective: DynamicsObjective,
    chunk_size: int,
    window_size: int,
    history_visibility: str,
    history_policy: str | None,
) -> None:
    geometry = resolve_dynamics_rollout_geometry(
        objective,
        fallback_frame_chunk_size=4,
        fallback_attention_window_size=30,
        fallback_history_stream_visibility="video_queries_video_only",
    )

    assert geometry.frame_chunk_size == chunk_size
    assert geometry.attention_window_size == window_size
    assert geometry.history_stream_visibility.value == history_visibility
    assert geometry.conditional_history_policy == history_policy


@pytest.mark.unit
def test_hidden_proprio_contract_is_identical_across_policy_architectures() -> None:
    batch = PolicyTrainBatch(
        actions=torch.zeros(1, 4, 7),
        extra={
            "proprio_context_state": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            "proprio_context_state_mask": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        },
    )
    dual = DualExpertConditioning(
        DualExpertPolicyConfig(
            program=VideoActionProgram.JOINT,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        )
    )
    parallel = ParallelStreamConditioning(
        ParallelStreamPolicyConfig(
            program=VideoActionProgram.JOINT,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        )
    )

    dual_context = dual.resolve_train_hidden_proprio_context(batch)
    parallel_context = parallel.resolve_train_hidden_proprio_context(
        batch,
        label="parallel-stream training",
    )

    assert dual_context is not None
    assert parallel_context is not None
    torch.testing.assert_close(dual_context.values, parallel_context.values)
    assert dual_context.granularity is ProprioContextGranularity.CHUNK
    assert parallel_context.granularity is ProprioContextGranularity.CHUNK

    strict_batch = PolicyTrainBatch(
        actions=batch.actions,
        extra={"proprio_context_state": torch.zeros(1, 1, 2)},
    )
    strict_dual = DualExpertConditioning(
        DualExpertPolicyConfig(
            program=VideoActionProgram.JOINT,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            sequence_contract=(
                VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
            ),
        )
    )
    strict_parallel = ParallelStreamConditioning(
        ParallelStreamPolicyConfig(
            program=VideoActionProgram.JOINT,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            sequence_contract=(
                VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
            ),
        )
    )
    for resolve in (
        lambda: strict_dual.resolve_train_hidden_proprio_context(strict_batch),
        lambda: strict_parallel.resolve_train_hidden_proprio_context(
            strict_batch,
            label="parallel-stream training",
        ),
    ):
        with pytest.raises(
            ValueError,
            match="requires frame-level `proprio_context_frames`",
        ):
            resolve()


@pytest.mark.unit
def test_shared_hidden_proprio_resolver_preserves_frame_granularity() -> None:
    context = resolve_hidden_proprio_context(
        {
            "proprio_context_frames": torch.tensor([[[1.0], [2.0]]]),
            "proprio_context_frames_mask": torch.tensor([[[0.0], [1.0]]]),
        },
        require_frame_aligned=True,
        label="test",
    )

    assert context.granularity is ProprioContextGranularity.FRAME
    assert context.values.squeeze(-1).tolist() == [[0.0, 2.0]]

    with pytest.raises(TypeError, match="proprio_context_frames.*must be a tensor"):
        resolve_hidden_proprio_context(
            {"proprio_context_frames": [[1.0], [2.0]]},
            require_frame_aligned=False,
            label="test",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("granularity", "values", "expected"),
    [
        (
            ProprioContextGranularity.FRAME,
            [9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [9.0, 9.0, 1.0, 1.0, 3.0, 3.0, 5.0],
        ),
        (
            ProprioContextGranularity.CHUNK,
            [9.0, 1.0, 3.0, 5.0],
            [9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0],
        ),
    ],
)
def test_shared_hidden_proprio_projection_owns_shifted_chunk_geometry(
    granularity: ProprioContextGranularity,
    values: list[float],
    expected: list[float],
) -> None:
    projected = project_hidden_proprio_context_to_frames(
        HiddenProprioContext(
            values=torch.tensor(values).reshape(1, -1, 1),
            granularity=granularity,
        ),
        num_frames=7,
        chunk_size=2,
        chunk_origin_frame=3,
        prefix_frames=1,
    )

    assert projected.squeeze(-1).tolist() == [expected]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("granularity", "values", "expected"),
    [
        (ProprioContextGranularity.FRAME, [1.0, 2.0, 3.0], [9.0, 1.0, 2.0]),
        (ProprioContextGranularity.CHUNK, [1.0, 3.0], [9.0, 1.0, 3.0]),
    ],
)
def test_shared_hidden_proprio_prefix_preserves_source_granularity(
    granularity: ProprioContextGranularity,
    values: list[float],
    expected: list[float],
) -> None:
    context = prepend_hidden_proprio_context(
        HiddenProprioContext(
            values=torch.tensor(values).reshape(1, -1, 1),
            granularity=granularity,
        ),
        prefix_state=torch.tensor([[[7.0], [9.0]]]),
        target_frame_count=2,
        label="test prefix",
    )

    assert context.granularity is granularity
    assert context.values.squeeze(-1).tolist() == [expected]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("program", "objective"),
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
def test_fixed_dynamics_programs_resolve_the_shared_rollout_contract(
    program: VideoActionProgram,
    objective: DynamicsObjective,
) -> None:
    clean_action = torch.zeros(1, 4, 7)
    clean_video = torch.zeros(1, 48, 1, 4, 4)
    request = DynamicsRolloutRequest(
        clean_action=(
            clean_action
            if objective == DynamicsObjective.ACTION_CONDITIONED_VIDEO
            else None
        ),
        clean_video=(
            clean_video
            if objective == DynamicsObjective.VIDEO_CONDITIONED_ACTION
            else None
        ),
    )

    plan = resolve_dynamics_rollout_plan(
        program=program,
        request=request,
    ).require_generation_inputs()

    assert plan.objective == objective
    assert plan.semantics.drop_text_conditioning is True
    assert plan.semantics.rollout_chunk_size_frames(fallback_chunk_size=4) == 1
    if objective == DynamicsObjective.ACTION_CONDITIONED_VIDEO:
        assert plan.history_action is clean_action


@pytest.mark.unit
def test_gjd_rollout_request_uses_the_same_fixed_dynamics_plan() -> None:
    clean_action = torch.randn(1, 4, 7)
    request = DynamicsRolloutRequest(
        objective=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        clean_action=clean_action,
    )

    gjd = resolve_dynamics_rollout_plan(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        request=request,
    )
    fixed = resolve_dynamics_rollout_plan(
        program=VideoActionProgram.FORWARD_DYNAMICS,
        request=DynamicsRolloutRequest(clean_action=clean_action),
    )

    assert gjd == fixed


@pytest.mark.unit
def test_dynamics_rollout_request_rejects_backend_shaped_or_wrong_modality_inputs() -> (
    None
):
    with pytest.raises(ValueError, match="clean_action.*3 dimensions"):
        DynamicsRolloutRequest(clean_action=torch.zeros(1, 7, 1, 4, 1))

    with pytest.raises(ValueError, match="clean_video.*only valid"):
        resolve_dynamics_rollout_plan(
            program=VideoActionProgram.FORWARD_DYNAMICS,
            request=DynamicsRolloutRequest(
                clean_video=torch.zeros(1, 48, 1, 4, 4),
            ),
        )

    with pytest.raises(ValueError, match="requires `clean_action`"):
        resolve_dynamics_rollout_plan(
            program=VideoActionProgram.FORWARD_DYNAMICS,
        ).require_generation_inputs()


def _target_only_metadata(
    objective: DynamicsObjective,
    *,
    drop_text: bool = True,
) -> SampleConstructionMetadata:
    metadata = SampleConstructionMetadata.from_mapping(
        {
            DYNAMICS_ROUTING_MODE_METADATA_KEY: objective.value,
            DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY: drop_text,
            DYNAMICS_ROUTING_SOURCE_METADATA_KEY: "real_demo",
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
    assert metadata is not None
    return metadata


@pytest.mark.unit
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
def test_fixed_and_gjd_programs_compile_the_same_training_plan(
    objective: DynamicsObjective,
    fixed_program: VideoActionProgram,
) -> None:
    metadata = _target_only_metadata(objective)

    gjd = resolve_dynamics_training_plan(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        sample_metadata=metadata,
        device=torch.device("cpu"),
    )
    fixed = resolve_dynamics_training_plan(
        program=fixed_program,
        sample_metadata=metadata,
        device=torch.device("cpu"),
    )

    assert gjd == fixed


@pytest.mark.unit
@pytest.mark.parametrize(
    "objective",
    [
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
    ],
)
def test_shared_conditional_training_plan_owns_modality_slots_and_losses(
    objective: DynamicsObjective,
) -> None:
    plan = resolve_dynamics_training_plan(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        sample_metadata=_target_only_metadata(objective),
        device=torch.device("cpu"),
    )
    assert plan is not None

    clean_video = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    noisy_video = clean_video + 10
    video_targets = clean_video + 20
    video_timesteps = torch.full((1, 3), 0.7)
    video_loss_mask = torch.ones_like(clean_video)
    clean_action = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    noisy_action = clean_action + 30
    action_targets = clean_action + 40
    action_timesteps = torch.full((1, 4), 0.5)
    action_loss_mask = torch.ones_like(clean_action)
    clean_action_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])

    result = apply_dynamics_training_plan(
        plan,
        clean_video=clean_video,
        noisy_video=noisy_video,
        video_targets=video_targets,
        video_timesteps=video_timesteps,
        video_loss_mask=video_loss_mask,
        clean_action=clean_action,
        noisy_action=noisy_action,
        action_targets=action_targets,
        action_timesteps=action_timesteps,
        action_loss_mask=action_loss_mask,
        clean_action_mask=clean_action_mask,
    )

    if objective == DynamicsObjective.ACTION_CONDITIONED_VIDEO:
        assert result.noisy_video is noisy_video
        assert result.video_targets is video_targets
        assert result.video_loss_mask is video_loss_mask
        torch.testing.assert_close(
            result.noisy_action,
            clean_action * clean_action_mask,
        )
        assert torch.count_nonzero(result.action_targets) == 0
        assert torch.count_nonzero(result.action_timesteps) == 0
        assert result.action_loss_mask is not None
        assert torch.count_nonzero(result.action_loss_mask) == 0
    else:
        assert result.noisy_video is clean_video
        assert torch.count_nonzero(result.video_targets) == 0
        assert torch.count_nonzero(result.video_timesteps) == 0
        assert torch.count_nonzero(result.video_loss_mask) == 0
        assert result.noisy_action is noisy_action
        assert result.action_targets is action_targets
        assert result.action_loss_mask is action_loss_mask


@pytest.mark.unit
def test_shared_conditional_plan_rejects_text_conditioning_override() -> None:
    with pytest.raises(ValueError, match="always removes task text"):
        resolve_dynamics_training_plan(
            program=VideoActionProgram.FORWARD_DYNAMICS,
            sample_metadata=_target_only_metadata(
                DynamicsObjective.ACTION_CONDITIONED_VIDEO,
                drop_text=False,
            ),
            device=torch.device("cpu"),
        )


@pytest.mark.unit
def test_shared_joint_plan_rejects_text_removal_override() -> None:
    with pytest.raises(ValueError, match="Joint dynamics always preserves task text"):
        resolve_dynamics_objective_semantics(
            DynamicsObjective.JOINT,
            drop_text_conditioning=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "objective",
    [None, DynamicsObjective.JOINT],
)
def test_target_only_layout_requires_explicit_conditional_route(
    objective: DynamicsObjective | None,
) -> None:
    metadata = dict(
        _target_only_metadata(DynamicsObjective.ACTION_CONDITIONED_VIDEO).raw
    )
    if objective is None:
        metadata.pop(DYNAMICS_ROUTING_MODE_METADATA_KEY)
    else:
        metadata[DYNAMICS_ROUTING_MODE_METADATA_KEY] = objective.value

    with pytest.raises(
        ValueError,
        match="requires an explicit conditional FDM/IDM route",
    ):
        resolve_dynamics_training_plan(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
            sample_metadata=SampleConstructionMetadata.from_mapping(metadata),
            device=torch.device("cpu"),
        )


@pytest.mark.unit
def test_shared_coupled_noise_plan_matches_sigmas_across_schedulers() -> None:
    torch.manual_seed(0)
    video_scheduler = FlowMatchScheduler(
        shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000
    )
    action_scheduler = FlowMatchScheduler(
        shift=1.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=500
    )
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
    assert torch.allclose(
        video_scheduler.sigma_for_timesteps(coupled.video_timesteps),
        coupled.sigma_values,
    )
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
    assert torch.equal(
        zero_loss_mask_like(mask, fallback_like=clean), torch.zeros_like(mask)
    )
    assert torch.equal(
        zero_loss_mask_like(None, fallback_like=clean), torch.zeros_like(clean)
    )

    half_clean = clean.to(dtype=torch.float16)
    half_mask = mask.to(dtype=torch.float32)
    force_clean_noisy_slot(artifact, half_clean, action_mask=half_mask)
    assert artifact["noisy_latents"].dtype == torch.float16


@pytest.mark.unit
def test_shared_metric_rollup_matches_m1_m5_generalist_shape() -> None:
    metrics: dict[str, torch.Tensor] = {}
    action_loss = torch.tensor(2.0)
    latent_loss = torch.tensor(3.0)

    add_dynamics_objective_metrics(
        metrics,
        namespace="joint_denoise",
        mode_value="action_conditioned_video",
        modes=DynamicsObjective,
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
    assert (
        metrics["joint_denoise/action_conditioned_video/action_flow_loss_sum"].item()
        == 2.0
    )
    assert (
        metrics["joint_denoise/action_conditioned_video/latent_flow_loss_sum"].item()
        == 3.0
    )
    assert (
        metrics["joint_denoise/action_conditioned_video/action_mse_sum"].item() == 2.0
    )
    assert (
        metrics["joint_denoise/action_conditioned_video/latent_mse_sum"].item() == 3.0
    )
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
