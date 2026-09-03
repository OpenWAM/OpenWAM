from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import (
    DualExpertPolicyConfig,
    DynamicsObjective,
    DynamicsRouteConfig,
    DynamicsRoutingConfig,
    DynamicsSource,
    ExperimentConfig,
    VideoActionProgram,
)
from open_wam.contracts import identify_video_latent_space
from open_wam.evals import libero_dual_expert_composition as composition
from open_wam.evals.libero_dual_expert_runtime import (
    DualExpertActionRoute,
    LiberoPolicyRuntimeRole,
)
from open_wam.models.policy_variants import (
    PolicyCompositionCapability,
    PolicyCompositionRngPolicy,
    PolicyGeneratedVideo,
    PolicyInferContext,
    PolicyInferenceOutputRequest,
    PolicyOutputModality,
    PolicyRecurrentHistoryPolicy,
)
from open_wam.pipelines import (
    PolicyVideoActionConsumerPlan,
    PolicyVideoProducerPlan,
    VariantRolloutSession,
)


def _producer_plan() -> PolicyVideoProducerPlan:
    return PolicyVideoProducerPlan(
        output_request=PolicyInferenceOutputRequest.video_only(),
        native_modalities=frozenset(PolicyOutputModality),
        recurrent_history_policy=PolicyRecurrentHistoryPolicy.NEXT_OBSERVATION,
    )


def _consumer_plan(
    *,
    rng_policy: PolicyCompositionRngPolicy = (
        PolicyCompositionRngPolicy.ISOLATED_STEP_SEED
    ),
    history_policy: PolicyRecurrentHistoryPolicy = (
        PolicyRecurrentHistoryPolicy.EXPLICIT_RECONCILIATION
    ),
) -> PolicyVideoActionConsumerPlan:
    return PolicyVideoActionConsumerPlan(
        capability=PolicyCompositionCapability.video_to_action(
            rng_policy=rng_policy
        ),
        recurrent_history_policy=history_policy,
    )


def _latent_identity(root, *, weights: bytes = b"weights"):
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"latent_channels": 48}\n')
    (root / "model.safetensors").write_bytes(weights)
    return identify_video_latent_space(
        root,
        encoder_family="test.WanVAE@1",
        encoding_contract="test.wan_latents.v1",
    )


def _consumer_args(*, route: str, config: str | None, checkpoint: str | None):
    return Namespace(
        dual_expert_gjd_action_route=route,
        dual_expert_action_only_rollout=False,
        action_consumer_cfg=config,
        action_consumer_checkpoint=checkpoint,
        action_consumer_set_overrides=[],
        action_consumer_runtime_device=None,
        action_consumer_action_device=None,
        action_consumer_frontend_device=None,
    )


def test_action_consumer_arguments_use_canonical_cli_surface() -> None:
    parser = ArgumentParser()
    composition.add_action_consumer_arguments(parser)

    args = parser.parse_args(
        [
            "--action-consumer-cfg",
            "consumer.yaml",
            "--action-consumer-checkpoint",
            "checkpoint_step_1",
            "--action-consumer-set",
            "data.local_root=/data",
            "--action-consumer-runtime-device",
            "cuda:1",
            "--action-consumer-action-device",
            "cuda:1",
            "--action-consumer-frontend-device",
            "cuda:1",
        ]
    )

    assert args.action_consumer_cfg == "consumer.yaml"
    assert args.action_consumer_checkpoint == "checkpoint_step_1"
    assert args.action_consumer_set_overrides == ["data.local_root=/data"]
    assert args.action_consumer_runtime_device == "cuda:1"
    assert args.action_consumer_action_device == "cuda:1"
    assert args.action_consumer_frontend_device == "cuda:1"


def test_video_action_consumer_arguments_are_scoped_to_the_composed_route() -> None:
    composed = _consumer_args(
        route=DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION.value,
        config="idm.yaml",
        checkpoint="checkpoint_step_1",
    )
    assert composition.validate_action_consumer_arguments(composed) is True
    assert composition.action_consumer_options_from_args(composed) == (
        composition.ActionConsumerLoadOptions(
            config="idm.yaml",
            checkpoint="checkpoint_step_1",
        )
    )

    with pytest.raises(ValueError, match="requires both"):
        composition.validate_action_consumer_arguments(
            _consumer_args(
                route=DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION.value,
                config="idm.yaml",
                checkpoint=None,
            )
        )
    with pytest.raises(ValueError, match="only valid"):
        composition.validate_action_consumer_arguments(
            _consumer_args(
                route=DualExpertActionRoute.JOINT.value,
                config="idm.yaml",
                checkpoint="checkpoint_step_1",
            )
        )

    action_only = _consumer_args(
        route=DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION.value,
        config="idm.yaml",
        checkpoint="checkpoint_step_1",
    )
    action_only.dual_expert_action_only_rollout = True
    with pytest.raises(ValueError, match="cannot be combined"):
        composition.validate_action_consumer_arguments(action_only)


def test_video_action_consumer_contract_accepts_vta_fixed_and_routed_idm() -> None:
    primary = ExperimentConfig(
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION
        )
    )
    fixed_idm = replace(
        primary,
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.INVERSE_DYNAMICS
        ),
    )
    fixed_report = composition.validate_video_action_composition_contract(
        primary,
        fixed_idm,
        producer_plan=_producer_plan(),
        consumer_plan=_consumer_plan(),
    )
    assert fixed_report["action_consumer_route_source"] == (
        "fixed_inverse_dynamics_program"
    )

    vta_report = composition.validate_video_action_composition_contract(
        primary,
        primary,
        producer_plan=_producer_plan(),
        consumer_plan=_consumer_plan(
            rng_policy=PolicyCompositionRngPolicy.CALLER_STREAM
        ),
    )
    assert vta_report["action_consumer_route_source"] == (
        "native_video_then_action_program"
    )

    routed_data = replace(
        primary.data,
        dynamics_routing=DynamicsRoutingConfig(
            routes=(
                DynamicsRouteConfig(
                    source=DynamicsSource.REAL_DEMO,
                    mode=DynamicsObjective.VIDEO_CONDITIONED_ACTION,
                    weight=1.0,
                ),
            )
        ),
    )
    routed_idm = replace(
        primary,
        data=routed_data,
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING
        ),
    )
    routed_report = composition.validate_video_action_composition_contract(
        primary,
        routed_idm,
        producer_plan=_producer_plan(),
        consumer_plan=_consumer_plan(),
    )
    assert routed_report["action_consumer_route_source"] == (
        "generalist_joint_denoising_training_route"
    )


def test_video_action_consumer_contract_rejects_inactive_route_and_geometry_drift() -> None:
    primary = ExperimentConfig(
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION
        )
    )
    inactive_gjd = replace(
        primary,
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING
        ),
    )
    with pytest.raises(ValueError, match="positive.*video_conditioned_action"):
        composition.validate_video_action_composition_contract(
            primary,
            inactive_gjd,
            producer_plan=_producer_plan(),
            consumer_plan=_consumer_plan(),
        )

    mismatched_idm = replace(
        primary,
        data=replace(primary.data, canonical_width=primary.data.canonical_width + 8),
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.INVERSE_DYNAMICS
        ),
    )
    with pytest.raises(ValueError, match="data.canonical_width"):
        composition.validate_video_action_composition_contract(
            primary,
            mismatched_idm,
            producer_plan=_producer_plan(),
            consumer_plan=_consumer_plan(),
        )


def test_config_preflight_does_not_treat_artifact_paths_as_latent_identity() -> None:
    primary = ExperimentConfig(
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION
        )
    )
    primary = replace(
        primary,
        backbone=replace(
            primary.backbone,
            pretrained_model_name_or_path="/mirror/producer/model",
            vae_subdir="/mirror/producer/vae",
        ),
    )
    consumer = replace(
        primary,
        backbone=replace(
            primary.backbone,
            pretrained_model_name_or_path="/mirror/consumer/model",
            vae_subdir="/mirror/consumer/vae",
        ),
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.INVERSE_DYNAMICS
        ),
    )

    report = composition.validate_video_action_composition_contract(
        primary,
        consumer,
        producer_plan=_producer_plan(),
        consumer_plan=_consumer_plan(),
    )

    assert "backbone.pretrained_model_name_or_path" not in report["validated_fields"]
    assert "backbone.vae_subdir" not in report["validated_fields"]


@pytest.mark.parametrize(
    "conditional_objective",
    (
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
    ),
)
def test_composition_rejects_gjd_producer_without_joint_training(
    conditional_objective: DynamicsObjective,
) -> None:
    baseline = ExperimentConfig(
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION
        )
    )
    conditional_only_data = replace(
        baseline.data,
        dynamics_routing=DynamicsRoutingConfig(
            routes=(
                DynamicsRouteConfig(
                    source=DynamicsSource.REAL_DEMO,
                    mode=conditional_objective,
                    weight=1.0,
                ),
            )
        ),
    )
    conditional_only_gjd = replace(
        baseline,
        data=conditional_only_data,
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING
        ),
    )

    with pytest.raises(ValueError, match="positive.*`joint` training route"):
        composition.validate_video_action_composition_contract(
            conditional_only_gjd,
            replace(
                baseline,
                policy_variant=DualExpertPolicyConfig(
                    program=VideoActionProgram.INVERSE_DYNAMICS
                ),
            ),
            producer_plan=_producer_plan(),
            consumer_plan=_consumer_plan(),
        )


def test_composition_rejects_fixed_fdm_producer() -> None:
    baseline = ExperimentConfig(
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION
        )
    )
    fixed_fdm = replace(
        baseline,
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.FORWARD_DYNAMICS
        ),
    )

    with pytest.raises(ValueError, match="FDM-only policy"):
        composition.validate_video_action_composition_contract(
            fixed_fdm,
            replace(
                baseline,
                policy_variant=DualExpertPolicyConfig(
                    program=VideoActionProgram.INVERSE_DYNAMICS
                ),
            ),
            producer_plan=_producer_plan(),
            consumer_plan=_consumer_plan(),
        )


def test_composition_accepts_gjd_producer_with_joint_training() -> None:
    baseline = ExperimentConfig(
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION
        )
    )
    joint_gjd = replace(
        baseline,
        data=replace(
            baseline.data,
            dynamics_routing=DynamicsRoutingConfig(
                routes=(
                    DynamicsRouteConfig(
                        source=DynamicsSource.REAL_DEMO,
                        mode=DynamicsObjective.JOINT,
                        weight=1.0,
                    ),
                )
            ),
        ),
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING
        ),
    )

    report = composition.validate_video_action_composition_contract(
        joint_gjd,
        replace(
            baseline,
            policy_variant=DualExpertPolicyConfig(
                program=VideoActionProgram.INVERSE_DYNAMICS
            ),
        ),
        producer_plan=_producer_plan(),
        consumer_plan=_consumer_plan(),
    )

    assert report["video_producer_route_source"] == (
        "generalist_joint_denoising_training_route"
    )


def test_video_action_consumer_loader_provides_only_the_clean_video_objective(
    monkeypatch,
    tmp_path,
) -> None:
    primary_config = object()
    external_config = object()
    primary_runtime = SimpleNamespace(
        config=primary_config,
        runtime_device=torch.device("cpu"),
        raw_window_frames=13,
        startup_model_obs_frames=1,
        startup_env_init_steps=5,
    )
    primary_options = SimpleNamespace(
        dual_expert_gjd_action_route=(
            DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION.value
        ),
        source="test",
        dual_expert_inference_window_size=30,
        dual_expert_rollout_frame_chunk_size=4,
        execute_action_steps=None,
        execute_frame_chunk_size=None,
        frontend_encode_mode="lingbot_streaming_vae",
        reset_policy_state_each_chunk=False,
        allow_deprecated_libero_config=False,
        allow_deprecated_frontend_encode_mode=False,
        checkpoint_load_policy="checkpoint-policy",
    )
    captured: dict[str, object] = {}
    latent_identity = _latent_identity(tmp_path / "vae")
    external_runtime = SimpleNamespace(
        config=external_config,
        runtime_device=torch.device("cpu"),
        action_device=torch.device("cpu"),
        pipeline=SimpleNamespace(
            visual_tower=SimpleNamespace(
                frontend=SimpleNamespace(latent_space_identity=latent_identity)
            ),
            policy_variant=SimpleNamespace(
                inference_capabilities=SimpleNamespace(
                    recurrent_history_policy=(
                        PolicyRecurrentHistoryPolicy.EXPLICIT_RECONCILIATION
                    )
                )
            ),
        ),
    )

    def _load(options):
        captured["options"] = options
        return external_runtime

    monkeypatch.setattr(composition, "load_dual_expert_libero_runtime", _load)
    monkeypatch.setattr(
        composition,
        "resolve_policy_video_producer_plan",
        lambda policy: _producer_plan(),
    )
    monkeypatch.setattr(
        composition,
        "resolve_policy_video_action_consumer_plan",
        lambda policy: _consumer_plan(),
    )
    monkeypatch.setattr(
        composition,
        "validate_video_action_composition_contract",
        lambda primary, consumer, *, producer_plan, consumer_plan: {
            "configs": (primary, consumer),
            "producer_plan": producer_plan,
            "consumer_plan": consumer_plan,
        },
    )
    primary_runtime.pipeline = SimpleNamespace(
        policy_variant=object(),
        visual_tower=SimpleNamespace(
            frontend=SimpleNamespace(latent_space_identity=latent_identity)
        ),
    )

    loaded = composition.load_video_action_composition(
        primary_runtime=primary_runtime,
        primary_options=primary_options,
        consumer_options=composition.ActionConsumerLoadOptions(
            config="idm.yaml",
            checkpoint="checkpoint_step_1",
        ),
    )

    options = captured["options"]
    assert options.provided_conditioning_modalities == (
        PolicyOutputModality.VIDEO,
    )
    assert options.runtime_role is (
        LiberoPolicyRuntimeRole.VIDEO_CONDITIONED_ACTION_CONSUMER
    )
    assert options.merge_checkpoint_runtime_config is False
    assert options.raw_window_frames == 13
    assert loaded is not None
    assert loaded.runtime is external_runtime
    assert loaded.producer_plan == _producer_plan()
    assert loaded.consumer_plan == _consumer_plan()
    assert loaded.compatibility_report["configs"] == (
        primary_config,
        external_config,
    )


def test_video_action_consumer_loader_rejects_split_packed_devices(monkeypatch) -> None:
    primary_runtime = SimpleNamespace(
        config=object(),
        runtime_device=torch.device("cpu"),
        raw_window_frames=13,
        startup_model_obs_frames=1,
        startup_env_init_steps=5,
    )
    primary_options = SimpleNamespace(
        dual_expert_gjd_action_route=(
            DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION.value
        ),
        source="test",
        dual_expert_inference_window_size=30,
        dual_expert_rollout_frame_chunk_size=4,
        execute_action_steps=None,
        execute_frame_chunk_size=None,
        frontend_encode_mode="lingbot_streaming_vae",
        reset_policy_state_each_chunk=False,
        allow_deprecated_libero_config=False,
        allow_deprecated_frontend_encode_mode=False,
        checkpoint_load_policy="checkpoint-policy",
    )
    monkeypatch.setattr(
        composition,
        "load_dual_expert_libero_runtime",
        lambda options: SimpleNamespace(
            config=object(),
            runtime_device=torch.device("cuda:0"),
            action_device=torch.device("cuda:1"),
        ),
    )
    monkeypatch.setattr(
        composition,
        "resolve_policy_video_producer_plan",
        lambda policy: _producer_plan(),
    )
    primary_runtime.pipeline = SimpleNamespace(policy_variant=object())

    with pytest.raises(ValueError, match="same device"):
        composition.load_video_action_composition(
            primary_runtime=primary_runtime,
            primary_options=primary_options,
            consumer_options=composition.ActionConsumerLoadOptions(
                config="idm.yaml",
                checkpoint="checkpoint_step_1",
            ),
        )


@pytest.mark.parametrize(
    ("rng_policy", "rollout_seed", "expected_seed", "expect_rng_restore"),
    (
        (PolicyCompositionRngPolicy.ISOLATED_STEP_SEED, 11, 14, True),
        (PolicyCompositionRngPolicy.CALLER_STREAM, 11, None, False),
    ),
)
def test_video_action_consumer_inference_hands_generated_video_to_public_runner(
    monkeypatch,
    tmp_path,
    rng_policy,
    rollout_seed,
    expected_seed,
    expect_rng_restore,
) -> None:
    calls: dict[str, object] = {}
    built_context = PolicyInferContext()
    rollout_output = SimpleNamespace(
        infer_output=SimpleNamespace(
            policy_output=SimpleNamespace(
                generation_frame_start=1,
                aux={"generation_frame_start": 1},
            )
        )
    )

    def _build_context(*args, **kwargs):
        calls["context_args"] = (args, kwargs)
        return built_context

    class _Runner:
        def reset(self, **kwargs):
            calls["reset"] = kwargs
            return VariantRolloutSession(**kwargs)

        def infer_prepared_step(self, *, session, context, visual_outputs):
            calls["infer"] = (session, context, visual_outputs)
            return rollout_output

    monkeypatch.setattr(composition, "_build_infer_context", _build_context)
    monkeypatch.setattr(
        composition,
        "seed_everywhere",
        lambda seed: calls.setdefault("seed", seed),
    )
    rng_snapshot = object()
    monkeypatch.setattr(
        composition,
        "snapshot_rng_state",
        lambda: rng_snapshot,
    )
    monkeypatch.setattr(
        composition,
        "restore_rng_state",
        lambda snapshot: calls.setdefault("restored_rng", snapshot),
    )
    latent_identity = _latent_identity(tmp_path / "vae")
    visual_outputs = SimpleNamespace(
        frontend=SimpleNamespace(
            video_latents=torch.zeros(1, 48, 1, 2, 2),
            latent_space_identity=latent_identity,
        )
    )
    runtime = SimpleNamespace(
        runner=_Runner(),
        action_device=torch.device("cpu"),
        runtime_device=torch.device("cpu"),
        config=object(),
    )
    generated = torch.randn(1, 48, 4, 2, 2, requires_grad=True)
    policy_state = object()

    output = composition.infer_video_conditioned_action(
        composition.VideoActionComposition(
            runtime=runtime,
            producer_plan=_producer_plan(),
            consumer_plan=_consumer_plan(rng_policy=rng_policy),
            compatibility_report={},
        ),
        session=VariantRolloutSession(
            policy_state=policy_state,
            task_text=("task",),
            text_context=torch.ones(1, 2, 3),
            negative_text_context=torch.zeros(1, 2, 3),
        ),
        visual_outputs=visual_outputs,
        model_obs_window=[{"observation.state": torch.zeros(8).numpy()}],
        prompt="task",
        generated_video=PolicyGeneratedVideo(
            latents=generated,
            frame_start=1,
            latent_space_identity=latent_identity,
        ),
        inference_window_size=30,
        reset_policy_state=False,
        rollout_seed=rollout_seed,
        chunk_index=3,
        producer_rng_device=torch.device("cpu"),
    )

    infer_session, infer_context, infer_visual_outputs = calls["infer"]
    assert output.rollout is rollout_output
    assert output.inference_seed == expected_seed
    if expected_seed is None:
        assert "seed" not in calls
    else:
        assert calls["seed"] == expected_seed
    if expect_rng_restore:
        assert calls["restored_rng"] is rng_snapshot
    else:
        assert "restored_rng" not in calls
    assert infer_session.policy_state is policy_state
    assert infer_visual_outputs is visual_outputs
    assert infer_context is not built_context
    assert infer_context.dynamics is None
    assert infer_context.video_conditioned_action is not None
    consumer_video = infer_context.video_conditioned_action.generated_video
    assert consumer_video.latents.shape == generated.shape
    assert not consumer_video.latents.requires_grad
    assert calls["context_args"][1]["dual_expert_rollout_frame_chunk_size"] is None


def test_video_action_consumer_rejects_latent_coordinate_mismatch() -> None:
    generated = PolicyGeneratedVideo(
        latents=torch.randn(1, 48, 4, 8, 16),
        frame_start=1,
    )

    with pytest.raises(RuntimeError, match="latent batch/channel/spatial geometry"):
        composition._validate_generated_video_tensor_geometry(
            generated_video=generated,
            observed_video=torch.randn(1, 32, 1, 8, 16),
        )


def test_video_action_consumer_rejects_non_video_consumer_tensor() -> None:
    generated = PolicyGeneratedVideo(
        latents=torch.randn(1, 48, 4, 8, 16),
        frame_start=1,
    )

    with pytest.raises(RuntimeError, match=r"\[B, C, T, H, W\]"):
        composition._validate_generated_video_tensor_geometry(
            generated_video=generated,
            observed_video=torch.randn(1, 48, 8, 16),
        )
