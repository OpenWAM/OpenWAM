"""LIBERO adapter for policy-video to video-conditioned-action composition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from open_wam.configs import (
    CurrentBlockCoupling,
    DynamicsObjective,
    VideoActionProgram,
)
from open_wam.evals.libero_dual_expert_inputs import _build_infer_context
from open_wam.evals.libero_dual_expert_runtime import (
    DualExpertActionRoute,
    DualExpertLiberoLoadOptions,
    DualExpertLiberoRuntime,
    LiberoPolicyRuntimeRole,
    load_dual_expert_libero_runtime,
    uses_video_action_composition,
)
from open_wam.evals.realtime_speculation import restore_rng_state, snapshot_rng_state
from open_wam.models.common.dynamics_objectives import (
    resolve_dynamics_objective_semantics,
)
from open_wam.models.policy_variants import (
    PolicyGeneratedVideo,
    PolicyRecurrentHistoryPolicy,
)
from open_wam.models.visual_tower import VisualStageOutputs
from open_wam.pipelines import (
    PolicyVideoProducerPlan,
    VariantRolloutSession,
    VariantRolloutStepOutput,
    build_video_conditioned_action_request,
    require_compatible_video_latent_spaces,
    resolve_policy_video_producer_plan,
)
from open_wam.utils import seed_everywhere


@dataclass(frozen=True)
class ExternalIdmLoadOptions:
    """External checkpoint and device choices for one composed rollout."""

    config: str | Path
    checkpoint: str | Path
    set_overrides: tuple[str, ...] = ()
    runtime_device: str | None = None
    action_device: str | None = None
    frontend_device: str | None = None


@dataclass(frozen=True)
class ExternalIdmComposition:
    """A policy video producer composed with an external action consumer."""

    runtime: DualExpertLiberoRuntime
    producer_plan: PolicyVideoProducerPlan
    compatibility_report: dict[str, object]


@dataclass(frozen=True)
class ExternalIdmStepOutput:
    """One external action-stage result and its deterministic seed."""

    rollout: VariantRolloutStepOutput
    inference_seed: int | None


def add_external_idm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared external-IDM CLI surface to a LIBERO driver."""

    parser.add_argument(
        "--action-consumer-cfg",
        dest="external_idm_cfg",
        type=str,
        default=None,
        help="Resolved config for the video-conditioned action consumer.",
    )
    parser.add_argument(
        "--action-consumer-checkpoint",
        dest="external_idm_checkpoint",
        type=str,
        default=None,
        help="Checkpoint file or checkpoint_step_* directory for the action consumer.",
    )
    parser.add_argument(
        "--action-consumer-set",
        dest="external_idm_set_overrides",
        action="append",
        default=[],
        help="Apply an override only to the action-consumer config.",
    )
    parser.add_argument(
        "--action-consumer-runtime-device",
        dest="external_idm_runtime_device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--action-consumer-action-device",
        dest="external_idm_action_device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--action-consumer-frontend-device",
        dest="external_idm_frontend_device",
        type=str,
        default=None,
    )


def validate_external_idm_arguments(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> bool:
    """Validate route-scoped external options and return whether they are active."""

    route = DualExpertActionRoute(args.dual_expert_gjd_action_route)
    uses_external_idm = uses_video_action_composition(route)
    supplied = any(
        getattr(args, name, None)
        for name in (
            "external_idm_cfg",
            "external_idm_checkpoint",
            "external_idm_set_overrides",
            "external_idm_runtime_device",
            "external_idm_action_device",
            "external_idm_frontend_device",
        )
    )
    if uses_external_idm and not (
        args.external_idm_cfg and args.external_idm_checkpoint
    ):
        _argument_error(
            parser,
            "generated_video_then_action requires both --action-consumer-cfg "
            "and --action-consumer-checkpoint.",
        )
    if uses_external_idm and bool(
        getattr(args, "dual_expert_action_only_rollout", False)
    ):
        _argument_error(
            parser,
            "generated_video_then_action requires the primary policy to produce "
            "video; it cannot be combined with --dual-expert-action-only-rollout.",
        )
    if not uses_external_idm and supplied:
        _argument_error(
            parser,
            "External IDM arguments are only valid with "
            "--dual-expert-action-route generated_video_then_action.",
        )
    return uses_external_idm


def external_idm_options_from_args(
    args: argparse.Namespace,
) -> ExternalIdmLoadOptions | None:
    """Build typed external options after argument validation."""

    if not validate_external_idm_arguments(args):
        return None
    return ExternalIdmLoadOptions(
        config=args.external_idm_cfg,
        checkpoint=args.external_idm_checkpoint,
        set_overrides=tuple(args.external_idm_set_overrides),
        runtime_device=args.external_idm_runtime_device,
        action_device=args.external_idm_action_device,
        frontend_device=args.external_idm_frontend_device,
    )


def load_external_idm_composition(
    *,
    primary_runtime: DualExpertLiberoRuntime,
    primary_options: DualExpertLiberoLoadOptions,
    external_options: ExternalIdmLoadOptions | None,
) -> ExternalIdmComposition | None:
    """Load and validate the optional action stage for one primary runtime."""

    route = DualExpertActionRoute(primary_options.dual_expert_gjd_action_route)
    if not uses_video_action_composition(route):
        if external_options is not None:
            raise ValueError(
                "External IDM load options require the generated-video external-IDM route."
            )
        return None
    if external_options is None:
        raise ValueError(
            "The generated-video external-IDM route requires external load options."
        )

    producer_plan = resolve_policy_video_producer_plan(
        primary_runtime.pipeline.policy_variant
    )
    runtime = load_dual_expert_libero_runtime(
        DualExpertLiberoLoadOptions(
            config=external_options.config,
            checkpoint=external_options.checkpoint,
            merge_checkpoint_runtime_config=False,
            set_overrides=external_options.set_overrides,
            source=f"{primary_options.source} external IDM",
            checkpoint_error="External IDM composition requires a checkpoint.",
            raw_window_frames=primary_runtime.raw_window_frames,
            startup_model_obs_frames=primary_runtime.startup_model_obs_frames,
            startup_env_init_steps=primary_runtime.startup_env_init_steps,
            dual_expert_inference_window_size=(
                primary_options.dual_expert_inference_window_size
            ),
            dual_expert_rollout_frame_chunk_size=(
                primary_options.dual_expert_rollout_frame_chunk_size
            ),
            dual_expert_action_only_rollout=False,
            dual_expert_gjd_action_route=route.value,
            execute_action_steps=primary_options.execute_action_steps,
            execute_frame_chunk_size=primary_options.execute_frame_chunk_size,
            frontend_encode_mode=primary_options.frontend_encode_mode,
            reset_policy_state_each_chunk=primary_options.reset_policy_state_each_chunk,
            runtime_device=(
                external_options.runtime_device or str(primary_runtime.runtime_device)
            ),
            action_device=external_options.action_device,
            frontend_device=external_options.frontend_device,
            decode_device=external_options.frontend_device,
            allow_deprecated_libero_config=(
                primary_options.allow_deprecated_libero_config
            ),
            allow_deprecated_frontend_encode_mode=(
                primary_options.allow_deprecated_frontend_encode_mode
            ),
            checkpoint_load_policy=primary_options.checkpoint_load_policy,
            runtime_role=(
                LiberoPolicyRuntimeRole.VIDEO_CONDITIONED_ACTION_CONSUMER
            ),
            provided_dynamics_objectives=(DynamicsObjective.VIDEO_CONDITIONED_ACTION,),
        )
    )
    if runtime.action_device != runtime.runtime_device:
        raise ValueError(
            "External packed IDM inference requires its runtime and action "
            "modules on the same device; got "
            f"runtime_device={runtime.runtime_device}, "
            f"action_device={runtime.action_device}."
        )
    consumer_history_policy = (
        runtime.pipeline.policy_variant.inference_capabilities.recurrent_history_policy
    )
    if (
        consumer_history_policy
        is not PolicyRecurrentHistoryPolicy.EXPLICIT_RECONCILIATION
    ):
        raise ValueError(
            "The video-conditioned action consumer must explicitly reconcile its "
            "speculative video/action history after execution; got "
            f"history_policy={consumer_history_policy.value!r}."
        )
    compatibility_report = dict(
        validate_external_idm_contract(
            primary_runtime.config,
            runtime.config,
            producer_plan=producer_plan,
        )
    )
    compatibility_report["video_latent_space"] = (
        require_compatible_video_latent_spaces(
            primary_runtime.pipeline.visual_tower.frontend.latent_space_identity,
            runtime.pipeline.visual_tower.frontend.latent_space_identity,
        )
    )
    return ExternalIdmComposition(
        runtime=runtime,
        producer_plan=producer_plan,
        compatibility_report=compatibility_report,
    )


def infer_external_idm_action(
    composition: ExternalIdmComposition,
    *,
    session: VariantRolloutSession,
    visual_outputs: VisualStageOutputs,
    model_obs_window: list[dict[str, Any]],
    prompt: str,
    generated_video: PolicyGeneratedVideo,
    inference_window_size: int | None,
    reset_policy_state: bool,
    rollout_seed: int | None,
    chunk_index: int,
) -> ExternalIdmStepOutput:
    """Predict actions from a generated video through the public runner API."""

    runtime = composition.runtime
    context = _build_infer_context(
        prompt,
        action_device=runtime.action_device,
        model_obs_window=model_obs_window,
        config=runtime.config,
        runtime_device=runtime.runtime_device,
        dual_expert_inference_window_size=inference_window_size,
        # The typed DynamicsRolloutRequest below owns consumer chunk geometry.
        # Do not duplicate it through the legacy DualExpert override.
        dual_expert_rollout_frame_chunk_size=None,
        dual_expert_action_only_rollout=False,
    )
    _validate_generated_video_tensor_geometry(
        generated_video=generated_video,
        observed_video=visual_outputs.frontend.video_latents,
    )
    require_compatible_video_latent_spaces(
        generated_video.latent_space_identity,
        visual_outputs.frontend.latent_space_identity,
    )
    consumer_video = PolicyGeneratedVideo(
        latents=generated_video.latents.detach().to(
            device=runtime.runtime_device,
            dtype=visual_outputs.frontend.video_latents.dtype,
        ),
        frame_start=generated_video.frame_start,
        latent_space_identity=generated_video.latent_space_identity,
    )
    context.dynamics = build_video_conditioned_action_request(consumer_video)
    infer_session = runtime.runner.reset(
        task_text=session.task_text,
        text_context=session.text_context,
        negative_text_context=session.negative_text_context,
    )
    infer_session.policy_state = None if reset_policy_state else session.policy_state
    inference_seed = None if rollout_seed is None else int(rollout_seed + chunk_index)
    rng_snapshot = snapshot_rng_state() if inference_seed is not None else None
    try:
        if inference_seed is not None:
            # Make action-consumer noise deterministic without advancing the
            # video producer's
            # Python, NumPy, CPU, or CUDA random streams.
            seed_everywhere(inference_seed)
        rollout = runtime.runner.infer_prepared_step(
            session=infer_session,
            context=context,
            visual_outputs=visual_outputs,
        )
    finally:
        restore_rng_state(rng_snapshot)
    _validate_generated_video_frame_alignment(
        generated_video=consumer_video,
        rollout=rollout,
    )
    return ExternalIdmStepOutput(
        rollout=rollout,
        inference_seed=inference_seed,
    )


def _validate_generated_video_tensor_geometry(
    *,
    generated_video: PolicyGeneratedVideo,
    observed_video: torch.Tensor,
) -> None:
    """Require the handoff to match the consumer's latent coordinate space."""

    generated = generated_video.latents
    if observed_video.ndim != 5:
        raise RuntimeError(
            "The action consumer must expose observed video as a "
            "[B, C, T, H, W] latent tensor; "
            f"got {tuple(observed_video.shape)}."
        )
    expected = (
        int(observed_video.shape[0]),
        int(observed_video.shape[1]),
        int(observed_video.shape[3]),
        int(observed_video.shape[4]),
    )
    actual = (
        int(generated.shape[0]),
        int(generated.shape[1]),
        int(generated.shape[3]),
        int(generated.shape[4]),
    )
    if actual != expected:
        raise RuntimeError(
            "Generated video does not match the action consumer's latent "
            "batch/channel/spatial geometry: "
            f"producer={actual}, consumer={expected}."
        )


def _validate_generated_video_frame_alignment(
    *,
    generated_video: PolicyGeneratedVideo,
    rollout: VariantRolloutStepOutput,
) -> None:
    """Reject producer/consumer temporal origins that would misalign tokens."""

    if generated_video.frame_start is None:
        raise RuntimeError(
            "Generated video is missing its temporal origin, so action-consumer "
            "alignment cannot be verified."
        )
    raw_consumer_start = (
        rollout.infer_output.policy_output.generation_frame_start
    )
    if raw_consumer_start is None:
        raise RuntimeError(
            "The video-conditioned action consumer did not report its generated "
            "frame origin, so temporal handoff parity cannot be verified."
        )
    consumer_start = int(raw_consumer_start)
    producer_start = int(generated_video.frame_start)
    if consumer_start != producer_start:
        raise RuntimeError(
            "Generated-video producer and action consumer use different temporal "
            "origins: "
            f"producer_frame_start={producer_start}, "
            f"consumer_frame_start={consumer_start}."
        )


def validate_external_idm_contract(
    primary_config: Any,
    idm_config: Any,
    *,
    producer_plan: PolicyVideoProducerPlan,
) -> dict[str, object]:
    """Require a lossless generated-video handoff into one action consumer."""

    primary_program = getattr(primary_config.policy_variant, "program", None)
    producer_route_source = _validate_video_producer_training_contract(
        primary_config
    )

    idm_program = VideoActionProgram(idm_config.policy_variant.program)
    if idm_program is VideoActionProgram.INVERSE_DYNAMICS:
        idm_route_source = "fixed_inverse_dynamics_program"
    elif idm_program is VideoActionProgram.GENERALIST_JOINT_DENOISING:
        active_idm_routes = tuple(
            route
            for route in idm_config.data.dynamics_routing.active_routes
            if route.mode is DynamicsObjective.VIDEO_CONDITIONED_ACTION
        )
        if not active_idm_routes:
            raise ValueError(
                "External GJD IDM checkpoint must declare a positive "
                "`video_conditioned_action` training route."
            )
        idm_route_source = "generalist_joint_denoising_training_route"
    else:
        raise ValueError(
            "External IDM composition requires `inverse_dynamics` or "
            "`generalist_joint_denoising`; "
            f"got program={idm_program.value!r}."
        )
    if (
        idm_config.policy_variant.current_block_coupling
        is not CurrentBlockCoupling.JOINT
    ):
        raise ValueError(
            "External IDM requires packed joint coupling so clean-video action "
            "prediction matches conditional training semantics."
        )

    compared = _generated_video_handoff_contract_fields(primary_config, idm_config)
    mismatches = {
        name: {"primary": primary, "external_idm": external}
        for name, (primary, external) in compared.items()
        if primary != external
    }
    if mismatches:
        details = "; ".join(
            f"{name}: primary={values['primary']!r}, "
            f"external_idm={values['external_idm']!r}"
            for name, values in sorted(mismatches.items())
        )
        raise ValueError(
            "Primary video generator and external IDM contracts differ: " + details
        )
    return {
        "route": DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION.value,
        "video_producer_program": (
            None
            if primary_program is None
            else getattr(primary_program, "value", str(primary_program))
        ),
        "video_producer": producer_plan.to_report(),
        "video_producer_route_source": producer_route_source,
        "external_idm_program": idm_program.value,
        "external_idm_route_source": idm_route_source,
        "validated_fields": sorted(compared),
    }


def _validate_video_producer_training_contract(primary_config: Any) -> str:
    """Reject routed checkpoints whose training contract never predicts video."""

    raw_program = getattr(primary_config.policy_variant, "program", None)
    try:
        program = VideoActionProgram(raw_program)
    except (TypeError, ValueError):
        # Video-only policy families use their own program enum. Their typed
        # inference capability remains the authoritative producer declaration.
        return "native_video_policy_program"
    if program is not VideoActionProgram.GENERALIST_JOINT_DENOISING:
        return "fixed_video_action_program"

    video_routes = tuple(
        route
        for route in primary_config.data.dynamics_routing.active_routes
        if resolve_dynamics_objective_semantics(route.mode).video_loss_active
    )
    if not video_routes:
        raise ValueError(
            "A generalist-joint-denoising video producer must declare at least "
            "one active training route with video supervision. A pure-IDM "
            "checkpoint cannot be used as the generated-video stage."
        )
    return "generalist_joint_denoising_training_route"


def build_composed_component_report(
    primary_runtime: DualExpertLiberoRuntime,
    composition: ExternalIdmComposition | None,
) -> dict[str, object]:
    """Return artifact metadata without mutating either runtime report."""

    report = dict(primary_runtime.component_report)
    if composition is None:
        return report
    report["external_idm_composition"] = {
        "checkpoint_file": str(composition.runtime.checkpoint_path.resolve()),
        "component_report": dict(composition.runtime.component_report),
        "compatibility": dict(composition.compatibility_report),
    }
    report["video_producer"] = composition.producer_plan.to_report()
    return report


def _generated_video_handoff_contract_fields(
    primary_config: Any,
    idm_config: Any,
) -> dict[str, tuple[object, object]]:
    primary_data = primary_config.data
    idm_data = idm_config.data
    primary_backbone = primary_config.backbone
    idm_backbone = idm_config.backbone
    return {
        "data.frame_stride": (primary_data.frame_stride, idm_data.frame_stride),
        "data.canonical_height": (
            primary_data.canonical_height,
            idm_data.canonical_height,
        ),
        "data.canonical_width": (
            primary_data.canonical_width,
            idm_data.canonical_width,
        ),
        "data.camera_names": (primary_data.camera_names, idm_data.camera_names),
        "data.latent_camera_names": (
            primary_data.latent_camera_names,
            idm_data.latent_camera_names,
        ),
        "data.view_layout": (primary_data.view_layout, idm_data.view_layout),
        "data.latent_temporal_layout": (
            primary_data.latent_temporal_layout,
            idm_data.latent_temporal_layout,
        ),
        "backbone.latent_channels": (
            primary_backbone.latent_channels,
            idm_backbone.latent_channels,
        ),
        "backbone.latent_stride": (
            primary_backbone.latent_stride,
            idm_backbone.latent_stride,
        ),
        "backbone.patch_size": (
            (
                primary_backbone.patch_size_t,
                primary_backbone.patch_size_h,
                primary_backbone.patch_size_w,
            ),
            (
                idm_backbone.patch_size_t,
                idm_backbone.patch_size_h,
                idm_backbone.patch_size_w,
            ),
        ),
    }


def _argument_error(
    parser: argparse.ArgumentParser | None,
    message: str,
) -> None:
    if parser is not None:
        parser.error(message)
    raise ValueError(message)


__all__ = [
    "ExternalIdmComposition",
    "ExternalIdmLoadOptions",
    "ExternalIdmStepOutput",
    "add_external_idm_arguments",
    "build_composed_component_report",
    "external_idm_options_from_args",
    "infer_external_idm_action",
    "load_external_idm_composition",
    "validate_external_idm_arguments",
    "validate_external_idm_contract",
]
