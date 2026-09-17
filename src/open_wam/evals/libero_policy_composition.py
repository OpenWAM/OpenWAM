"""LIBERO adapter for policy-video to video-conditioned-action composition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from open_wam.models.training_provenance import PolicyTrainingProvenance
from open_wam.evals.libero_policy_inputs import _build_infer_context
from open_wam.evals.libero_policy_runtime import (
    PolicyActionRoute,
    LiberoPolicyLoadOptions,
    LiberoPolicyRuntime,
    LiberoPolicyRuntimeRole,
    load_libero_policy_runtime,
    uses_video_action_composition,
)
from open_wam.models.policy_variants import (
    PolicyGeneratedVideo,
    PolicyOutputModality,
)
from open_wam.models.visual_tower import VisualStageOutputs
from open_wam.pipelines import (
    PolicyVideoActionConsumerPlan,
    PolicyVideoProducerPlan,
    VariantRolloutSession,
    VariantRolloutStepOutput,
    require_compatible_video_latent_spaces,
    resolve_policy_video_action_consumer_plan,
    resolve_policy_video_producer_plan,
)


@dataclass(frozen=True)
class ActionConsumerLoadOptions:
    """Action-consumer checkpoint and device choices for one composed rollout."""

    config: str | Path
    checkpoint: str | Path
    set_overrides: tuple[str, ...] = ()
    runtime_device: str | None = None
    action_device: str | None = None
    frontend_device: str | None = None


@dataclass(frozen=True)
class VideoActionComposition:
    """A policy video producer composed with an external action consumer."""

    runtime: LiberoPolicyRuntime
    producer_plan: PolicyVideoProducerPlan
    consumer_plan: PolicyVideoActionConsumerPlan
    compatibility_report: dict[str, object]


@dataclass(frozen=True)
class VideoActionConsumerStepOutput:
    """One composed action-stage result and its deterministic seed."""

    rollout: VariantRolloutStepOutput
    inference_seed: int | None


def add_action_consumer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the video-conditioned action-consumer options to a LIBERO driver."""

    parser.add_argument(
        "--action-consumer-cfg",
        dest="action_consumer_cfg",
        type=str,
        default=None,
        help="Resolved config for the video-conditioned action consumer.",
    )
    parser.add_argument(
        "--action-consumer-checkpoint",
        dest="action_consumer_checkpoint",
        type=str,
        default=None,
        help="Checkpoint file or checkpoint_step_* directory for the action consumer.",
    )
    parser.add_argument(
        "--action-consumer-set",
        dest="action_consumer_set_overrides",
        action="append",
        default=[],
        help="Apply an override only to the action-consumer config.",
    )
    parser.add_argument(
        "--action-consumer-runtime-device",
        dest="action_consumer_runtime_device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--action-consumer-action-device",
        dest="action_consumer_action_device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--action-consumer-frontend-device",
        dest="action_consumer_frontend_device",
        type=str,
        default=None,
    )


def validate_action_consumer_arguments(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> bool:
    """Validate route-scoped external options and return whether they are active."""

    route = PolicyActionRoute(args.policy_action_route)
    uses_composition = uses_video_action_composition(route)
    supplied = any(
        getattr(args, name, None)
        for name in (
            "action_consumer_cfg",
            "action_consumer_checkpoint",
            "action_consumer_set_overrides",
            "action_consumer_runtime_device",
            "action_consumer_action_device",
            "action_consumer_frontend_device",
        )
    )
    if uses_composition and not (
        args.action_consumer_cfg and args.action_consumer_checkpoint
    ):
        _argument_error(
            parser,
            "generated_video_then_action requires both --action-consumer-cfg "
            "and --action-consumer-checkpoint.",
        )
    if uses_composition and bool(getattr(args, "action_only_rollout", False)):
        _argument_error(
            parser,
            "generated_video_then_action requires the primary policy to produce "
            "video; it cannot be combined with --action-only-rollout.",
        )
    if not uses_composition and supplied:
        _argument_error(
            parser,
            "Action-consumer arguments are only valid with "
            "--action-route generated_video_then_action.",
        )
    return uses_composition


def action_consumer_options_from_args(
    args: argparse.Namespace,
) -> ActionConsumerLoadOptions | None:
    """Build typed external options after argument validation."""

    if not validate_action_consumer_arguments(args):
        return None
    return ActionConsumerLoadOptions(
        config=args.action_consumer_cfg,
        checkpoint=args.action_consumer_checkpoint,
        set_overrides=tuple(args.action_consumer_set_overrides),
        runtime_device=args.action_consumer_runtime_device,
        action_device=args.action_consumer_action_device,
        frontend_device=args.action_consumer_frontend_device,
    )


def load_video_action_composition(
    *,
    primary_runtime: LiberoPolicyRuntime,
    primary_options: LiberoPolicyLoadOptions,
    consumer_options: ActionConsumerLoadOptions | None,
) -> VideoActionComposition | None:
    """Load and validate the optional action stage for one primary runtime."""

    route = PolicyActionRoute(primary_options.policy_action_route)
    if not uses_video_action_composition(route):
        if consumer_options is not None:
            raise ValueError(
                "Action-consumer options require the generated-video composition route."
            )
        return None
    if consumer_options is None:
        raise ValueError(
            "The generated-video composition route requires action-consumer options."
        )

    producer_plan = resolve_policy_video_producer_plan(
        primary_runtime.pipeline.policy_variant,
        training=PolicyTrainingProvenance.from_routes(
            primary_runtime.config.data.dynamics_routing.active_routes
        ),
    )
    runtime = load_libero_policy_runtime(
        LiberoPolicyLoadOptions(
            config=consumer_options.config,
            checkpoint=consumer_options.checkpoint,
            merge_checkpoint_runtime_config=False,
            set_overrides=consumer_options.set_overrides,
            source=f"{primary_options.source} action consumer",
            checkpoint_error="Video/action composition requires a consumer checkpoint.",
            raw_window_frames=primary_runtime.raw_window_frames,
            startup_model_obs_frames=primary_runtime.startup_model_obs_frames,
            startup_env_init_steps=primary_runtime.startup_env_init_steps,
            inference_window_size=(primary_options.inference_window_size),
            rollout_frame_chunk_size=(primary_options.rollout_frame_chunk_size),
            action_only_rollout=False,
            policy_action_route=route.value,
            execute_action_steps=primary_options.execute_action_steps,
            execute_frame_chunk_size=primary_options.execute_frame_chunk_size,
            frontend_encode_mode=primary_options.frontend_encode_mode,
            reset_policy_state_each_chunk=primary_options.reset_policy_state_each_chunk,
            runtime_device=(
                consumer_options.runtime_device or str(primary_runtime.runtime_device)
            ),
            action_device=consumer_options.action_device,
            frontend_device=consumer_options.frontend_device,
            decode_device=consumer_options.frontend_device,
            allow_deprecated_libero_config=(
                primary_options.allow_deprecated_libero_config
            ),
            allow_deprecated_frontend_encode_mode=(
                primary_options.allow_deprecated_frontend_encode_mode
            ),
            checkpoint_load_policy=primary_options.checkpoint_load_policy,
            runtime_role=(LiberoPolicyRuntimeRole.VIDEO_CONDITIONED_ACTION_CONSUMER),
            provided_conditioning_modalities=(PolicyOutputModality.VIDEO,),
        )
    )
    if runtime.action_device != runtime.runtime_device:
        raise ValueError(
            "Video-conditioned action inference requires its runtime and action "
            "modules on the same device; got "
            f"runtime_device={runtime.runtime_device}, "
            f"action_device={runtime.action_device}."
        )
    consumer_plan = resolve_policy_video_action_consumer_plan(
        runtime.pipeline.policy_variant,
        training=PolicyTrainingProvenance.from_routes(
            runtime.config.data.dynamics_routing.active_routes
        ),
    )
    compatibility_report = dict(
        validate_video_action_composition_contract(
            primary_runtime.config,
            runtime.config,
            producer_plan=producer_plan,
            consumer_plan=consumer_plan,
        )
    )
    compatibility_report["video_latent_space"] = require_compatible_video_latent_spaces(
        primary_runtime.pipeline.visual_tower.frontend.latent_space_identity,
        runtime.pipeline.visual_tower.frontend.latent_space_identity,
    )
    return VideoActionComposition(
        runtime=runtime,
        producer_plan=producer_plan,
        consumer_plan=consumer_plan,
        compatibility_report=compatibility_report,
    )


def infer_video_conditioned_action(
    composition: VideoActionComposition,
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
    producer_rng_device: torch.device | str,
) -> VideoActionConsumerStepOutput:
    """Predict actions from a generated video through the public runner API."""

    runtime = composition.runtime
    context = _build_infer_context(
        prompt,
        action_device=runtime.action_device,
        model_obs_window=model_obs_window,
        config=runtime.config,
        runtime_device=runtime.runtime_device,
        inference_window_size=inference_window_size,
        # The typed generated-video request below owns consumer chunk geometry.
        # The generated artifact owns its temporal extent.
        rollout_frame_chunk_size=None,
        action_only_rollout=False,
    )
    rollout = composition.consumer_plan.infer(
        runtime.runner,
        session=replace(session, policy_state=None) if reset_policy_state else session,
        context=context,
        visual_outputs=visual_outputs,
        generated_video=generated_video,
        producer_device=producer_rng_device,
        rollout_seed=rollout_seed,
        step_index=chunk_index,
    )
    return VideoActionConsumerStepOutput(
        rollout=rollout,
        inference_seed=composition.consumer_plan.resolve_step_seed(
            rollout_seed=rollout_seed,
            step_index=chunk_index,
        ),
    )


def validate_video_action_composition_contract(
    primary_config: Any,
    consumer_config: Any,
    *,
    producer_plan: PolicyVideoProducerPlan,
    consumer_plan: PolicyVideoActionConsumerPlan,
) -> dict[str, object]:
    """Require a lossless generated-video handoff into one action consumer."""

    primary_program = getattr(primary_config.policy_variant, "program", None)
    consumer_program = getattr(consumer_config.policy_variant, "program", None)

    compared = _generated_video_handoff_contract_fields(
        primary_config,
        consumer_config,
    )
    mismatches = {
        name: {"producer": producer, "consumer": consumer}
        for name, (producer, consumer) in compared.items()
        if producer != consumer
    }
    if mismatches:
        details = "; ".join(
            f"{name}: producer={values['producer']!r}, consumer={values['consumer']!r}"
            for name, values in sorted(mismatches.items())
        )
        raise ValueError(
            "Video producer and action consumer contracts differ: " + details
        )
    return {
        "route": PolicyActionRoute.GENERATED_VIDEO_THEN_ACTION.value,
        "video_producer_program": (
            None
            if primary_program is None
            else getattr(primary_program, "value", str(primary_program))
        ),
        "video_producer": producer_plan.to_report(),
        "action_consumer_program": getattr(consumer_program, "value", consumer_program),
        "action_consumer": consumer_plan.to_report(),
        "validated_fields": sorted(compared),
    }




def build_composed_component_report(
    primary_runtime: LiberoPolicyRuntime,
    composition: VideoActionComposition | None,
) -> dict[str, object]:
    """Return artifact metadata without mutating either runtime report."""

    report = dict(primary_runtime.component_report)
    if composition is None:
        return report
    report["video_action_composition"] = {
        "checkpoint_file": str(composition.runtime.checkpoint_path.resolve()),
        "component_report": dict(composition.runtime.component_report),
        "compatibility": dict(composition.compatibility_report),
    }
    report["video_producer"] = composition.producer_plan.to_report()
    report["action_consumer"] = composition.consumer_plan.to_report()
    return report


def _generated_video_handoff_contract_fields(
    primary_config: Any,
    consumer_config: Any,
) -> dict[str, tuple[object, object]]:
    primary_data = primary_config.data
    consumer_data = consumer_config.data
    primary_backbone = primary_config.backbone
    consumer_backbone = consumer_config.backbone
    return {
        "data.frame_stride": (
            primary_data.frame_stride,
            consumer_data.frame_stride,
        ),
        "data.canonical_height": (
            primary_data.canonical_height,
            consumer_data.canonical_height,
        ),
        "data.canonical_width": (
            primary_data.canonical_width,
            consumer_data.canonical_width,
        ),
        "data.camera_names": (
            primary_data.camera_names,
            consumer_data.camera_names,
        ),
        "data.latent_camera_names": (
            primary_data.latent_camera_names,
            consumer_data.latent_camera_names,
        ),
        "data.view_layout": (
            primary_data.view_layout,
            consumer_data.view_layout,
        ),
        "data.latent_temporal_layout": (
            primary_data.latent_temporal_layout,
            consumer_data.latent_temporal_layout,
        ),
        "backbone.latent_channels": (
            primary_backbone.latent_channels,
            consumer_backbone.latent_channels,
        ),
        "backbone.latent_stride": (
            primary_backbone.latent_stride,
            consumer_backbone.latent_stride,
        ),
        "backbone.patch_size": (
            (
                primary_backbone.patch_size_t,
                primary_backbone.patch_size_h,
                primary_backbone.patch_size_w,
            ),
            (
                consumer_backbone.patch_size_t,
                consumer_backbone.patch_size_h,
                consumer_backbone.patch_size_w,
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
    "ActionConsumerLoadOptions",
    "VideoActionComposition",
    "VideoActionConsumerStepOutput",
    "action_consumer_options_from_args",
    "add_action_consumer_arguments",
    "build_composed_component_report",
    "infer_video_conditioned_action",
    "load_video_action_composition",
    "validate_action_consumer_arguments",
    "validate_video_action_composition_contract",
]
