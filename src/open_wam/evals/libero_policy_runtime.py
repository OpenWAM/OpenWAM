"""Reusable checkpoint/config/device composition for policy rollouts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import torch

from open_wam.configs import (
    ExperimentConfig,
    ReferenceCoreInitMode,
    load_experiment_config,
    read_yaml_with_local_paths,
)
from open_wam.evals.libero_visualization import resolve_device as _resolve_device
from open_wam.models.common.rollout_history import (
    resolve_execute_action_steps as _resolve_shared_execute_action_steps,
)
from open_wam.models.policy_variants import (
    PolicyOutputModality,
    PolicyCompositionCapability,
)
from open_wam.models.visual_tower import resolve_runtime_backbone_dir
from open_wam.pipelines import (
    VariantPipeline,
    VariantRolloutRunner,
    build_variant_pipeline_from_config,
)
from open_wam.runtime.checkpoint_artifacts import is_usable_transformer_dir
from open_wam.runtime.rollout_temporal import ResolvedRolloutTemporalContract
from open_wam.runtime.checkpoints import (
    CheckpointCompatibilityPolicy,
    load_pipeline_checkpoint,
    resolve_checkpoint_file,
    resolve_checkpoint_step_dir_from_transformer_dir,
)
from open_wam.utils import (
    apply_config_overrides,
    merge_runtime_config_from_checkpoint,
    parse_override_assignments,
)
from open_wam.utils.libero_paradigm import (
    require_current_libero_policy_paradigm,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

CURRENT_FRONTEND_ENCODE_MODE = "lingbot_streaming_vae"
DEPRECATED_FRONTEND_ENCODE_MODE = "rolling_offline"


class PolicyActionRoute(str, Enum):
    """Action source selected by the maintained Policy LIBERO runner."""

    NATIVE = "native"
    JOINT_VIDEO_THEN_IDM = "joint_video_then_idm"
    GENERATED_VIDEO_THEN_ACTION = "generated_video_then_action"


class LiberoPolicyRuntimeRole(str, Enum):
    """Integration role used to validate one loaded LIBERO policy runtime."""

    NATIVE_POLICY = "native_policy"
    VIDEO_PRODUCER = "video_producer"
    VIDEO_CONDITIONED_ACTION_CONSUMER = "video_conditioned_action_consumer"


GJD_ACTION_ROUTES = frozenset(
    {
        PolicyActionRoute.NATIVE.value,
        PolicyActionRoute.JOINT_VIDEO_THEN_IDM.value,
    }
)
POLICY_ACTION_ROUTES = frozenset(
    {
        *GJD_ACTION_ROUTES,
        PolicyActionRoute.GENERATED_VIDEO_THEN_ACTION.value,
    }
)
VIDEO_ACTION_COMPOSITION_ROUTES = frozenset(
    {PolicyActionRoute.GENERATED_VIDEO_THEN_ACTION}
)


def uses_video_action_composition(route: PolicyActionRoute | str) -> bool:
    """Return whether a rollout composes policy video with another action model."""

    return PolicyActionRoute(route) in VIDEO_ACTION_COMPOSITION_ROUTES


__all__ = [
    "CURRENT_FRONTEND_ENCODE_MODE",
    "DEPRECATED_FRONTEND_ENCODE_MODE",
    "POLICY_ACTION_ROUTES",
    "GJD_ACTION_ROUTES",
    "VIDEO_ACTION_COMPOSITION_ROUTES",
    "PolicyActionRoute",
    "LiberoPolicyLoadOptions",
    "LiberoPolicyRuntime",
    "LiberoPolicyRuntimeRole",
    "load_libero_policy_runtime",
    "print_rollout_event",
    "uses_video_action_composition",
]


_resolve_execute_action_steps = _resolve_shared_execute_action_steps


@dataclass
class LiberoPolicyRuntime:
    """Loaded model resources shared across one or many LIBERO episodes."""

    config: ExperimentConfig
    checkpoint_path: Path
    pipeline: VariantPipeline
    runner: VariantRolloutRunner
    component_report: dict[str, object]
    runtime_device: torch.device
    action_device: torch.device
    frontend_device: torch.device
    decode_device: torch.device
    raw_window_frames: int
    startup_model_obs_frames: int
    startup_env_init_steps: int
    use_lingbot_streaming_vae: bool


@dataclass(frozen=True)
class LiberoPolicyLoadOptions:
    """Inputs needed to load and validate one reusable policy runtime."""

    config: str | Path
    checkpoint: str | Path | None
    merge_checkpoint_runtime_config: bool
    set_overrides: tuple[str, ...]
    source: str
    checkpoint_error: str
    raw_window_frames: int | None
    startup_model_obs_frames: int
    startup_env_init_steps: int
    inference_window_size: int | None
    rollout_frame_chunk_size: int | None
    action_only_rollout: bool
    policy_action_route: str
    execute_action_steps: int | None
    execute_frame_chunk_size: int | None
    frontend_encode_mode: str
    reset_policy_state_each_chunk: bool
    runtime_device: str | None
    action_device: str | None
    frontend_device: str | None
    decode_device: str | None
    allow_deprecated_libero_config: bool
    allow_deprecated_frontend_encode_mode: bool
    checkpoint_load_policy: CheckpointCompatibilityPolicy = (
        CheckpointCompatibilityPolicy.ALLOW_CHECKPOINT_SUPERSET
    )
    runtime_role: LiberoPolicyRuntimeRole = LiberoPolicyRuntimeRole.NATIVE_POLICY
    provided_conditioning_modalities: tuple[PolicyOutputModality, ...] = ()
    component_report_extra: Mapping[str, object] = field(default_factory=dict)


def load_libero_policy_runtime(options: LiberoPolicyLoadOptions) -> LiberoPolicyRuntime:
    """Load one validated runtime shared by single and batch rollout drivers."""

    runtime_role = LiberoPolicyRuntimeRole(options.runtime_role)
    _validate_runtime_role_inputs(
        runtime_role,
        action_route=options.policy_action_route,
        provided_modalities=options.provided_conditioning_modalities,
    )
    config_path = Path(options.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(
        config_path,
        checkpoint_runtime_compat=options.allow_deprecated_libero_config,
    )
    checkpoint_path = _resolve_policy_checkpoint_path(
        config_path=config_path,
        checkpoint_arg=(
            None if options.checkpoint is None else str(options.checkpoint)
        ),
        runtime_backbone_artifact_path=resolve_runtime_backbone_dir(config.backbone),
    )
    if checkpoint_path is None:
        raise ValueError(options.checkpoint_error)
    config, checkpoint_runtime_config_path = _maybe_merge_checkpoint_runtime_config(
        config,
        checkpoint_path,
        merge_enabled=options.merge_checkpoint_runtime_config,
    )
    if options.set_overrides:
        config = apply_config_overrides(
            config,
            parse_override_assignments(options.set_overrides),
        )
    require_current_libero_policy_paradigm(
        config,
        config_path=config_path,
        source=options.source,
        allow_deprecated=options.allow_deprecated_libero_config,
    )
    transformer_dir = checkpoint_path.parent / "transformer"
    if is_usable_transformer_dir(transformer_dir):
        config = replace(
            config,
            backbone=replace(
                config.backbone,
                runtime_backbone_artifact_path=str(transformer_dir.resolve()),
                reference_core_init_mode=ReferenceCoreInitMode.FULL,
            ),
        )

    runtime_device = _resolve_device(options.runtime_device)
    action_device = _resolve_device(
        options.action_device,
        fallback=runtime_device,
    )
    frontend_device = _resolve_device(
        options.frontend_device,
        fallback=runtime_device,
    )
    decode_device = _resolve_device(
        options.decode_device,
        fallback=frontend_device,
    )
    startup_model_obs_frames = int(options.startup_model_obs_frames)
    if startup_model_obs_frames <= 0:
        raise ValueError(
            "Expected --startup-model-obs-frames to be positive, "
            f"got {startup_model_obs_frames}."
        )
    startup_env_init_steps = int(options.startup_env_init_steps)
    if startup_env_init_steps <= 0:
        raise ValueError(
            "Expected --startup-env-init-steps to be positive, "
            f"got {startup_env_init_steps}."
        )
    if (
        options.inference_window_size is not None
        and int(options.inference_window_size) <= 0
    ):
        raise ValueError(
            "Expected --inference-window-size to be positive when provided, "
            f"got {options.inference_window_size}."
        )
    if options.rollout_frame_chunk_size is not None:
        rollout_frame_chunk_size = int(options.rollout_frame_chunk_size)
        configured_frame_chunk_size = _frame_chunk_size(config)
        if rollout_frame_chunk_size <= 0:
            raise ValueError(
                "Expected --rollout-frame-chunk-size to be positive when "
                f"provided, got {options.rollout_frame_chunk_size}."
            )
        if rollout_frame_chunk_size > configured_frame_chunk_size:
            raise ValueError(
                "--rollout-frame-chunk-size cannot exceed configured "
                "inference.frame_chunk_size, "
                f"got override={rollout_frame_chunk_size}, "
                f"configured={configured_frame_chunk_size}."
            )
    _require_current_frontend_encode_mode(
        options.frontend_encode_mode,
        allow_deprecated=options.allow_deprecated_frontend_encode_mode,
        source=options.source,
    )
    use_lingbot_streaming_vae = (
        options.frontend_encode_mode == CURRENT_FRONTEND_ENCODE_MODE
    )
    if use_lingbot_streaming_vae and startup_model_obs_frames != 1:
        raise ValueError(
            "`--frontend-encode-mode lingbot_streaming_vae` expects "
            "`--startup-model-obs-frames 1` to match LingBot-VA's first-frame "
            "bootstrap."
        )
    if use_lingbot_streaming_vae and options.reset_policy_state_each_chunk:
        raise ValueError(
            "`--frontend-encode-mode lingbot_streaming_vae` requires persistent "
            "policy/frontend state; drop `--reset-policy-state-each-chunk`."
        )

    pipeline = build_variant_pipeline_from_config(config)
    temporal = ResolvedRolloutTemporalContract.from_pipeline(pipeline)
    raw_window_frames = (
        int(options.raw_window_frames)
        if options.raw_window_frames is not None
        else temporal.raw_window_frames(int(config.data.num_frames))
    )
    if startup_model_obs_frames > raw_window_frames:
        raise ValueError(
            "--startup-model-obs-frames must be <= --raw-window-frames, "
            f"got startup_model_obs_frames={startup_model_obs_frames}, "
            f"raw_window_frames={raw_window_frames}."
        )
    _validate_libero_policy_runtime(
        pipeline, runtime_role=runtime_role,
        provided_modalities=options.provided_conditioning_modalities,
    )
    if runtime_role is not LiberoPolicyRuntimeRole.VIDEO_PRODUCER:
        _resolve_execute_action_steps(
            options.execute_action_steps,
            execute_frame_chunk_size=options.execute_frame_chunk_size,
            action_horizon=temporal.prediction_controls,
            action_per_frame=temporal.controls_per_frame,
        )
    checkpoint_report = load_pipeline_checkpoint(
        pipeline,
        checkpoint_path,
        compatibility=options.checkpoint_load_policy,
    )
    if checkpoint_report.missing_keys:
        print(f"viz.checkpoint_missing_keys {len(checkpoint_report.missing_keys)}")
    if checkpoint_report.unexpected_keys:
        print(
            f"viz.checkpoint_unexpected_keys {len(checkpoint_report.unexpected_keys)}"
        )
    pipeline.to(device=runtime_device)
    for module in pipeline.module_topology().action_expert_modules:
        module.to(device=action_device)
    pipeline.eval()
    runner = VariantRolloutRunner(pipeline)
    component_report = _build_component_report(
        config,
        pipeline,
        runtime_device=runtime_device,
        action_device=action_device,
        frontend_device=frontend_device,
        decode_device=decode_device,
        raw_window_frames=raw_window_frames,
        inference_window_size=options.inference_window_size,
        rollout_frame_chunk_size=options.rollout_frame_chunk_size,
        action_only_rollout=options.action_only_rollout,
        policy_action_route=options.policy_action_route,
    )
    component_report.update(
        {
            **dict(options.component_report_extra),
            "feature_cache_scope": pipeline.policy_variant.inference_capabilities.feature_cache_scope.value,
            "checkpoint_file": str(checkpoint_path.resolve()),
            "checkpoint_runtime_config_path": (
                None
                if checkpoint_runtime_config_path is None
                else str(checkpoint_runtime_config_path)
            ),
            "checkpoint_runtime_config_merged": (
                checkpoint_runtime_config_path is not None
            ),
            "pipeline_training_mode": bool(pipeline.training),
            "runtime_role": runtime_role.value,
            "frontend_encode_mode": str(options.frontend_encode_mode),
            "rollout_frame_chunk_size": (
                None
                if options.rollout_frame_chunk_size is None
                else int(options.rollout_frame_chunk_size)
            ),
            "execute_action_steps": (
                None
                if options.execute_action_steps is None
                else int(options.execute_action_steps)
            ),
            "execute_frame_chunk_size": (
                None
                if options.execute_frame_chunk_size is None
                else int(options.execute_frame_chunk_size)
            ),
        }
    )
    if options.provided_conditioning_modalities:
        component_report["provided_conditioning_modalities"] = [
            PolicyOutputModality(modality).value
            for modality in options.provided_conditioning_modalities
        ]
    _print_log("load_report", component_report)
    return LiberoPolicyRuntime(
        config=config,
        checkpoint_path=checkpoint_path,
        pipeline=pipeline,
        runner=runner,
        component_report=component_report,
        runtime_device=runtime_device,
        action_device=action_device,
        frontend_device=frontend_device,
        decode_device=decode_device,
        raw_window_frames=raw_window_frames,
        startup_model_obs_frames=startup_model_obs_frames,
        startup_env_init_steps=startup_env_init_steps,
        use_lingbot_streaming_vae=use_lingbot_streaming_vae,
    )


def _validate_libero_policy_runtime(
    pipeline: VariantPipeline,
    *,
    runtime_role: LiberoPolicyRuntimeRole,
    provided_modalities: tuple[PolicyOutputModality, ...] = (),
) -> None:
    """Validate the requested role by products, not by architecture names."""

    role = LiberoPolicyRuntimeRole(runtime_role)
    capabilities = pipeline.policy_variant.inference_capabilities
    capabilities.require_future_inputs(frozenset(provided_modalities))
    if role is LiberoPolicyRuntimeRole.VIDEO_CONDITIONED_ACTION_CONSUMER:
        if not capabilities.supports_composition(
            PolicyCompositionCapability.video_to_action()
        ):
            raise ValueError(
                "Action consumers must support generated-video conditioning."
            )
    required = (
        PolicyOutputModality.VIDEO
        if role is LiberoPolicyRuntimeRole.VIDEO_PRODUCER
        else PolicyOutputModality.ACTION
    )
    if required not in capabilities.native_modalities:
        raise ValueError(
            f"LIBERO runtime role {role.value!r} requires {required.value} output."
        )


def _validate_runtime_role_inputs(
    runtime_role: LiberoPolicyRuntimeRole,
    *,
    action_route: PolicyActionRoute | str,
    provided_modalities: tuple[PolicyOutputModality, ...],
) -> None:
    """Require explicit clean inputs for conditional consumer runtimes."""

    role = LiberoPolicyRuntimeRole(runtime_role)
    composed_route = uses_video_action_composition(action_route)
    if composed_route and role is LiberoPolicyRuntimeRole.NATIVE_POLICY:
        raise ValueError(
            "The generated-video action route requires an explicit video-producer "
            "or video-conditioned-action-consumer runtime role."
        )
    if not composed_route and role is not LiberoPolicyRuntimeRole.NATIVE_POLICY:
        raise ValueError(
            f"LIBERO runtime role {role.value!r} requires the "
            "generated-video action composition route."
        )
    available = frozenset(
        PolicyOutputModality(modality) for modality in provided_modalities
    )
    expected = (
        frozenset({PolicyOutputModality.VIDEO})
        if role is LiberoPolicyRuntimeRole.VIDEO_CONDITIONED_ACTION_CONSUMER
        else frozenset()
    )
    if available != expected:
        raise ValueError(
            f"LIBERO runtime role {role.value!r} requires provided conditioning "
            f"modalities {sorted(item.value for item in expected)!r}; got "
            f"{sorted(item.value for item in available)!r}. Only the "
            "video-conditioned action consumer receives a clean future tensor "
            "from this composition route."
        )


def _require_current_frontend_encode_mode(
    frontend_encode_mode: str,
    *,
    allow_deprecated: bool,
    source: str,
) -> None:
    if frontend_encode_mode == CURRENT_FRONTEND_ENCODE_MODE:
        return
    if allow_deprecated:
        return
    raise ValueError(
        f"{source} frontend encode mode {frontend_encode_mode!r} is deprecated. "
        f"Use `--frontend-encode-mode {CURRENT_FRONTEND_ENCODE_MODE}`. Pass "
        "`--allow-deprecated-frontend-encode-mode` only for historical debugging."
    )




def _maybe_merge_checkpoint_runtime_config(
    config,
    checkpoint_path: Path,
    *,
    merge_enabled: bool,
):
    if not merge_enabled:
        return config, None
    return merge_runtime_config_from_checkpoint(config, checkpoint_path)


def _resolve_policy_checkpoint_path(
    *,
    config_path: Path,
    checkpoint_arg: str | None,
    runtime_backbone_artifact_path: str | Path | None,
) -> Path | None:
    if checkpoint_arg is not None:
        return resolve_checkpoint_file(Path(checkpoint_arg))
    raw = read_yaml_with_local_paths(config_path)
    raw_checkpoint = raw.get("checkpoint_path")
    if raw_checkpoint is not None:
        return resolve_checkpoint_file(Path(str(raw_checkpoint)))
    if runtime_backbone_artifact_path is None:
        return None
    try:
        return resolve_checkpoint_file(
            resolve_checkpoint_step_dir_from_transformer_dir(
                runtime_backbone_artifact_path
            )
        )
    except (FileNotFoundError, ValueError):
        return None


def _frame_chunk_size(config) -> int:
    return int(config.inference.frame_chunk_size)


def _build_component_report(
    config,
    pipeline,
    *,
    runtime_device: torch.device,
    action_device: torch.device,
    frontend_device: torch.device,
    decode_device: torch.device,
    raw_window_frames: int,
    inference_window_size: int | None,
    rollout_frame_chunk_size: int | None,
    action_only_rollout: bool,
    policy_action_route: str,
) -> dict[str, object]:
    backbone = config.backbone
    policy_variant = pipeline.policy_variant
    policy_config = policy_variant.config
    action_expert = getattr(policy_variant, "action_expert", None)
    runtime_backbone_dir = resolve_runtime_backbone_dir(backbone)
    program = getattr(policy_config, "program", None)
    return {
        "pipeline": "variant_rollout",
        "runtime_device": str(runtime_device),
        "action_device": str(action_device),
        "frontend_device": str(frontend_device),
        "decode_device": str(decode_device),
        "raw_window_frames": int(raw_window_frames),
        "inference_window_size": (
            None if inference_window_size is None else int(inference_window_size)
        ),
        "rollout_frame_chunk_size": (
            None if rollout_frame_chunk_size is None else int(rollout_frame_chunk_size)
        ),
        "action_only_rollout": bool(action_only_rollout),
        "policy_action_route": str(policy_action_route),
        "config_name": config.name,
        "policy_variant_class": policy_variant.__class__.__name__,
        "program": None if program is None else getattr(program, "value", str(program)),
        "condition_mode": (
            None
            if not hasattr(policy_config, "condition_mode")
            else str(policy_config.condition_mode)
        ),
        "video_prefix_frames": (
            None
            if not hasattr(policy_config, "video_prefix_frames")
            else int(policy_config.video_prefix_frames)
        ),
        "current_block_coupling": (
            None
            if not hasattr(policy_config, "current_block_coupling")
            else getattr(
                policy_config.current_block_coupling,
                "value",
                str(policy_config.current_block_coupling),
            )
        ),
        "backbone_hidden_size": int(backbone.hidden_size),
        "backbone_num_layers": int(backbone.num_layers),
        "action_hidden_size": (
            None
            if action_expert is None
            else int(getattr(action_expert, "hidden_size", 0))
        ),
        "action_num_layers": (
            None
            if not hasattr(policy_config, "num_action_layers")
            else int(policy_config.num_action_layers)
        ),
        "action_horizon": int(config.data.action_schema.action_horizon),
        "action_dim": int(config.data.action_schema.action_dim),
        "trainable_parameters": _count_trainable_parameters(pipeline),
        "total_parameters": sum(
            parameter.numel() for parameter in pipeline.parameters()
        ),
        "backbone_pretrained_root": str(backbone.pretrained_model_name_or_path),
        "runtime_backbone_artifact_path": (
            None if runtime_backbone_dir is None else str(runtime_backbone_dir)
        ),
        "config_sha256": _sha256_if_exists(
            None
            if runtime_backbone_dir is None
            else runtime_backbone_dir / "config.json"
        ),
    }


def _sha256_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_rollout_event(label: str, payload: dict[str, object]) -> None:
    """Write one stable JSON event for CLI and batch-driver progress logs."""

    print(f"[{label}] {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


_print_log = print_rollout_event


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
