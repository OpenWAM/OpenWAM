"""Reusable checkpoint/config/device composition for M5 and GJD rollouts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

from open_wam.configs import (
    ExperimentConfig,
    ReferenceCoreInitMode,
    load_experiment_config,
    read_yaml_with_local_paths,
)
from open_wam.data.latent_temporal import raw_window_frames_for_latents
from open_wam.evals.libero_visualization import resolve_device as _resolve_device
from open_wam.models.common.rollout_history import (
    resolve_execute_action_steps as _resolve_shared_execute_action_steps,
)
from open_wam.models.policy_variants.mot.runtime_routing import (
    ensure_mot_inference_backend,
)
from open_wam.pipelines import (
    VariantPipeline,
    VariantRolloutRunner,
    build_variant_pipeline_from_config,
)
from open_wam.runtime.checkpoints import (
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
LIVE_SIM_MOT_GENERALIST_ROLLOUT_MODES = frozenset(
    {
        "joint",
        "vanilla_joint_rollout",
    }
)
MOT_GJD_ACTION_ROUTES = frozenset(
    {
        "joint",
        "joint_video_then_idm",
    }
)
OFFLINE_DIAGNOSTIC_MOT_GENERALIST_ROLLOUT_MODES = frozenset(
    {
        "clean_action_feedback",
        "forced_action_joint_fdm",
        "action_conditioned_video",
        "video_conditioned_action",
        "fdm",
        "idm",
    }
)

__all__ = [
    "CURRENT_FRONTEND_ENCODE_MODE",
    "DEPRECATED_FRONTEND_ENCODE_MODE",
    "MOT_GJD_ACTION_ROUTES",
    "MotLiberoLoadOptions",
    "MotLiberoRuntime",
    "load_mot_libero_runtime",
    "print_rollout_event",
]


_default_raw_window_frames = raw_window_frames_for_latents
_resolve_execute_action_steps = _resolve_shared_execute_action_steps


@dataclass
class MotLiberoRuntime:
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
class MotLiberoLoadOptions:
    """Inputs needed to load and validate one reusable M5/GJD runtime."""

    config: str | Path
    checkpoint: str | Path | None
    merge_checkpoint_runtime_config: bool
    set_overrides: tuple[str, ...]
    source: str
    checkpoint_error: str
    raw_window_frames: int | None
    startup_model_obs_frames: int
    startup_env_init_steps: int
    mot_inference_window_size: int | None
    mot_rollout_frame_chunk_size: int | None
    mot_action_only_rollout: bool
    mot_generalist_rollout_mode: str | None
    mot_gjd_action_route: str
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
    component_report_extra: Mapping[str, object] = field(default_factory=dict)


def load_mot_libero_runtime(options: MotLiberoLoadOptions) -> MotLiberoRuntime:
    """Load one validated runtime shared by single and batch rollout drivers."""

    _validate_live_sim_mot_generalist_rollout_mode(
        options.mot_generalist_rollout_mode
    )
    config_path = Path(options.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    _validate_mot_config(config)
    checkpoint_path = _resolve_mot_checkpoint_path(
        config_path=config_path,
        checkpoint_arg=(
            None if options.checkpoint is None else str(options.checkpoint)
        ),
        transformer_subdir=str(config.backbone.transformer_subdir),
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
    _validate_mot_config(config)
    require_current_libero_policy_paradigm(
        config,
        config_path=config_path,
        source=options.source,
        allow_deprecated=options.allow_deprecated_libero_config,
    )
    transformer_dir = checkpoint_path.parent / "transformer"
    if transformer_dir.is_dir():
        object.__setattr__(
            config.backbone,
            "transformer_subdir",
            str(transformer_dir.resolve()),
        )
        object.__setattr__(
            config.backbone,
            "reference_core_init_mode",
            ReferenceCoreInitMode.FULL,
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
    raw_window_frames = (
        int(options.raw_window_frames)
        if options.raw_window_frames is not None
        else _default_raw_window_frames(int(config.data.num_frames))
    )
    startup_model_obs_frames = int(options.startup_model_obs_frames)
    if startup_model_obs_frames <= 0:
        raise ValueError(
            "Expected --startup-model-obs-frames to be positive, "
            f"got {startup_model_obs_frames}."
        )
    if startup_model_obs_frames > raw_window_frames:
        raise ValueError(
            "--startup-model-obs-frames must be <= --raw-window-frames, "
            f"got startup_model_obs_frames={startup_model_obs_frames}, "
            f"raw_window_frames={raw_window_frames}."
        )
    startup_env_init_steps = int(options.startup_env_init_steps)
    if startup_env_init_steps <= 0:
        raise ValueError(
            "Expected --startup-env-init-steps to be positive, "
            f"got {startup_env_init_steps}."
        )
    if (
        options.mot_inference_window_size is not None
        and int(options.mot_inference_window_size) <= 0
    ):
        raise ValueError(
            "Expected --mot-inference-window-size to be positive when provided, "
            f"got {options.mot_inference_window_size}."
        )
    if options.mot_rollout_frame_chunk_size is not None:
        rollout_frame_chunk_size = int(options.mot_rollout_frame_chunk_size)
        configured_frame_chunk_size = _frame_chunk_size(config)
        if rollout_frame_chunk_size <= 0:
            raise ValueError(
                "Expected --mot-rollout-frame-chunk-size to be positive when "
                f"provided, got {options.mot_rollout_frame_chunk_size}."
            )
        if rollout_frame_chunk_size > configured_frame_chunk_size:
            raise ValueError(
                "--mot-rollout-frame-chunk-size cannot exceed configured "
                "inference.frame_chunk_size, "
                f"got override={rollout_frame_chunk_size}, "
                f"configured={configured_frame_chunk_size}."
            )
    if (
        options.execute_action_steps is not None
        or options.execute_frame_chunk_size is not None
    ):
        _resolve_execute_action_steps(
            options.execute_action_steps,
            execute_frame_chunk_size=options.execute_frame_chunk_size,
            action_horizon=int(config.data.action_schema.action_horizon),
            action_per_frame=_action_per_frame(config),
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
    checkpoint_report = load_pipeline_checkpoint(pipeline, checkpoint_path)
    if checkpoint_report.missing_keys:
        print(f"viz.checkpoint_missing_keys {len(checkpoint_report.missing_keys)}")
    if checkpoint_report.unexpected_keys:
        print(
            "viz.checkpoint_unexpected_keys "
            f"{len(checkpoint_report.unexpected_keys)}"
        )
    pipeline.to(device=runtime_device)
    if hasattr(pipeline.policy_variant, "_maybe_initialize_action_expert"):
        pipeline.policy_variant._maybe_initialize_action_expert(
            pipeline.visual_tower
        )
    mot_inference_backend = ensure_mot_inference_backend(pipeline, config)
    if mot_inference_backend["legacy_split_cache_restored_this_call"]:
        _print_log(
            "stage",
            {"name": "mot_legacy_cache_inference_blocks_restored"},
        )
    if hasattr(pipeline.policy_variant, "action_expert"):
        pipeline.policy_variant.action_expert.to(device=action_device)
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
        mot_inference_window_size=options.mot_inference_window_size,
        mot_rollout_frame_chunk_size=options.mot_rollout_frame_chunk_size,
        mot_action_only_rollout=options.mot_action_only_rollout,
        mot_generalist_rollout_mode=options.mot_generalist_rollout_mode,
        mot_gjd_action_route=options.mot_gjd_action_route,
    )
    component_report.update(
        {
            "mot_inference_backend": mot_inference_backend,
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
            "frontend_encode_mode": str(options.frontend_encode_mode),
            "mot_rollout_frame_chunk_size": (
                None
                if options.mot_rollout_frame_chunk_size is None
                else int(options.mot_rollout_frame_chunk_size)
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
            **dict(options.component_report_extra),
        }
    )
    _print_log("load_report", component_report)
    return MotLiberoRuntime(
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


def _validate_mot_config(config) -> None:
    if str(config.policy_variant.name) != "mot":
        raise ValueError(
            "run_libero_mot_visualization.py requires a `mot` policy variant, "
            f"got policy_variant.name={config.policy_variant.name!r}."
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


def _validate_live_sim_mot_generalist_rollout_mode(mode: str | None) -> None:
    if mode is None or mode in LIVE_SIM_MOT_GENERALIST_ROLLOUT_MODES:
        return
    if mode in OFFLINE_DIAGNOSTIC_MOT_GENERALIST_ROLLOUT_MODES:
        supported = ", ".join(sorted(LIVE_SIM_MOT_GENERALIST_ROLLOUT_MODES))
        raise ValueError(
            f"--mot-generalist-rollout-mode={mode!r} is an offline diagnostic mode, not a live sim rollout mode. "
            "It requires ground-truth clean action and/or video condition tensors that this LIBERO visualization "
            f"script does not provide. Use one of [{supported}] here, or use "
            "open_wam.evals.dynamics.cli for offline FDM/IDM diagnostics."
        )
    raise ValueError(f"Unsupported --mot-generalist-rollout-mode={mode!r}.")


def _maybe_merge_checkpoint_runtime_config(
    config,
    checkpoint_path: Path,
    *,
    merge_enabled: bool,
):
    if not merge_enabled:
        return config, None
    return merge_runtime_config_from_checkpoint(config, checkpoint_path)


def _resolve_mot_checkpoint_path(
    *,
    config_path: Path,
    checkpoint_arg: str | None,
    transformer_subdir: str | None,
) -> Path | None:
    if checkpoint_arg is not None:
        return resolve_checkpoint_file(Path(checkpoint_arg))
    raw = read_yaml_with_local_paths(config_path)
    raw_checkpoint = raw.get("checkpoint_path")
    if raw_checkpoint is not None:
        return resolve_checkpoint_file(Path(str(raw_checkpoint)))
    if transformer_subdir is None:
        return None
    try:
        return resolve_checkpoint_file(
            resolve_checkpoint_step_dir_from_transformer_dir(transformer_subdir)
        )
    except (FileNotFoundError, ValueError):
        return None


def _frame_chunk_size(config) -> int:
    frame_chunk_size = max(1, int(config.inference.frame_chunk_size))
    action_horizon = int(config.data.action_schema.action_horizon)
    if action_horizon % frame_chunk_size != 0:
        raise ValueError(
            "MoT rollout expects action_horizon to divide by inference.frame_chunk_size, "
            f"got action_horizon={action_horizon}, frame_chunk_size={frame_chunk_size}."
        )
    return frame_chunk_size


def _action_per_frame(config) -> int:
    return max(1, int(config.data.action_schema.action_horizon) // _frame_chunk_size(config))


def _build_component_report(
    config,
    pipeline,
    *,
    runtime_device: torch.device,
    action_device: torch.device,
    frontend_device: torch.device,
    decode_device: torch.device,
    raw_window_frames: int,
    mot_inference_window_size: int | None,
    mot_rollout_frame_chunk_size: int | None,
    mot_action_only_rollout: bool,
    mot_generalist_rollout_mode: str | None,
    mot_gjd_action_route: str,
) -> dict[str, object]:
    backbone = config.backbone
    policy_variant = pipeline.policy_variant
    action_expert = getattr(policy_variant, "action_expert", None)
    return {
        "pipeline": "open_wam_mot",
        "runtime_device": str(runtime_device),
        "action_device": str(action_device),
        "frontend_device": str(frontend_device),
        "decode_device": str(decode_device),
        "raw_window_frames": int(raw_window_frames),
        "mot_inference_window_size": (
            None if mot_inference_window_size is None else int(mot_inference_window_size)
        ),
        "mot_rollout_frame_chunk_size": (
            None if mot_rollout_frame_chunk_size is None else int(mot_rollout_frame_chunk_size)
        ),
        "mot_action_only_rollout": bool(mot_action_only_rollout),
        "mot_generalist_rollout_mode": mot_generalist_rollout_mode,
        "mot_gjd_action_route": str(mot_gjd_action_route),
        "config_name": config.name,
        "policy_variant_class": policy_variant.__class__.__name__,
        "runtime_mode": str(policy_variant.config.runtime_mode),
        "condition_mode": str(policy_variant.config.condition_mode),
        "video_prefix_frames": int(policy_variant.config.video_prefix_frames),
        "video_can_attend_action": bool(getattr(policy_variant.config, "video_can_attend_action", False)),
        "backbone_hidden_size": int(backbone.hidden_size),
        "backbone_num_layers": int(backbone.num_layers),
        "action_hidden_size": (
            None if action_expert is None else int(getattr(action_expert, "hidden_size", 0))
        ),
        "action_num_layers": int(policy_variant.config.num_action_layers),
        "action_horizon": int(config.data.action_schema.action_horizon),
        "action_dim": int(config.data.action_schema.action_dim),
        "trainable_parameters": _count_trainable_parameters(pipeline),
        "total_parameters": sum(parameter.numel() for parameter in pipeline.parameters()),
        "backbone_pretrained_root": str(backbone.pretrained_model_name_or_path),
        "transformer_subdir": str(backbone.transformer_subdir),
        "config_sha256": _sha256_if_exists(Path(backbone.pretrained_model_name_or_path) / "transformer" / "config.json")
        if backbone.pretrained_model_name_or_path
        else None,
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
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
