from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from open_wam.configs import (
    ActionDecoderName,
    DynamicsObjective,
    ExperimentConfig,
    SampleWeightMode,
    VideoActionProgram,
)
from open_wam.configs.policy_video_action import resolve_fixed_conditioning_mode

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_policy_architecture(config: ExperimentConfig) -> str:
    return str(config.policy_variant.name)


def _resolve_policy_program(
    config: ExperimentConfig,
) -> VideoActionProgram | None:
    program = getattr(config.policy_variant, "program", None)
    return None if program is None else VideoActionProgram(program)


def _resolve_workload_family(config: ExperimentConfig) -> str:
    if config.action_decoder.name is ActionDecoderName.VIDEO_ONLY:
        return "video_pretrain"
    return "policy_train"


def _resolve_git_metadata() -> dict[str, str | bool | None]:
    def _run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        value = completed.stdout.strip()
        return value or None

    dirty_blob = _run_git("status", "--porcelain")
    return {
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": (bool(dirty_blob) if dirty_blob is not None else None),
    }


def build_run_tracking_metadata(
    config: ExperimentConfig,
    *,
    run_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    architecture = _resolve_policy_architecture(config)
    program = _resolve_policy_program(config)
    program_label = None if program is None else program.value
    workload_family = _resolve_workload_family(config)
    attach_site = getattr(config.policy_variant, "attach_site", None)
    runtime_mode = getattr(config.policy_variant, "runtime_mode", None)
    current_block_coupling = getattr(
        config.policy_variant, "current_block_coupling", None
    )
    reference_profile = getattr(config.policy_variant, "reference_profile", None)
    generalist_mode_text_token = bool(
        getattr(config.policy_variant, "generalist_mode_text_token", False)
    )
    fixed_conditioning_mode = resolve_fixed_conditioning_mode(config.policy_variant)
    history_stream_visibility = getattr(
        config.policy_variant,
        "history_stream_visibility",
        None,
    )
    sample_construction = getattr(config.data, "sample_construction", None)
    dynamics_routing = getattr(config.data, "dynamics_routing", None)
    dynamics_routes = (
        tuple(dynamics_routing.active_routes) if dynamics_routing is not None else ()
    )
    if dynamics_routes:
        mode_probabilities = dynamics_routing.mode_probabilities()
    elif program is VideoActionProgram.GENERALIST_JOINT_DENOISING:
        mode_probabilities = {
            mode: float(mode == DynamicsObjective.JOINT) for mode in DynamicsObjective
        }
    else:
        mode_probabilities = None
    gjd_ablation = (
        _resolve_generalist_ablation(
            mode_probabilities,
            generalist_mode_text_token=generalist_mode_text_token,
        )
        if program is VideoActionProgram.GENERALIST_JOINT_DENOISING
        else None
    )
    checkpoint_dir = (
        Path(config.trainer.checkpoint_dir)
        if config.trainer.checkpoint_dir
        else output_dir / "checkpoints"
    )
    metadata: dict[str, Any] = {
        "tracking_schema_version": 4,
        "framework": "open_wam",
        "experiment_name": config.name,
        "run_name": run_name,
        "run_slug": run_name,
        "architecture": architecture,
        "program": program_label,
        "workload_family": workload_family,
        "policy_variant": str(config.policy_variant.name),
        "runtime_mode": (str(runtime_mode) if runtime_mode is not None else None),
        "current_block_coupling": (
            str(current_block_coupling) if current_block_coupling is not None else None
        ),
        "reference_profile": reference_profile,
        "dynamics_routing_routes": [
            {
                "source": route.source.value,
                "mode": route.mode.value,
                "weight": float(route.weight),
            }
            for route in dynamics_routes
        ],
        "dynamics_objective_probabilities": (
            {
                mode.value: float(probability)
                for mode, probability in mode_probabilities.items()
            }
            if mode_probabilities is not None
            else None
        ),
        "gjd_ablation": gjd_ablation,
        "dynamics_routing_enabled": bool(dynamics_routes),
        "generalist_mode_text_token": generalist_mode_text_token,
        "fixed_conditioning_mode": (
            fixed_conditioning_mode.value
            if fixed_conditioning_mode is not None
            else None
        ),
        "dynamics_routing_train_latent_root": (
            dynamics_routing.train_latent_root if dynamics_routing is not None else None
        ),
        "dynamics_routing_val_latent_root": (
            dynamics_routing.val_latent_root if dynamics_routing is not None else None
        ),
        "history_stream_visibility": (
            str(history_stream_visibility)
            if history_stream_visibility is not None
            else None
        ),
        "action_decoder": str(config.action_decoder.name),
        "attach_site": (str(attach_site) if attach_site is not None else None),
        "dataset_name": config.data.dataset_name,
        "dataset_type": config.data.dataset_type,
        "sample_construction_mode": (
            str(sample_construction.mode) if sample_construction is not None else None
        ),
        "segment_min_frames": (
            int(sample_construction.segment_min_frames)
            if sample_construction is not None
            and sample_construction.segment_min_frames is not None
            else None
        ),
        "segment_max_frames": (
            int(sample_construction.segment_max_frames)
            if sample_construction is not None
            and sample_construction.segment_max_frames is not None
            else None
        ),
        "segment_frames": (
            int(sample_construction.segment_frames)
            if sample_construction is not None
            and sample_construction.segment_frames is not None
            else None
        ),
        "start_padding_frames": (
            int(sample_construction.start_padding_frames)
            if sample_construction is not None
            else 0
        ),
        "target_alignment": (
            str(sample_construction.target_alignment)
            if sample_construction is not None
            else None
        ),
        "rollout_context_policy": (
            str(sample_construction.rollout_context_policy)
            if sample_construction is not None
            else None
        ),
        "rollout_context_frames": (
            int(sample_construction.rollout_context_frames)
            if sample_construction is not None
            and sample_construction.rollout_context_frames is not None
            else None
        ),
        "tail_padding_policy": (
            str(sample_construction.tail_padding_policy)
            if sample_construction is not None
            else None
        ),
        "padded_target_policy": (
            str(sample_construction.padded_target_policy)
            if sample_construction is not None
            else None
        ),
        "task_start_power": (
            float(sample_construction.task_start_power)
            if sample_construction is not None
            else None
        ),
        "demo_count_power": (
            float(sample_construction.demo_count_power)
            if sample_construction is not None
            else None
        ),
        "trajectory_start_power": (
            float(sample_construction.trajectory_start_power)
            if sample_construction is not None
            else None
        ),
        "sample_weight_mode": (
            str(sample_construction.sample_weight_mode)
            if sample_construction is not None
            and sample_construction.sample_weight_mode != SampleWeightMode.UNIFORM
            else None
        ),
        "sample_order_mode": (
            str(sample_construction.sample_order_mode)
            if sample_construction is not None
            else None
        ),
        "sample_weight_length_power": (
            float(sample_construction.sample_weight_length_power)
            if sample_construction is not None
            and sample_construction.sample_weight_length_power is not None
            else None
        ),
        "backbone_implementation": str(config.backbone.implementation),
        "backbone_runtime_backbone_artifact_path": (
            config.backbone.runtime_backbone_artifact_path
        ),
        "backbone_transformer_subdir": config.backbone.transformer_subdir,
        "runtime": str(config.trainer.runtime),
        "batch_adapter": str(config.trainer.batch_adapter),
        "strategy": str(config.trainer.strategy),
        "accelerator": str(config.trainer.accelerator),
        "precision": str(config.trainer.precision),
        "num_frames": int(config.data.num_frames),
        "action_dim": int(config.data.action_schema.action_dim),
        "action_horizon": int(config.data.action_schema.action_horizon),
        "state_dim": int(config.data.action_schema.state_dim),
        "state_horizon": int(config.data.action_schema.state_horizon),
        "enabled_objectives": [
            str(value) for value in config.training.enabled_objectives
        ],
        "trainable_components": [
            str(value) for value in config.training.trainable_components
        ],
        "frozen_components": [
            str(value) for value in config.training.frozen_components
        ],
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "resume_from": config.trainer.resume_from,
    }
    metadata["run_title"] = build_run_title(metadata)
    metadata.update(_resolve_git_metadata())
    return metadata


def build_default_wandb_project(tracking_metadata: dict[str, Any]) -> str:
    return f"openwam-{tracking_metadata['dataset_name']}-{tracking_metadata['workload_family'].replace('_', '-')}"


def resolve_wandb_project(
    config: ExperimentConfig, tracking_metadata: dict[str, Any]
) -> str:
    if config.trainer.wandb_project is not None:
        return config.trainer.wandb_project
    return build_default_wandb_project(tracking_metadata)


def build_wandb_group(tracking_metadata: dict[str, Any]) -> str:
    parts = [
        str(tracking_metadata["dataset_name"]),
        str(tracking_metadata["architecture"]),
    ]
    if tracking_metadata.get("program"):
        parts.append(str(tracking_metadata["program"]))
    group = "/".join(parts)
    if tracking_metadata.get("gjd_ablation"):
        group = f"{group}/{tracking_metadata['gjd_ablation']}"
    return group


def build_wandb_job_type(tracking_metadata: dict[str, Any]) -> str:
    return str(tracking_metadata["workload_family"])


def build_run_title(tracking_metadata: dict[str, Any]) -> str:
    parts = [
        str(tracking_metadata["dataset_name"]),
        str(tracking_metadata["architecture"]),
    ]
    if tracking_metadata.get("program"):
        parts.append(str(tracking_metadata["program"]))
    if tracking_metadata.get("gjd_ablation"):
        parts.append(f"gjd:{tracking_metadata['gjd_ablation']}")
    parts.append(str(tracking_metadata["run_slug"]))
    return " · ".join(parts)


def build_wandb_tags(tracking_metadata: dict[str, Any]) -> tuple[str, ...]:
    ordered_tags = [
        "framework:open_wam",
        f"dataset:{tracking_metadata['dataset_name']}",
        f"dataset_type:{tracking_metadata['dataset_type']}",
        f"workload:{tracking_metadata['workload_family']}",
        f"architecture:{tracking_metadata['architecture']}",
        f"variant:{tracking_metadata['policy_variant']}",
        f"decoder:{tracking_metadata['action_decoder']}",
    ]
    if tracking_metadata.get("program"):
        ordered_tags.append(f"program:{tracking_metadata['program']}")
    if tracking_metadata.get("git_dirty") is True:
        ordered_tags.append("dirty_worktree")
    if tracking_metadata.get("runtime_mode"):
        ordered_tags.append(f"runtime_mode:{tracking_metadata['runtime_mode']}")
    if tracking_metadata.get("current_block_coupling"):
        ordered_tags.append(f"coupling:{tracking_metadata['current_block_coupling']}")
    if tracking_metadata.get("reference_profile"):
        ordered_tags.append(
            f"reference_profile:{tracking_metadata['reference_profile']}"
        )
    if tracking_metadata.get("sample_construction_mode"):
        ordered_tags.append(f"sample:{tracking_metadata['sample_construction_mode']}")
    if tracking_metadata.get("segment_frames") is not None:
        ordered_tags.append(f"segment_frames:{tracking_metadata['segment_frames']}")
    segment_min_frames = tracking_metadata.get("segment_min_frames")
    segment_max_frames = tracking_metadata.get("segment_max_frames")
    if (
        segment_min_frames is not None
        and segment_max_frames is not None
        and segment_min_frames == segment_max_frames
    ):
        ordered_tags.append(f"segment_frames:{tracking_metadata['segment_min_frames']}")
    if int(tracking_metadata.get("start_padding_frames") or 0) > 0:
        ordered_tags.append(
            f"start_padding_frames:{tracking_metadata['start_padding_frames']}"
        )
    if (
        tracking_metadata.get("target_alignment")
        and tracking_metadata["target_alignment"] != "legacy"
    ):
        ordered_tags.append(f"target_alignment:{tracking_metadata['target_alignment']}")
    if (
        tracking_metadata.get("target_alignment")
        and tracking_metadata["target_alignment"] != "legacy"
        and tracking_metadata.get("rollout_context_policy")
    ):
        ordered_tags.append(
            f"rollout_context:{tracking_metadata['rollout_context_policy']}"
        )
    if tracking_metadata.get("sample_weight_mode"):
        ordered_tags.append(f"sample_weight:{tracking_metadata['sample_weight_mode']}")
    if tracking_metadata.get("sample_order_mode"):
        ordered_tags.append(f"sample_order:{tracking_metadata['sample_order_mode']}")
    if tracking_metadata.get("history_stream_visibility"):
        ordered_tags.append(
            f"history_visibility:{tracking_metadata['history_stream_visibility']}"
        )
    if tracking_metadata.get("dynamics_routing_enabled"):
        ordered_tags.append("dynamics_routing:enabled")
    if tracking_metadata.get("gjd_ablation"):
        ordered_tags.append(
            f"gjd:{tracking_metadata['architecture']}:{tracking_metadata['gjd_ablation']}"
        )
    if tracking_metadata.get("generalist_mode_text_token") is True:
        ordered_tags.append("generalist_mode_text_token")
    if tracking_metadata.get("fixed_conditioning_mode"):
        ordered_tags.append(
            f"conditioning_mode:{tracking_metadata['fixed_conditioning_mode']}"
        )
    deduped: list[str] = []
    for tag in ordered_tags:
        if tag not in deduped:
            deduped.append(tag)
    return tuple(deduped)


def _resolve_generalist_ablation(
    probs: Mapping[Any, float] | None,
    *,
    generalist_mode_text_token: bool,
) -> str | None:
    if probs is None:
        return None

    def _mode_value(mode: Any) -> str:
        return str(getattr(mode, "value", mode))

    normalized = {_mode_value(mode): float(prob) for mode, prob in probs.items()}

    def _close(key: str, value: float) -> bool:
        return abs(float(normalized.get(key, 0.0)) - float(value)) <= 1e-6

    if (
        _close("joint", 1.0)
        and _close("action_conditioned_video", 0.0)
        and _close("video_conditioned_action", 0.0)
    ):
        base = "pure_joint"
    elif (
        _close("joint", 0.0)
        and _close("action_conditioned_video", 1.0)
        and _close("video_conditioned_action", 0.0)
    ):
        base = "pure_fdm"
    elif (
        _close("joint", 0.0)
        and _close("action_conditioned_video", 0.0)
        and _close("video_conditioned_action", 1.0)
    ):
        base = "pure_idm"
    elif (
        _close("joint", 0.6)
        and _close("action_conditioned_video", 0.2)
        and _close("video_conditioned_action", 0.2)
    ):
        base = "vanilla"
    else:
        base = "custom"

    if not generalist_mode_text_token:
        return base
    return "mode_token" if base == "vanilla" else f"{base}_mode_token"
