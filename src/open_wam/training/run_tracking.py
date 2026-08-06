from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from open_wam.configs import (
    ExperimentConfig,
    ParallelStreamVariantProfile,
    PolicyVariantName,
    SampleOrderMode,
    SampleWeightMode,
)
from open_wam.configs.policy_video_action import resolve_fixed_conditioning_mode

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_policy_architecture(config: ExperimentConfig) -> str:
    return str(config.policy_variant.name)


def _resolve_policy_program(config: ExperimentConfig) -> str | None:
    program = getattr(config.policy_variant, "program", None)
    if program is not None:
        return str(program)
    variant_profile = getattr(config.policy_variant, "variant_profile", None)
    if _is_generalist_joint_denoising_profile(variant_profile):
        return "generalist_joint_denoising"
    return None


def _resolve_workload_family(config: ExperimentConfig) -> str:
    if config.policy_variant.name == PolicyVariantName.CAUSAL_VIDEO_PREDICTION:
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
    workload_family = _resolve_workload_family(config)
    attach_site = getattr(config.policy_variant, "attach_site", None)
    runtime_mode = getattr(config.policy_variant, "runtime_mode", None)
    variant_profile = getattr(config.policy_variant, "variant_profile", None)
    current_block_coupling = getattr(config.policy_variant, "current_block_coupling", None)
    reference_profile = getattr(config.policy_variant, "reference_profile", None)
    generalist_denoising_mode_probs = getattr(
        config.policy_variant,
        "generalist_denoising_mode_probs",
        None,
    )
    generalist_training_paradigm = getattr(config.policy_variant, "generalist_training_paradigm", None)
    generalist_mode_text_token = bool(getattr(config.policy_variant, "generalist_mode_text_token", False))
    fixed_conditioning_mode = resolve_fixed_conditioning_mode(config.policy_variant)
    gjd_ablation = (
        _resolve_generalist_ablation(
            generalist_denoising_mode_probs,
            generalist_mode_text_token=generalist_mode_text_token,
        )
        if program == "generalist_joint_denoising"
        else None
    )
    preserve_video_pretrain_history = getattr(config.policy_variant, "preserve_video_pretrain_history", None)
    train_video_condition_source = getattr(config.policy_variant, "train_video_condition_source", None)
    sample_construction = getattr(config.data, "sample_construction", None)
    dynamics_mixture = getattr(config.data, "generalist_dynamics_mixture", None)
    checkpoint_dir = Path(config.trainer.checkpoint_dir) if config.trainer.checkpoint_dir else output_dir / "checkpoints"
    metadata: dict[str, Any] = {
        "tracking_schema_version": 2,
        "framework": "open_wam",
        "experiment_name": config.name,
        "run_name": run_name,
        "run_slug": run_name,
        "architecture": architecture,
        "program": program,
        "workload_family": workload_family,
        "policy_variant": str(config.policy_variant.name),
        "runtime_mode": (str(runtime_mode) if runtime_mode is not None else None),
        "variant_profile": (str(variant_profile) if variant_profile is not None else None),
        "current_block_coupling": (str(current_block_coupling) if current_block_coupling is not None else None),
        "reference_profile": reference_profile,
        "generalist_denoising_mode_probs": (
            {str(mode): float(prob) for mode, prob in generalist_denoising_mode_probs.items()}
            if generalist_denoising_mode_probs is not None
            else None
        ),
        "gjd_ablation": gjd_ablation,
        "generalist_training_paradigm": (
            str(generalist_training_paradigm) if generalist_training_paradigm is not None else None
        ),
        "generalist_mode_text_token": generalist_mode_text_token,
        "fixed_conditioning_mode": (
            fixed_conditioning_mode.value
            if fixed_conditioning_mode is not None
            else None
        ),
        "generalist_dynamics_train_latent_root": (
            dynamics_mixture.train_latent_root if dynamics_mixture is not None else None
        ),
        "generalist_dynamics_val_latent_root": (
            dynamics_mixture.val_latent_root if dynamics_mixture is not None else None
        ),
        "preserve_video_pretrain_history": preserve_video_pretrain_history,
        "action_decoder": str(config.action_decoder.name),
        "attach_site": (str(attach_site) if attach_site is not None else None),
        "dataset_name": config.data.dataset_name,
        "dataset_type": config.data.dataset_type,
        "sample_construction_mode": (
            str(sample_construction.mode) if sample_construction is not None else None
        ),
        "segment_min_frames": (
            int(sample_construction.segment_min_frames)
            if sample_construction is not None and sample_construction.segment_min_frames is not None
            else None
        ),
        "segment_max_frames": (
            int(sample_construction.segment_max_frames)
            if sample_construction is not None and sample_construction.segment_max_frames is not None
            else None
        ),
        "segment_frames": (
            int(sample_construction.segment_frames)
            if sample_construction is not None and sample_construction.segment_frames is not None
            else None
        ),
        "start_padding_frames": (
            int(sample_construction.start_padding_frames)
            if sample_construction is not None
            else 0
        ),
        "target_alignment": (
            str(sample_construction.target_alignment) if sample_construction is not None else None
        ),
        "rollout_context_policy": (
            str(sample_construction.rollout_context_policy) if sample_construction is not None else None
        ),
        "rollout_context_frames": (
            int(sample_construction.rollout_context_frames)
            if sample_construction is not None and sample_construction.rollout_context_frames is not None
            else None
        ),
        "tail_padding_policy": (
            str(sample_construction.tail_padding_policy) if sample_construction is not None else None
        ),
        "padded_target_policy": (
            str(sample_construction.padded_target_policy) if sample_construction is not None else None
        ),
        "task_start_power": (
            float(sample_construction.task_start_power) if sample_construction is not None else None
        ),
        "demo_count_power": (
            float(sample_construction.demo_count_power) if sample_construction is not None else None
        ),
        "trajectory_start_power": (
            float(sample_construction.trajectory_start_power) if sample_construction is not None else None
        ),
        "sample_weight_mode": (
            str(sample_construction.sample_weight_mode)
            if sample_construction is not None and sample_construction.sample_weight_mode != SampleWeightMode.UNIFORM
            else None
        ),
        "sample_order_mode": (
            str(sample_construction.sample_order_mode)
            if sample_construction is not None and sample_construction.sample_order_mode != SampleOrderMode.EPOCH_ORDER
            else None
        ),
        "sample_weight_length_power": (
            float(sample_construction.sample_weight_length_power)
            if sample_construction is not None and sample_construction.sample_weight_length_power is not None
            else None
        ),
        "backbone_implementation": str(config.backbone.implementation),
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
        "enabled_objectives": [str(value) for value in config.training.enabled_objectives],
        "trainable_components": [str(value) for value in config.training.trainable_components],
        "frozen_components": [str(value) for value in config.training.frozen_components],
        "train_video_condition_source": (
            str(train_video_condition_source) if train_video_condition_source is not None else None
        ),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "resume_from": config.trainer.resume_from,
    }
    metadata["run_title"] = build_run_title(metadata)
    metadata.update(_resolve_git_metadata())
    return metadata


def build_default_wandb_project(tracking_metadata: dict[str, Any]) -> str:
    return f"openwam-{tracking_metadata['dataset_name']}-{tracking_metadata['workload_family'].replace('_', '-')}"


def resolve_wandb_project(config: ExperimentConfig, tracking_metadata: dict[str, Any]) -> str:
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
    if tracking_metadata.get("train_video_condition_source"):
        ordered_tags.append(f"train_video_condition:{tracking_metadata['train_video_condition_source']}")
    if tracking_metadata.get("runtime_mode"):
        ordered_tags.append(f"runtime_mode:{tracking_metadata['runtime_mode']}")
    if tracking_metadata.get("variant_profile") and tracking_metadata["variant_profile"] != "standard":
        ordered_tags.append(f"variant_profile:{tracking_metadata['variant_profile']}")
    if tracking_metadata.get("current_block_coupling"):
        ordered_tags.append(f"coupling:{tracking_metadata['current_block_coupling']}")
    if tracking_metadata.get("reference_profile"):
        ordered_tags.append(f"reference_profile:{tracking_metadata['reference_profile']}")
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
        ordered_tags.append(f"start_padding_frames:{tracking_metadata['start_padding_frames']}")
    if tracking_metadata.get("target_alignment") and tracking_metadata["target_alignment"] != "legacy":
        ordered_tags.append(f"target_alignment:{tracking_metadata['target_alignment']}")
    if (
        tracking_metadata.get("target_alignment")
        and tracking_metadata["target_alignment"] != "legacy"
        and tracking_metadata.get("rollout_context_policy")
    ):
        ordered_tags.append(f"rollout_context:{tracking_metadata['rollout_context_policy']}")
    if tracking_metadata.get("sample_weight_mode"):
        ordered_tags.append(f"sample_weight:{tracking_metadata['sample_weight_mode']}")
    if tracking_metadata.get("sample_order_mode"):
        ordered_tags.append(f"sample_order:{tracking_metadata['sample_order_mode']}")
    if tracking_metadata.get("preserve_video_pretrain_history") is True:
        ordered_tags.append("video_pretrain_history:preserved")
    if tracking_metadata.get("generalist_training_paradigm"):
        ordered_tags.append(f"generalist_paradigm:{tracking_metadata['generalist_training_paradigm']}")
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


def _is_generalist_joint_denoising_profile(variant_profile: Any) -> bool:
    return (
        variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
        or str(getattr(variant_profile, "value", variant_profile)) == "generalist_joint_denoising"
    )


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
