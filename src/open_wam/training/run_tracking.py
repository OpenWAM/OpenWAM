from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping

from open_wam.configs import (
    ExperimentConfig,
    ParallelStreamVariantProfile,
    PolicyVariantName,
    SampleOrderMode,
    SampleWeightMode,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_method_family(config: ExperimentConfig) -> str:
    policy_name = config.policy_variant.name
    if policy_name == PolicyVariantName.PARALLEL_STREAM:
        return "method_1"
    if policy_name == PolicyVariantName.REGISTER_ATTACHED:
        return "method_2"
    if policy_name == PolicyVariantName.VIDEO_SEQUENCE_POLICY:
        return "method_3"
    if policy_name in {PolicyVariantName.POST_LATENT, PolicyVariantName.POST_DECODED}:
        return "method_4"
    if policy_name == PolicyVariantName.MOT:
        return "method_5"
    if policy_name == PolicyVariantName.CAUSAL_VIDEO_PREDICTION:
        return "causal_video_prediction"
    return str(policy_name)


def _resolve_method_label(method_family: str) -> str:
    return {
        "method_1": "m1",
        "method_2": "m2",
        "method_3": "m3",
        "method_4": "m4",
        "method_5": "m5",
        "causal_video_prediction": "causal",
    }.get(method_family, method_family)


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
    method_family = _resolve_method_family(config)
    method_label = _resolve_method_label(method_family)
    workload_family = _resolve_workload_family(config)
    attach_site = getattr(config.policy_variant, "attach_site", None)
    runtime_mode = getattr(config.policy_variant, "runtime_mode", None)
    variant_profile = getattr(config.policy_variant, "variant_profile", None)
    current_block_coupling = getattr(config.policy_variant, "current_block_coupling", None)
    reference_profile = getattr(config.policy_variant, "reference_profile", None)
    joint_denoise_training_mode_probs = getattr(config.policy_variant, "joint_denoise_training_mode_probs", None)
    mot_generalist_training_mode_probs = getattr(config.policy_variant, "mot_generalist_training_mode_probs", None)
    generalist_training_paradigm = getattr(config.policy_variant, "generalist_training_paradigm", None)
    generalist_mode_text_token = bool(getattr(config.policy_variant, "generalist_mode_text_token", False))
    if _is_m1_generalist_joint_denoising_profile(variant_profile):
        m1_generalist_ablation = _resolve_generalist_ablation(
            joint_denoise_training_mode_probs,
            generalist_mode_text_token=generalist_mode_text_token,
        )
    else:
        m1_generalist_ablation = None
    mot_generalist_ablation = _resolve_generalist_ablation(
        mot_generalist_training_mode_probs,
        generalist_mode_text_token=generalist_mode_text_token,
    )
    gjd_ablation = m1_generalist_ablation or mot_generalist_ablation
    preserve_video_pretrain_history = getattr(config.policy_variant, "preserve_video_pretrain_history", None)
    train_video_condition_source = getattr(config.policy_variant, "train_video_condition_source", None)
    sample_construction = getattr(config.data, "sample_construction", None)
    dynamics_mixture = getattr(config.data, "generalist_dynamics_mixture", None)
    checkpoint_dir = Path(config.trainer.checkpoint_dir) if config.trainer.checkpoint_dir else output_dir / "checkpoints"
    metadata: dict[str, Any] = {
        "tracking_schema_version": 1,
        "framework": "open_wam",
        "experiment_name": config.name,
        "run_name": run_name,
        "run_slug": run_name,
        "method_family": method_family,
        "method_label": method_label,
        "workload_family": workload_family,
        "policy_variant": str(config.policy_variant.name),
        "runtime_mode": (str(runtime_mode) if runtime_mode is not None else None),
        "variant_profile": (str(variant_profile) if variant_profile is not None else None),
        "current_block_coupling": (str(current_block_coupling) if current_block_coupling is not None else None),
        "reference_profile": reference_profile,
        "joint_denoise_training_mode_probs": (
            {str(mode): float(prob) for mode, prob in joint_denoise_training_mode_probs.items()}
            if joint_denoise_training_mode_probs is not None
            else None
        ),
        "mot_generalist_training_mode_probs": (
            {str(mode): float(prob) for mode, prob in mot_generalist_training_mode_probs.items()}
            if mot_generalist_training_mode_probs is not None
            else None
        ),
        "gjd_ablation": gjd_ablation,
        "m1_generalist_ablation": m1_generalist_ablation,
        "mot_generalist_ablation": mot_generalist_ablation,
        "generalist_training_paradigm": (
            str(generalist_training_paradigm) if generalist_training_paradigm is not None else None
        ),
        "generalist_mode_text_token": generalist_mode_text_token,
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
    group = (
        f"{tracking_metadata['dataset_name']}/"
        f"{tracking_metadata['method_label']}/"
        f"{tracking_metadata['policy_variant']}"
    )
    if tracking_metadata.get("gjd_ablation"):
        group = f"{group}/{tracking_metadata['gjd_ablation']}"
    return group


def build_wandb_job_type(tracking_metadata: dict[str, Any]) -> str:
    return str(tracking_metadata["workload_family"])


def build_run_title(tracking_metadata: dict[str, Any]) -> str:
    parts = [
        str(tracking_metadata["dataset_name"]),
        str(tracking_metadata["method_label"]),
        str(tracking_metadata["policy_variant"]),
    ]
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
        f"method:{tracking_metadata['method_label']}",
        f"method_family:{tracking_metadata['method_family']}",
        f"variant:{tracking_metadata['policy_variant']}",
        f"decoder:{tracking_metadata['action_decoder']}",
    ]
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
        ordered_tags.append(f"gjd:{tracking_metadata['method_label']}:{tracking_metadata['gjd_ablation']}")
    if tracking_metadata.get("m1_generalist_ablation"):
        ordered_tags.append(f"m1_gjd:{tracking_metadata['m1_generalist_ablation']}")
    if tracking_metadata.get("mot_generalist_ablation"):
        ordered_tags.append(f"mot_gjd:{tracking_metadata['mot_generalist_ablation']}")
    if tracking_metadata.get("generalist_mode_text_token") is True:
        ordered_tags.append("generalist_mode_text_token")
    deduped: list[str] = []
    for tag in ordered_tags:
        if tag not in deduped:
            deduped.append(tag)
    return tuple(deduped)


def _is_m1_generalist_joint_denoising_profile(variant_profile: Any) -> bool:
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
