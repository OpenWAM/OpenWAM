from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from open_wam.configs import ExperimentConfig, PolicyVariantName


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
    preserve_video_pretrain_history = getattr(config.policy_variant, "preserve_video_pretrain_history", None)
    train_video_condition_source = getattr(config.policy_variant, "train_video_condition_source", None)
    sample_construction = getattr(config.data, "sample_construction", None)
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
        "start_padding_frames": (
            int(sample_construction.start_padding_frames)
            if sample_construction is not None
            else 0
        ),
        "sample_weight_mode": (
            str(sample_construction.sample_weight_mode) if sample_construction is not None else None
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
    return (
        f"{tracking_metadata['dataset_name']}/"
        f"{tracking_metadata['method_label']}/"
        f"{tracking_metadata['policy_variant']}"
    )


def build_wandb_job_type(tracking_metadata: dict[str, Any]) -> str:
    return str(tracking_metadata["workload_family"])


def build_run_title(tracking_metadata: dict[str, Any]) -> str:
    return (
        f"{tracking_metadata['dataset_name']} · "
        f"{tracking_metadata['method_label']} · "
        f"{tracking_metadata['policy_variant']} · "
        f"{tracking_metadata['run_slug']}"
    )


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
    if tracking_metadata.get("sample_weight_mode"):
        ordered_tags.append(f"sample_weight:{tracking_metadata['sample_weight_mode']}")
    if tracking_metadata.get("preserve_video_pretrain_history") is True:
        ordered_tags.append("video_pretrain_history:preserved")
    deduped: list[str] = []
    for tag in ordered_tags:
        if tag not in deduped:
            deduped.append(tag)
    return tuple(deduped)
