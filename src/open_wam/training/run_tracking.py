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
        "action_decoder": str(config.action_decoder.name),
        "attach_site": (str(attach_site) if attach_site is not None else None),
        "dataset_name": config.data.dataset_name,
        "dataset_type": config.data.dataset_type,
        "backbone_implementation": str(config.backbone.implementation),
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
    deduped: list[str] = []
    for tag in ordered_tags:
        if tag not in deduped:
            deduped.append(tag)
    return tuple(deduped)
