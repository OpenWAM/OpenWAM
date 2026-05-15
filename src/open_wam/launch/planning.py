from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from open_wam.training import load_training_cli_config, parse_train_cli

from .preflight import preflight_checkpoint, preflight_dataset
from .types import (
    CheckpointSpec,
    ClusterProfile,
    ConfigOverride,
    EvalHook,
    LaunchJob,
    LaunchMatrix,
    LaunchSpec,
    LaunchValidation,
    MethodProfile,
)
from .wrappers import resolve_wrapper_train_argv


def build_launch_jobs(
    *,
    matrix: LaunchMatrix,
    cluster: ClusterProfile,
    specs: tuple[LaunchSpec, ...],
    output_root: Path,
    num_steps: int,
    account: str | None,
    partition: str | None,
    wandb_project: str,
    wandb_mode: str,
    wandb_entity: str | None,
) -> tuple[LaunchJob, ...]:
    jobs: list[LaunchJob] = []
    for index, spec in enumerate(specs):
        method = matrix.method_profiles[spec.method_profile]
        dataset = matrix.dataset_profiles[spec.dataset_profile]
        checkpoint = matrix.checkpoints[spec.checkpoint]
        resource = matrix.resources[spec.resources]
        hooks = tuple(matrix.eval_hooks[name] for name in spec.eval_hooks)
        save_root = output_root / spec.save_root_name
        command = train_command(
            spec=spec,
            method=method,
            dataset_root=dataset.root,
            checkpoint=checkpoint,
            save_root=save_root,
            num_steps=num_steps,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            wandb_entity=wandb_entity,
        )
        env = {
            "NGPU": str(resource.gpus),
            "LOG_RANK": "0",
            "CONFIG_NAME": spec.config_name,
            "WANDB_MODE": wandb_mode,
            "WANDB_PROJECT": wandb_project,
        }
        if wandb_entity:
            env["WANDB_ENTITY"] = wandb_entity
        jobs.append(
            LaunchJob(
                spec=spec,
                index=index,
                save_root=save_root,
                sbatch_path=Path(),
                command=command,
                env=env,
                slurm={
                    "account": account or cluster.default_account or "",
                    "partition": partition or cluster.default_partition or "",
                    "gpus": resource.gpus,
                    "cpus_per_task": resource.cpus_per_task,
                    "mem": resource.mem,
                    "time": resource.time,
                },
                method_key=method.method_key,
                dataset=dataset,
                checkpoint=checkpoint,
                resource=resource,
                eval_hooks=hooks,
            )
        )
    return tuple(jobs)


def train_command(
    *,
    spec: LaunchSpec,
    method: MethodProfile,
    dataset_root: Path,
    checkpoint: CheckpointSpec,
    save_root: Path,
    num_steps: int,
    wandb_project: str,
    wandb_mode: str,
    wandb_entity: str | None,
) -> list[str]:
    command = [
        "bash",
        method.launcher,
        "--save-root",
        str(save_root),
        "--transformer-subdir",
        str(checkpoint.transformer_subdir),
        "--dataset-root",
        str(dataset_root),
        "--num-steps",
        str(num_steps),
    ]
    if wandb_mode == "disabled":
        command.append("--disable-wandb")
    else:
        command.extend(["--enable-wandb", "--wandb-project", wandb_project, "--wandb-mode", wandb_mode])
        if wandb_entity:
            command.extend(["--wandb-entity", wandb_entity])
    command.extend(overrides_to_cli_args((*method.default_overrides, *spec.overrides)))
    return command


def overrides_to_cli_args(overrides: tuple[ConfigOverride, ...]) -> list[str]:
    args: list[str] = []
    for override in overrides:
        args.extend(["--set", f"{override.key}={json.dumps(override.value, separators=(',', ':'))}"])
    return args


def validate_launch_job(*, job: LaunchJob, matrix: LaunchMatrix, repo_root: Path) -> LaunchValidation:
    method = matrix.method_profiles[job.spec.method_profile]
    errors: list[str] = []
    policy_type_name: str | None = None
    try:
        inspection_config = load_training_cli_config(
            parse_train_cli(["--config-name", job.spec.config_name]),
            env={},
        )
        policy_type = type(inspection_config.policy_variant)
        policy_type_name = f"{policy_type.__module__}.{policy_type.__name__}"
        if method.allowed_policy_types and not isinstance(
            inspection_config.policy_variant,
            method.allowed_policy_types,
        ):
            allowed = ", ".join(f"{kind.__module__}.{kind.__name__}" for kind in method.allowed_policy_types)
            errors.append(
                f"method_profile {method.name!r} expects policy_variant in ({allowed}), got {policy_type_name}"
            )
    except Exception as exc:  # pragma: no cover - exact message covered by callers
        errors.append(f"base config load failed: {exc}")
    if not errors:
        try:
            load_training_cli_config(
                parse_train_cli(resolve_wrapper_train_argv(job, repo_root=repo_root)),
                env=job.env,
            )
        except Exception as exc:
            errors.append(f"wrapper final config load failed: {exc}")

    config_path = repo_root / "configs" / "experiments" / f"{job.spec.config_name}.yaml"
    launcher_path = repo_root / method.launcher
    if not config_path.is_file():
        errors.append(f"missing config: {config_path}")
    if not launcher_path.is_file():
        errors.append(f"missing launcher: {launcher_path}")
    return LaunchValidation(
        key=job.spec.name,
        config_name=job.spec.config_name,
        method_profile=method.name,
        policy_variant_type=policy_type_name,
        ok=not errors,
        errors=tuple(errors),
    )


def preflight_launch_job(job: LaunchJob) -> dict[str, Any]:
    dataset_preflight = preflight_dataset(job.dataset)
    missing = [*dataset_preflight.missing, *preflight_checkpoint(job.checkpoint)]
    return {
        "key": job.spec.name,
        "data_root": dataset_preflight.data_root,
        "transformer_subdir": str(job.checkpoint.transformer_subdir),
        "repo_count": dataset_preflight.repo_count,
        "latent_cameras": list(dataset_preflight.latent_cameras),
        "latent_counts": dataset_preflight.latent_counts,
        "missing": list(missing),
        "warnings": list(dataset_preflight.warnings),
    }


def job_with_id(job: LaunchJob, job_id: str) -> LaunchJob:
    return replace(job, slurm={**job.slurm, "job_id": job_id})


def job_report(job: LaunchJob) -> dict[str, Any]:
    payload = {
        "key": job.spec.name,
        "name": job.spec.name,
        "label": job.spec.label,
        "config_name": job.spec.config_name,
        "save_root_name": job.spec.save_root_name,
        "dataset_profile": job.spec.dataset_profile,
        "method_profile": job.spec.method_profile,
        "method_key": job.method_key,
        "checkpoint": job.spec.checkpoint,
        "resources": job.spec.resources,
        "data_local_root": str(job.dataset.root),
        "transformer_subdir": str(job.checkpoint.transformer_subdir),
        "eval_hooks": [asdict(hook) for hook in job.eval_hooks],
        "extra_overrides": [(override.key, override.value) for override in job.spec.overrides],
        "index": job.index,
        "save_root": str(job.save_root),
        "sbatch_path": str(job.sbatch_path),
        "command": job.command,
        "env": job.env,
        "slurm": job.slurm,
    }
    job_id = job.slurm.get("job_id")
    if job_id:
        payload["job_id"] = str(job_id)
    return payload


def manifest_payload(
    *,
    matrix: LaunchMatrix,
    cluster: ClusterProfile,
    run_id: str,
    repo_root: Path,
    output_root: Path,
    log_root: Path,
    sbatch_dir: Path,
    slurm_dir: Path,
    account: str | None,
    partition: str | None,
    num_steps: int,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
    preflight_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    blocked_cases: list[dict[str, Any]],
    jobs: tuple[LaunchJob, ...],
    test_only_results: list[dict[str, Any]] | None,
    submitted_ids: list[str],
    submission_started: bool = False,
) -> dict[str, Any]:
    eval_jobs = [
        job
        for job in jobs
        if job.eval_hooks and (not submission_started or bool(job.slurm.get("job_id")))
    ]
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_matrix": str(matrix.source_path),
        "source_recipe": matrix.source_recipe,
        "cluster": cluster.name,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "log_root": str(log_root),
        "sbatch_dir": str(sbatch_dir),
        "slurm_dir": str(slurm_dir),
        "account": account or cluster.default_account,
        "partition": partition or cluster.default_partition,
        "num_steps": num_steps,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_mode": wandb_mode,
        "case_group_aliases": matrix.case_groups,
        "preflight": preflight_rows,
        "validation": validation_rows,
        "blocked_cases": blocked_cases,
        "jobs": [job_report(job) for job in jobs],
        "eval_ready_jobs": [job_report(job) for job in eval_jobs],
        "submission_started": submission_started,
        "submitted": bool(submitted_ids),
        "slurm_job_ids": submitted_ids,
    }
    if test_only_results is not None:
        manifest["test_only"] = test_only_results
    return manifest


def namespace_from_defaults(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)
