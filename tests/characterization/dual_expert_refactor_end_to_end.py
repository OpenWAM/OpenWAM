from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dual_expert_refactor_artifacts import (
    sha256_file,
    tensor_fingerprint,
    write_json_atomic,
)
from .dual_expert_refactor_contract import (
    DEFAULT_INFERENCE_CONTRACT,
    GJD_INFERENCE_CONTRACT,
    CharacterizationAssets,
    DualExpertMethodSpec,
    DualExpertTrainingProfile,
    gjd_ablation_overrides,
    ground_truth_training_overrides,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def stage_model_only_checkpoint(source: Path, root: Path) -> Path:
    """Expose weights without sibling counters/full-state promotion."""

    destination = root / "model_only" / "model_state.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        temporary.symlink_to(source.expanduser().resolve())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_training_cli_smoke_command(
    *,
    method: DualExpertMethodSpec,
    assets: CharacterizationAssets,
    checkpoint: Path,
    save_root: Path,
    world_size: int,
) -> list[str]:
    overrides = _runtime_asset_overrides(assets, include_counterfactual=method.is_gjd)
    if method.is_gjd:
        overrides.update(gjd_ablation_overrides(method))
    else:
        overrides.update(
            ground_truth_training_overrides(DualExpertTrainingProfile.FULL_SEGMENT_W64)
        )
    overrides.update(
        {
            "data.num_workers": 0,
            "trainer.strategy": "fsdp",
            "trainer.accelerator": "gpu",
            "trainer.precision": "bf16-mixed",
            "trainer.checkpoint_mode": "model_only",
            "trainer.enable_checkpointing": False,
            "trainer.save_interval": None,
            "trainer.export_runtime_backbone": False,
            "trainer.enable_jsonl_logging": True,
            "trainer.enable_wandb": False,
            "trainer.wandb_mode": "disabled",
            "trainer.validation_interval": None,
            "trainer.limit_train_batches": 1,
            "trainer.limit_val_batches": 0,
            "training.gradient_accumulation_steps": 1,
            "training.num_steps": 1,
        }
    )
    return [
        _find_torchrun(),
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "open_wam.cli.train",
        "--config-name",
        method.config_name,
        "--save-root",
        str(save_root),
        "--resume-from",
        str(checkpoint),
        "--devices",
        str(world_size),
        "--expected-world-size",
        str(world_size),
        "--disable-wandb",
        *_override_arguments(overrides),
    ]


def build_libero_rollout_command(
    *,
    method: DualExpertMethodSpec,
    assets: CharacterizationAssets,
    output_root: Path,
    task_id: int,
    episode_idx: int,
    seed: int,
    max_timestep: int | None = None,
    max_chunks: int | None = None,
) -> list[str]:
    contract = GJD_INFERENCE_CONTRACT if method.is_gjd else DEFAULT_INFERENCE_CONTRACT
    resolved_max_timestep = (
        contract.max_timestep if max_timestep is None else int(max_timestep)
    )
    resolved_max_chunks = contract.max_chunks if max_chunks is None else int(max_chunks)
    overrides = _runtime_asset_overrides(
        assets,
        include_counterfactual=method.is_gjd,
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_libero_dual_expert_visualization.py"),
        "--cfg",
        str(assets.checkpoint_config_for(method.asset_id)),
        "--checkpoint",
        str(assets.checkpoint_for(method.asset_id)),
        "--frontend-encode-mode",
        contract.frontend_encode_mode,
        "--dual-expert-inference-window-size",
        str(contract.inference_window_size),
        "--benchmark",
        "libero_10",
        "--task-id",
        str(task_id),
        "--episode-idx",
        str(episode_idx),
        "--max-timestep",
        str(resolved_max_timestep),
        "--max-chunks",
        str(resolved_max_chunks),
        "--startup-model-obs-frames",
        str(contract.startup_model_obs_frames),
        "--startup-env-init-steps",
        str(contract.startup_env_init_steps),
        "--output-dir",
        str(output_root),
        "--suffix",
        f"characterization_{method.asset_id}_task{task_id}_ep{episode_idx}",
        "--runtime-device",
        "cuda:0",
        "--action-device",
        "cuda:0",
        "--frontend-device",
        "cuda:0",
        "--decode-device",
        "cuda:0",
        "--seed",
        str(seed),
        "--allow-deprecated-libero-config",
        *_override_arguments(overrides),
    ]
    return command


def characterization_environment(
    *,
    cuda_devices: str,
    fsdp_cpu_offload: bool,
    disable_nccl_shm: bool = False,
    libero_repo_root: Path | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "OPEN_WAM_ENABLE_FIXED128_ROLLOUT_CONTEXT": "0",
            "OPEN_WAM_FSDP_CPU_OFFLOAD": "1" if fsdp_cpu_offload else "0",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCHINDUCTOR_COMPILE_THREADS": environment.get(
                "TORCHINDUCTOR_COMPILE_THREADS",
                "1",
            ),
            "MUJOCO_GL": environment.get("MUJOCO_GL", "egl"),
        }
    )
    if libero_repo_root is not None:
        environment["LIBERO_REPO_ROOT"] = str(libero_repo_root)
    if disable_nccl_shm:
        environment["NCCL_SHM_DISABLE"] = "1"
        environment["NCCL_CUMEM_HOST_ENABLE"] = "0"
    return environment


def run_training_cli_smoke(
    *,
    method: DualExpertMethodSpec,
    assets: CharacterizationAssets,
    output_root: Path,
    cuda_devices: str,
    world_size: int,
    fsdp_cpu_offload: bool,
    disable_nccl_shm: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = stage_model_only_checkpoint(
        assets.checkpoint_for(method.asset_id),
        output_root / "staged",
    )
    save_root = output_root / "run"
    command = build_training_cli_smoke_command(
        method=method,
        assets=assets,
        checkpoint=checkpoint,
        save_root=save_root,
        world_size=world_size,
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=characterization_environment(
            cuda_devices=cuda_devices,
            fsdp_cpu_offload=fsdp_cpu_offload,
            disable_nccl_shm=disable_nccl_shm,
        ),
        check=True,
    )
    metrics_path = save_root / "metrics.jsonl"
    records = _load_jsonl(metrics_path)
    train_metrics = [
        record
        for record in records
        if record.get("type") == "metrics" and record.get("phase") == "train"
    ]
    if len(train_metrics) != 1 or int(train_metrics[0].get("step", -1)) != 1:
        raise AssertionError(
            "Training CLI smoke expected exactly one optimizer-step metrics record."
        )
    report = {
        "schema_version": 1,
        "asset_id": method.asset_id,
        "returncode": completed.returncode,
        "optimizer_steps": len(train_metrics),
        "final_step": int(train_metrics[-1]["step"]),
        "metric_keys": sorted(train_metrics[-1]["metrics"]),
        "metrics": {
            key: float(value)
            for key, value in sorted(train_metrics[-1]["metrics"].items())
        },
    }
    write_json_atomic(output_root / "training_cli_smoke.json", report)
    return report


def run_libero_rollout(
    *,
    method: DualExpertMethodSpec,
    assets: CharacterizationAssets,
    output_root: Path,
    cuda_device: str,
    task_id: int,
    episode_idx: int,
    seed: int,
    libero_repo_root: Path | None,
    max_timestep: int | None = None,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    asset_root = output_root / method.asset_id
    artifact_root = asset_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    command = build_libero_rollout_command(
        method=method,
        assets=assets,
        output_root=artifact_root,
        task_id=task_id,
        episode_idx=episode_idx,
        seed=seed,
        max_timestep=max_timestep,
        max_chunks=max_chunks,
    )
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=characterization_environment(
            cuda_devices=cuda_device,
            fsdp_cpu_offload=False,
            libero_repo_root=libero_repo_root,
        ),
        check=True,
    )
    report = collect_libero_rollout_report(
        method=method,
        artifact_root=artifact_root,
    )
    write_json_atomic(asset_root / "rollout_report.json", report)
    return report


def collect_libero_rollout_report(
    *,
    method: DualExpertMethodSpec,
    artifact_root: Path,
) -> dict[str, Any]:
    summary_paths = [
        path
        for path in artifact_root.rglob("*.json")
        if not path.name.endswith(("_chunks.json", "_load_report.json"))
    ]
    summaries: list[tuple[Path, dict[str, Any]]] = []
    for path in summary_paths:
        payload = _load_json(path)
        if _is_libero_summary(payload):
            summaries.append((path, dict(payload)))
    if len(summaries) != 1:
        raise AssertionError(
            f"Expected one LIBERO summary for {method.asset_id}, found "
            f"{[str(path) for path, _ in summaries]}."
        )
    summary_path, summary = summaries[0]
    action_path = Path(str(summary["action_trace_path"]))
    if not action_path.is_file():
        raise FileNotFoundError(f"Missing rollout action trace: {action_path}")
    actions = np.asarray(
        [record["action"] for record in _load_jsonl(action_path)],
        dtype=np.float32,
    )
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise AssertionError(f"Expected a [T, 7] action trace, got {actions.shape}.")
    video_path = Path(str(summary["video_path"]))
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing rollout video: {video_path}")
    chunk_path = summary_path.with_name(f"{summary_path.stem}_chunks.json")
    chunks = _load_json(chunk_path)
    if not isinstance(chunks, list):
        raise TypeError(f"Expected a chunk log list in {chunk_path}.")
    infer_chunks = [
        _project_rollout_chunk(item)
        for item in chunks
        if isinstance(item, Mapping) and item.get("phase") == "infer"
    ]
    if len(infer_chunks) != int(summary["chunk_count"]):
        raise AssertionError(
            f"Rollout summary reports {summary['chunk_count']} chunks but "
            f"{len(infer_chunks)} infer logs were saved."
        )
    action_tensor = torch.from_numpy(actions)
    return {
        "schema_version": 1,
        "asset_id": method.asset_id,
        "benchmark": summary["benchmark"],
        "task_id": int(summary["task_id"]),
        "episode_idx": int(summary["episode_idx"]),
        "seed": int(summary["seed"]),
        "success": bool(summary["success"]),
        "terminal": bool(summary["terminal"]),
        "chunk_count": int(summary["chunk_count"]),
        "env_timestep": int(summary["env_timestep"]),
        "action_count": int(summary["action_count"]),
        "startup_model_obs_frames": int(summary["startup_model_obs_frames"]),
        "startup_env_init_steps": int(summary["startup_env_init_steps"]),
        "action_trace_sha256": _array_sha256(actions),
        "action_trace": tensor_fingerprint(action_tensor),
        "comparison_video_size_bytes": video_path.stat().st_size,
        "comparison_video_sha256": sha256_file(video_path),
        "inference_chunks": infer_chunks,
    }


def _runtime_asset_overrides(
    assets: CharacterizationAssets,
    *,
    include_counterfactual: bool,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "data.local_root": str(assets.dataset_root),
        "data.empty_text_embedding_path": str(assets.empty_text_embedding),
        "backbone.pretrained_model_name_or_path": str(assets.base_model_root),
        "backbone.transformer_subdir": str(assets.video_transformer_root),
    }
    if include_counterfactual:
        if (
            assets.counterfactual_train_root is None
            or assets.counterfactual_val_root is None
        ):
            raise ValueError(
                "GJD end-to-end characterization requires counterfactual roots."
            )
        overrides.update(
            {
                "data.dynamics_routing.train_latent_root": str(
                    assets.counterfactual_train_root
                ),
                "data.dynamics_routing.val_latent_root": str(
                    assets.counterfactual_val_root
                ),
            }
        )
    return overrides


def _override_arguments(overrides: Mapping[str, Any]) -> list[str]:
    arguments: list[str] = []
    for key, value in overrides.items():
        arguments.extend(["--set", f"{key}={_serialize_override(value)}"])
    return arguments


def _serialize_override(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        value = enum_value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _find_torchrun() -> str:
    torchrun = shutil.which("torchrun")
    if torchrun is not None:
        return torchrun
    sibling = Path(sys.executable).with_name("torchrun")
    if sibling.is_file():
        return str(sibling)
    raise FileNotFoundError(
        f"Could not find torchrun beside {sys.executable} or in PATH."
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected JSONL artifact at {path}.")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _is_libero_summary(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("pipeline") == "open_wam_dual_expert"
        and "action_trace_path" in value
    )


def _project_rollout_chunk(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "chunk_index",
            "model_obs_frames",
            "video_latent_frames",
            "frontend_path",
            "action_shape",
            "execute_action_steps",
            "configured_frame_chunk_size",
            "rollout_frame_chunk_size",
            "execute_frame_chunk_size",
            "predicted_latents_shape",
            "dual_expert_gjd_action_route",
            "first_action_preview",
            "policy_debug",
        )
    }


def _array_sha256(array: np.ndarray) -> str:
    import hashlib

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()
