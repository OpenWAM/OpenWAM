from __future__ import annotations

import gc
import json
import os
import shutil
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from safetensors.torch import save_file
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from open_wam.configs import CheckpointMode, ExperimentConfig
from open_wam.configs.enums import serialize_enum_values

from .checkpoint_export import _remap_packed_video_blocks_into_backbone
from .checkpoint_storage import (
    _atomic_torch_save,
    _is_rank_zero,
    _load_sibling_train_state,
    _serialize_config,
    _serialize_runtime_backbone_config,
    _wait_for_file,
)
from .state import TrainState

_CHECKPOINT_COMPATIBILITY_EXPORTS = (
    asdict,
    is_dataclass,
    os,
    serialize_enum_values,
    time,
)


def _save_state_dict_options() -> StateDictOptions:
    return StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        strict=False,
    )


def _load_state_dict_options(
    *,
    broadcast_from_rank0: bool = False,
) -> StateDictOptions:
    return StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        strict=False,
        broadcast_from_rank0=broadcast_from_rank0,
    )


def _release_unused_device_memory() -> None:
    """Drop Python and CUDA allocator caches before memory-heavy checkpoint ops."""

    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        # ipc_collect can fail if CUDA is not initialized for this rank yet.
        pass


def _densify_optimizer_state_dict(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optim_state_dict: dict[str, Any],
    options: StateDictOptions,
) -> dict[str, Any]:
    """Fill missing optimizer-state entries for trainable-but-unused parameters.

    Some runs legitimately save sparse optimizer state because not every
    trainable parameter receives a gradient before the checkpoint is written.
    `set_optimizer_state_dict(...)` expects the current optimizer structure,
    so we rebuild the param-group layout from the current optimizer and overlay
    whatever state/hyperparameters were present in the checkpoint.
    """

    current_state_dict = get_optimizer_state_dict(model, optimizer, options=options)
    current_state = current_state_dict.get("state", {})
    loaded_state = optim_state_dict.get("state", {})
    dense_state = {key: loaded_state.get(key, {}) for key in current_state}

    loaded_groups = list(optim_state_dict.get("param_groups", []))
    dense_groups: list[dict[str, Any]] = []
    for index, current_group in enumerate(current_state_dict.get("param_groups", [])):
        merged_group = dict(current_group)
        if index < len(loaded_groups):
            for key, value in loaded_groups[index].items():
                if key == "params":
                    continue
                merged_group[key] = value
        dense_groups.append(merged_group)

    return {
        "state": dense_state,
        "param_groups": dense_groups,
    }


def _optimizer_state_presence_contract(
    optimizer_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_keys": tuple(optimizer_state.get("state", {})),
        "param_groups": tuple(
            tuple(group.get("params", ()))
            for group in optimizer_state.get("param_groups", ())
        ),
    }


def _prune_synthetic_optimizer_state(
    *,
    optimizer: torch.optim.Optimizer,
    loaded_state_contract: dict[str, Any],
) -> None:
    """Remove state PyTorch synthesizes for parameters absent from a sparse save.

    ``set_optimizer_state_dict`` materializes zero AdamW moments for every
    parameter listed in a loaded param group, even when that parameter has no
    entry in ``state``. Its synthesized scalar step inherits the current saved
    step, which changes the first future update for a previously unused
    parameter. Preserve the checkpoint's sparse-state semantics by mapping the
    saved parameter-group names back to runtime parameters and pruning entries
    that were not actually serialized.
    """

    loaded_state_keys = set(loaded_state_contract.get("state_keys", ()))
    loaded_groups = list(loaded_state_contract.get("param_groups", ()))
    runtime_groups = list(optimizer.param_groups)
    if len(loaded_groups) != len(runtime_groups):
        raise ValueError(
            "Loaded optimizer param-group count does not match the runtime: "
            f"{len(loaded_groups)} != {len(runtime_groups)}."
        )
    for group_index, (loaded_group, runtime_group) in enumerate(
        zip(loaded_groups, runtime_groups, strict=True)
    ):
        loaded_names = list(loaded_group)
        runtime_parameters = list(runtime_group.get("params", ()))
        if len(loaded_names) != len(runtime_parameters):
            raise ValueError(
                "Loaded optimizer parameter count does not match runtime group "
                f"{group_index}: {len(loaded_names)} != "
                f"{len(runtime_parameters)}."
            )
        for loaded_name, parameter in zip(
            loaded_names,
            runtime_parameters,
            strict=True,
        ):
            if loaded_name not in loaded_state_keys:
                optimizer.state.pop(parameter, None)


def _is_dtensor(value: object) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return isinstance(value, DTensor)


def _iter_model_state_tensors(model: nn.Module):
    yield from model.parameters(recurse=True)
    yield from model.buffers(recurse=True)


def _non_scalar_model_state_devices(model: nn.Module) -> set[torch.device]:
    return {
        value.device
        for value in _iter_model_state_tensors(model)
        if torch.is_tensor(value) and value.dim() > 0
    }


@contextmanager
def _cpu_align_non_dtensor_state_for_full_load(model: nn.Module):
    """Temporarily align mixed CPU-offload FSDP state so DCP full-state load works."""

    moved: list[tuple[torch.Tensor, torch.device]] = []
    for value in _iter_model_state_tensors(model):
        if not torch.is_tensor(value) or value.dim() == 0 or _is_dtensor(value):
            continue
        original_device = value.device
        if original_device.type == "cpu":
            continue
        moved.append((value, original_device))
        value.data = value.data.to("cpu")
    try:
        yield
    finally:
        for value, original_device in moved:
            value.data = value.data.to(original_device)


def _set_model_state_dict(
    model: nn.Module, model_state_dict: dict[str, Any], options: StateDictOptions
) -> None:
    devices = _non_scalar_model_state_devices(model)
    if dist.is_initialized() and torch.device("cpu") in devices and len(devices) > 1:
        with _cpu_align_non_dtensor_state_for_full_load(model):
            set_model_state_dict(model, model_state_dict, options=options)
        return
    set_model_state_dict(model, model_state_dict, options=options)


class CheckpointManager:
    """Own save/load/export behavior for the composable training runtime."""

    def __init__(
        self,
        *,
        root_dir: Path,
        config: ExperimentConfig,
        checkpoint_mode: CheckpointMode | str,
        max_checkpoints_to_keep: int | None = None,
        export_runtime_backbone: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.checkpoint_mode = checkpoint_mode
        if max_checkpoints_to_keep is not None:
            if (
                isinstance(max_checkpoints_to_keep, bool)
                or int(max_checkpoints_to_keep) <= 0
            ):
                raise ValueError(
                    "`max_checkpoints_to_keep` must be a positive integer or None."
                )
            max_checkpoints_to_keep = int(max_checkpoints_to_keep)
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.export_runtime_backbone = export_runtime_backbone

    def checkpoint_dir_for_step(self, step: int) -> Path:
        return self.root_dir / f"checkpoint_step_{step}"

    def save(
        self,
        *,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_state: TrainState,
        strategy_state: dict[str, object] | None = None,
    ) -> Path:
        checkpoint_dir = self.checkpoint_dir_for_step(step)
        payload_marker = checkpoint_dir / ".checkpoint_payload_complete"
        completion_marker = checkpoint_dir / ".checkpoint_complete"
        if _is_rank_zero():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for marker in (payload_marker, completion_marker):
                if marker.exists():
                    marker.unlink()
        if dist.is_initialized():
            dist.barrier()

        _release_unused_device_memory()
        if dist.is_initialized():
            dist.barrier()

        save_options = _save_state_dict_options()
        model_state_dict = get_model_state_dict(model, options=save_options)
        resolved_mode = CheckpointMode(self.checkpoint_mode)
        payload: dict[str, Any] = {
            "model_state_dict": model_state_dict,
            "train_state": train_state.state_dict(),
        }
        if resolved_mode == CheckpointMode.FULL_TRAINING_STATE:
            payload.update(
                {
                    "optimizer_state_dict": (
                        get_optimizer_state_dict(model, optimizer, options=save_options)
                        if optimizer is not None
                        else None
                    ),
                    "scheduler_state_dict": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "strategy_state_dict": strategy_state,
                }
            )

        if _is_rank_zero():
            self._write_resolved_config(checkpoint_dir)
            if resolved_mode == CheckpointMode.MODEL_ONLY:
                self._write_model_state_checkpoint(
                    checkpoint_dir, payload["model_state_dict"]
                )
            elif resolved_mode == CheckpointMode.FULL_TRAINING_STATE:
                _atomic_torch_save(payload, checkpoint_dir / "full_training_state.pt")
                # Always write a lightweight model-only checkpoint alongside the
                # resumable training checkpoint so eval / visualization paths
                # can skip optimizer-state deserialization.
                self._write_model_state_checkpoint(
                    checkpoint_dir, payload["model_state_dict"]
                )
            else:
                raise ValueError(
                    f"Unsupported checkpoint_mode {self.checkpoint_mode!r}."
                )

            with (checkpoint_dir / "train_state.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(train_state.state_dict(), handle, indent=2, sort_keys=True)
            payload_marker.write_text("ok\n", encoding="utf-8")
        elif dist.is_initialized():
            _wait_for_file(payload_marker)

        if self.export_runtime_backbone:
            self._export_runtime_backbone(checkpoint_dir, model)

        del payload
        del model_state_dict
        _release_unused_device_memory()
        if dist.is_initialized():
            dist.barrier()

        if _is_rank_zero():
            completion_marker.write_text("ok\n", encoding="utf-8")
            self._prune_old_checkpoints(
                keep=self.max_checkpoints_to_keep, preserve=checkpoint_dir
            )
        elif dist.is_initialized():
            _wait_for_file(completion_marker)
        if dist.is_initialized():
            dist.barrier()
        return checkpoint_dir

    def load(
        self,
        *,
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple[TrainState, dict[str, object]]:
        checkpoint_path = self.resolve_checkpoint_path(path)
        distributed = dist.is_initialized()
        is_rank_zero = _is_rank_zero()
        load_options = _load_state_dict_options(
            broadcast_from_rank0=distributed,
        )
        payload = (
            torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=False,
            )
            if is_rank_zero or not distributed
            else {}
        )
        if not isinstance(payload, dict):
            raise TypeError(
                f"Expected checkpoint mapping at {checkpoint_path}, "
                f"got {type(payload).__name__}."
            )

        model_state = payload.get("model_state_dict", {})
        optimizer_state = payload.get("optimizer_state_dict")
        optimizer_state_contract = (
            _optimizer_state_presence_contract(optimizer_state)
            if isinstance(optimizer_state, dict)
            else None
        )
        metadata_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"model_state_dict", "optimizer_state_dict"}
        }
        if metadata_payload.get("train_state") is None and is_rank_zero:
            sibling_train_state = _load_sibling_train_state(checkpoint_path)
            if sibling_train_state is not None:
                metadata_payload["train_state"] = sibling_train_state
        if distributed:
            metadata_object: list[dict[str, object] | None] = [
                metadata_payload if is_rank_zero else None
            ]
            optimizer_contract_object: list[dict[str, Any] | None] = [
                optimizer_state_contract if is_rank_zero else None
            ]
            dist.broadcast_object_list(metadata_object, src=0)
            dist.broadcast_object_list(optimizer_contract_object, src=0)
            metadata_payload = metadata_object[0] or {}
            optimizer_state_contract = optimizer_contract_object[0]

        _set_model_state_dict(model, model_state, options=load_options)
        if optimizer is not None and optimizer_state_contract is not None:
            optimizer_state_for_load = (
                optimizer_state if isinstance(optimizer_state, dict) else {}
            )
            try:
                set_optimizer_state_dict(
                    model,
                    optimizer,
                    optim_state_dict=optimizer_state_for_load,
                    options=load_options,
                )
            except Exception:
                dense_optimizer_state = _densify_optimizer_state_dict(
                    model=model,
                    optimizer=optimizer,
                    optim_state_dict=optimizer_state_for_load,
                    options=load_options,
                )
                set_optimizer_state_dict(
                    model,
                    optimizer,
                    optim_state_dict=dense_optimizer_state,
                    options=load_options,
                )
            _prune_synthetic_optimizer_state(
                optimizer=optimizer,
                loaded_state_contract=optimizer_state_contract,
            )
        scheduler_state = metadata_payload.get("scheduler_state_dict")
        if scheduler is not None and isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)
        raw_train_state = metadata_payload.get("train_state")
        train_state = TrainState.from_state_dict(raw_train_state)
        train_state.last_checkpoint_path = str(checkpoint_path.parent)
        train_state.resume_source = str(checkpoint_path)
        return train_state, payload if not distributed else metadata_payload

    def resolve_checkpoint_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_file():
            if (
                CheckpointMode(self.checkpoint_mode)
                == CheckpointMode.FULL_TRAINING_STATE
                and candidate.name == "model_state.pt"
            ):
                full_state = candidate.parent / "full_training_state.pt"
                if full_state.is_file():
                    warnings.warn(
                        "Promoting model_state.pt resume path to sibling full_training_state.pt "
                        "because trainer.checkpoint_mode=full_training_state. Pass a checkpoint "
                        "directory or full_training_state.pt for exact GJD resumes.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return full_state
            return candidate
        if (candidate / "full_training_state.pt").exists():
            return candidate / "full_training_state.pt"
        if (candidate / "model_state.pt").exists():
            return candidate / "model_state.pt"
        latest = self.find_latest_checkpoint(candidate)
        if latest is not None:
            if (latest / "full_training_state.pt").exists():
                return latest / "full_training_state.pt"
            return latest / "model_state.pt"
        raise FileNotFoundError(
            f"Unable to resolve a checkpoint file from {candidate}."
        )

    def find_latest_checkpoint(self, root: str | Path) -> Path | None:
        checkpoint_dirs = self._complete_checkpoint_dirs(Path(root))
        return checkpoint_dirs[-1] if checkpoint_dirs else None

    def _complete_checkpoint_dirs(self, root: Path | None = None) -> list[Path]:
        root_path = self.root_dir if root is None else Path(root)
        candidate_dirs: list[Path] = []
        for path in root_path.glob("checkpoint_step_*"):
            if not path.is_dir():
                continue
            try:
                int(path.name.split("_")[-1])
            except ValueError:
                continue
            candidate_dirs.append(path)
        # New checkpoints write a completion marker after payload, optional
        # exports, and pruning finish. Once a run root has marker-aware
        # checkpoints, ignore later unmarked dirs without touching their
        # payload files because interrupted full-state writes on network
        # filesystems can make those stat calls block. Older tests/checkpoints
        # did not have markers, so preserve the legacy behavior when no markers
        # are present under the queried root.
        marked_checkpoint_dirs = [
            checkpoint_dir
            for checkpoint_dir in candidate_dirs
            if (checkpoint_dir / ".checkpoint_complete").exists()
        ]
        scan_dirs = marked_checkpoint_dirs if marked_checkpoint_dirs else candidate_dirs
        checkpoint_dirs = [
            checkpoint_dir
            for checkpoint_dir in scan_dirs
            if (checkpoint_dir / "full_training_state.pt").exists()
            or (checkpoint_dir / "model_state.pt").exists()
        ]
        return sorted(checkpoint_dirs, key=lambda path: int(path.name.split("_")[-1]))

    def _prune_old_checkpoints(self, *, keep: int | None, preserve: Path) -> list[Path]:
        if keep is None:
            return []
        checkpoint_dirs = self._complete_checkpoint_dirs()
        if len(checkpoint_dirs) <= int(keep):
            return []
        preserve = preserve.resolve()
        removed: list[Path] = []
        for checkpoint_dir in checkpoint_dirs[
            : max(0, len(checkpoint_dirs) - int(keep))
        ]:
            if checkpoint_dir.resolve() == preserve:
                continue
            shutil.rmtree(checkpoint_dir)
            removed.append(checkpoint_dir)
        return removed

    def _write_resolved_config(self, checkpoint_dir: Path) -> None:
        with (checkpoint_dir / "resolved_config.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(_serialize_config(self.config), handle, sort_keys=False)

    def _write_model_state_checkpoint(
        self,
        checkpoint_dir: Path,
        model_state_dict: dict[str, torch.Tensor],
    ) -> None:
        _atomic_torch_save(
            {"model_state_dict": model_state_dict}, checkpoint_dir / "model_state.pt"
        )

    def _export_runtime_backbone(self, checkpoint_dir: Path, model: nn.Module) -> None:
        pipeline = getattr(model, "pipeline", model)
        visual_tower = getattr(pipeline, "visual_tower", None)
        if visual_tower is None or getattr(visual_tower, "action_dim", None) is None:
            return
        backbone = visual_tower.get_runtime_backbone(
            action_dim=int(visual_tower.action_dim)
        )
        backbone_state_dict = get_model_state_dict(
            backbone, options=_save_state_dict_options()
        )
        # DualExpert packed-coupling path: video_block weights live under
        # policy_variant.packed_block_stack.packed_blocks.{i}.video_block.* and
        # visual_tower.core.blocks is empty. Re-key those into blocks.{i}.* so
        # the exported transformer/ matches the LingBot loader layout that
        # exact parallel-stream and visualization scripts expect.
        policy_variant = getattr(pipeline, "policy_variant", None)
        packed_block_stack = getattr(policy_variant, "packed_block_stack", None)
        if packed_block_stack is not None:
            stack_state_dict = get_model_state_dict(
                packed_block_stack, options=_save_state_dict_options()
            )
            backbone_state_dict = _remap_packed_video_blocks_into_backbone(
                backbone_state_dict=backbone_state_dict,
                stack_state_dict=stack_state_dict,
            )
        if not _is_rank_zero():
            return
        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        state_dict_bf16 = {
            key: value.detach().cpu().to(torch.bfloat16)
            if torch.is_floating_point(value)
            else value.detach().cpu()
            for key, value in backbone_state_dict.items()
        }
        save_file(
            state_dict_bf16, transformer_dir / "diffusion_pytorch_model.safetensors"
        )
        config_payload = _serialize_runtime_backbone_config(
            getattr(backbone, "config", self.config.backbone)
        )
        with (transformer_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config_payload, handle, indent=2, sort_keys=True, default=str)
