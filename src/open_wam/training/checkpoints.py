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

from open_wam.artifacts import load_tensor_artifact as _load_tensor_artifact
from open_wam.configs import CheckpointMode, ExperimentConfig
from open_wam.configs.enums import serialize_enum_values
from open_wam.configs.runtime_backbone_components import (
    is_complete_runtime_backbone_selection as _is_complete_runtime_backbone_selection,
)
from open_wam.runtime.checkpoint_artifacts import (
    CheckpointOperation as _CheckpointOperation,
    checkpoint_step as _checkpoint_step,
    sorted_checkpoint_dirs as _sorted_checkpoint_dirs,
)
from open_wam.runtime.checkpoints import (
    normalize_checkpoint_state_dict as _normalize_checkpoint_state_dict,
    notify_checkpoint_loaded as _notify_checkpoint_loaded,
    resolve_checkpoint_file as _resolve_checkpoint_file,
)
from open_wam.runtime.runtime_backbone_manifest import (
    RuntimeBackboneManifest as _RuntimeBackboneManifest,
)
from open_wam.runtime.runtime_backbone_manifest import (
    write_runtime_backbone_manifest as _write_runtime_backbone_manifest,
)

from .checkpoint_export import (
    merge_state_dict_overlay as _merge_state_dict_overlay,
)
from .checkpoint_storage import (
    _atomic_torch_save,
    _is_rank_zero,
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

_FULL_TRAINING_STATE_KEYS = frozenset(
    {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "strategy_state_dict",
        "train_state",
    }
)
_TRAIN_STATE_CURSOR_KEYS = frozenset(
    {
        "global_step",
        "optimizer_step",
        "epoch_index",
        "next_batch_index",
        "seen_batches",
    }
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


def _filter_unexpected_model_state(
    expected_keys: frozenset[str],
    model_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Apply non-strict load semantics before rank-zero state broadcast.

    PyTorch's full-state broadcast indexes every checkpoint key in the local
    model state before ``strict=False`` can discard unexpected entries. This
    raises ``KeyError`` for checkpoints that contain parameters from a removed
    optional component. Filtering against the current parameter-and-buffer
    contract preserves normal non-strict loading while leaving missing and
    shape-mismatched current keys to the state-dict loader.
    """

    unexpected_keys = sorted(set(model_state_dict) - expected_keys)
    if not unexpected_keys:
        return model_state_dict
    preview = ", ".join(unexpected_keys[:8])
    suffix = "" if len(unexpected_keys) <= 8 else ", ..."
    warnings.warn(
        "Ignoring "
        f"{len(unexpected_keys)} checkpoint key(s) absent from the current model "
        f"during non-strict model-state load: {preview}{suffix}",
        RuntimeWarning,
        stacklevel=2,
    )
    return {
        key: value for key, value in model_state_dict.items() if key in expected_keys
    }


def _validate_full_training_state_payload(
    payload: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    missing_keys = sorted(_FULL_TRAINING_STATE_KEYS.difference(payload))
    if missing_keys:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a full training-state checkpoint; "
            f"missing: {', '.join(missing_keys)}."
        )
    if not isinstance(payload["model_state_dict"], dict):
        raise TypeError("`model_state_dict` must be a mapping.")
    if not isinstance(payload["train_state"], dict):
        raise TypeError("`train_state` must be a mapping.")
    missing_cursor_keys = sorted(
        _TRAIN_STATE_CURSOR_KEYS.difference(payload["train_state"])
    )
    if missing_cursor_keys:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has incomplete `train_state`; "
            f"missing cursor fields: {', '.join(missing_cursor_keys)}."
        )


def _raise_checkpoint_validation_error(error: Exception | None) -> None:
    if not dist.is_initialized():
        if error is not None:
            raise error
        return
    serialized_error: list[tuple[str, str] | None] = [
        (type(error).__name__, str(error)) if error is not None else None
    ]
    dist.broadcast_object_list(serialized_error, src=0)
    if serialized_error[0] is None:
        return
    error_type, message = serialized_error[0]
    if error_type == torch.OutOfMemoryError.__name__:
        raise torch.OutOfMemoryError(message)
    if error_type == TypeError.__name__:
        raise TypeError(message)
    if error_type == ValueError.__name__:
        raise ValueError(message)
    raise RuntimeError(message)


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
        runtime_backbone_export_keys: frozenset[str] | None = None,
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
        export_components = tuple(
            self.config.trainer.runtime_backbone_export_components
        )
        narrow_export = not _is_complete_runtime_backbone_selection(export_components)
        if (
            export_runtime_backbone
            and narrow_export
            and runtime_backbone_export_keys is None
        ):
            raise ValueError(
                "Scoped runtime-backbone export keys must be resolved before "
                "distributed strategy wrapping."
            )
        if not narrow_export and runtime_backbone_export_keys is not None:
            raise ValueError(
                "A complete runtime-backbone export must not provide scoped state keys."
            )
        self.runtime_backbone_export_keys = runtime_backbone_export_keys

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
        resolved_mode = CheckpointMode(self.checkpoint_mode)
        if (
            resolved_mode == CheckpointMode.FULL_TRAINING_STATE
            and not train_state.is_optimizer_boundary(
                self.config.training.gradient_accumulation_steps
            )
        ):
            raise ValueError(
                "Full training-state checkpoints require an optimizer boundary; "
                "partially accumulated gradients are not serialized."
            )
        checkpoint_dir = self.checkpoint_dir_for_step(step)
        payload_marker = checkpoint_dir / ".checkpoint_payload_complete"
        completion_marker = checkpoint_dir / ".checkpoint_complete"
        error_marker = checkpoint_dir / ".checkpoint_error"
        wait_timeout_seconds = float(self.config.trainer.distributed_timeout_seconds)
        if _is_rank_zero():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for marker in (payload_marker, completion_marker, error_marker):
                marker.unlink(missing_ok=True)
        if dist.is_initialized():
            dist.barrier()

        _release_unused_device_memory()
        if dist.is_initialized():
            dist.barrier()

        save_options = _save_state_dict_options()
        model_state_dict = get_model_state_dict(model, options=save_options)
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
            try:
                self._write_resolved_config(checkpoint_dir)
                if resolved_mode == CheckpointMode.MODEL_ONLY:
                    self._write_model_state_checkpoint(
                        checkpoint_dir, payload["model_state_dict"]
                    )
                elif resolved_mode == CheckpointMode.FULL_TRAINING_STATE:
                    _atomic_torch_save(
                        payload, checkpoint_dir / "full_training_state.pt"
                    )
                    # Always write a lightweight model-only checkpoint alongside
                    # the resumable training checkpoint so eval / visualization
                    # paths can skip optimizer-state deserialization.
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
                    json.dump(
                        train_state.state_dict(), handle, indent=2, sort_keys=True
                    )
                payload_marker.write_text("ok\n", encoding="utf-8")
            except BaseException as error:
                self._record_checkpoint_failure(
                    error_marker=error_marker,
                    invalid_markers=(payload_marker, completion_marker),
                    error=error,
                )
                raise
        elif dist.is_initialized():
            _wait_for_file(
                payload_marker,
                timeout_seconds=wait_timeout_seconds,
                error_marker=error_marker,
            )

        try:
            if self.export_runtime_backbone:
                self._export_runtime_backbone(checkpoint_dir, model)

            del payload
            del model_state_dict
            _release_unused_device_memory()

            if _is_rank_zero():
                self._prune_old_checkpoints(
                    keep=self.max_checkpoints_to_keep, preserve=checkpoint_dir
                )
                completion_marker.write_text("ok\n", encoding="utf-8")
        except BaseException as error:
            if _is_rank_zero():
                self._record_checkpoint_failure(
                    error_marker=error_marker,
                    invalid_markers=(completion_marker,),
                    error=error,
                )
            raise

        if not _is_rank_zero() and dist.is_initialized():
            _wait_for_file(
                completion_marker,
                timeout_seconds=wait_timeout_seconds,
                error_marker=error_marker,
            )
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
        checkpoint_path = self.resolve_checkpoint_path(
            path,
            operation=_CheckpointOperation.RESUME_TRAINING,
        )
        distributed = dist.is_initialized()
        is_rank_zero = _is_rank_zero()
        current_model_state = model.state_dict()
        expected_model_state_keys = frozenset(current_model_state)
        del current_model_state
        load_options = _load_state_dict_options(
            broadcast_from_rank0=distributed,
        )
        payload: dict[str, Any] = {}
        validation_error: Exception | None = None
        if is_rank_zero:
            try:
                loaded_payload = _load_tensor_artifact(
                    checkpoint_path, map_location=map_location
                )
                if not isinstance(loaded_payload, dict):
                    raise TypeError(
                        f"Expected checkpoint mapping at {checkpoint_path}, "
                        f"got {type(loaded_payload).__name__}."
                    )
                payload = loaded_payload
                _validate_full_training_state_payload(payload, checkpoint_path)
                checkpoint_train_state = TrainState.from_state_dict(
                    payload["train_state"]
                )
                if not checkpoint_train_state.is_optimizer_boundary(
                    self.config.training.gradient_accumulation_steps
                ):
                    raise ValueError(
                        f"Checkpoint {checkpoint_path} contains partially accumulated "
                        "gradient state, which cannot be resumed exactly."
                    )
                missing_model_keys = sorted(
                    expected_model_state_keys.difference(payload["model_state_dict"])
                )
                if missing_model_keys:
                    preview = ", ".join(missing_model_keys[:8])
                    suffix = "" if len(missing_model_keys) <= 8 else ", ..."
                    raise ValueError(
                        f"Checkpoint {checkpoint_path} is missing "
                        f"{len(missing_model_keys)} current model key(s): "
                        f"{preview}{suffix}."
                    )
                if optimizer is not None and not isinstance(
                    payload["optimizer_state_dict"], dict
                ):
                    raise ValueError(
                        f"Checkpoint {checkpoint_path} has no optimizer state to resume."
                    )
                if scheduler is not None and not isinstance(
                    payload["scheduler_state_dict"], dict
                ):
                    raise ValueError(
                        f"Checkpoint {checkpoint_path} has no scheduler state to resume."
                    )
            except Exception as error:
                validation_error = error
        _raise_checkpoint_validation_error(validation_error)

        model_state = payload.get("model_state_dict", {})
        if distributed and is_rank_zero and isinstance(model_state, dict):
            model_state = _filter_unexpected_model_state(
                expected_model_state_keys,
                model_state,
            )
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
        _notify_checkpoint_loaded(
            model,
            loaded_state_keys=expected_model_state_keys,
            missing_state_keys=frozenset(),
        )
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
            except (KeyError, ValueError, RuntimeError, TypeError) as error:
                if isinstance(error, torch.OutOfMemoryError):
                    raise
                # Retry with a densified optimizer state to tolerate legacy
                # sparse-tensor / partial-shard checkpoints. OOM and exception
                # classes unrelated to optimizer structure surface directly.
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

    def initialize_weights(
        self,
        *,
        path: str | Path,
        model: nn.Module,
        map_location: str | torch.device = "cpu",
    ) -> Path:
        """Initialize model tensors without restoring training progress."""

        checkpoint_path = self.resolve_checkpoint_path(
            path,
            operation=_CheckpointOperation.INITIALIZE_WEIGHTS,
        )
        distributed = dist.is_initialized()
        is_rank_zero = _is_rank_zero()
        expected_state_keys = frozenset(model.state_dict())
        payload: dict[str, Any] = {}
        model_state: dict[str, torch.Tensor] = {}
        validation_error: Exception | None = None
        if is_rank_zero:
            try:
                loaded_payload = _load_tensor_artifact(
                    checkpoint_path, map_location=map_location
                )
                if not isinstance(loaded_payload, dict):
                    raise TypeError(
                        f"Expected checkpoint mapping at {checkpoint_path}, "
                        f"got {type(loaded_payload).__name__}."
                    )
                payload = loaded_payload
                model_state = _normalize_checkpoint_state_dict(payload)
                if not frozenset(model_state).intersection(expected_state_keys):
                    raise ValueError(
                        f"Checkpoint {checkpoint_path} has no parameters matching "
                        "the current model."
                    )
            except Exception as error:
                validation_error = error
        _raise_checkpoint_validation_error(validation_error)
        loaded_state_keys = frozenset(model_state).intersection(expected_state_keys)
        missing_state_keys = expected_state_keys.difference(loaded_state_keys)
        if distributed:
            lifecycle_keys: list[tuple[frozenset[str], frozenset[str]] | None] = [
                (loaded_state_keys, missing_state_keys) if is_rank_zero else None
            ]
            dist.broadcast_object_list(lifecycle_keys, src=0)
            loaded_state_keys, missing_state_keys = lifecycle_keys[0] or (
                frozenset(),
                expected_state_keys,
            )
        model_state = _filter_unexpected_model_state(
            expected_state_keys,
            model_state,
        )
        _set_model_state_dict(
            model,
            model_state,
            options=_load_state_dict_options(broadcast_from_rank0=distributed),
        )
        _notify_checkpoint_loaded(
            model,
            loaded_state_keys=loaded_state_keys,
            missing_state_keys=missing_state_keys,
        )
        return checkpoint_path

    @staticmethod
    def resolve_checkpoint_path(
        path: str | Path,
        *,
        operation: _CheckpointOperation | str = _CheckpointOperation.RESUME_TRAINING,
    ) -> Path:
        return _resolve_checkpoint_file(path, operation=operation)

    def find_latest_checkpoint(self, root: str | Path) -> Path | None:
        checkpoint_dirs = self._complete_checkpoint_dirs(Path(root))
        return checkpoint_dirs[-1] if checkpoint_dirs else None

    def _complete_checkpoint_dirs(self, root: Path | None = None) -> list[Path]:
        root_path = self.root_dir if root is None else Path(root)
        scan_dirs = [
            checkpoint_dir
            for checkpoint_dir in _sorted_checkpoint_dirs(root_path)
            if _checkpoint_step(checkpoint_dir) >= 0
        ]
        checkpoint_dirs = [
            checkpoint_dir
            for checkpoint_dir in scan_dirs
            if (checkpoint_dir / "full_training_state.pt").exists()
            or (checkpoint_dir / "model_state.pt").exists()
        ]
        return checkpoint_dirs

    def _prune_old_checkpoints(self, *, keep: int | None, preserve: Path) -> list[Path]:
        if keep is None:
            return []
        checkpoint_dirs = self._complete_checkpoint_dirs()
        preserve = preserve.resolve()
        if preserve.is_dir() and all(
            checkpoint_dir.resolve() != preserve for checkpoint_dir in checkpoint_dirs
        ):
            # The current checkpoint is intentionally unmarked until pruning
            # succeeds. Count it toward retention without exposing it as a
            # complete checkpoint to readers.
            checkpoint_dirs.append(preserve)
            checkpoint_dirs.sort(key=lambda path: int(path.name.split("_")[-1]))
        if len(checkpoint_dirs) <= int(keep):
            return []
        removed: list[Path] = []
        remove_count = len(checkpoint_dirs) - int(keep)
        removable = [
            checkpoint_dir
            for checkpoint_dir in checkpoint_dirs
            if checkpoint_dir.resolve() != preserve
        ]
        for checkpoint_dir in removable[:remove_count]:
            shutil.rmtree(checkpoint_dir)
            removed.append(checkpoint_dir)
        return removed

    @staticmethod
    def _record_checkpoint_failure(
        *,
        error_marker: Path,
        invalid_markers: tuple[Path, ...],
        error: BaseException,
    ) -> None:
        """Invalidate success markers and best-effort signal rank-zero failure."""

        for marker in invalid_markers:
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            error_marker.write_text(
                f"{type(error).__name__}: {error}\n",
                encoding="utf-8",
            )
        except OSError:
            pass

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
        topology = pipeline.module_topology()
        backbone = visual_tower.get_runtime_backbone(
            action_dim=int(visual_tower.action_dim)
        )
        backbone_state_dict = get_model_state_dict(
            backbone, options=_save_state_dict_options()
        )
        export_components = tuple(
            self.config.trainer.runtime_backbone_export_components
        )
        narrow_export = not _is_complete_runtime_backbone_selection(export_components)
        selected_state_keys = (
            set(self.runtime_backbone_export_keys or ()) if narrow_export else None
        )
        for overlay in topology.runtime_backbone_state_overlays:
            overlay_state_dict = get_model_state_dict(
                overlay.module,
                options=_save_state_dict_options(),
            )
            backbone_state_dict = _merge_state_dict_overlay(
                base_state_dict=backbone_state_dict,
                overlay_state_dict=overlay_state_dict,
                map_key=overlay.map_key,
                exclusive_target_prefixes=overlay.exclusive_target_prefixes,
            )
        if not _is_rank_zero():
            return
        if selected_state_keys is not None:
            unavailable = sorted(selected_state_keys - set(backbone_state_dict))
            if unavailable:
                raise ValueError(
                    "Runtime-backbone component ownership resolved keys absent from "
                    f"the exported state: {unavailable[:20]}."
                )
            backbone_state_dict = {
                key: value
                for key, value in backbone_state_dict.items()
                if key in selected_state_keys
            }
            if not backbone_state_dict:
                raise ValueError(
                    "Runtime-backbone export selectors resolved no state tensors."
                )
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
        _write_runtime_backbone_manifest(
            transformer_dir,
            _RuntimeBackboneManifest(
                components=export_components,
                state_keys=tuple(sorted(state_dict_bf16)),
            ),
        )
