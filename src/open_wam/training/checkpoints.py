from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time
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

from .state import TrainState


def _serialize_config(config: ExperimentConfig) -> dict[str, Any]:
    if is_dataclass(config):
        return serialize_enum_values(asdict(config))
    raise TypeError(f"Expected dataclass config, got {type(config).__name__}.")


def _serialize_runtime_backbone_config(backbone_config: object) -> dict[str, Any]:
    if is_dataclass(backbone_config):
        return serialize_enum_values(asdict(backbone_config))
    if isinstance(backbone_config, dict):
        return serialize_enum_values(dict(backbone_config))
    return {"repr": repr(backbone_config)}


def _is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _wait_for_file(path: Path, *, timeout_seconds: float = 7200.0, poll_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for checkpoint completion marker: {path}")
        time.sleep(float(poll_seconds))


def _save_state_dict_options() -> StateDictOptions:
    return StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        strict=False,
    )


def _load_state_dict_options() -> StateDictOptions:
    return StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        strict=False,
    )


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
    dense_state = {
        key: loaded_state.get(key, {})
        for key in current_state
    }

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


class CheckpointManager:
    """Own save/load/export behavior for the composable training runtime."""

    def __init__(
        self,
        *,
        root_dir: Path,
        config: ExperimentConfig,
        checkpoint_mode: CheckpointMode | str,
        export_runtime_backbone: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.checkpoint_mode = checkpoint_mode
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
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "strategy_state_dict": strategy_state,
                }
            )

        if _is_rank_zero():
            self._write_resolved_config(checkpoint_dir)
            if resolved_mode == CheckpointMode.MODEL_ONLY:
                self._write_model_state_checkpoint(checkpoint_dir, payload["model_state_dict"])
            elif resolved_mode == CheckpointMode.FULL_TRAINING_STATE:
                torch.save(payload, checkpoint_dir / "full_training_state.pt")
                # Always write a lightweight model-only checkpoint alongside the
                # resumable training checkpoint so eval / visualization paths
                # can skip optimizer-state deserialization.
                self._write_model_state_checkpoint(checkpoint_dir, payload["model_state_dict"])
            else:
                raise ValueError(f"Unsupported checkpoint_mode {self.checkpoint_mode!r}.")

            with (checkpoint_dir / "train_state.json").open("w", encoding="utf-8") as handle:
                json.dump(train_state.state_dict(), handle, indent=2, sort_keys=True)
            payload_marker.write_text("ok\n", encoding="utf-8")
        elif dist.is_initialized():
            _wait_for_file(payload_marker)

        if self.export_runtime_backbone:
            self._export_runtime_backbone(checkpoint_dir, model)

        if _is_rank_zero():
            completion_marker.write_text("ok\n", encoding="utf-8")
        elif dist.is_initialized():
            _wait_for_file(completion_marker)
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
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        load_options = _load_state_dict_options()
        set_model_state_dict(model, payload["model_state_dict"], options=load_options)
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer is not None and isinstance(optimizer_state, dict):
            try:
                set_optimizer_state_dict(model, optimizer, optim_state_dict=optimizer_state, options=load_options)
            except Exception:
                dense_optimizer_state = _densify_optimizer_state_dict(
                    model=model,
                    optimizer=optimizer,
                    optim_state_dict=optimizer_state,
                    options=load_options,
                )
                set_optimizer_state_dict(model, optimizer, optim_state_dict=dense_optimizer_state, options=load_options)
        scheduler_state = payload.get("scheduler_state_dict")
        if scheduler is not None and isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)
        train_state = TrainState.from_state_dict(payload.get("train_state"))
        train_state.last_checkpoint_path = str(checkpoint_path.parent)
        train_state.resume_source = str(checkpoint_path)
        return train_state, payload

    def resolve_checkpoint_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_file():
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
        raise FileNotFoundError(f"Unable to resolve a checkpoint file from {candidate}.")

    def find_latest_checkpoint(self, root: str | Path) -> Path | None:
        root_path = Path(root)
        checkpoint_dirs = sorted(
            [
                path
                for path in root_path.glob("checkpoint_step_*")
                if path.is_dir() and ((path / "full_training_state.pt").exists() or (path / "model_state.pt").exists())
            ],
            key=lambda path: int(path.name.split("_")[-1]),
        )
        return checkpoint_dirs[-1] if checkpoint_dirs else None

    def _write_resolved_config(self, checkpoint_dir: Path) -> None:
        with (checkpoint_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_serialize_config(self.config), handle, sort_keys=False)

    def _write_model_state_checkpoint(
        self,
        checkpoint_dir: Path,
        model_state_dict: dict[str, torch.Tensor],
    ) -> None:
        torch.save({"model_state_dict": model_state_dict}, checkpoint_dir / "model_state.pt")

    def _export_runtime_backbone(self, checkpoint_dir: Path, model: nn.Module) -> None:
        pipeline = getattr(model, "pipeline", model)
        visual_tower = getattr(pipeline, "visual_tower", None)
        if visual_tower is None or getattr(visual_tower, "action_dim", None) is None:
            return
        backbone = visual_tower.get_runtime_backbone(action_dim=int(visual_tower.action_dim))
        backbone_state_dict = get_model_state_dict(backbone, options=_save_state_dict_options())
        # MoT packed-coupling path: video_block weights live under
        # policy_variant.packed_block_stack.packed_blocks.{i}.video_block.* and
        # visual_tower.core.blocks is empty. Re-key those into blocks.{i}.* so
        # the exported transformer/ matches the LingBot loader layout that
        # method-1 / visualization scripts expect.
        policy_variant = getattr(pipeline, "policy_variant", None)
        packed_block_stack = getattr(policy_variant, "packed_block_stack", None)
        if packed_block_stack is not None:
            stack_state_dict = get_model_state_dict(packed_block_stack, options=_save_state_dict_options())
            backbone_state_dict = _remap_packed_video_blocks_into_backbone(
                backbone_state_dict=backbone_state_dict,
                stack_state_dict=stack_state_dict,
            )
        if not _is_rank_zero():
            return
        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        state_dict_bf16 = {
            key: value.detach().cpu().to(torch.bfloat16) if torch.is_floating_point(value) else value.detach().cpu()
            for key, value in backbone_state_dict.items()
        }
        save_file(state_dict_bf16, transformer_dir / "diffusion_pytorch_model.safetensors")
        config_payload = _serialize_runtime_backbone_config(getattr(backbone, "config", self.config.backbone))
        with (transformer_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config_payload, handle, indent=2, sort_keys=True, default=str)


def _remap_packed_video_blocks_into_backbone(
    *,
    backbone_state_dict: dict[str, torch.Tensor],
    stack_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Move ``packed_blocks.{i}.video_block.*`` entries under ``blocks.{i}.*``.

    After ownership transfer in ``MoTPolicyVariant.attach_visual_tower``, the
    visual_tower core no longer owns its blocks; running ``state_dict()`` on
    the core therefore drops every ``blocks.{i}.*`` weight. The packed stack
    holds the canonical video block weights under
    ``packed_blocks.{i}.video_block.*``; this helper re-keys them so the
    exported runtime backbone state dict is a drop-in replacement for the
    pre-surgery layout. ``action_block.*`` entries are intentionally skipped —
    they belong to the action expert export path, not the video runtime
    backbone.
    """

    if any(key.startswith("blocks.") for key in backbone_state_dict):
        raise ValueError(
            "Runtime backbone state dict already contains `blocks.*` keys; "
            "packed-coupling remap would clobber them. Investigate why "
            "visual_tower.core kept its block weights despite the packed "
            "stack being attached."
        )
    remapped: dict[str, torch.Tensor] = dict(backbone_state_dict)
    prefix = "packed_blocks."
    video_marker = ".video_block."
    for key, tensor in stack_state_dict.items():
        if not key.startswith(prefix):
            continue
        marker_index = key.find(video_marker, len(prefix))
        if marker_index == -1:
            # action_block.* (or any other future child) — not part of the
            # video runtime backbone export.
            continue
        block_index_str = key[len(prefix) : marker_index]
        if not block_index_str.isdigit():
            continue
        suffix = key[marker_index + len(video_marker) :]
        remapped[f"blocks.{block_index_str}.{suffix}"] = tensor
    return remapped
