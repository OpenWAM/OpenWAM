from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file
from torch import nn

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
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._write_resolved_config(checkpoint_dir)

        payload = {
            "model_state_dict": model.state_dict(),
            "train_state": train_state.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "strategy_state_dict": strategy_state,
        }
        if self.checkpoint_mode == CheckpointMode.MODEL_ONLY:
            torch.save({"model_state_dict": payload["model_state_dict"]}, checkpoint_dir / "model_state.pt")
        elif self.checkpoint_mode == CheckpointMode.FULL_TRAINING_STATE:
            torch.save(payload, checkpoint_dir / "full_training_state.pt")
        else:
            raise ValueError(f"Unsupported checkpoint_mode {self.checkpoint_mode!r}.")

        with (checkpoint_dir / "train_state.json").open("w", encoding="utf-8") as handle:
            json.dump(train_state.state_dict(), handle, indent=2, sort_keys=True)

        if self.export_runtime_backbone:
            self._export_runtime_backbone(checkpoint_dir, model)
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
        model.load_state_dict(payload["model_state_dict"])
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer is not None and isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
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

    def _export_runtime_backbone(self, checkpoint_dir: Path, model: nn.Module) -> None:
        pipeline = getattr(model, "pipeline", model)
        visual_tower = getattr(pipeline, "visual_tower", None)
        if visual_tower is None or getattr(visual_tower, "action_dim", None) is None:
            return
        backbone = visual_tower.get_runtime_backbone(action_dim=int(visual_tower.action_dim))
        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        state_dict_bf16 = {
            key: value.detach().cpu().to(torch.bfloat16) if torch.is_floating_point(value) else value.detach().cpu()
            for key, value in backbone.state_dict().items()
        }
        save_file(state_dict_bf16, transformer_dir / "diffusion_pytorch_model.safetensors")
        config_payload = _serialize_runtime_backbone_config(getattr(backbone, "config", self.config.backbone))
        with (transformer_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config_payload, handle, indent=2, sort_keys=True, default=str)
