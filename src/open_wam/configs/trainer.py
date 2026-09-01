from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .coercion import coerce_enum
from .enums import (
    BatchAdapterName,
    CheckpointMode,
    LoopPolicyName,
    StrategyName,
    TrainerAccelerator,
    TrainerPrecision,
    TrainerRuntimeName,
    TrainingComponentSelector,
    WandBMode,
    coerce_fields,
)
from .runtime_backbone_components import (
    validate_runtime_backbone_components,
)


@dataclass(frozen=True)
class TrainerConfig:
    """Runtime/launcher config stored separately from model configs."""

    # Loop-shape knobs
    max_epochs: int = 1
    limit_train_batches: int = 2
    limit_val_batches: int = 1
    validation_interval: int | None = None
    log_every_n_steps: int = 1

    # Device/runtime-selection knobs. Worker creation belongs to the external
    # launcher; `devices` is retained as a compatibility process-count expectation.
    accelerator: TrainerAccelerator = TrainerAccelerator.CPU
    devices: int = 1
    precision: TrainerPrecision = TrainerPrecision.FP32
    enable_checkpointing: bool = False
    enable_model_summary: bool = False
    runtime: TrainerRuntimeName = TrainerRuntimeName.COMPOSABLE
    batch_adapter: BatchAdapterName = BatchAdapterName.VIEWS
    loop_policy: LoopPolicyName = LoopPolicyName.EPOCHS
    strategy: StrategyName = StrategyName.SINGLE_DEVICE
    distributed_timeout_seconds: int = 1800
    default_root_dir: str | None = None

    # Checkpoint/export knobs
    checkpoint_dir: str | None = None
    save_interval: int | None = None
    checkpoint_mode: CheckpointMode = CheckpointMode.FULL_TRAINING_STATE
    max_checkpoints_to_keep: int | None = None
    export_runtime_backbone: bool = False
    runtime_backbone_export_components: tuple[TrainingComponentSelector, ...] = (
        TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE,
    )
    initialize_weights_from: str | None = None
    resume_from: str | None = None

    # Logging/tracking knobs
    enable_jsonl_logging: bool = False
    metrics_filename: str = "metrics.jsonl"
    enable_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: WandBMode = WandBMode.DISABLED
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.runtime == "lightning":
            raise ValueError(
                "`trainer.runtime=lightning` is no longer supported. "
                "Use the production `composable` runtime."
            )
        if self.strategy == "lightning":
            raise ValueError(
                "`trainer.strategy=lightning` is no longer supported. "
                "Use `single_device`, `ddp`, or `fsdp`."
            )
        coerce_fields(
            self,
            enum_fields={
                "accelerator": TrainerAccelerator,
                "precision": TrainerPrecision,
                "runtime": TrainerRuntimeName,
                "batch_adapter": BatchAdapterName,
                "loop_policy": LoopPolicyName,
                "strategy": StrategyName,
                "checkpoint_mode": CheckpointMode,
                "wandb_mode": WandBMode,
            },
        )
        object.__setattr__(
            self,
            "runtime_backbone_export_components",
            validate_runtime_backbone_components(
                self.runtime_backbone_export_components,
                scope="`trainer.runtime_backbone_export_components`",
            ),
        )
        if self.initialize_weights_from is not None and self.resume_from is not None:
            raise ValueError(
                "Choose either `trainer.initialize_weights_from` or "
                "`trainer.resume_from`, not both."
            )
        if isinstance(self.devices, bool) or int(self.devices) <= 0:
            raise ValueError("`trainer.devices` must be a positive integer.")
        object.__setattr__(self, "devices", int(self.devices))
        if self.validation_interval is not None:
            if isinstance(self.validation_interval, bool) or int(self.validation_interval) <= 0:
                raise ValueError("`trainer.validation_interval` must be a positive integer or null.")
            object.__setattr__(self, "validation_interval", int(self.validation_interval))
        if self.max_checkpoints_to_keep is not None:
            if isinstance(self.max_checkpoints_to_keep, bool) or int(self.max_checkpoints_to_keep) <= 0:
                raise ValueError("`trainer.max_checkpoints_to_keep` must be a positive integer or null.")
            object.__setattr__(self, "max_checkpoints_to_keep", int(self.max_checkpoints_to_keep))
        if (
            isinstance(self.distributed_timeout_seconds, bool)
            or int(self.distributed_timeout_seconds) <= 0
        ):
            raise ValueError(
                "`trainer.distributed_timeout_seconds` must be a positive integer."
            )
        object.__setattr__(
            self,
            "distributed_timeout_seconds",
            int(self.distributed_timeout_seconds),
        )


def parse_trainer_config(raw_value: Mapping[str, Any] | None) -> TrainerConfig:
    """Parse generic loop, device, checkpoint, and logging controls."""

    raw = raw_value or {}
    return TrainerConfig(
        max_epochs=raw.get("max_epochs", 1),
        limit_train_batches=raw.get("limit_train_batches", 2),
        limit_val_batches=raw.get("limit_val_batches", 1),
        validation_interval=raw.get("validation_interval"),
        log_every_n_steps=raw.get("log_every_n_steps", 1),
        accelerator=coerce_enum(
            TrainerAccelerator,
            raw.get("accelerator", "cpu"),
        ),
        devices=raw.get("devices", 1),
        precision=coerce_enum(
            TrainerPrecision,
            raw.get("precision", "32-true"),
        ),
        enable_checkpointing=raw.get("enable_checkpointing", False),
        enable_model_summary=raw.get("enable_model_summary", False),
        runtime=coerce_enum(
            TrainerRuntimeName,
            raw.get("runtime", "composable"),
        ),
        batch_adapter=coerce_enum(
            BatchAdapterName,
            raw.get("batch_adapter", "views"),
        ),
        loop_policy=coerce_enum(
            LoopPolicyName,
            raw.get("loop_policy", "epochs"),
        ),
        strategy=coerce_enum(
            StrategyName,
            raw.get("strategy", "single_device"),
        ),
        distributed_timeout_seconds=raw.get("distributed_timeout_seconds", 1800),
        default_root_dir=raw.get("default_root_dir"),
        checkpoint_dir=raw.get("checkpoint_dir"),
        save_interval=raw.get("save_interval"),
        checkpoint_mode=coerce_enum(
            CheckpointMode,
            raw.get("checkpoint_mode", "full_training_state"),
        ),
        max_checkpoints_to_keep=raw.get("max_checkpoints_to_keep"),
        export_runtime_backbone=raw.get("export_runtime_backbone", False),
        runtime_backbone_export_components=raw.get(
            "runtime_backbone_export_components",
            (TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE,),
        ),
        initialize_weights_from=raw.get("initialize_weights_from"),
        resume_from=raw.get("resume_from"),
        enable_jsonl_logging=raw.get("enable_jsonl_logging", False),
        metrics_filename=raw.get("metrics_filename", "metrics.jsonl"),
        enable_wandb=raw.get("enable_wandb", False),
        wandb_project=raw.get("wandb_project"),
        wandb_entity=raw.get("wandb_entity"),
        wandb_mode=coerce_enum(
            WandBMode,
            raw.get("wandb_mode", "disabled"),
        ),
        run_name=raw.get("run_name"),
    )
