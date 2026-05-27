from __future__ import annotations

from dataclasses import dataclass

from .enums import (
    BatchAdapterName,
    CheckpointMode,
    LoopPolicyName,
    StrategyName,
    TrainerAccelerator,
    TrainerPrecision,
    TrainerRuntimeName,
    WandBMode,
    coerce_fields,
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

    # Device/runtime-selection knobs
    accelerator: TrainerAccelerator = TrainerAccelerator.CPU
    devices: int = 1
    precision: TrainerPrecision = TrainerPrecision.FP32
    enable_checkpointing: bool = False
    enable_model_summary: bool = False
    runtime: TrainerRuntimeName = TrainerRuntimeName.LIGHTNING
    batch_adapter: BatchAdapterName = BatchAdapterName.VIEWS
    loop_policy: LoopPolicyName = LoopPolicyName.EPOCHS
    strategy: StrategyName = StrategyName.LIGHTNING
    default_root_dir: str | None = None

    # Checkpoint/export knobs
    checkpoint_dir: str | None = None
    save_interval: int | None = None
    checkpoint_mode: CheckpointMode = CheckpointMode.FULL_TRAINING_STATE
    max_checkpoints_to_keep: int | None = None
    export_runtime_backbone: bool = False
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
        if self.validation_interval is not None:
            if isinstance(self.validation_interval, bool) or int(self.validation_interval) <= 0:
                raise ValueError("`trainer.validation_interval` must be a positive integer or null.")
            object.__setattr__(self, "validation_interval", int(self.validation_interval))
        if self.max_checkpoints_to_keep is not None:
            if isinstance(self.max_checkpoints_to_keep, bool) or int(self.max_checkpoints_to_keep) <= 0:
                raise ValueError("`trainer.max_checkpoints_to_keep` must be a positive integer or null.")
            object.__setattr__(self, "max_checkpoints_to_keep", int(self.max_checkpoints_to_keep))
