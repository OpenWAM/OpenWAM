from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from open_wam.configs import BatchAdapterName, ExperimentConfig, LoopPolicyName, StrategyName, TrainerRuntimeName
from open_wam.configs.enums import serialize_enum_values
from open_wam.data import (
    WAMSample,
    build_train_val_datasets,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    collate_wam_samples,
)
from open_wam.data.latent_contracts import LatentWAMSample
from open_wam.pipelines import build_variant_pipeline_from_config

from .checkpoints import CheckpointManager
from .controls import TrainabilityReport, apply_training_component_controls
from .logging import CompositeLogSink, ConsoleLogSink, JsonlLogSink, NoopLogSink, WandBLogSink
from .loop_policies import EpochLoopPolicy, StepLoopPolicy
from .optim import build_optimizer, build_scheduler
from .state import TrainState
from .step_executor import PipelineTrainStepExecutor, build_batch_adapter
from .strategies import build_training_strategy


class TrainingRuntime:
    """Composable training runtime built from general, decoupled components."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        model: torch.nn.Module,
        strategy,
        train_loader: DataLoader,
        val_loader: DataLoader,
        step_executor: PipelineTrainStepExecutor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_manager: CheckpointManager,
        log_sink: CompositeLogSink,
        train_state: TrainState,
        trainability_report: TrainabilityReport,
    ) -> None:
        self.config = config
        self.model = model
        self.strategy = strategy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.step_executor = step_executor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.log_sink = log_sink
        self.train_state = train_state
        self.trainability_report = trainability_report

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "TrainingRuntime":
        strategy = build_training_strategy(config.trainer)
        model = build_variant_pipeline_from_config(config)
        trainability_report = apply_training_component_controls(model, config.training)
        model = strategy.prepare_model(model)
        batch_adapter = build_batch_adapter(config.trainer.batch_adapter)
        step_executor = PipelineTrainStepExecutor(
            pipeline=model,
            batch_adapter=batch_adapter,
            training_config=config.training,
        )
        train_loader, val_loader = build_runtime_dataloaders(config, strategy)
        optimizer = build_optimizer(model, config.training)
        scheduler = build_scheduler(optimizer, config.training)
        output_dir = resolve_runtime_output_dir(config)
        checkpoint_root = Path(config.trainer.checkpoint_dir) if config.trainer.checkpoint_dir else output_dir / "checkpoints"
        checkpoint_manager = CheckpointManager(
            root_dir=checkpoint_root,
            config=config,
            checkpoint_mode=config.trainer.checkpoint_mode,
            export_runtime_backbone=config.trainer.export_runtime_backbone,
        )
        run_name = config.trainer.run_name or config.name
        train_state = TrainState(run_name=run_name)
        log_sink = build_log_sink(config=config, output_dir=output_dir, run_name=run_name, strategy=strategy)
        runtime = cls(
            config=config,
            model=model,
            strategy=strategy,
            train_loader=train_loader,
            val_loader=val_loader,
            step_executor=step_executor,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_manager=checkpoint_manager,
            log_sink=log_sink,
            train_state=train_state,
            trainability_report=trainability_report,
        )
        if config.trainer.resume_from is not None:
            runtime.resume(config.trainer.resume_from)
        return runtime

    def resume(self, checkpoint_path: str) -> None:
        train_state, payload = self.checkpoint_manager.load(
            path=checkpoint_path,
            model=self.strategy.unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.strategy.device,
        )
        self.train_state = train_state
        self.strategy.load_state_dict(payload.get("strategy_state_dict") if isinstance(payload, dict) else None)
        self.log_sink.log_event(
            name="resume",
            payload={"checkpoint_path": checkpoint_path, "optimizer_step": self.train_state.optimizer_step},
        )

    def run(self) -> TrainState:
        self.log_sink.log_event(
            name="run_start",
            payload={
                "run_name": self.train_state.run_name,
                "runtime": self.config.trainer.runtime,
                "batch_adapter": self.config.trainer.batch_adapter,
                "loop_policy": self.config.trainer.loop_policy,
                "strategy": self.config.trainer.strategy,
                "output_dir": str(resolve_runtime_output_dir(self.config)),
                "enabled_objectives": self.trainability_report.enabled_objectives,
                "trainable_components": self.trainability_report.trainable_components,
                "frozen_components": self.trainability_report.frozen_components,
                "trainable_parameters": self.trainability_report.trainable_parameters,
                "total_parameters": self.trainability_report.total_parameters,
            },
        )
        self.strategy.zero_grad(self.optimizer)
        try:
            if self.config.trainer.loop_policy == LoopPolicyName.STEPS:
                max_steps = self.config.training.num_steps
                if max_steps is None:
                    raise ValueError("`training.num_steps` is required when `trainer.loop_policy = steps`.")
                self._run_step_loop(StepLoopPolicy(
                    max_steps=max_steps,
                    limit_train_batches=self.config.trainer.limit_train_batches,
                    limit_val_batches=self.config.trainer.limit_val_batches,
                ))
            else:
                self._run_epoch_loop(EpochLoopPolicy(
                    max_epochs=self.config.trainer.max_epochs,
                    limit_train_batches=self.config.trainer.limit_train_batches,
                    limit_val_batches=self.config.trainer.limit_val_batches,
                ))
        finally:
            self.log_sink.close()
            self.strategy.close()
        return self.train_state

    def _run_epoch_loop(self, policy: EpochLoopPolicy) -> None:
        while policy.should_continue(self.train_state):
            train_sampler = getattr(self.train_loader, "sampler", None)
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(self.train_state.epoch_index)
            for batch_idx, batch in enumerate(self.train_loader):
                if policy.limit_train_batches is not None and batch_idx >= policy.limit_train_batches:
                    break
                self._train_micro_step(batch)
            self._run_validation(limit_batches=policy.limit_val_batches)
            self.train_state.epoch_index += 1
        self._save_checkpoint(final=True)

    def _run_step_loop(self, policy: StepLoopPolicy) -> None:
        for _, batch in enumerate(_cycle(self.train_loader)):
            self._train_micro_step(batch)
            if not policy.should_continue(self.train_state):
                break
        self._run_validation(limit_batches=policy.limit_val_batches)
        self._save_checkpoint(final=True)

    def _train_micro_step(self, batch) -> None:
        device_batch = self.step_executor.batch_adapter.move_to_device(batch, self.strategy.device)
        self.model.train()
        with self.strategy.autocast_context():
            result = self.step_executor.forward_train(device_batch)
            loss = result.loss / max(1, self.config.training.gradient_accumulation_steps)
        self.strategy.backward(loss)
        self.train_state.global_step += 1
        self.train_state.seen_batches += 1

        should_update = self.train_state.global_step % max(1, self.config.training.gradient_accumulation_steps) == 0
        if not should_update:
            return

        self.strategy.unscale_(self.optimizer)
        if self.config.training.max_grad_norm is not None:
            self.strategy.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        self.strategy.optimizer_step(self.optimizer)
        self.scheduler.step()
        self.strategy.zero_grad(self.optimizer)
        self.train_state.optimizer_step += 1

        metric_payload = {name: float(value.item()) for name, value in result.metrics.items()}
        if (
            self.config.trainer.log_every_n_steps <= 1
            or self.train_state.optimizer_step % self.config.trainer.log_every_n_steps == 0
        ):
            self.log_sink.log_metrics(step=self.train_state.optimizer_step, phase="train", metrics=metric_payload)
        if self._should_save_checkpoint():
            self._save_checkpoint(final=False)

    def _run_validation(self, *, limit_batches: int | None) -> None:
        self.model.eval()
        metric_totals: dict[str, float] = {}
        batch_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                device_batch = self.step_executor.batch_adapter.move_to_device(batch, self.strategy.device)
                with self.strategy.autocast_context():
                    result = self.step_executor.forward_train(device_batch)
                for name, value in result.metrics.items():
                    metric_totals[name] = metric_totals.get(name, 0.0) + float(value.item())
                batch_count += 1
        if batch_count == 0:
            return
        averaged = {name: value / batch_count for name, value in metric_totals.items()}
        self.log_sink.log_metrics(step=self.train_state.optimizer_step, phase="val", metrics=averaged)

    def _should_save_checkpoint(self) -> bool:
        save_interval = self.config.trainer.save_interval
        if save_interval is None or save_interval <= 0:
            return False
        return self.train_state.optimizer_step > 0 and self.train_state.optimizer_step % save_interval == 0

    def _save_checkpoint(self, *, final: bool) -> None:
        should_write = (
            self.config.trainer.enable_checkpointing
            or self.config.trainer.save_interval is not None
            or self.config.trainer.export_runtime_backbone
        )
        if not should_write:
            return
        if not self.strategy.is_main_process:
            self.strategy.barrier()
            return
        checkpoint_dir = self.checkpoint_manager.save(
            step=self.train_state.optimizer_step,
            model=self.strategy.unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_state=self.train_state,
            strategy_state=self.strategy.state_dict(),
        )
        self.train_state.last_checkpoint_path = str(checkpoint_dir)
        self.log_sink.log_event(
            name="checkpoint_saved",
            payload={"path": str(checkpoint_dir), "final": final, "optimizer_step": self.train_state.optimizer_step},
        )
        self.strategy.barrier()


def build_runtime_dataloaders(config: ExperimentConfig, strategy) -> tuple[DataLoader, DataLoader]:
    if config.trainer.batch_adapter == BatchAdapterName.LATENTS:
        train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
        train_sampler = (
            DistributedSampler(train_dataset, shuffle=True, num_replicas=strategy.world_size, rank=strategy.rank)
            if strategy.distributed
            else None
        )
        val_sampler = (
            DistributedSampler(val_dataset, shuffle=False, num_replicas=strategy.world_size, rank=strategy.rank)
            if strategy.distributed
            else None
        )
        return (
            DataLoader(
                train_dataset,
                batch_size=config.data.train_batch_size,
                shuffle=train_sampler is None,
                num_workers=config.data.num_workers,
                sampler=train_sampler,
                collate_fn=collate_latent_wam_samples,
            ),
            DataLoader(
                val_dataset,
                batch_size=config.data.val_batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                sampler=val_sampler,
                collate_fn=collate_latent_wam_samples,
            ),
        )
    train_dataset, val_dataset = build_train_val_datasets(config.data)
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, num_replicas=strategy.world_size, rank=strategy.rank)
        if strategy.distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False, num_replicas=strategy.world_size, rank=strategy.rank)
        if strategy.distributed
        else None
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=config.data.train_batch_size,
            shuffle=train_sampler is None,
            num_workers=config.data.num_workers,
            sampler=train_sampler,
            collate_fn=collate_wam_samples,
        ),
        DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            sampler=val_sampler,
            collate_fn=collate_wam_samples,
        ),
    )


def build_log_sink(*, config: ExperimentConfig, output_dir: Path, run_name: str, strategy=None) -> CompositeLogSink:
    if strategy is not None and not strategy.is_main_process:
        return CompositeLogSink([NoopLogSink()])
    sinks = [ConsoleLogSink()]
    if config.trainer.enable_jsonl_logging:
        sinks.append(JsonlLogSink(output_dir / config.trainer.metrics_filename))
    if config.trainer.enable_wandb:
        sinks.append(
            WandBLogSink(
                project=config.trainer.wandb_project,
                entity=config.trainer.wandb_entity,
                mode=config.trainer.wandb_mode,
                run_name=run_name,
                config_payload=serialize_enum_values(asdict(config)),
            )
        )
    return CompositeLogSink(sinks)


def resolve_runtime_output_dir(config: ExperimentConfig) -> Path:
    root = Path(config.trainer.default_root_dir) if config.trainer.default_root_dir else Path("runs")
    return root / (config.trainer.run_name or config.name)


def should_use_composable_runtime(config: ExperimentConfig) -> bool:
    if config.trainer.runtime != TrainerRuntimeName.LIGHTNING:
        return True
    if config.trainer.batch_adapter != BatchAdapterName.VIEWS:
        return True
    if config.trainer.loop_policy != LoopPolicyName.EPOCHS:
        return True
    if config.trainer.enable_jsonl_logging or config.trainer.enable_wandb:
        return True
    if config.trainer.save_interval is not None or config.trainer.resume_from is not None:
        return True
    if config.trainer.strategy not in {StrategyName.LIGHTNING, StrategyName.SINGLE_DEVICE}:
        return True
    return False


def _cycle(loader: DataLoader) -> Iterator[WAMSample | LatentWAMSample]:
    while True:
        for batch in loader:
            yield batch
