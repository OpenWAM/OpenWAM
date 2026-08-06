from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from open_wam.configs import (
    AuxiliaryValidationTaskConfig,
    ExperimentConfig,
    LoopPolicyName,
)
from open_wam.data.artifacts import DatasetArtifactStatus
from open_wam.data.registries import preflight_dataset_artifacts
from open_wam.pipelines import build_variant_pipeline_from_config

from .auxiliary_validation import (
    AuxiliaryValidationDataset,
    AuxiliaryValidationRun,
    _auxiliary_validation_summary_metrics,
    _resolve_auxiliary_validation_source,
    build_auxiliary_validation_runs,
)
from .checkpoints import CheckpointManager
from .controls import TrainabilityReport, apply_training_component_controls
from .data_loading import (
    _uses_dynamics_routing,
    _validate_dynamics_source_sampling,
    build_runtime_dataloaders,
)
from .launch import DistributedLaunchContext, validate_training_launch
from .logging import (
    CompositeLogSink,
    ConsoleLogSink,
    JsonlLogSink,
    NoopLogSink,
    WandBLogSink,
    build_log_sink,
)
from .loop_policies import EpochLoopPolicy, StepLoopPolicy
from .optim import (
    _is_floating_dtype,
    _normalize_optimizer_state_dtypes,
    _optimizer_state_target_dtype,
    build_optimizer,
    build_scheduler,
)
from .state import TrainState
from .step_executor import PipelineTrainStepExecutor, build_batch_adapter
from .strategies import build_training_strategy

# Keep historical runtime-module lookups stable while canonical owners remain
# role-specific. These names are compatibility aliases, not extension points.
_RUNTIME_COMPATIBILITY_EXPORTS = (
    AuxiliaryValidationDataset,
    ConsoleLogSink,
    JsonlLogSink,
    NoopLogSink,
    WandBLogSink,
    _is_floating_dtype,
    _optimizer_state_target_dtype,
    _resolve_auxiliary_validation_source,
    _uses_dynamics_routing,
    _validate_dynamics_source_sampling,
)


def _local_tensor_view(tensor: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None
    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


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
        dataset_artifacts: tuple[DatasetArtifactStatus, ...] = (),
        auxiliary_validation_runs: tuple[AuxiliaryValidationRun, ...] = (),
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
        self.dataset_artifacts = dataset_artifacts
        self.auxiliary_validation_runs = auxiliary_validation_runs
        self._last_validation_optimizer_step: int | None = None
        self._accumulated_train_metrics: dict[str, list[torch.Tensor]] = {}

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        *,
        launch_context: DistributedLaunchContext | None = None,
    ) -> TrainingRuntime:
        resolved_launch_context = launch_context or DistributedLaunchContext.from_env()
        validate_training_launch(config.trainer, resolved_launch_context)
        dataset_artifacts = preflight_dataset_artifacts(config.data)
        strategy = build_training_strategy(
            config.trainer,
            launch_context=resolved_launch_context,
        )
        model = build_variant_pipeline_from_config(config)
        visual_tower = getattr(model, "visual_tower", None)
        policy_variant = getattr(model, "policy_variant", None)
        action_dim = getattr(visual_tower, "action_dim", None)
        if visual_tower is not None and action_dim is not None:
            # Initialize reference runtime weights before FSDP/DDP wrapping so
            # shared-core state dict keys stay in the replica module namespace.
            visual_tower.get_runtime_backbone(action_dim=action_dim)
        if visual_tower is not None and policy_variant is not None:
            # Variant-owned warm starts must happen before strategy wrapping so
            # replicated modules all inherit the same initialized weights.
            policy_variant.initialize_for_training(visual_tower)
        trainability_report = apply_training_component_controls(model, config.training)
        model = strategy.prepare_model(model)
        batch_adapter = build_batch_adapter(config.trainer.batch_adapter)
        step_executor = PipelineTrainStepExecutor(
            pipeline=model,
            batch_adapter=batch_adapter,
            training_config=config.training,
        )
        train_loader, val_loader = build_runtime_dataloaders(config, strategy)
        auxiliary_validation_runs = build_auxiliary_validation_runs(
            config,
            strategy,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        optimizer = build_optimizer(model, config.training)
        scheduler = build_scheduler(optimizer, config.training)
        output_dir = resolve_runtime_output_dir(config)
        checkpoint_root = Path(config.trainer.checkpoint_dir) if config.trainer.checkpoint_dir else output_dir / "checkpoints"
        checkpoint_manager = CheckpointManager(
            root_dir=checkpoint_root,
            config=config,
            checkpoint_mode=config.trainer.checkpoint_mode,
            max_checkpoints_to_keep=config.trainer.max_checkpoints_to_keep,
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
            dataset_artifacts=dataset_artifacts,
            auxiliary_validation_runs=auxiliary_validation_runs,
        )
        if config.trainer.resume_from is not None:
            runtime.resume(config.trainer.resume_from)
        return runtime

    def resume(self, checkpoint_path: str) -> None:
        current_run_name = self.train_state.run_name
        train_state, payload = self.checkpoint_manager.load(
            path=checkpoint_path,
            model=self.strategy.unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location="cpu",
        )
        _normalize_optimizer_state_dtypes(self.optimizer)
        if train_state.run_name is None:
            train_state.run_name = current_run_name
        self.train_state = train_state
        self.strategy.load_state_dict(payload.get("strategy_state_dict") if isinstance(payload, dict) else None)
        self.log_sink.log_event(
            name="resume",
            payload={
                "checkpoint_path": checkpoint_path,
                "resolved_checkpoint_path": self.train_state.resume_source,
                "optimizer_step": self.train_state.optimizer_step,
            },
        )

    def run(self) -> TrainState:
        train_video_condition_source = getattr(self.config.policy_variant, "train_video_condition_source", None)
        self.log_sink.log_event(
            name="run_start",
            payload={
                "run_name": self.train_state.run_name,
                "runtime": self.config.trainer.runtime,
                "batch_adapter": self.config.trainer.batch_adapter,
                "loop_policy": self.config.trainer.loop_policy,
                "strategy": self.config.trainer.strategy,
                "launch": self.strategy.launch_context.to_dict(),
                "dataset_artifacts": [
                    status.to_dict() for status in self.dataset_artifacts
                ],
                "output_dir": str(resolve_runtime_output_dir(self.config)),
                "enabled_objectives": self.trainability_report.enabled_objectives,
                "trainable_components": self.trainability_report.trainable_components,
                "frozen_components": self.trainability_report.frozen_components,
                "train_video_condition_source": train_video_condition_source,
                "validation_interval": self.config.trainer.validation_interval,
                "auxiliary_validation_tasks": [
                    {
                        "name": run.config.name,
                        "phase": run.config.phase,
                        "dataset_split": run.config.dataset_split.value,
                        "source": run.config.source.value,
                        "resolved_source": run.resolved_source,
                        "mode_override": (
                            None if run.config.mode_override is None else run.config.mode_override.value
                        ),
                        "max_batches": run.config.max_batches,
                    }
                    for run in self.auxiliary_validation_runs
                ],
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
            _set_sampler_epoch(self.train_loader, self.train_state.epoch_index)
            resume_batch_idx = self._current_epoch_resume_batch_index()
            if resume_batch_idx > 0 and self.strategy.is_main_process:
                self.log_sink.log_event(
                    name="resume_epoch_cursor",
                    payload={
                        "epoch_index": self.train_state.epoch_index,
                        "skip_batches": resume_batch_idx,
                        "seen_batches": self.train_state.seen_batches,
                    },
                )
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx < resume_batch_idx:
                    continue
                if policy.limit_train_batches is not None and batch_idx >= policy.limit_train_batches:
                    break
                previous_optimizer_step = self.train_state.optimizer_step
                self._train_micro_step(batch)
                if self._should_run_validation_interval(previous_optimizer_step=previous_optimizer_step):
                    self._run_all_validation(limit_batches=policy.limit_val_batches)
                if self.train_state.optimizer_step != previous_optimizer_step and self._should_save_checkpoint():
                    self._save_checkpoint(final=False)
            self._run_all_validation(limit_batches=policy.limit_val_batches)
            self.train_state.epoch_index += 1
        self._save_checkpoint(final=True)

    def _run_step_loop(self, policy: StepLoopPolicy) -> None:
        while policy.should_continue(self.train_state):
            _set_sampler_epoch(self.train_loader, self.train_state.epoch_index)
            resume_batch_idx = self._current_epoch_resume_batch_index()
            if resume_batch_idx > 0 and self.strategy.is_main_process:
                self.log_sink.log_event(
                    name="resume_step_loop_cursor",
                    payload={
                        "epoch_index": self.train_state.epoch_index,
                        "skip_batches": resume_batch_idx,
                        "seen_batches": self.train_state.seen_batches,
                    },
                )
            saw_batch = False
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx < resume_batch_idx:
                    continue
                if policy.limit_train_batches is not None and batch_idx >= policy.limit_train_batches:
                    break
                saw_batch = True
                previous_optimizer_step = self.train_state.optimizer_step
                self._train_micro_step(batch)
                if self._should_run_validation_interval(previous_optimizer_step=previous_optimizer_step):
                    self._run_all_validation(limit_batches=policy.limit_val_batches)
                if self.train_state.optimizer_step != previous_optimizer_step and self._should_save_checkpoint():
                    self._save_checkpoint(final=False)
                if not policy.should_continue(self.train_state):
                    break
            if not saw_batch:
                raise ValueError("Step-loop training received no batches from the train dataloader.")
            self.train_state.epoch_index += 1
        self._run_all_validation(limit_batches=policy.limit_val_batches)
        self._save_checkpoint(final=True)

    def _current_epoch_resume_batch_index(self) -> int:
        if self.train_state.resume_source is None or self.train_state.seen_batches <= 0:
            return 0
        try:
            epoch_batches = len(self.train_loader)
        except TypeError:
            return 0
        if epoch_batches <= 0:
            return 0
        if self.config.trainer.limit_train_batches is not None:
            epoch_batches = min(epoch_batches, int(self.config.trainer.limit_train_batches))
        if epoch_batches <= 0:
            return 0
        return int(self.train_state.seen_batches % epoch_batches)

    def _train_micro_step(self, batch) -> None:
        device_batch = self.step_executor.batch_adapter.move_to_device(batch, self.strategy.device)
        self.model.train()
        gradient_accumulation_steps = max(1, self.config.training.gradient_accumulation_steps)
        should_update = (self.train_state.global_step + 1) % gradient_accumulation_steps == 0
        self.strategy.set_gradient_sync(self.model, enabled=should_update)
        with self.strategy.autocast_context():
            result = self.step_executor.forward_train(device_batch)
            loss = result.loss / gradient_accumulation_steps
        self.strategy.backward(loss)
        self.train_state.global_step += 1
        self.train_state.seen_batches += 1
        self._accumulate_train_metrics(result.metrics)

        if not should_update:
            return

        self.strategy.unscale_(self.optimizer)
        if self.config.training.max_grad_norm is not None:
            grad_norm = self.strategy.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        else:
            grad_norm = None
        if grad_norm is not None and not torch.isfinite(grad_norm):
            self._report_nonfinite_gradients()
            raise RuntimeError(f"Non-finite gradient norm detected before optimizer step: {grad_norm.item()}.")
        _normalize_optimizer_state_dtypes(self.optimizer)
        self.strategy.optimizer_step(self.optimizer)
        self.scheduler.step()
        self.strategy.zero_grad(self.optimizer)
        self.strategy.set_gradient_sync(self.model, enabled=True)
        self.train_state.optimizer_step += 1

        metric_payload = self._finalize_accumulated_train_metrics()
        if "latent_mse" in metric_payload:
            metric_payload["latent_loss"] = metric_payload["latent_mse"]
        if "action_mse" in metric_payload:
            metric_payload["action_loss"] = metric_payload["action_mse"]
        metric_payload["lr"] = float(self.scheduler.get_last_lr()[0])
        if grad_norm is not None:
            metric_payload["grad_norm"] = float(grad_norm.item())
        if (
            self.config.trainer.log_every_n_steps <= 1
            or self.train_state.optimizer_step % self.config.trainer.log_every_n_steps == 0
        ):
            self.log_sink.log_metrics(step=self.train_state.optimizer_step, phase="train", metrics=metric_payload)

    def _report_nonfinite_gradients(self, *, limit: int = 20) -> None:
        diagnostics: list[dict[str, object]] = []
        for name, param in self.model.named_parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            local_grad = _local_tensor_view(grad)
            finite = torch.isfinite(local_grad)
            if bool(finite.all().item()):
                continue
            nonfinite_count = int((~finite).sum().item())
            finite_abs = local_grad.detach().float().abs().masked_fill(~finite, 0.0)
            diagnostics.append(
                {
                    "rank": int(getattr(self.strategy, "rank", 0)),
                    "name": name,
                    "shape": tuple(int(value) for value in local_grad.shape),
                    "nonfinite_count": nonfinite_count,
                    "max_finite_abs": float(finite_abs.max().item()) if finite_abs.numel() else 0.0,
                }
            )
            if len(diagnostics) >= limit:
                break
        if self.strategy.is_main_process:
            self.log_sink.log_event(
                name="nonfinite_gradients",
                payload={"diagnostics": diagnostics, "limit": int(limit)},
            )
        if os.getenv("OPEN_WAM_DEBUG_NONFINITE_GRADS", "0") == "1":
            for item in diagnostics:
                print(f"[open_wam][nonfinite_grad] {item}", flush=True)

    def _run_all_validation(self, *, limit_batches: int | None) -> None:
        current_step = int(self.train_state.optimizer_step)
        if getattr(self, "_last_validation_optimizer_step", None) == current_step:
            return
        ran_any = bool(self._run_validation(limit_batches=limit_batches))
        for run in getattr(self, "auxiliary_validation_runs", ()):
            ran = self._run_validation(
                loader=run.loader,
                phase=run.config.phase,
                limit_batches=run.config.max_batches,
                task=run.config,
            )
            ran_any = bool(ran) or ran_any
        if ran_any:
            self._last_validation_optimizer_step = current_step

    def _run_validation(
        self,
        *,
        loader: DataLoader | None = None,
        phase: str = "val",
        limit_batches: int | None,
        task: AuxiliaryValidationTaskConfig | None = None,
    ) -> bool:
        if limit_batches is not None and int(limit_batches) <= 0:
            return False
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        metric_totals: dict[str, float] = {}
        batch_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                device_batch = self.step_executor.batch_adapter.move_to_device(batch, self.strategy.device)
                with self.strategy.autocast_context():
                    result = self.step_executor.forward_train(device_batch)
                for name, value in result.metrics.items():
                    metric_totals[name] = metric_totals.get(name, 0.0) + float(value.item())
                batch_count += 1
        global_batch_count = float(
            self._distributed_sum(torch.tensor(float(batch_count), device=self.strategy.device)).item()
        )
        if global_batch_count <= 0.0:
            return False
        averaged = {
            name: float(self._distributed_sum(torch.tensor(value, device=self.strategy.device)).item())
            / global_batch_count
            for name, value in metric_totals.items()
        }
        if task is not None:
            averaged.update(
                _auxiliary_validation_summary_metrics(
                    task=task,
                    metrics=averaged,
                    batch_count=global_batch_count,
                )
            )
        self.log_sink.log_metrics(step=self.train_state.optimizer_step, phase=phase, metrics=averaged)
        return True

    def _should_run_validation_interval(self, *, previous_optimizer_step: int) -> bool:
        trainer_config = getattr(getattr(self, "config", None), "trainer", None)
        interval = getattr(trainer_config, "validation_interval", None)
        if interval is None or interval <= 0:
            return False
        current_step = int(self.train_state.optimizer_step)
        if current_step <= 0 or current_step == int(previous_optimizer_step):
            return False
        if current_step % int(interval) != 0:
            return False
        return getattr(self, "_last_validation_optimizer_step", None) != current_step

    def _should_save_checkpoint(self) -> bool:
        trainer_config = getattr(getattr(self, "config", None), "trainer", None)
        save_interval = getattr(trainer_config, "save_interval", None)
        if save_interval is None or save_interval <= 0:
            return False
        return self.train_state.optimizer_step > 0 and self.train_state.optimizer_step % save_interval == 0

    def _save_checkpoint(self, *, final: bool) -> None:
        should_write = (
            self.config.trainer.enable_checkpointing
            or (self.config.trainer.save_interval is not None and self.config.trainer.save_interval > 0)
        )
        if not should_write:
            return
        checkpoint_dir = self.checkpoint_manager.checkpoint_dir_for_step(self.train_state.optimizer_step)
        if final and self.train_state.last_checkpoint_path == str(checkpoint_dir):
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
        if self.strategy.is_main_process:
            self.log_sink.log_event(
                name="checkpoint_saved",
                payload={"path": str(checkpoint_dir), "final": final, "optimizer_step": self.train_state.optimizer_step},
            )
        self.strategy.barrier()

    def _accumulate_train_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        gradient_accumulation_steps = max(1, self.config.training.gradient_accumulation_steps)
        for name, value in metrics.items():
            scaled_value = value.detach() / gradient_accumulation_steps
            self._accumulated_train_metrics.setdefault(name, []).append(scaled_value)

    def _finalize_accumulated_train_metrics(self) -> dict[str, float]:
        finalized: dict[str, float] = {}
        for name, values in self._accumulated_train_metrics.items():
            if not values:
                continue
            accumulated = torch.stack(values).sum()
            finalized[name] = float(self._distributed_mean(accumulated).item())
            finalized[f"max_{name}"] = float(self._distributed_max(accumulated).item())
        self._accumulated_train_metrics = {}
        return finalized

    def _distributed_mean(self, value: torch.Tensor) -> torch.Tensor:
        reduced = value.detach().float().clone()
        if dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced = reduced / float(dist.get_world_size())
        return reduced

    def _distributed_sum(self, value: torch.Tensor) -> torch.Tensor:
        reduced = value.detach().float().clone()
        if dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

    def _distributed_max(self, value: torch.Tensor) -> torch.Tensor:
        reduced = value.detach().float().clone()
        if dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
        return reduced


def _set_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    set_epoch = getattr(getattr(loader, "sampler", None), "set_epoch", None)
    if callable(set_epoch):
        set_epoch(int(epoch))


def resolve_runtime_output_dir(config: ExperimentConfig) -> Path:
    root = Path(config.trainer.default_root_dir) if config.trainer.default_root_dir else Path("runs")
    return root / (config.trainer.run_name or config.name)
