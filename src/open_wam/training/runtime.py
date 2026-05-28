from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from open_wam.configs import (
    AuxiliaryValidationTaskConfig,
    BatchAdapterName,
    ExperimentConfig,
    LoopPolicyName,
    StrategyName,
    TrainerRuntimeName,
)
from open_wam.configs.enums import (
    AuxiliaryValidationSource,
    DataSplit,
    GeneralistTrainingParadigm,
    SampleOrderMode,
    SampleWeightMode,
    serialize_enum_values,
)
from open_wam.configs.variant_semantics import (
    GENERALIST_TRAINING_BUCKET_METADATA_KEY,
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
)
from open_wam.data import (
    build_generalist_dynamics_mixture_datasets,
    build_train_val_datasets,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    collate_wam_samples,
    resolve_dataset_loader_spec,
)
from open_wam.pipelines import build_variant_pipeline_from_config

from .checkpoints import CheckpointManager
from .controls import TrainabilityReport, apply_training_component_controls
from .logging import CompositeLogSink, ConsoleLogSink, JsonlLogSink, NoopLogSink, WandBLogSink
from .loop_policies import EpochLoopPolicy, StepLoopPolicy
from .optim import build_optimizer, build_scheduler
from .run_tracking import (
    build_run_title,
    build_run_tracking_metadata,
    build_wandb_group,
    build_wandb_job_type,
    build_wandb_tags,
    resolve_wandb_project,
)
from .state import TrainState
from .step_executor import PipelineTrainStepExecutor, build_batch_adapter
from .strategies import build_training_strategy


@dataclass(frozen=True)
class AuxiliaryValidationRun:
    """Runtime-ready auxiliary validation task."""

    config: AuxiliaryValidationTaskConfig
    loader: DataLoader
    resolved_source: str


def _is_floating_dtype(dtype: torch.dtype | None) -> bool:
    if dtype is None:
        return False
    return torch.empty((), dtype=dtype).is_floating_point()


def _optimizer_state_target_dtype(parameter: object) -> torch.dtype | None:
    grad = getattr(parameter, "grad", None)
    grad_dtype = getattr(grad, "dtype", None)
    if _is_floating_dtype(grad_dtype):
        return grad_dtype
    parameter_dtype = getattr(parameter, "dtype", None)
    if _is_floating_dtype(parameter_dtype):
        return parameter_dtype
    return None


def _normalize_optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        if not isinstance(state, dict):
            continue
        state_dtype = _optimizer_state_target_dtype(parameter)
        if state_dtype is None:
            continue
        for key, value in list(state.items()):
            if key == "step":
                continue
            if torch.is_tensor(value) and torch.is_floating_point(value) and value.dtype != state_dtype:
                state[key] = value.to(dtype=state_dtype)


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
        self.auxiliary_validation_runs = auxiliary_validation_runs
        self._last_validation_optimizer_step: int | None = None
        self._accumulated_train_metrics: dict[str, list[torch.Tensor]] = {}

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "TrainingRuntime":
        strategy = build_training_strategy(config.trainer)
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


def build_runtime_dataloaders(config: ExperimentConfig, strategy) -> tuple[DataLoader, DataLoader]:
    if _uses_mixed_dynamics_paradigm(config):
        _validate_mixed_dynamics_source_sampling(config)
    if config.trainer.batch_adapter == BatchAdapterName.LATENTS:
        train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
        if _uses_mixed_dynamics_paradigm(config):
            if config.data.train_batch_size != 1 or config.data.val_batch_size != 1:
                raise ValueError(
                    "`generalist_training_paradigm = mixed_dynamics` currently requires "
                    "`data.train_batch_size = data.val_batch_size = 1` because mixed samples may have "
                    "different temporal lengths and GJD runtimes use one forced mode per segment."
                )
            train_dataset, val_dataset = build_generalist_dynamics_mixture_datasets(
                data_config=config.data,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
        train_loader_spec = resolve_dataset_loader_spec(
            train_dataset,
            split="train",
            world_size=strategy.world_size,
            rank=strategy.rank,
        )
        train_sampler = train_loader_spec.sampler
        if train_sampler is None and strategy.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                shuffle=True,
                num_replicas=strategy.world_size,
                rank=strategy.rank,
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
                shuffle=train_sampler is None and train_loader_spec.shuffle,
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
    train_loader_spec = resolve_dataset_loader_spec(
        train_dataset,
        split="train",
        world_size=strategy.world_size,
        rank=strategy.rank,
    )
    val_loader_spec = resolve_dataset_loader_spec(
        val_dataset,
        split="val",
        world_size=strategy.world_size,
        rank=strategy.rank,
    )
    train_sampler = train_loader_spec.sampler
    if train_sampler is None and strategy.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=strategy.world_size, rank=strategy.rank)
    val_sampler = val_loader_spec.sampler
    if val_sampler is None and strategy.distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=strategy.world_size, rank=strategy.rank)
    return (
        DataLoader(
            train_dataset,
            batch_size=config.data.train_batch_size,
            shuffle=train_sampler is None and train_loader_spec.shuffle,
            num_workers=config.data.num_workers,
            sampler=train_sampler,
            collate_fn=collate_wam_samples,
        ),
        DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            shuffle=val_loader_spec.shuffle,
            num_workers=config.data.num_workers,
            sampler=val_sampler,
            collate_fn=collate_wam_samples,
        ),
    )


class AuxiliaryValidationDataset(Dataset):
    """Apply validation-only metadata overrides without changing source datasets."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        task: AuxiliaryValidationTaskConfig,
    ) -> None:
        self.dataset = dataset
        self.task = task

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        metadata = dict(getattr(sample, "metadata", {}) or {})
        if self.task.mode_override is not None:
            metadata[GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY] = self.task.mode_override.value
            metadata[GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY] = self.task.should_drop_text
            metadata.setdefault(GENERALIST_TRAINING_SOURCE_METADATA_KEY, "auxiliary_validation")
            metadata.setdefault(GENERALIST_TRAINING_BUCKET_METADATA_KEY, self.task.name)
            metadata["generalist_validation_task"] = self.task.name
            metadata["generalist_validation_phase"] = self.task.phase
            metadata["generalist_validation_requested_source"] = self.task.source.value
        updates = {"metadata": metadata}
        if self.task.should_drop_text:
            if hasattr(sample, "task_text"):
                updates["task_text"] = None
            if hasattr(sample, "text_context"):
                text_context = getattr(sample, "text_context")
                negative_text_context = getattr(sample, "negative_text_context", None)
                if negative_text_context is not None:
                    updates["text_context"] = negative_text_context.clone()
                elif text_context is not None:
                    updates["text_context"] = torch.zeros_like(text_context)
        return replace(sample, **updates)


def build_auxiliary_validation_runs(
    config: ExperimentConfig,
    strategy,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[AuxiliaryValidationRun, ...]:
    runs: list[AuxiliaryValidationRun] = []
    seen_phases: set[str] = set()
    for task in config.validation.auxiliary_tasks:
        if not task.enabled or task.max_batches == 0:
            continue
        if task.phase in seen_phases:
            raise ValueError(f"Duplicate auxiliary validation report prefix {task.phase!r}.")
        seen_phases.add(task.phase)
        source_loader = train_loader if task.dataset_split == DataSplit.TRAIN else val_loader
        source_dataset, resolved_source = _resolve_auxiliary_validation_source(source_loader.dataset, task=task)
        dataset = AuxiliaryValidationDataset(source_dataset, task=task)
        sampler = (
            DistributedSampler(dataset, shuffle=False, num_replicas=strategy.world_size, rank=strategy.rank)
            if strategy.distributed
            else None
        )
        runs.append(
            AuxiliaryValidationRun(
                config=task,
                loader=DataLoader(
                    dataset,
                    batch_size=source_loader.batch_size,
                    shuffle=False,
                    num_workers=source_loader.num_workers,
                    sampler=sampler,
                    collate_fn=source_loader.collate_fn,
                    pin_memory=source_loader.pin_memory,
                ),
                resolved_source=resolved_source,
            )
        )
    return tuple(runs)


def _resolve_auxiliary_validation_source(
    dataset: Dataset,
    *,
    task: AuxiliaryValidationTaskConfig,
) -> tuple[Dataset, str]:
    if task.source == AuxiliaryValidationSource.DATASET:
        return dataset, AuxiliaryValidationSource.DATASET.value
    if task.source == AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS_IF_AVAILABLE:
        return _resolve_named_auxiliary_validation_source(
            dataset,
            task=task,
            source=AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS,
            fallback=(dataset, AuxiliaryValidationSource.DATASET.value),
        )
    return _resolve_named_auxiliary_validation_source(dataset, task=task, source=task.source)


def _resolve_named_auxiliary_validation_source(
    dataset: Dataset,
    *,
    task: AuxiliaryValidationTaskConfig,
    source: AuxiliaryValidationSource,
    fallback: tuple[Dataset, str] | None = None,
) -> tuple[Dataset, str]:
    build_source_view = getattr(dataset, "build_source_view", None)
    if callable(build_source_view):
        view = build_source_view(
            source=source.value,
            mode=task.mode_override.value if task.mode_override is not None else "joint",
            bucket_name=task.name,
            drop_text=task.should_drop_text,
        )
        if isinstance(view, Dataset):
            return view, source.value
    attribute_by_source = {
        AuxiliaryValidationSource.REAL_DEMO: "real_dataset",
        AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS: "counterfactual_dataset",
    }
    attribute = attribute_by_source.get(source)
    if attribute is not None and hasattr(dataset, attribute):
        resolved = getattr(dataset, attribute)
        if isinstance(resolved, Dataset):
            return resolved, source.value
    if fallback is not None:
        return fallback
    raise ValueError(
        f"Auxiliary validation task {task.name!r} requested source {task.source.value!r}, "
        f"but the selected {task.dataset_split.value!r} dataset does not expose that source."
    )


def _auxiliary_validation_summary_metrics(
    *,
    task: AuxiliaryValidationTaskConfig,
    metrics: dict[str, float],
    batch_count: float,
) -> dict[str, float]:
    summary: dict[str, float] = {"count": float(batch_count)}
    for namespace in ("joint_denoise", "mot_generalist"):
        action_active_key = f"{namespace}/action_loss_active"
        latent_active_key = f"{namespace}/latent_loss_active"
        if action_active_key in metrics:
            summary["action_loss_active"] = metrics[action_active_key]
        if latent_active_key in metrics:
            summary["latent_loss_active"] = metrics[latent_active_key]
        if task.mode_override is None:
            continue
        mode = task.mode_override.value
        mode_count_key = f"{namespace}/{mode}/count"
        if mode_count_key in metrics:
            summary["mode_fraction"] = metrics[mode_count_key]
    return summary


def _uses_mixed_dynamics_paradigm(config: ExperimentConfig) -> bool:
    paradigm = getattr(config.policy_variant, "generalist_training_paradigm", None)
    return paradigm == GeneralistTrainingParadigm.MIXED_DYNAMICS


def _validate_mixed_dynamics_source_sampling(config: ExperimentConfig) -> None:
    if config.trainer.batch_adapter != BatchAdapterName.LATENTS:
        raise ValueError(
            "`policy_variant.generalist_training_paradigm=mixed_dynamics` requires "
            "`trainer.batch_adapter=latents` because the mixed-dynamics source mixture wraps latent datasets."
        )
    sample_construction = config.data.sample_construction
    if sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT:
        raise ValueError(
            "`data.sample_construction.sample_order_mode=replacement` is not supported with "
            "`policy_variant.generalist_training_paradigm=mixed_dynamics` because the mixed-dynamics "
            "wrapper owns source sampling."
        )
    if sample_construction.sample_weight_mode != SampleWeightMode.UNIFORM:
        raise ValueError(
            "`data.sample_construction.sample_weight_mode` must be `uniform` with "
            "`policy_variant.generalist_training_paradigm=mixed_dynamics` because the mixed-dynamics "
            "wrapper owns source sampling."
        )


def build_log_sink(*, config: ExperimentConfig, output_dir: Path, run_name: str, strategy=None) -> CompositeLogSink:
    if strategy is not None and not strategy.is_main_process:
        return CompositeLogSink([NoopLogSink()])
    sinks = [ConsoleLogSink()]
    tracking_metadata = build_run_tracking_metadata(config, run_name=run_name, output_dir=output_dir)
    resolved_project = resolve_wandb_project(config, tracking_metadata)
    resolved_group = build_wandb_group(tracking_metadata)
    resolved_job_type = build_wandb_job_type(tracking_metadata)
    resolved_tags = build_wandb_tags(tracking_metadata)
    tracking_metadata["wandb_project"] = resolved_project
    tracking_metadata["wandb_group"] = resolved_group
    tracking_metadata["wandb_job_type"] = resolved_job_type
    tracking_metadata["wandb_tags"] = list(resolved_tags)
    if config.trainer.enable_jsonl_logging:
        sinks.append(JsonlLogSink(output_dir / config.trainer.metrics_filename))
    if config.trainer.enable_wandb:
        config_payload = serialize_enum_values(asdict(config))
        config_payload["tracking"] = tracking_metadata
        sinks.append(
            WandBLogSink(
                project=resolved_project,
                entity=config.trainer.wandb_entity,
                mode=config.trainer.wandb_mode,
                run_name=build_run_title(tracking_metadata),
                group=resolved_group,
                job_type=resolved_job_type,
                tags=resolved_tags,
                config_payload=config_payload,
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
