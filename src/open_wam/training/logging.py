from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Protocol

from open_wam.configs import ExperimentConfig
from open_wam.configs.enums import serialize_enum_values

from .run_tracking import (
    build_run_title,
    build_run_tracking_metadata,
    build_wandb_group,
    build_wandb_job_type,
    build_wandb_tags,
    resolve_wandb_project,
)


class LogSink(Protocol):
    def log_metrics(self, *, step: int, phase: str, metrics: dict[str, float]) -> None: ...
    def log_event(self, *, name: str, payload: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class CompositeLogSink:
    """Broadcast logs to a list of sinks."""

    def __init__(self, sinks: list[LogSink]) -> None:
        self.sinks = sinks

    def log_metrics(self, *, step: int, phase: str, metrics: dict[str, float]) -> None:
        for sink in self.sinks:
            sink.log_metrics(step=step, phase=phase, metrics=metrics)

    def log_event(self, *, name: str, payload: dict[str, Any]) -> None:
        for sink in self.sinks:
            sink.log_event(name=name, payload=payload)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


class NoopLogSink:
    """Drop all metrics and events."""

    def log_metrics(self, *, step: int, phase: str, metrics: dict[str, float]) -> None:
        return None

    def log_event(self, *, name: str, payload: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class ConsoleLogSink:
    """Small stdout logger for training progress."""

    def log_metrics(self, *, step: int, phase: str, metrics: dict[str, float]) -> None:
        metric_blob = " ".join(f"{key}={value:.6f}" for key, value in sorted(metrics.items()))
        print(f"[{phase}] step={step} {metric_blob}")

    def log_event(self, *, name: str, payload: dict[str, Any]) -> None:
        print(f"[event] {name}: {json.dumps(payload, sort_keys=True, default=str)}")

    def close(self) -> None:
        return None


class JsonlLogSink:
    """Append metrics and events to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log_metrics(self, *, step: int, phase: str, metrics: dict[str, float]) -> None:
        self._handle.write(json.dumps({"type": "metrics", "step": step, "phase": phase, "metrics": metrics}) + "\n")
        self._handle.flush()

    def log_event(self, *, name: str, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps({"type": "event", "name": name, "payload": payload}, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class WandBLogSink:
    """Optional WandB sink created only when WandB is enabled in config."""

    def __init__(
        self,
        *,
        project: str | None,
        entity: str | None,
        mode: str,
        run_name: str,
        group: str | None,
        job_type: str | None,
        tags: tuple[str, ...] | list[str],
        config_payload: dict[str, Any],
    ) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("WandB logging was requested but the `wandb` package is not installed.") from exc
        self._wandb = wandb
        self._use_contiguous_wandb_steps = os.environ.get(
            "OPEN_WAM_WANDB_CONTIGUOUS_STEPS",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self._run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            name=run_name,
            group=group,
            job_type=job_type,
            tags=list(tags),
            config=config_payload,
        )
        if self._use_contiguous_wandb_steps:
            self._wandb.define_metric("trainer/global_step")
            self._wandb.define_metric("*", step_metric="trainer/global_step")

    def log_metrics(self, *, step: int, phase: str, metrics: dict[str, float]) -> None:
        payload = {f"{phase}/{key}": value for key, value in metrics.items()}
        if self._use_contiguous_wandb_steps:
            payload["trainer/global_step"] = step
            payload[f"{phase}/global_step"] = step
            self._wandb.log(payload)
            return
        self._wandb.log(payload, step=step)

    def log_event(self, *, name: str, payload: dict[str, Any]) -> None:
        self._wandb.log({f"event/{name}": payload})

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


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
