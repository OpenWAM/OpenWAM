from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


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

    def log_metrics(self, *, step: int, phase: str, metrics: dict[str, float]) -> None:
        self._wandb.log({f"{phase}/{key}": value for key, value in metrics.items()}, step=step)

    def log_event(self, *, name: str, payload: dict[str, Any]) -> None:
        self._wandb.log({f"event/{name}": payload})

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
