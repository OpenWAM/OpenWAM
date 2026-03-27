from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainState:
    """Mutable runtime state shared across training components."""

    global_step: int = 0
    optimizer_step: int = 0
    epoch_index: int = 0
    seen_batches: int = 0
    run_name: str | None = None
    resume_source: str | None = None
    last_checkpoint_path: str | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "optimizer_step": self.optimizer_step,
            "epoch_index": self.epoch_index,
            "seen_batches": self.seen_batches,
            "run_name": self.run_name,
            "resume_source": self.resume_source,
            "last_checkpoint_path": self.last_checkpoint_path,
            "best_metrics": dict(self.best_metrics),
        }

    @classmethod
    def from_state_dict(cls, raw: dict[str, Any] | None) -> "TrainState":
        payload = raw or {}
        return cls(
            global_step=int(payload.get("global_step", 0)),
            optimizer_step=int(payload.get("optimizer_step", 0)),
            epoch_index=int(payload.get("epoch_index", 0)),
            seen_batches=int(payload.get("seen_batches", 0)),
            run_name=payload.get("run_name"),
            resume_source=payload.get("resume_source"),
            last_checkpoint_path=payload.get("last_checkpoint_path"),
            best_metrics={str(key): float(value) for key, value in dict(payload.get("best_metrics", {})).items()},
        )
