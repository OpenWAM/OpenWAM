from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from open_wam.configs import StrategyName, TrainerConfig


class LaunchEnvironment(str, Enum):
    """Process environment used to start one training worker."""

    SINGLE_PROCESS = "single_process"
    TORCH_DISTRIBUTED = "torch_distributed"


@dataclass(frozen=True)
class DistributedLaunchContext:
    """Validated process coordinates supplied by an external launcher.

    Open-WAM deliberately does not spawn workers inside the training runtime.
    Launchers such as ``torchrun`` own process creation and communicate the
    resulting topology through the standard Torch distributed environment.
    """

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int | None
    environment: LaunchEnvironment

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> DistributedLaunchContext:
        values = os.environ if env is None else env
        coordinate_names = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        present = tuple(
            name for name in coordinate_names if values.get(name) not in (None, "")
        )
        if not present:
            if values.get("LOCAL_WORLD_SIZE") not in (None, ""):
                raise ValueError(
                    "Incomplete Torch distributed launch environment. "
                    "LOCAL_WORLD_SIZE requires RANK, LOCAL_RANK, and WORLD_SIZE."
                )
            return cls(
                rank=0,
                local_rank=0,
                world_size=1,
                local_world_size=1,
                environment=LaunchEnvironment.SINGLE_PROCESS,
            )
        if len(present) != len(coordinate_names):
            missing = ", ".join(
                name for name in coordinate_names if name not in present
            )
            raise ValueError(
                "Incomplete Torch distributed launch environment. "
                f"Missing {missing}; provide RANK, LOCAL_RANK, and WORLD_SIZE "
                "together. "
                "Use torchrun or map the scheduler's rank variables explicitly."
            )

        rank = _parse_non_negative_int(values, "RANK")
        local_rank = _parse_non_negative_int(values, "LOCAL_RANK")
        world_size = _parse_positive_int(values, "WORLD_SIZE")
        local_world_size = (
            _parse_positive_int(values, "LOCAL_WORLD_SIZE")
            if values.get("LOCAL_WORLD_SIZE") not in (None, "")
            else None
        )
        if rank >= world_size:
            raise ValueError(
                f"Invalid distributed coordinates: RANK={rank} must be smaller than "
                f"WORLD_SIZE={world_size}."
            )
        if local_world_size is not None and local_rank >= local_world_size:
            raise ValueError(
                "Invalid distributed coordinates: "
                f"LOCAL_RANK={local_rank} must be smaller than "
                f"LOCAL_WORLD_SIZE={local_world_size}."
            )
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            environment=LaunchEnvironment.TORCH_DISTRIBUTED,
        )

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "environment": self.environment.value,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
        }


def validate_training_launch(
    trainer_config: TrainerConfig,
    launch_context: DistributedLaunchContext,
    *,
    expected_world_size: int | None = None,
) -> None:
    """Fail before model allocation when launch topology and config disagree."""

    configured_expectation = (
        int(trainer_config.devices) if int(trainer_config.devices) > 1 else None
    )
    expected = expected_world_size
    if (
        expected is not None
        and configured_expectation is not None
        and int(expected) != configured_expectation
    ):
        raise ValueError(
            "Conflicting launch expectations: "
            f"trainer.devices={configured_expectation} "
            f"but expected_world_size={expected}."
        )
    if expected is None and configured_expectation is not None:
        # Values above one historically expressed launch intent. Enforce that
        # intent instead of silently allocating the complete model on one worker.
        expected = configured_expectation
    if expected is not None:
        if isinstance(expected, bool) or int(expected) <= 0:
            raise ValueError("Expected world size must be a positive integer.")
        expected = int(expected)
        if launch_context.world_size != expected:
            raise ValueError(
                f"Training expected {expected} process(es), but the launcher provided "
                f"WORLD_SIZE={launch_context.world_size}. Open-WAM does not spawn "
                "workers "
                f"from config; launch with `torchrun --nproc-per-node={expected} ...` "
                "or correct the expectation."
            )

    if (
        trainer_config.strategy is StrategyName.SINGLE_DEVICE
        and launch_context.distributed
    ):
        raise ValueError(
            "A multi-process launch cannot use `trainer.strategy=single_device`; "
            "every rank would train an independent model. Select `ddp` or `fsdp`, "
            "or launch one process."
        )


def _parse_non_negative_int(env: Mapping[str, str], name: str) -> int:
    value = _parse_int(env, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _parse_positive_int(env: Mapping[str, str], name: str) -> int:
    value = _parse_int(env, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _parse_int(env: Mapping[str, str], name: str) -> int:
    raw = env.get(name)
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


__all__ = [
    "DistributedLaunchContext",
    "LaunchEnvironment",
    "validate_training_launch",
]
