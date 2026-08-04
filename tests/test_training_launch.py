from __future__ import annotations

import pytest

from open_wam.configs import StrategyName, TrainerConfig
from open_wam.training.launch import (
    DistributedLaunchContext,
    LaunchEnvironment,
    validate_training_launch,
)


def test_launch_context_defaults_to_single_process() -> None:
    context = DistributedLaunchContext.from_env({})

    assert context.to_dict() == {
        "environment": "single_process",
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "local_world_size": 1,
    }
    assert context.environment is LaunchEnvironment.SINGLE_PROCESS
    assert context.distributed is False


def test_launch_context_parses_standard_torch_coordinates() -> None:
    context = DistributedLaunchContext.from_env(
        {
            "RANK": "5",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
        }
    )

    assert context.rank == 5
    assert context.local_rank == 1
    assert context.world_size == 8
    assert context.local_world_size == 4
    assert context.distributed is True


@pytest.mark.parametrize(
    "env",
    [
        {"WORLD_SIZE": "4"},
        {"RANK": "0", "WORLD_SIZE": "4"},
        {"RANK": "0", "LOCAL_RANK": "0"},
    ],
)
def test_launch_context_rejects_partial_coordinates(env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="Incomplete Torch distributed launch"):
        DistributedLaunchContext.from_env(env)


def test_launch_context_rejects_orphaned_local_world_size() -> None:
    with pytest.raises(ValueError, match="LOCAL_WORLD_SIZE requires"):
        DistributedLaunchContext.from_env({"LOCAL_WORLD_SIZE": "4"})


def test_launch_preflight_rejects_unlaunched_multi_process_intent() -> None:
    with pytest.raises(ValueError, match="expected 4 process"):
        validate_training_launch(
            TrainerConfig(strategy=StrategyName.FSDP),
            DistributedLaunchContext.from_env({}),
            expected_world_size=4,
        )


def test_launch_preflight_honors_legacy_devices_above_one() -> None:
    with pytest.raises(ValueError, match="expected 4 process"):
        validate_training_launch(
            TrainerConfig(strategy=StrategyName.FSDP, devices=4),
            DistributedLaunchContext.from_env({}),
        )


def test_launch_preflight_rejects_duplicate_single_device_workers() -> None:
    context = DistributedLaunchContext.from_env(
        {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}
    )

    with pytest.raises(ValueError, match="strategy=single_device"):
        validate_training_launch(TrainerConfig(), context)


def test_launch_preflight_accepts_matching_distributed_strategy() -> None:
    context = DistributedLaunchContext.from_env(
        {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
    )

    validate_training_launch(
        TrainerConfig(strategy=StrategyName.FSDP),
        context,
        expected_world_size=4,
    )


def test_launch_preflight_rejects_conflicting_configured_expectation() -> None:
    context = DistributedLaunchContext.from_env(
        {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}
    )

    with pytest.raises(ValueError, match="Conflicting launch expectations"):
        validate_training_launch(
            TrainerConfig(strategy=StrategyName.FSDP, devices=4),
            context,
            expected_world_size=2,
        )


@pytest.mark.parametrize("devices", [True, 0, -1])
def test_trainer_rejects_invalid_device_expectation(devices: object) -> None:
    with pytest.raises(ValueError, match=r"trainer\.devices"):
        TrainerConfig(devices=devices)  # type: ignore[arg-type]
