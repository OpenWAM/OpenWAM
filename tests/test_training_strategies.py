from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from open_wam.configs import StrategyName, TrainerAccelerator, TrainerConfig
from open_wam.training.strategies import (
    _apply_composable_fsdp_sharding,
    build_training_strategy,
)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = torch.nn.Linear(2, 2)
        self.attn2 = torch.nn.Linear(2, 2)
        self.ffn = torch.nn.Linear(2, 2)


class _Pipeline(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root_projection = torch.nn.Linear(2, 2)
        self.visual_tower = torch.nn.Module()
        self.visual_tower.core = torch.nn.Module()
        self.visual_tower.core.blocks = torch.nn.ModuleList([_Block()])
        self.policy_variant = torch.nn.Module()
        self.policy_variant.action_expert = torch.nn.Module()
        self.policy_variant.action_expert.blocks = torch.nn.ModuleList([_Block()])
        self.policy_variant.packed_block_stack = None


def test_composable_fsdp_shards_nested_blocks_then_pipeline_root(
    monkeypatch,
) -> None:
    model = _Pipeline()
    calls: list[tuple[torch.nn.Module, dict[str, object]]] = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", fake_fully_shard)

    result = _apply_composable_fsdp_sharding(
        model,
        mesh=SimpleNamespace(),
        mp_policy=SimpleNamespace(),
    )

    assert result is model
    assert calls[-1][0] is model
    assert [module for module, _ in calls[:-1]] == [
        model.visual_tower.core.blocks[0].attn1,
        model.visual_tower.core.blocks[0].attn2,
        model.visual_tower.core.blocks[0].ffn,
        model.visual_tower.core.blocks[0],
        model.policy_variant.action_expert.blocks[0].attn1,
        model.policy_variant.action_expert.blocks[0].attn2,
        model.policy_variant.action_expert.blocks[0].ffn,
        model.policy_variant.action_expert.blocks[0],
    ]
    assert all(kwargs["reshard_after_forward"] is True for _, kwargs in calls[:-1])
    assert calls[-1][1]["reshard_after_forward"] is False


def test_distributed_strategy_uses_configured_process_group_timeout(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda **kwargs: calls.append(kwargs),
    )

    strategy = build_training_strategy(
        TrainerConfig(
            accelerator=TrainerAccelerator.CPU,
            strategy=StrategyName.DDP,
            distributed_timeout_seconds=42,
        )
    )

    assert strategy.distributed_timeout_seconds == 42
    assert calls == [
        {
            "backend": "gloo",
            "rank": 0,
            "world_size": 2,
            "timeout": timedelta(seconds=42),
        }
    ]


def test_distributed_strategy_rejects_initialized_group_coordinate_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="process group disagrees"):
        build_training_strategy(
            TrainerConfig(
                accelerator=TrainerAccelerator.CPU,
                strategy=StrategyName.DDP,
            )
        )


def test_single_device_strategy_rejects_preinitialized_multi_rank_group(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="SingleDeviceStrategy"):
        build_training_strategy(TrainerConfig())


@pytest.mark.parametrize("value", [True, 0, -1])
def test_trainer_rejects_invalid_distributed_timeout(value: object) -> None:
    with pytest.raises(ValueError, match="distributed_timeout_seconds"):
        TrainerConfig(distributed_timeout_seconds=value)
