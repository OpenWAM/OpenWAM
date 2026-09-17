from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from open_wam.configs import (
    StrategyName,
    TrainerAccelerator,
    TrainerConfig,
    TrainerPrecision,
)
from open_wam.models.policy_variants import PolicyModuleTopology
from open_wam.training.strategies import (
    SingleDeviceStrategy,
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

    def module_topology(self) -> PolicyModuleTopology:
        return PolicyModuleTopology(
            visual_runtime_modules=(self.visual_tower.core,),
            action_expert_modules=(self.policy_variant.action_expert,),
            fsdp_block_stacks=(
                self.visual_tower.core,
                self.policy_variant.action_expert,
            ),
        )


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


@pytest.mark.parametrize("torch_launch", [False, True])
def test_single_rank_fsdp_keeps_sharding_and_offload(monkeypatch, torch_launch):
    from open_wam.training import strategies

    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    if torch_launch:
        for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}.items():
            monkeypatch.setenv(key, value)
    for key in ("MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: calls.append(kw))
    monkeypatch.setattr(
        strategies, "_resolve_device", lambda *a, **kw: torch.device("cuda:0")
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    mesh = object()
    monkeypatch.setattr(strategies, "init_device_mesh", lambda *a: mesh)
    monkeypatch.setenv("OPEN_WAM_FSDP_CPU_OFFLOAD", "1")
    shards = []
    monkeypatch.setattr(
        "torch.distributed.fsdp.fully_shard", lambda module, **kw: shards.append(kw)
    )
    model = _Pipeline()
    monkeypatch.setattr(model, "to", lambda **kw: model)
    strategy = build_training_strategy(
        TrainerConfig(
            accelerator=TrainerAccelerator.GPU,
            strategy=StrategyName.FSDP,
        )
    )

    assert strategy.distributed is False
    assert calls[0]["world_size"] == 1
    assert isinstance(calls[0]["store"], dist.HashStore)
    assert strategy.prepare_model(model) is model
    assert shards and all(kw["mesh"] is mesh for kw in shards)
    assert all(
        isinstance(kw["offload_policy"], torch.distributed.fsdp.CPUOffloadPolicy)
        for kw in shards
    )
    sync = []
    monkeypatch.setattr(
        strategies,
        "_set_gradient_sync_recursive",
        lambda m, enabled: sync.append(enabled),
    )
    strategy.set_gradient_sync(model, False)
    strategy.set_gradient_sync(model, True)
    assert sync == [False, True]
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    closed = []
    monkeypatch.setattr(dist, "destroy_process_group", lambda: closed.append(True))
    strategy.close()
    assert closed == [True]


def test_single_rank_ddp_remains_unwrapped(monkeypatch):
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda **kw: pytest.fail("DDP initialized a single-rank group"),
    )
    strategy = build_training_strategy(
        TrainerConfig(
            accelerator=TrainerAccelerator.CPU,
            strategy=StrategyName.DDP,
        )
    )
    model = _Pipeline()
    assert strategy.prepare_model(model) is model
    assert not strategy._owns_process_group


@pytest.mark.parametrize("borrowed_process_group", [False, True])
def test_single_rank_fsdp_accumulation_and_group_ownership(
    monkeypatch, borrowed_process_group
):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    class Projection(torch.nn.Linear):
        def module_topology(self):
            return PolicyModuleTopology(visual_runtime_modules=())

    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPEN_WAM_FSDP_CPU_OFFLOAD", "0")
    if borrowed_process_group:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    strategy = None
    try:
        strategy = build_training_strategy(
            TrainerConfig(
                accelerator=TrainerAccelerator.CPU,
                strategy=StrategyName.FSDP,
                precision=TrainerPrecision.FP32,
            )
        )
        assert strategy._owns_process_group is not borrowed_process_group
        assert not strategy.distributed
        plain = Projection(4, 4)
        sharded = strategy.prepare_model(deepcopy(plain))
        optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3)
                      for m in (plain, sharded)]
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
        for _ in range(2):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for index in range(3):
                strategy.set_gradient_sync(sharded, enabled=index == 2)
                batch = inputs + index / 10
                (plain(batch).square().mean() / 3).backward()
                strategy.backward(sharded(batch).square().mean() / 3)
            expected_norm = torch.nn.utils.clip_grad_norm_(plain.parameters(), 0.5)
            actual_norm = strategy.clip_grad_norm_(sharded.parameters(), 0.5)
            torch.testing.assert_close(actual_norm, expected_norm)
            for optimizer in optimizers:
                strategy.optimizer_step(optimizer)
            torch.testing.assert_close(
                get_model_state_dict(sharded, options=StateDictOptions(full_state_dict=True)),
                plain.state_dict(), rtol=1e-6, atol=1e-7,
            )
        strategy.barrier()
        strategy.close()
        strategy.close()
        assert dist.is_initialized() is borrowed_process_group
        if not borrowed_process_group:
            dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
            strategy.close()
            assert dist.is_initialized(), "A closed strategy must not destroy a later run's group."
    finally:
        if strategy is not None:
            strategy.close()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sharded_source", [False, True])
def test_single_rank_fsdp_offload_full_state_resume(
    tmp_path, monkeypatch, sharded_source
):
    import json
    from copy import deepcopy

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    from open_wam.configs import CheckpointMode
    from open_wam.training.state import TrainState
    from tests.test_checkpoint_protocol import _manager

    class Projection(torch.nn.Linear):
        def module_topology(self):
            return PolicyModuleTopology(visual_runtime_modules=())

    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPEN_WAM_FSDP_CPU_OFFLOAD", "1")
    strategy = build_training_strategy(
        TrainerConfig(
            accelerator=TrainerAccelerator.GPU,
            strategy=StrategyName.FSDP,
            precision=TrainerPrecision.FP32,
        )
    )
    manager = _manager(tmp_path, checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE)
    inputs = torch.ones(2, 4, device="cuda")
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)

    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()

    try:
        model = (
            strategy.prepare_model(Projection(4, 4))
            if sharded_source
            else Projection(4, 4).cuda()
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        update(model, optimizer)
        scheduler.step()
        checkpoint = manager.save(
            step=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_state=TrainState(optimizer_step=1),
        )
        saved_optimizer = deepcopy(
            get_optimizer_state_dict(model, optimizer, options=options)
        )
        update(model, optimizer)
        expected = get_model_state_dict(model, options=options)
        resumed = strategy.prepare_model(Projection(4, 4))
        restored_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        restored_scheduler = torch.optim.lr_scheduler.StepLR(
            restored_optimizer, 1, gamma=0.9
        )
        state, _ = manager.load(
            path=checkpoint,
            model=resumed,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
        )
        assert state.optimizer_step == 1
        assert restored_scheduler.state_dict() == scheduler.state_dict()
        restored_state = get_optimizer_state_dict(
            resumed, restored_optimizer, options=options
        )
        # DCP normalizes the Adam betas tuple to a list without changing values.
        assert json.dumps(restored_state["param_groups"], sort_keys=True) == json.dumps(
            saved_optimizer["param_groups"], sort_keys=True
        )
        torch.testing.assert_close(
            restored_state["state"],
            saved_optimizer["state"],
            rtol=0,
            atol=0,
        )
        assert all(p.device.type == "cpu" for p in resumed.parameters())
        assert all(
            v.device.type == "cpu"
            for s in restored_optimizer.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )
        update(resumed, restored_optimizer)
        torch.testing.assert_close(
            get_model_state_dict(resumed, options=options),
            expected,
            rtol=1e-6,
            atol=1e-7,
        )
    finally:
        strategy.close()


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


@pytest.mark.parametrize("strategy_state", [None, {}, {"grad_scaler": None}])
def test_single_device_strategy_requires_enabled_gradient_scaler_state(
    strategy_state: dict[str, object] | None,
) -> None:
    strategy = SingleDeviceStrategy(
        accelerator=TrainerAccelerator.CPU,
        precision=TrainerPrecision.FP32,
    )
    strategy.grad_scaler = SimpleNamespace(is_enabled=lambda: True)

    with pytest.raises(ValueError, match="requires `grad_scaler` state"):
        strategy.load_state_dict(strategy_state)


def test_single_device_strategy_restores_enabled_gradient_scaler_state() -> None:
    strategy = SingleDeviceStrategy(
        accelerator=TrainerAccelerator.CPU,
        precision=TrainerPrecision.FP32,
    )
    restored: list[dict[str, object]] = []
    strategy.grad_scaler = SimpleNamespace(
        is_enabled=lambda: True,
        load_state_dict=restored.append,
    )
    scaler_state = {"scale": 65536.0}

    strategy.load_state_dict({"grad_scaler": scaler_state})

    assert restored == [scaler_state]


@pytest.mark.parametrize("value", [True, 0, -1])
def test_trainer_rejects_invalid_distributed_timeout(value: object) -> None:
    with pytest.raises(ValueError, match="distributed_timeout_seconds"):
        TrainerConfig(distributed_timeout_seconds=value)
