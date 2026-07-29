from __future__ import annotations

from types import SimpleNamespace

import torch

from open_wam.training.strategies import _apply_composable_fsdp_sharding


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
