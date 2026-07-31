from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.models.policy_variants.mot.cache_execution import (
    forward_action_with_video_and_action_cache,
    forward_action_with_video_cache,
    prefill_video_kv_cache,
)
from open_wam.models.policy_variants.mot.contracts import (
    MoTActionCache,
    MoTActionLayerCache,
    MoTVideoCache,
    MoTVideoLayerCache,
)
from open_wam.models.policy_variants.mot.modules import (
    MoTActionExpert,
    MoTActionPreprocessOutput,
)


class _FakeVisualTower:
    def __init__(self, entries: tuple[SimpleNamespace, ...]) -> None:
        self.entries = entries
        self.calls: list[dict[str, object]] = []

    def prefill_exact_video_cache(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(self_attention_kv=self.entries)


@pytest.mark.parametrize("detach_cache", [False, True])
def test_prefill_video_kv_cache_preserves_execution_contract(
    detach_cache: bool,
) -> None:
    key = torch.randn(1, 2, 3, 8, requires_grad=True)
    value = torch.randn(1, 2, 3, 8, requires_grad=True)
    tower = _FakeVisualTower((SimpleNamespace(key=key, value=value),))
    observed_prefix = torch.randn(1, 4, 2, 2, 2)
    attention_mask = torch.ones(3, 3, dtype=torch.bool)
    cross_attention_mask = torch.ones(1, 3, 2, dtype=torch.bool)

    cache = prefill_video_kv_cache(
        visual_tower=tower,
        observed_prefix=observed_prefix,
        text_context=None,
        frame_start=7,
        attention_mask=attention_mask,
        cross_attention_mask=cross_attention_mask,
        detach_cache=detach_cache,
    )

    assert cache.video_seq_len == 3
    assert len(cache.layers) == 1
    assert cache.layers[0].key.data_ptr() == key.data_ptr()
    assert cache.layers[0].value.data_ptr() == value.data_ptr()
    assert cache.layers[0].key.requires_grad is (not detach_cache)
    assert cache.layers[0].value.requires_grad is (not detach_cache)
    assert len(tower.calls) == 1
    call = tower.calls[0]
    assert call["observed_prefix"] is observed_prefix
    assert call["text_context"] is None
    assert call["frame_start"] == 7
    assert call["cache_name"] == "mot_video_prefill"
    assert call["attention_mask"] is attention_mask
    assert call["cross_attention_mask"] is cross_attention_mask
    assert call["detach_cache"] is detach_cache


def _build_action_inputs() -> tuple[
    MoTActionExpert,
    torch.Tensor,
    MoTActionPreprocessOutput,
    MoTVideoCache,
]:
    torch.manual_seed(7)
    expert = MoTActionExpert(
        hidden_size=16,
        action_dim=4,
        num_layers=1,
        num_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        text_dim=8,
        freq_dim=8,
    )
    action_tokens = torch.randn(1, 3, 4, requires_grad=True)
    action_pre = expert.pre_dit(
        action_tokens=action_tokens,
        timestep=torch.zeros(1, 3),
        context=torch.randn(1, 2, 8),
    )
    video_cache = MoTVideoCache(
        layers=(
            MoTVideoLayerCache(
                key=torch.randn(1, 2, 4, 8),
                value=torch.randn(1, 2, 4, 8),
            ),
        ),
        video_seq_len=4,
    )
    return expert, action_tokens, action_pre, video_cache


def test_forward_action_with_video_cache_preserves_backward_flow() -> None:
    expert, action_tokens, action_pre, video_cache = _build_action_inputs()

    hidden_states = forward_action_with_video_cache(
        action_expert=expert,
        action_pre=action_pre,
        video_cache=video_cache,
        attention_mask=torch.ones(7, 7, dtype=torch.bool),
    )
    hidden_states.square().mean().backward()

    assert hidden_states.shape == (1, 3, 16)
    assert action_tokens.grad is not None
    assert torch.isfinite(action_tokens.grad).all()


@pytest.mark.parametrize("include_action_cache", [False, True])
def test_forward_action_with_video_and_action_cache_returns_detached_fresh_kv(
    include_action_cache: bool,
) -> None:
    expert, action_tokens, action_pre, video_cache = _build_action_inputs()
    action_cache = (
        MoTActionCache(
            layers=(
                MoTActionLayerCache(
                    key=torch.randn(1, 2, 2, 8),
                    value=torch.randn(1, 2, 2, 8),
                ),
            ),
            action_seq_len=2,
        )
        if include_action_cache
        else None
    )
    total_tokens = 4 + (2 if include_action_cache else 0) + 3

    hidden_states, fresh_cache = forward_action_with_video_and_action_cache(
        action_expert=expert,
        action_pre=action_pre,
        video_cache=video_cache,
        action_cache=action_cache,
        attention_mask=torch.ones(
            total_tokens,
            total_tokens,
            dtype=torch.bool,
        ),
    )
    hidden_states.square().mean().backward()

    assert hidden_states.shape == (1, 3, 16)
    assert action_tokens.grad is not None
    assert torch.isfinite(action_tokens.grad).all()
    assert fresh_cache.action_seq_len == 3
    assert len(fresh_cache.layers) == 1
    assert fresh_cache.layers[0].key.shape == (1, 2, 3, 8)
    assert fresh_cache.layers[0].value.shape == (1, 2, 3, 8)
    assert not fresh_cache.layers[0].key.requires_grad
    assert not fresh_cache.layers[0].value.requires_grad
