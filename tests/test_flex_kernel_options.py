"""Kernel presets and their actual attention-call boundary."""

import pytest
import torch

from open_wam.models.common.attention_backends import shared_flex_kernel_options
from open_wam.models.visual_tower import shared_transformer_support as transformer


pytestmark = pytest.mark.unit

EXPECTED_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_M1": 32,
    "BLOCK_N1": 64,
    "BLOCK_M2": 64,
    "BLOCK_N2": 32,
}


def test_shared_preset_preserves_exact_values_and_returns_independent_dictionaries():
    first = shared_flex_kernel_options()
    second = shared_flex_kernel_options()
    assert first == second == EXPECTED_OPTIONS
    first["BLOCK_M"] = 1
    first["extra"] = 2
    assert second == shared_flex_kernel_options() == EXPECTED_OPTIONS


@pytest.mark.parametrize("has_block_mask", [False, True])
def test_attention_passes_kernel_options_only_with_a_block_mask(monkeypatch, has_block_mask):
    block_mask = object() if has_block_mask else None
    monkeypatch.setattr(
        transformer, "select_attention_profile_mask",
        lambda *args, **kwargs: (None, block_mask),
    )
    calls = []

    def attention(**kwargs):
        calls.append(kwargs)
        return kwargs["value"]

    monkeypatch.setattr(transformer, "apply_attention_backend", attention)
    layer = transformer.SharedTransformerAttention(dim=8, heads=2, dim_head=4, eps=1e-6)
    tokens = torch.zeros(1, 3, 8)
    result, _ = layer(tokens, tokens, tokens)

    assert result.shape == tokens.shape
    assert len(calls) == 1
    assert calls[0]["block_mask"] is block_mask
    assert calls[0]["kernel_options"] == (EXPECTED_OPTIONS if has_block_mask else None)
