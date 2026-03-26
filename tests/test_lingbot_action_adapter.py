from __future__ import annotations

import torch

from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.models.policy_variants.parallel_stream.action_adapter import (
    LingbotActionAdapter,
    build_action_adapter_spec,
)


def _build_libero_adapter() -> LingbotActionAdapter:
    config = ParallelStreamPolicyConfig(
        runtime_mode="lingbot_exact",
        reference_profile="libero",
    )
    spec = build_action_adapter_spec(config, model_action_dim=30)
    assert spec is not None
    return LingbotActionAdapter(spec)


def test_to_raw_action_sequence_matches_float32_quantile_math() -> None:
    adapter = _build_libero_adapter()
    model_action = torch.zeros(1, 1, 30, dtype=torch.bfloat16)

    raw_action = adapter.to_raw_action_sequence(model_action)
    expected = torch.tensor(
        [[[0.12053614854812622, 0.005357623100280762, 4.76837158203125e-07, 0.02517908811569214, 0.01232193410396576, 0.03910765051841736, 4.76837158203125e-07]]],
        dtype=torch.float32,
    )
    wrong = torch.tensor(
        [[[0.12109375, 0.0078125, 0.0, 0.025390625, 0.0126953125, 0.0390625, 0.0]]],
        dtype=torch.float32,
    )

    assert raw_action is not None
    assert raw_action.dtype == torch.float32
    assert torch.allclose(raw_action, expected)
    assert not torch.allclose(raw_action, wrong)


def test_to_model_action_sequence_matches_float32_quantile_math_before_cast() -> None:
    adapter = _build_libero_adapter()
    raw_action = torch.tensor(
        [[[0.12053614854812622, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    model_action = adapter.to_model_action_sequence(
        raw_action,
        action_space="raw",
        dtype=torch.bfloat16,
    )
    expected = torch.tensor(
        [[[0.0, -0.006309688091278076, -5.364418029785156e-07, -0.17216408252716064, -0.07165384292602539, -0.12829673290252686, -1.0, -1.0]]],
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16)
    wrong = torch.tensor(
        [[[0.0, -0.0078125, 0.0, -0.171875, -0.07421875, -0.12890625, -1.0, -1.0]]],
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16)

    assert model_action.dtype == torch.bfloat16
    assert torch.equal(model_action[..., :8], expected)
    assert not torch.equal(model_action[..., :8], wrong)
