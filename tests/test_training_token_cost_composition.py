"""Token admission is selected at composition, not inside the training loop."""

from dataclasses import replace

import pytest

from open_wam.configs import BatchingConfig, DualExpertPolicyConfig, ExperimentConfig, ParallelStreamPolicyConfig
from open_wam.pipelines.token_cost import build_training_token_cost_fn


def test_disabled_budget_does_not_require_a_supported_policy():
    assert build_training_token_cost_fn(ExperimentConfig()) is None


@pytest.mark.parametrize("program", ["video_then_action", "joint"])
def test_supported_counter_is_bound_to_the_experiment(program, monkeypatch):
    config = ExperimentConfig()
    config = replace(config,
        data=replace(config.data, batching=BatchingConfig(mode="packed", max_tokens=100)),
        policy_variant=DualExpertPolicyConfig(program=program),
    )
    counter = build_training_token_cost_fn(config)
    assert callable(counter)
    assert counter.keywords == {"config": config}
    import open_wam.models.policy_variants.dual_expert.token_cost as costs
    sentinel = object()

    def count(*, config: ExperimentConfig, batch: object):
        assert config is counter.keywords["config"]
        assert batch is sentinel
        return (17,)

    monkeypatch.setattr(costs, "dual_expert_token_costs", count)
    assert counter(sentinel) == (17,)


def test_unsupported_policy_fails_before_model_construction():
    config = ExperimentConfig()
    config = replace(config,
        data=replace(config.data, batching=BatchingConfig(mode="packed", max_tokens=100)),
        policy_variant=ParallelStreamPolicyConfig(program="joint"),
    )
    with pytest.raises(ValueError, match="VTA/Joint"):
        build_training_token_cost_fn(config)
