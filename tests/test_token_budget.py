"""Order, budget, and equal-sample gradient contracts for token subdivision."""

from __future__ import annotations

import random

import pytest
import torch

from open_wam.training.token_budget import (
    plan_token_microbatches,
    token_microbatch_loss_scale,
)


pytestmark = pytest.mark.unit


def test_greedy_plan_preserves_sample_order_and_exact_budget_boundaries():
    assert plan_token_microbatches([2, 3, 1, 4, 2], max_tokens=5) == (
        (0, 1),
        (2, 3),
        (4,),
    )
    assert plan_token_microbatches([5], max_tokens=5) == ((0,),)
    assert plan_token_microbatches([1, 1, 1], max_tokens=3) == ((0, 1, 2),)


def test_target_splits_earliest_groups_deterministically_without_empty_groups():
    costs = [2, 3, 1, 4, 2]
    assert plan_token_microbatches(costs, max_tokens=5, target_microbatches=3) == (
        (0, 1),
        (2, 3),
        (4,),
    )
    assert plan_token_microbatches(costs, max_tokens=5, target_microbatches=4) == (
        (0,),
        (1,),
        (2, 3),
        (4,),
    )
    assert plan_token_microbatches(costs, max_tokens=5, target_microbatches=5) == (
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
    )


def test_randomized_plans_conserve_indices_order_and_budget():
    rng = random.Random(7291)
    for _ in range(300):
        budget = rng.randint(1, 128)
        costs = [rng.randint(1, budget) for _ in range(rng.randint(1, 32))]
        before = costs.copy()
        greedy = plan_token_microbatches(costs, max_tokens=budget)
        target = rng.randint(len(greedy), len(costs))
        groups = plan_token_microbatches(
            costs, max_tokens=budget, target_microbatches=target
        )
        assert costs == before
        assert len(groups) == target
        assert all(groups)
        assert tuple(index for group in groups for index in group) == tuple(
            range(len(costs))
        )
        assert all(sum(costs[index] for index in group) <= budget for group in groups)
        assert groups == plan_token_microbatches(
            tuple(costs), max_tokens=budget, target_microbatches=target
        )
        # A greedy boundary can only occur when the next item would overflow.
        for group, following in zip(greedy, greedy[1:]):
            assert sum(costs[index] for index in group) + costs[following[0]] > budget


def test_greedy_group_count_matches_minimum_contiguous_partition():
    rng = random.Random(622)
    for _ in range(100):
        budget = rng.randint(1, 20)
        costs = [rng.randint(1, budget) for _ in range(rng.randint(1, 9))]
        optimum = [0] + [len(costs) + 1] * len(costs)
        for end in range(1, len(costs) + 1):
            for start in range(end):
                if sum(costs[start:end]) <= budget:
                    optimum[end] = min(optimum[end], optimum[start] + 1)
        assert len(plan_token_microbatches(costs, max_tokens=budget)) == optimum[-1]


def test_rank_plans_can_split_to_shared_maximum_without_repeating_samples():
    rank_costs = ([2, 2, 2, 2, 2], [5, 5, 5, 5, 5], [4, 1, 2, 3, 4])
    target = max(
        len(plan_token_microbatches(costs, max_tokens=5)) for costs in rank_costs
    )
    for costs in rank_costs:
        groups = plan_token_microbatches(
            costs, max_tokens=5, target_microbatches=target
        )
        assert len(groups) == target
        assert tuple(index for group in groups for index in group) == tuple(
            range(len(costs))
        )
        assert all(sum(costs[index] for index in group) <= 5 for group in groups)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.0, "2", None])
def test_planner_rejects_non_positive_integer_budget(invalid):
    with pytest.raises(ValueError, match="max_tokens.*positive integer"):
        plan_token_microbatches([1], max_tokens=invalid)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.0, "2", None])
def test_planner_rejects_non_positive_integer_costs(invalid):
    with pytest.raises(ValueError, match=r"costs\[1\].*positive integer"):
        plan_token_microbatches([1, invalid], max_tokens=10)


@pytest.mark.parametrize("invalid", [None, 1, {1, 2}, "12", b"12"])
def test_planner_requires_sequence_input(invalid):
    with pytest.raises(TypeError, match="sequence"):
        plan_token_microbatches(invalid, max_tokens=10)


def test_planner_rejects_empty_logical_batch():
    with pytest.raises(ValueError, match="at least one sample"):
        plan_token_microbatches([], max_tokens=10)


def test_planner_rejects_single_over_budget_sample_before_planning():
    with pytest.raises(ValueError, match="Sample 1.*cost 11.*max_tokens=10"):
        plan_token_microbatches([1, 11, 2], max_tokens=10)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 2.0, "2"])
def test_planner_rejects_non_positive_integer_target(invalid):
    with pytest.raises(ValueError, match="target_microbatches.*positive integer"):
        plan_token_microbatches([1, 1], max_tokens=10, target_microbatches=invalid)


@pytest.mark.parametrize("target", [1, 5])
def test_planner_rejects_target_outside_feasible_range(target):
    with pytest.raises(ValueError, match=r"group count \(2\).*sample count \(4\)"):
        plan_token_microbatches([2, 2, 2, 2], max_tokens=4, target_microbatches=target)


@pytest.mark.parametrize("accumulation", [1, 2, 7])
def test_loss_scales_sum_to_one_accumulation_share(accumulation):
    counts = (1, 3, 2)
    scales = [token_microbatch_loss_scale(count, 6, accumulation) for count in counts]
    assert sum(scales) == pytest.approx(1.0 / accumulation)
    for count, scale in zip(counts, scales, strict=True):
        assert scale / count == pytest.approx(1.0 / (6 * accumulation))


@pytest.mark.parametrize("field", range(3))
@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.0, "2", None])
def test_loss_scale_requires_positive_integers(field, invalid):
    arguments = [2, 5, 3]
    arguments[field] = invalid
    names = ("sample_count", "logical_sample_count", "accumulation_steps")
    with pytest.raises(ValueError, match=f"{names[field]}.*positive integer"):
        token_microbatch_loss_scale(*arguments)


def test_loss_scale_rejects_more_samples_than_the_logical_batch():
    with pytest.raises(ValueError, match="sample_count cannot exceed"):
        token_microbatch_loss_scale(5, 4, 1)


@pytest.mark.parametrize("target", [None, 4, 6])
@pytest.mark.parametrize("accumulation", [1, 3])
def test_subdivision_preserves_equal_sample_loss_and_linear_gradient(
    target, accumulation
):
    costs = [2, 3, 1, 4, 2, 1]
    groups = plan_token_microbatches(
        costs, max_tokens=5, target_microbatches=target
    )
    weights = torch.tensor([0.25, -0.4], dtype=torch.float64, requires_grad=True)
    inputs = torch.arange(12, dtype=torch.float64).reshape(6, 2) / 7.0
    targets = torch.tensor([0.1, -0.3, 0.7, 0.2, 1.1, -0.4], dtype=torch.float64)
    sample_losses = (inputs @ weights - targets).square()
    reference = sample_losses.mean() / accumulation
    subdivided = torch.stack(
        [
            sample_losses[list(group)].mean()
            * token_microbatch_loss_scale(len(group), len(costs), accumulation)
            for group in groups
        ]
    ).sum()
    expected_gradient = torch.autograd.grad(reference, weights, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(subdivided, weights)[0]
    torch.testing.assert_close(subdivided, reference, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-14, rtol=1e-14)
