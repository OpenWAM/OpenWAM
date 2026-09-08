"""Pure planning and loss-accounting helpers for token-budget microbatches.

The caller owns sample selection, token-cost measurement, and any distributed
coordination. These helpers only subdivide an already selected logical batch.
"""

from __future__ import annotations

from collections.abc import Sequence


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def plan_token_microbatches(
    costs: Sequence[int],
    *,
    max_tokens: int,
    target_microbatches: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Partition original sample indices into nonempty, budget-bounded groups.

    Greedy contiguous grouping preserves sample order and uses the minimum
    number of groups for the supplied positive costs. A distributed caller can
    pass an agreed ``target_microbatches`` to request more groups: the earliest
    splittable groups are deterministically peeled into leading singletons.
    Splitting never changes the retained samples, their order, or their costs.

    A single over-budget sample raises rather than being truncated, skipped, or
    executed outside the budget. The target must be between the greedy group
    count and the number of samples; empty synchronization groups are forbidden.
    """
    budget = _positive_integer(max_tokens, "max_tokens")
    if isinstance(costs, (str, bytes)) or not isinstance(costs, Sequence):
        raise TypeError("costs must be a nonempty sequence of positive integers.")
    if not costs:
        raise ValueError("costs must contain at least one sample.")
    checked_costs = tuple(
        _positive_integer(cost, f"costs[{index}]")
        for index, cost in enumerate(costs)
    )
    for index, cost in enumerate(checked_costs):
        if cost > budget:
            raise ValueError(
                f"Sample {index} has token cost {cost}, exceeding max_tokens={budget}. "
                "Increase the budget or explicitly change the sample construction."
            )

    groups: list[tuple[int, ...]] = []
    start = 0
    running_cost = 0
    for index, cost in enumerate(checked_costs):
        if running_cost + cost > budget:
            groups.append(tuple(range(start, index)))
            start = index
            running_cost = 0
        running_cost += cost
    groups.append(tuple(range(start, len(checked_costs))))

    if target_microbatches is None:
        return tuple(groups)
    target = _positive_integer(target_microbatches, "target_microbatches")
    if not len(groups) <= target <= len(checked_costs):
        raise ValueError(
            "target_microbatches must be between the minimum required group count "
            f"({len(groups)}) and the logical sample count ({len(checked_costs)}), "
            f"got {target}."
        )

    extra_groups = target - len(groups)
    refined: list[tuple[int, ...]] = []
    for group in groups:
        split_count = min(extra_groups, len(group) - 1)
        refined.extend((index,) for index in group[:split_count])
        refined.append(group[split_count:])
        extra_groups -= split_count
    return tuple(refined)


def token_microbatch_loss_scale(
    sample_count: int,
    logical_sample_count: int,
    accumulation_steps: int,
) -> float:
    """Scale a microbatch's mean loss to its logical-batch accumulation share.

    Multiplying a mean over ``sample_count`` samples by ``n / (N * A)`` gives
    every original sample weight ``1 / (N * A)``, independent of its token
    length and the chosen subdivision. The caller must not apply an additional
    accumulation divisor after using this scale.
    """
    count = _positive_integer(sample_count, "sample_count")
    logical_count = _positive_integer(logical_sample_count, "logical_sample_count")
    accumulation = _positive_integer(accumulation_steps, "accumulation_steps")
    if count > logical_count:
        raise ValueError("sample_count cannot exceed logical_sample_count.")
    return count / (logical_count * accumulation)
