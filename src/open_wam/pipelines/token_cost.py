"""Compose a policy-owned token counter for generic training infrastructure."""

from collections.abc import Callable
from functools import partial

from open_wam.configs import ExperimentConfig, VideoActionProgram
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig


def _count_dual_expert_tokens(batch: object, *, config: ExperimentConfig) -> tuple[int, ...]:
    from open_wam.models.policy_variants.dual_expert.token_cost import dual_expert_token_costs

    return dual_expert_token_costs(config=config, batch=batch)


def build_training_token_cost_fn(
    config: ExperimentConfig,
) -> Callable[[object], tuple[int, ...]] | None:
    """Return CPU admission accounting, or None for unchanged fixed batching."""
    if config.data.batching.max_tokens is None:
        return None
    if (
        not isinstance(config.policy_variant, DualExpertPolicyConfig)
        or config.policy_variant.program not in {
            VideoActionProgram.VIDEO_THEN_ACTION,
            VideoActionProgram.JOINT,
        }
    ):
        raise ValueError("Token-budget training currently supports DualExpert VTA/Joint only.")
    return partial(_count_dual_expert_tokens, config=config)
