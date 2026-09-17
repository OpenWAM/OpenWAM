"""Validate assembled execution without changing parameter ownership."""

from __future__ import annotations

from typing import Any

def ensure_dual_expert_policy_variant_inference_backend(
    *,
    policy_variant: Any,
    visual_tower: Any,
    policy_config: Any,
) -> None:
    """Check the one paired executor without selecting a program-specific route."""
    if (
        policy_variant.packed_block_stack is None
        or not visual_tower.core.execution_blocks
        or not policy_variant.action_expert.execution_blocks
    ):
        raise RuntimeError(
            "Dual Expert inference requires an assembled paired block stack."
        )


def ensure_dual_expert_inference_backend(
    pipeline: Any,
    config: Any,
) -> dict[str, object]:
    """Validate an assembled pipeline without mutating its modules."""
    ensure_dual_expert_policy_variant_inference_backend(
        policy_variant=pipeline.policy_variant,
        visual_tower=pipeline.visual_tower,
        policy_config=config.policy_variant,
    )
    return {
        "policy_variant": "dual_expert",
        "backend": "paired_transformer",
    }


__all__ = [
    "ensure_dual_expert_inference_backend",
    "ensure_dual_expert_policy_variant_inference_backend",
]
