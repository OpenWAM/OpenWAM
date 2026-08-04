"""DualExpert inference-backend selection and ownership restoration."""

from __future__ import annotations

from typing import Any

from .runtime_routes import resolve_dual_expert_runtime_route


def ensure_dual_expert_policy_variant_inference_backend(
    *,
    policy_variant: Any,
    visual_tower: Any,
    policy_config: Any,
    allow_module_mutation: bool = True,
) -> dict[str, object]:
    """Route dual-expert inference to the backend implied by the config.

    Packed training transfers video/action blocks into a packed owner for FSDP.
    Some rollout modes intentionally run the older split-cache backend instead.
    This helper is safe to call from scripts and from the policy variant itself,
    so generic eval paths cannot silently keep the packed backend for those
    legacy split-cache rollout contracts.
    """

    route = resolve_dual_expert_runtime_route(policy_config)
    if not route.requires_legacy_block_restore:
        return {
            "policy_variant": "dual_expert",
            "backend": "split_cache"
            if route.uses_split_cache_rollout
            else "packed_coupling",
            "route": route.to_report(),
            "legacy_split_cache_required": False,
            "legacy_split_cache_ready": False,
            "legacy_split_cache_restored_this_call": False,
        }

    restore = getattr(
        policy_variant, "restore_packed_blocks_for_legacy_inference", None
    )
    if not callable(restore):
        raise RuntimeError(
            "dual-expert config requires legacy split-cache inference, but the policy variant "
            "does not expose `restore_packed_blocks_for_legacy_inference`."
        )

    already_restored_before = bool(
        getattr(policy_variant, "_legacy_inference_blocks_restored", False)
    )
    if not already_restored_before and not allow_module_mutation:
        raise RuntimeError(
            "dual-expert legacy split-cache inference requires a one-way module ownership restore, "
            "but this call disallows module mutation. Run rollout/eval with a dedicated "
            "inference-only pipeline, or skip inference validation for this packed dual-expert mode."
        )
    restored = False if already_restored_before else bool(restore(visual_tower))
    already_restored = bool(
        getattr(policy_variant, "_legacy_inference_blocks_restored", False)
    )
    if not restored and not already_restored:
        raise RuntimeError(
            "dual-expert legacy split-cache inference was requested, but packed block ownership "
            "was not restored. Refusing to run a different inference backend silently."
        )
    return {
        "policy_variant": "dual_expert",
        "backend": "legacy_split_cache",
        "route": route.to_report(),
        "legacy_split_cache_required": True,
        "legacy_split_cache_ready": bool(already_restored),
        "legacy_split_cache_restored_this_call": bool(restored),
    }


def ensure_dual_expert_inference_backend(
    pipeline: Any,
    config: Any,
    *,
    allow_module_mutation: bool = True,
) -> dict[str, object]:
    """Route an assembled dual-expert pipeline to the backend implied by the config."""

    return ensure_dual_expert_policy_variant_inference_backend(
        policy_variant=getattr(pipeline, "policy_variant", None),
        visual_tower=getattr(pipeline, "visual_tower", None),
        policy_config=getattr(config, "policy_variant", None),
        allow_module_mutation=allow_module_mutation,
    )


__all__ = [
    "ensure_dual_expert_inference_backend",
    "ensure_dual_expert_policy_variant_inference_backend",
]
