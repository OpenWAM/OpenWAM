from __future__ import annotations

from typing import Any

from open_wam.configs import CurrentBlockCoupling


MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS = frozenset(
    {
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    }
)


def mot_policy_requires_legacy_split_cache_inference(policy_config: Any) -> bool:
    """Return whether an M5 policy config requires the split-cache rollout path."""

    raw_coupling = getattr(policy_config, "current_block_coupling", None)
    if raw_coupling is None:
        return False
    return CurrentBlockCoupling(raw_coupling) in MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS


def should_use_mot_legacy_split_cache_inference(config: Any) -> bool:
    """Return whether this M5 config should use the legacy split-cache rollout path."""

    policy_variant = getattr(config, "policy_variant", None)
    return mot_policy_requires_legacy_split_cache_inference(policy_variant)


def ensure_mot_policy_variant_inference_backend(
    *,
    policy_variant: Any,
    visual_tower: Any,
    policy_config: Any,
    allow_module_mutation: bool = True,
) -> dict[str, object]:
    """Route M5 inference to the backend implied by the config.

    Packed training transfers video/action blocks into a packed owner for FSDP.
    Some rollout modes intentionally run the older split-cache backend instead.
    This helper is safe to call from scripts and from the policy variant itself,
    so generic eval paths cannot silently keep the packed backend for those
    legacy split-cache rollout contracts.
    """

    if not mot_policy_requires_legacy_split_cache_inference(policy_config):
        return {
            "policy_variant": "mot",
            "backend": "packed_coupling",
            "legacy_split_cache_required": False,
            "legacy_split_cache_ready": False,
            "legacy_split_cache_restored_this_call": False,
        }

    restore = getattr(policy_variant, "restore_packed_blocks_for_legacy_inference", None)
    if not callable(restore):
        raise RuntimeError(
            "M5 config requires legacy split-cache inference, but the policy variant "
            "does not expose `restore_packed_blocks_for_legacy_inference`."
        )

    already_restored_before = bool(getattr(policy_variant, "_legacy_inference_blocks_restored", False))
    if not already_restored_before and not allow_module_mutation:
        raise RuntimeError(
            "M5 legacy split-cache inference requires a one-way module ownership restore, "
            "but this call disallows module mutation. Run rollout/eval with a dedicated "
            "inference-only pipeline, or skip inference validation for this packed M5 mode."
        )
    restored = False if already_restored_before else bool(restore(visual_tower))
    already_restored = bool(getattr(policy_variant, "_legacy_inference_blocks_restored", False))
    if not restored and not already_restored:
        raise RuntimeError(
            "M5 legacy split-cache inference was requested, but packed block ownership "
            "was not restored. Refusing to run a different inference backend silently."
        )
    return {
        "policy_variant": "mot",
        "backend": "legacy_split_cache",
        "legacy_split_cache_required": True,
        "legacy_split_cache_ready": bool(already_restored),
        "legacy_split_cache_restored_this_call": bool(restored),
    }


def ensure_mot_inference_backend(
    pipeline: Any,
    config: Any,
    *,
    allow_module_mutation: bool = True,
) -> dict[str, object]:
    """Route an assembled M5 pipeline to the backend implied by the config."""

    return ensure_mot_policy_variant_inference_backend(
        policy_variant=getattr(pipeline, "policy_variant", None),
        visual_tower=getattr(pipeline, "visual_tower", None),
        policy_config=getattr(config, "policy_variant", None),
        allow_module_mutation=allow_module_mutation,
    )


def resolve_mot_rollout_history_frames(*, window_size: int, frame_chunk_size: int) -> int:
    """History frames visible to the current chunk under block-id windowing.

    Video and action chunks occupy alternating block ids, so odd attention
    windows do not expose an extra complete same-stream history chunk. That
    gives floor semantics for per-stream lookback, matching Method-1 cache
    retention and M5's fixed-128 rollout-history contract.
    """

    chunk = max(1, int(frame_chunk_size))
    window = max(1, int(window_size))
    return max(chunk, (window // 2) * chunk)


def resolve_mot_rollout_cache_window_frames(*, window_size: int, frame_chunk_size: int) -> int:
    """Total cached clean frames to retain: visible history plus the current chunk."""

    chunk = max(1, int(frame_chunk_size))
    return resolve_mot_rollout_history_frames(
        window_size=window_size,
        frame_chunk_size=chunk,
    ) + chunk
