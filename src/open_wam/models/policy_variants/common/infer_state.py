from __future__ import annotations

from open_wam.models.visual_tower import VisualTower

from ..contracts import PolicyInferState, RolloutCursor
from .rollout import advance_rollout_cursor


def prepare_default_runtime_infer_state(
    visual_tower: VisualTower,
    *,
    previous_state: PolicyInferState | None,
    stage: str,
    payload: dict[str, object] | None = None,
) -> PolicyInferState:
    """Initialize or pass through a simple rollout state over shared cache infra.

    `post_latent` and `post_decoded` do not own elaborate rollout semantics.
    They still should use the same cursor/cache vocabulary as the joint
    variants, but without duplicating the same initialization code in every
    simple variant.
    """

    if previous_state is not None:
        return previous_state
    cursor = RolloutCursor()
    return PolicyInferState(
        step_index=0,
        cursor=cursor,
        cache=visual_tower.resolve_runtime_cache_state(
            None,
            cursor=cursor,
            stage=stage,
            payload=payload,
        ),
    )


def advance_default_runtime_infer_state(
    visual_tower: VisualTower,
    *,
    infer_state: PolicyInferState,
    stage: str,
    payload: dict[str, object] | None = None,
    payload_updates: dict[str, object] | None = None,
) -> PolicyInferState:
    """Advance a simple rollout state using the shared cursor/cache lifecycle.

    This deliberately does not invent any variant-specific semantics. It only
    advances the common rollout cursor and lets the visual tower own cache
    lifecycle bookkeeping.
    """

    next_cursor = advance_rollout_cursor(infer_state.cursor)
    current_cache = visual_tower.resolve_runtime_cache_state(
        infer_state.cache,
        cursor=infer_state.cursor,
        stage=stage,
        payload=payload,
    )
    next_cache = visual_tower.advance_runtime_cache_state(
        current_cache,
        next_cursor=next_cursor,
        payload_updates=payload_updates,
    )
    return PolicyInferState(
        step_index=infer_state.step_index + 1,
        cursor=next_cursor,
        cache=next_cache,
    )
