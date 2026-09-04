from __future__ import annotations

from types import SimpleNamespace

import pytest

from open_wam.configs import CurrentBlockCoupling, VideoActionProgram
from open_wam.models.policy_variants import PolicyTemporalGeometry
from open_wam.models.policy_variants.dual_expert.inference_backend import (
    ensure_dual_expert_inference_backend,
    ensure_dual_expert_policy_variant_inference_backend,
)
from open_wam.models.policy_variants.dual_expert.rollout_geometry import (
    resolve_dual_expert_action_only_rollout,
    resolve_dual_expert_rollout_frame_chunk_size,
)
from open_wam.models.policy_variants.dual_expert.runtime_routes import (
    DualExpertRuntimeRouteKind,
    dual_expert_policy_requires_split_cache_inference,
    resolve_dual_expert_runtime_route,
    should_use_dual_expert_split_cache_inference,
)


class _FakeDualExpertPolicy:
    def __init__(self, *, restored: bool = False, expose_restore: bool = True) -> None:
        self.restore_calls = 0
        self._split_cache_inference_blocks_restored = restored
        if expose_restore:
            self.restore_packed_blocks_for_split_cache_inference = self._restore

    def _restore(self, visual_tower) -> bool:
        self.restore_calls += 1
        self.visual_tower_seen = visual_tower
        self._split_cache_inference_blocks_restored = True
        return True


def _config(program: VideoActionProgram | str):
    return SimpleNamespace(
        policy_variant=SimpleNamespace(name="dual_expert", program=program)
    )


def test_dual_expert_program_owns_runtime_route_and_coupling() -> None:
    expected = {
        VideoActionProgram.VIDEO_THEN_ACTION: (
            DualExpertRuntimeRouteKind.SPLIT_CACHE,
            CurrentBlockCoupling.VIDEO_THEN_ACTION,
        ),
        VideoActionProgram.DECOUPLED_SAME_STEP: (
            DualExpertRuntimeRouteKind.SPLIT_CACHE,
            CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        ),
        VideoActionProgram.ACTION_THEN_VIDEO: (
            DualExpertRuntimeRouteKind.PACKED_COUPLING,
            CurrentBlockCoupling.ACTION_THEN_VIDEO,
        ),
        VideoActionProgram.JOINT: (
            DualExpertRuntimeRouteKind.PACKED_COUPLING,
            CurrentBlockCoupling.JOINT,
        ),
        VideoActionProgram.VIDEO_NOISY_TO_ACTION: (
            DualExpertRuntimeRouteKind.PACKED_COUPLING,
            CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        ),
        VideoActionProgram.ACTION_NOISY_TO_VIDEO: (
            DualExpertRuntimeRouteKind.PACKED_COUPLING,
            CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
        ),
        VideoActionProgram.GENERALIST_JOINT_DENOISING: (
            DualExpertRuntimeRouteKind.PACKED_COUPLING,
            CurrentBlockCoupling.JOINT,
        ),
        VideoActionProgram.FORWARD_DYNAMICS: (
            DualExpertRuntimeRouteKind.PACKED_COUPLING,
            CurrentBlockCoupling.JOINT,
        ),
        VideoActionProgram.INVERSE_DYNAMICS: (
            DualExpertRuntimeRouteKind.PACKED_COUPLING,
            CurrentBlockCoupling.JOINT,
        ),
    }

    for program, (kind, coupling) in expected.items():
        route = resolve_dual_expert_runtime_route(_config(program))
        assert route.kind is kind
        assert route.program is program
        assert route.current_block_coupling is coupling
        assert route.requires_block_restore is (
            kind is DualExpertRuntimeRouteKind.SPLIT_CACHE
        )


def test_dual_expert_runtime_route_rejects_missing_program() -> None:
    config = SimpleNamespace(policy_variant=SimpleNamespace(name="dual_expert"))

    with pytest.raises(ValueError, match="requires `policy_variant.program`"):
        resolve_dual_expert_runtime_route(config)


def test_runtime_route_does_not_infer_architecture_from_program_alone() -> None:
    route = resolve_dual_expert_runtime_route(
        SimpleNamespace(program=VideoActionProgram.VIDEO_THEN_ACTION)
    )

    assert route.kind is DualExpertRuntimeRouteKind.NOT_DUAL_EXPERT


def test_dual_expert_rollout_override_preserves_configured_action_token_density() -> None:
    context = SimpleNamespace(
        extra={
            "dual_expert_rollout_frame_chunk_size": 2,
        }
    )

    assert resolve_dual_expert_rollout_frame_chunk_size(
        context,
        default_frame_chunk_size=4,
        base_action_horizon=16,
    ) == (2, 8, 4)


def test_dual_expert_rollout_honors_resolved_temporal_geometry() -> None:
    context = SimpleNamespace(
        extra={},
        temporal_geometry=PolicyTemporalGeometry(
            frame_chunk_size=2,
            attention_window_size=17,
        ),
    )

    assert resolve_dual_expert_rollout_frame_chunk_size(
        context,
        default_frame_chunk_size=4,
        base_action_horizon=16,
    ) == (2, 8, 4)


def test_dual_expert_action_only_rollout_is_limited_to_action_safe_couplings() -> None:
    context = SimpleNamespace(extra={"dual_expert_action_only_rollout": True})

    assert resolve_dual_expert_action_only_rollout(
        context,
        current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
    )
    with pytest.raises(ValueError, match="action-only-safe"):
        resolve_dual_expert_action_only_rollout(
            context,
            current_block_coupling=CurrentBlockCoupling.JOINT,
        )


def test_dual_expert_split_cache_selection_is_program_driven() -> None:
    assert should_use_dual_expert_split_cache_inference(
        _config(VideoActionProgram.VIDEO_THEN_ACTION)
    )
    assert should_use_dual_expert_split_cache_inference(
        _config(VideoActionProgram.DECOUPLED_SAME_STEP)
    )
    assert not should_use_dual_expert_split_cache_inference(
        _config(VideoActionProgram.JOINT)
    )
    assert dual_expert_policy_requires_split_cache_inference(
        _config(VideoActionProgram.VIDEO_THEN_ACTION).policy_variant
    )
    assert not dual_expert_policy_requires_split_cache_inference(
        _config(VideoActionProgram.ACTION_THEN_VIDEO).policy_variant
    )


def test_ensure_dual_expert_inference_backend_restores_split_cache_blocks() -> None:
    policy = _FakeDualExpertPolicy()
    visual_tower = object()
    pipeline = SimpleNamespace(policy_variant=policy, visual_tower=visual_tower)

    report = ensure_dual_expert_inference_backend(
        pipeline, _config(VideoActionProgram.VIDEO_THEN_ACTION)
    )

    assert report["backend"] == "split_cache"
    assert report["block_restore_required"] is True
    assert report["block_restore_ready"] is True
    assert report["block_restore_performed"] is True
    assert report["route"]["program"] == "video_then_action"
    assert policy.restore_calls == 1
    assert policy.visual_tower_seen is visual_tower


def test_ensure_dual_expert_inference_backend_keeps_packed_backend() -> None:
    policy = _FakeDualExpertPolicy()
    pipeline = SimpleNamespace(policy_variant=policy, visual_tower=object())

    report = ensure_dual_expert_inference_backend(
        pipeline, _config(VideoActionProgram.JOINT)
    )

    assert report["backend"] == "packed_coupling"
    assert report["block_restore_required"] is False
    assert report["block_restore_ready"] is False
    assert report["block_restore_performed"] is False
    assert report["route"]["current_block_coupling"] == "joint"
    assert policy.restore_calls == 0


def test_ensure_dual_expert_policy_backend_can_disallow_module_mutation() -> None:
    policy = _FakeDualExpertPolicy()

    with pytest.raises(RuntimeError, match="disallows module mutation"):
        ensure_dual_expert_policy_variant_inference_backend(
            policy_variant=policy,
            visual_tower=object(),
            policy_config=_config(VideoActionProgram.VIDEO_THEN_ACTION).policy_variant,
            allow_module_mutation=False,
        )

    assert policy.restore_calls == 0
    assert policy._split_cache_inference_blocks_restored is False


def test_ensure_dual_expert_policy_backend_restore_is_idempotent() -> None:
    policy = _FakeDualExpertPolicy(restored=True)

    report = ensure_dual_expert_policy_variant_inference_backend(
        policy_variant=policy,
        visual_tower=object(),
        policy_config=_config(VideoActionProgram.VIDEO_THEN_ACTION).policy_variant,
    )

    assert report["backend"] == "split_cache"
    assert report["block_restore_required"] is True
    assert report["block_restore_ready"] is True
    assert report["block_restore_performed"] is False
    assert policy.restore_calls == 0


def test_ensure_dual_expert_inference_backend_rejects_missing_restore_hook() -> None:
    pipeline = SimpleNamespace(
        policy_variant=_FakeDualExpertPolicy(expose_restore=False),
        visual_tower=object(),
    )

    with pytest.raises(RuntimeError, match="requires split-cache inference"):
        ensure_dual_expert_inference_backend(
            pipeline, _config(VideoActionProgram.DECOUPLED_SAME_STEP)
        )
