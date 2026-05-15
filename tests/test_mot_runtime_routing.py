from __future__ import annotations

from types import SimpleNamespace

import pytest

from open_wam.models.policy_variants.mot.runtime_routing import (
    MoTRuntimeRouteKind,
    ensure_mot_inference_backend,
    ensure_mot_policy_variant_inference_backend,
    mot_policy_requires_legacy_split_cache_inference,
    resolve_mot_runtime_route,
    should_use_mot_legacy_split_cache_inference,
)


class _FakeMoTPolicy:
    def __init__(self, *, restored: bool = False, expose_restore: bool = True) -> None:
        self.restore_calls = 0
        self._legacy_inference_blocks_restored = restored
        if expose_restore:
            self.restore_packed_blocks_for_legacy_inference = self._restore

    def _restore(self, visual_tower) -> bool:
        self.restore_calls += 1
        self.visual_tower_seen = visual_tower
        self._legacy_inference_blocks_restored = True
        return True


def _config(current_block_coupling: str | None):
    return SimpleNamespace(policy_variant=SimpleNamespace(current_block_coupling=current_block_coupling))


def _mot_config(*, runtime_mode: str, current_block_coupling: str | None = None):
    return SimpleNamespace(
        policy_variant=SimpleNamespace(
            name="mot",
            runtime_mode=runtime_mode,
            current_block_coupling=current_block_coupling,
        )
    )


def test_mot_runtime_route_taxonomy_marks_legacy_and_current_paths() -> None:
    legacy_prefill = resolve_mot_runtime_route(_mot_config(runtime_mode="video_prefill_action_denoise"))
    legacy_joint = resolve_mot_runtime_route(_mot_config(runtime_mode="joint_denoise"))
    split_default = resolve_mot_runtime_route(_mot_config(runtime_mode="non_joint_two_stream"))
    split_explicit = resolve_mot_runtime_route(
        _mot_config(runtime_mode="non_joint_two_stream", current_block_coupling="video_then_action")
    )
    native = resolve_mot_runtime_route(
        _mot_config(runtime_mode="non_joint_two_stream", current_block_coupling="joint")
    )

    assert legacy_prefill.kind is MoTRuntimeRouteKind.LEGACY_VIDEO_PREFILL
    assert legacy_joint.kind is MoTRuntimeRouteKind.LEGACY_JOINT_DENOISE
    assert split_default.kind is MoTRuntimeRouteKind.SPLIT_CACHE_NON_JOINT
    assert split_default.uses_split_cache_rollout
    assert not split_default.requires_legacy_block_restore
    assert split_explicit.kind is MoTRuntimeRouteKind.SPLIT_CACHE_NON_JOINT
    assert split_explicit.requires_legacy_block_restore
    assert native.kind is MoTRuntimeRouteKind.NATIVE_PACKED_COUPLING
    assert native.uses_native_packed_rollout
    assert not native.supports_realtime_history_controls


def test_mot_runtime_routing_selects_legacy_only_for_split_cache_couplings() -> None:
    assert should_use_mot_legacy_split_cache_inference(_config("video_then_action")) is True
    assert should_use_mot_legacy_split_cache_inference(_config("decoupled_same_step")) is True
    assert should_use_mot_legacy_split_cache_inference(_config("joint")) is False
    assert should_use_mot_legacy_split_cache_inference(_config(None)) is False
    assert mot_policy_requires_legacy_split_cache_inference(_config("video_then_action").policy_variant) is True
    assert mot_policy_requires_legacy_split_cache_inference(_config("action_then_video").policy_variant) is False


def test_ensure_mot_inference_backend_restores_required_legacy_blocks() -> None:
    policy = _FakeMoTPolicy()
    visual_tower = object()
    pipeline = SimpleNamespace(policy_variant=policy, visual_tower=visual_tower)

    report = ensure_mot_inference_backend(pipeline, _config("video_then_action"))

    assert report["backend"] == "legacy_split_cache"
    assert report["legacy_split_cache_required"] is True
    assert report["legacy_split_cache_ready"] is True
    assert report["legacy_split_cache_restored_this_call"] is True
    assert policy.restore_calls == 1
    assert policy.visual_tower_seen is visual_tower


def test_ensure_mot_inference_backend_keeps_packed_backend_for_native_couplings() -> None:
    policy = _FakeMoTPolicy()
    pipeline = SimpleNamespace(policy_variant=policy, visual_tower=object())

    report = ensure_mot_inference_backend(pipeline, _config("joint"))

    assert report["backend"] == "packed_coupling"
    assert report["legacy_split_cache_required"] is False
    assert report["legacy_split_cache_ready"] is False
    assert report["legacy_split_cache_restored_this_call"] is False
    assert policy.restore_calls == 0


def test_ensure_mot_inference_backend_reports_default_non_joint_split_cache_backend() -> None:
    policy = _FakeMoTPolicy()
    pipeline = SimpleNamespace(policy_variant=policy, visual_tower=object())

    report = ensure_mot_inference_backend(pipeline, _mot_config(runtime_mode="non_joint_two_stream"))

    assert report["backend"] == "split_cache"
    assert report["legacy_split_cache_required"] is False
    assert report["legacy_split_cache_ready"] is False
    assert report["legacy_split_cache_restored_this_call"] is False
    assert report["route"]["kind"] == "split_cache_non_joint"
    assert report["route"]["uses_split_cache_rollout"] is True
    assert policy.restore_calls == 0


def test_ensure_mot_policy_variant_inference_backend_can_disallow_module_mutation() -> None:
    policy = _FakeMoTPolicy()

    with pytest.raises(RuntimeError, match="disallows module mutation"):
        ensure_mot_policy_variant_inference_backend(
            policy_variant=policy,
            visual_tower=object(),
            policy_config=_config("video_then_action").policy_variant,
            allow_module_mutation=False,
        )

    assert policy.restore_calls == 0
    assert policy._legacy_inference_blocks_restored is False


def test_ensure_mot_policy_variant_inference_backend_is_idempotent() -> None:
    policy = _FakeMoTPolicy(restored=True)

    report = ensure_mot_policy_variant_inference_backend(
        policy_variant=policy,
        visual_tower=object(),
        policy_config=_config("video_then_action").policy_variant,
    )

    assert report["backend"] == "legacy_split_cache"
    assert report["legacy_split_cache_required"] is True
    assert report["legacy_split_cache_ready"] is True
    assert report["legacy_split_cache_restored_this_call"] is False
    assert policy.restore_calls == 0


def test_ensure_mot_inference_backend_rejects_silent_legacy_misroute() -> None:
    pipeline = SimpleNamespace(
        policy_variant=_FakeMoTPolicy(expose_restore=False),
        visual_tower=object(),
    )

    with pytest.raises(RuntimeError, match="legacy split-cache inference"):
        ensure_mot_inference_backend(pipeline, _config("decoupled_same_step"))
