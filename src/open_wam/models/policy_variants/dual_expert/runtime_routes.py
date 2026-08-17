"""Typed DualExpert runtime routes and deterministic route selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from open_wam.configs import (
    CurrentBlockCoupling,
    DualExpertRuntimeMode,
    PolicyVariantName,
)

DUAL_EXPERT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS = frozenset(
    {
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    }
)


class DualExpertRuntimeRouteKind(str, Enum):
    """Inference route family for dual-expert/DualExpert rollouts."""

    NOT_DUAL_EXPERT = "not_dual_expert"
    LEGACY_VIDEO_PREFILL = "legacy_video_prefill"
    LEGACY_JOINT_DENOISE = "legacy_joint_denoise"
    SPLIT_CACHE_NON_JOINT = "split_cache_non_joint"
    NATIVE_PACKED_COUPLING = "native_packed_coupling"


@dataclass(frozen=True)
class DualExpertRuntimeRoute:
    kind: DualExpertRuntimeRouteKind
    runtime_mode: DualExpertRuntimeMode | None
    current_block_coupling: CurrentBlockCoupling | None
    resolved_current_block_coupling: CurrentBlockCoupling | None
    requires_legacy_block_restore: bool = False
    uses_split_cache_rollout: bool = False
    uses_stateful_realtime_session: bool = False
    supports_realtime_history_controls: bool = False

    @property
    def is_dual_expert(self) -> bool:
        return self.kind is not DualExpertRuntimeRouteKind.NOT_DUAL_EXPERT

    @property
    def uses_native_packed_rollout(self) -> bool:
        return self.kind is DualExpertRuntimeRouteKind.NATIVE_PACKED_COUPLING

    def to_report(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "runtime_mode": None
            if self.runtime_mode is None
            else self.runtime_mode.value,
            "current_block_coupling": (
                None
                if self.current_block_coupling is None
                else self.current_block_coupling.value
            ),
            "resolved_current_block_coupling": (
                None
                if self.resolved_current_block_coupling is None
                else self.resolved_current_block_coupling.value
            ),
            "requires_legacy_block_restore": bool(self.requires_legacy_block_restore),
            "uses_split_cache_rollout": bool(self.uses_split_cache_rollout),
            "uses_stateful_realtime_session": bool(self.uses_stateful_realtime_session),
            "supports_realtime_history_controls": bool(
                self.supports_realtime_history_controls
            ),
        }


def resolve_dual_expert_runtime_route(config_or_policy_config: Any) -> DualExpertRuntimeRoute:
    """Resolve the single DualExpert inference route that scripts and policy code should use."""

    policy_config = _policy_config(config_or_policy_config)
    if not _looks_like_dual_expert_policy_config(policy_config):
        return DualExpertRuntimeRoute(
            kind=DualExpertRuntimeRouteKind.NOT_DUAL_EXPERT,
            runtime_mode=None,
            current_block_coupling=None,
            resolved_current_block_coupling=None,
        )

    explicit_coupling = _coerce_current_block_coupling(
        getattr(policy_config, "current_block_coupling", None)
    )
    runtime_mode = _coerce_runtime_mode(
        getattr(policy_config, "runtime_mode", None),
        explicit_coupling=explicit_coupling,
    )
    resolved_coupling = _resolve_current_block_coupling(
        runtime_mode=runtime_mode,
        explicit_coupling=explicit_coupling,
    )

    if runtime_mode == DualExpertRuntimeMode.NON_JOINT_TWO_STREAM:
        if resolved_coupling in DUAL_EXPERT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS:
            return DualExpertRuntimeRoute(
                kind=DualExpertRuntimeRouteKind.SPLIT_CACHE_NON_JOINT,
                runtime_mode=runtime_mode,
                current_block_coupling=explicit_coupling,
                resolved_current_block_coupling=resolved_coupling,
                requires_legacy_block_restore=explicit_coupling is not None,
                uses_split_cache_rollout=True,
                uses_stateful_realtime_session=True,
                supports_realtime_history_controls=True,
            )
        return DualExpertRuntimeRoute(
            kind=DualExpertRuntimeRouteKind.NATIVE_PACKED_COUPLING,
            runtime_mode=runtime_mode,
            current_block_coupling=explicit_coupling,
            resolved_current_block_coupling=resolved_coupling,
        )

    if runtime_mode == DualExpertRuntimeMode.JOINT_DENOISE:
        return DualExpertRuntimeRoute(
            kind=DualExpertRuntimeRouteKind.LEGACY_JOINT_DENOISE,
            runtime_mode=runtime_mode,
            current_block_coupling=explicit_coupling,
            resolved_current_block_coupling=resolved_coupling,
        )

    return DualExpertRuntimeRoute(
        kind=DualExpertRuntimeRouteKind.LEGACY_VIDEO_PREFILL,
        runtime_mode=runtime_mode,
        current_block_coupling=explicit_coupling,
        resolved_current_block_coupling=resolved_coupling,
    )


def dual_expert_policy_requires_legacy_split_cache_inference(policy_config: Any) -> bool:
    """Return whether an dual-expert policy config must restore split-cache module ownership."""

    return resolve_dual_expert_runtime_route(policy_config).requires_legacy_block_restore


def should_use_dual_expert_legacy_split_cache_inference(config: Any) -> bool:
    """Return whether this dual-expert config must restore split-cache module ownership."""

    return dual_expert_policy_requires_legacy_split_cache_inference(_policy_config(config))


def _policy_config(config_or_policy_config: Any) -> Any:
    return getattr(config_or_policy_config, "policy_variant", config_or_policy_config)


def _looks_like_dual_expert_policy_config(policy_config: Any) -> bool:
    raw_name = getattr(policy_config, "name", None)
    if raw_name is None:
        return hasattr(policy_config, "runtime_mode") or hasattr(
            policy_config, "current_block_coupling"
        )
    return _enum_value(raw_name) == PolicyVariantName.DUAL_EXPERT.value


def _coerce_runtime_mode(
    raw_value: object,
    *,
    explicit_coupling: CurrentBlockCoupling | None,
) -> DualExpertRuntimeMode:
    if raw_value is None:
        if explicit_coupling is not None:
            return DualExpertRuntimeMode.NON_JOINT_TWO_STREAM
        return DualExpertRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE
    return DualExpertRuntimeMode(_enum_value(raw_value))


def _coerce_current_block_coupling(raw_value: object) -> CurrentBlockCoupling | None:
    if raw_value is None:
        return None
    return CurrentBlockCoupling(_enum_value(raw_value))


def _resolve_current_block_coupling(
    *,
    runtime_mode: DualExpertRuntimeMode,
    explicit_coupling: CurrentBlockCoupling | None,
) -> CurrentBlockCoupling:
    if explicit_coupling is not None:
        return explicit_coupling
    if runtime_mode == DualExpertRuntimeMode.JOINT_DENOISE:
        return CurrentBlockCoupling.JOINT
    return CurrentBlockCoupling.VIDEO_THEN_ACTION


def _enum_value(raw_value: object) -> object:
    return getattr(raw_value, "value", raw_value)


__all__ = [
    "DUAL_EXPERT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS",
    "DualExpertRuntimeRoute",
    "DualExpertRuntimeRouteKind",
    "dual_expert_policy_requires_legacy_split_cache_inference",
    "resolve_dual_expert_runtime_route",
    "should_use_dual_expert_legacy_split_cache_inference",
]
