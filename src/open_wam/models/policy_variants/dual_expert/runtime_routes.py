"""Typed Dual Expert inference routes derived from public programs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from open_wam.configs import CurrentBlockCoupling, PolicyVariantName, VideoActionProgram
from open_wam.configs.policy_video_action import current_block_coupling_for_program

DUAL_EXPERT_SPLIT_CACHE_INFERENCE_PROGRAMS = frozenset(
    {
        VideoActionProgram.VIDEO_THEN_ACTION,
        VideoActionProgram.DECOUPLED_SAME_STEP,
    }
)
DUAL_EXPERT_SPLIT_CACHE_INFERENCE_COUPLINGS = frozenset(
    current_block_coupling_for_program(program)
    for program in DUAL_EXPERT_SPLIT_CACHE_INFERENCE_PROGRAMS
)


class DualExpertRuntimeRouteKind(str, Enum):
    """Numerical inference backend selected for a Dual Expert program."""

    NOT_DUAL_EXPERT = "not_dual_expert"
    SPLIT_CACHE = "split_cache"
    PACKED_COUPLING = "packed_coupling"


@dataclass(frozen=True)
class DualExpertRuntimeRoute:
    kind: DualExpertRuntimeRouteKind
    program: VideoActionProgram | None
    current_block_coupling: CurrentBlockCoupling | None
    requires_block_restore: bool = False
    uses_stateful_realtime_session: bool = False
    supports_realtime_history_controls: bool = False

    @property
    def is_dual_expert(self) -> bool:
        return self.kind is not DualExpertRuntimeRouteKind.NOT_DUAL_EXPERT

    @property
    def uses_split_cache_rollout(self) -> bool:
        return self.kind is DualExpertRuntimeRouteKind.SPLIT_CACHE

    @property
    def uses_native_packed_rollout(self) -> bool:
        return self.kind is DualExpertRuntimeRouteKind.PACKED_COUPLING

    def to_report(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "program": None if self.program is None else self.program.value,
            "current_block_coupling": (
                None
                if self.current_block_coupling is None
                else self.current_block_coupling.value
            ),
            "requires_block_restore": bool(self.requires_block_restore),
            "uses_split_cache_rollout": bool(self.uses_split_cache_rollout),
            "uses_stateful_realtime_session": bool(self.uses_stateful_realtime_session),
            "supports_realtime_history_controls": bool(
                self.supports_realtime_history_controls
            ),
        }


def resolve_dual_expert_runtime_route(
    config_or_policy_config: Any,
) -> DualExpertRuntimeRoute:
    """Resolve the inference backend owned by one Dual Expert program."""

    policy_config = _policy_config(config_or_policy_config)
    if not _looks_like_dual_expert_policy_config(policy_config):
        return DualExpertRuntimeRoute(
            kind=DualExpertRuntimeRouteKind.NOT_DUAL_EXPERT,
            program=None,
            current_block_coupling=None,
        )

    raw_program = getattr(policy_config, "program", None)
    if raw_program is None:
        raise ValueError(
            "Dual Expert runtime routing requires `policy_variant.program`."
        )
    program = VideoActionProgram(_enum_value(raw_program))
    coupling = current_block_coupling_for_program(program)
    if program in DUAL_EXPERT_SPLIT_CACHE_INFERENCE_PROGRAMS:
        return DualExpertRuntimeRoute(
            kind=DualExpertRuntimeRouteKind.SPLIT_CACHE,
            program=program,
            current_block_coupling=coupling,
            requires_block_restore=True,
            uses_stateful_realtime_session=True,
            supports_realtime_history_controls=True,
        )
    return DualExpertRuntimeRoute(
        kind=DualExpertRuntimeRouteKind.PACKED_COUPLING,
        program=program,
        current_block_coupling=coupling,
    )


def dual_expert_policy_requires_split_cache_inference(policy_config: Any) -> bool:
    """Return whether a Dual Expert program uses split-cache rollout."""

    return resolve_dual_expert_runtime_route(policy_config).requires_block_restore


def should_use_dual_expert_split_cache_inference(config: Any) -> bool:
    """Return whether a Dual Expert experiment uses split-cache rollout."""

    return dual_expert_policy_requires_split_cache_inference(_policy_config(config))


def _policy_config(config_or_policy_config: Any) -> Any:
    return getattr(config_or_policy_config, "policy_variant", config_or_policy_config)


def _looks_like_dual_expert_policy_config(policy_config: Any) -> bool:
    raw_name = getattr(policy_config, "name", None)
    return (
        raw_name is not None
        and _enum_value(raw_name) == PolicyVariantName.DUAL_EXPERT.value
    )


def _enum_value(raw_value: object) -> object:
    return getattr(raw_value, "value", raw_value)


__all__ = [
    "DUAL_EXPERT_SPLIT_CACHE_INFERENCE_COUPLINGS",
    "DUAL_EXPERT_SPLIT_CACHE_INFERENCE_PROGRAMS",
    "DualExpertRuntimeRoute",
    "DualExpertRuntimeRouteKind",
    "dual_expert_policy_requires_split_cache_inference",
    "resolve_dual_expert_runtime_route",
    "should_use_dual_expert_split_cache_inference",
]
