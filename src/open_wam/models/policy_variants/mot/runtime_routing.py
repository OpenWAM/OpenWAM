from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from open_wam.configs import CurrentBlockCoupling, MoTRuntimeMode, PolicyVariantName
from open_wam.configs.enums import RolloutContextPolicy, SampleTargetAlignment


class MoTRuntimeRouteKind(str, Enum):
    """Inference route family for Method-5/MoT rollouts."""

    NOT_MOT = "not_mot"
    LEGACY_VIDEO_PREFILL = "legacy_video_prefill"
    LEGACY_JOINT_DENOISE = "legacy_joint_denoise"
    SPLIT_CACHE_NON_JOINT = "split_cache_non_joint"
    NATIVE_PACKED_COUPLING = "native_packed_coupling"


MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS = frozenset(
    {
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    }
)


@dataclass(frozen=True)
class MoTRuntimeRoute:
    kind: MoTRuntimeRouteKind
    runtime_mode: MoTRuntimeMode | None
    current_block_coupling: CurrentBlockCoupling | None
    resolved_current_block_coupling: CurrentBlockCoupling | None
    requires_legacy_block_restore: bool = False
    uses_split_cache_rollout: bool = False
    uses_stateful_realtime_session: bool = False
    supports_realtime_history_controls: bool = False

    @property
    def is_mot(self) -> bool:
        return self.kind is not MoTRuntimeRouteKind.NOT_MOT

    @property
    def uses_native_packed_rollout(self) -> bool:
        return self.kind is MoTRuntimeRouteKind.NATIVE_PACKED_COUPLING

    def to_report(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "runtime_mode": None if self.runtime_mode is None else self.runtime_mode.value,
            "current_block_coupling": (
                None if self.current_block_coupling is None else self.current_block_coupling.value
            ),
            "resolved_current_block_coupling": (
                None
                if self.resolved_current_block_coupling is None
                else self.resolved_current_block_coupling.value
            ),
            "requires_legacy_block_restore": bool(self.requires_legacy_block_restore),
            "uses_split_cache_rollout": bool(self.uses_split_cache_rollout),
            "uses_stateful_realtime_session": bool(self.uses_stateful_realtime_session),
            "supports_realtime_history_controls": bool(self.supports_realtime_history_controls),
        }


def resolve_mot_runtime_route(config_or_policy_config: Any) -> MoTRuntimeRoute:
    """Resolve the single MoT inference route that scripts and policy code should use."""

    policy_config = _policy_config(config_or_policy_config)
    if not _looks_like_mot_policy_config(policy_config):
        return MoTRuntimeRoute(
            kind=MoTRuntimeRouteKind.NOT_MOT,
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

    if runtime_mode == MoTRuntimeMode.NON_JOINT_TWO_STREAM:
        if resolved_coupling in MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS:
            return MoTRuntimeRoute(
                kind=MoTRuntimeRouteKind.SPLIT_CACHE_NON_JOINT,
                runtime_mode=runtime_mode,
                current_block_coupling=explicit_coupling,
                resolved_current_block_coupling=resolved_coupling,
                requires_legacy_block_restore=explicit_coupling is not None,
                uses_split_cache_rollout=True,
                uses_stateful_realtime_session=True,
                supports_realtime_history_controls=True,
            )
        return MoTRuntimeRoute(
            kind=MoTRuntimeRouteKind.NATIVE_PACKED_COUPLING,
            runtime_mode=runtime_mode,
            current_block_coupling=explicit_coupling,
            resolved_current_block_coupling=resolved_coupling,
        )

    if runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
        return MoTRuntimeRoute(
            kind=MoTRuntimeRouteKind.LEGACY_JOINT_DENOISE,
            runtime_mode=runtime_mode,
            current_block_coupling=explicit_coupling,
            resolved_current_block_coupling=resolved_coupling,
        )

    return MoTRuntimeRoute(
        kind=MoTRuntimeRouteKind.LEGACY_VIDEO_PREFILL,
        runtime_mode=runtime_mode,
        current_block_coupling=explicit_coupling,
        resolved_current_block_coupling=resolved_coupling,
    )


def mot_policy_requires_legacy_split_cache_inference(policy_config: Any) -> bool:
    """Return whether an M5 policy config must restore split-cache module ownership."""

    return resolve_mot_runtime_route(policy_config).requires_legacy_block_restore


def should_use_mot_legacy_split_cache_inference(config: Any) -> bool:
    """Return whether this M5 config must restore split-cache module ownership."""

    return mot_policy_requires_legacy_split_cache_inference(_policy_config(config))


def resolve_mot_sequence_actions_per_frame(*, action_horizon: int, frame_chunk_size: int) -> int:
    """Resolve low-level control actions represented by one generated video frame."""

    action_horizon = int(action_horizon)
    frame_chunk_size = int(frame_chunk_size)
    if frame_chunk_size <= 0:
        raise ValueError(f"Expected frame_chunk_size > 0, got {frame_chunk_size}.")
    if action_horizon <= 0:
        raise ValueError(f"Expected action_horizon > 0, got {action_horizon}.")
    if action_horizon % frame_chunk_size != 0:
        raise ValueError(
            "MoT realtime rollout expects action_horizon to divide by inference.frame_chunk_size, "
            f"got action_horizon={action_horizon}, frame_chunk_size={frame_chunk_size}."
        )
    return action_horizon // frame_chunk_size


def resolve_mot_sequence_execution_action_offset(
    config_or_policy_config: Any,
    *,
    action_horizon: int,
    frame_chunk_size: int,
) -> int:
    """Resolve action-index offset between model output and executable actions.

    Strict rollout-parity M5 emits only executable generated actions, including
    split-cache routes when the full experiment config is available. Older
    split-cache/legacy M5 routes can still include the observed frame's action
    group in the returned chunk, so they keep the historical one-frame
    execution reindexing behind the legacy config contract.
    """

    route = resolve_mot_runtime_route(config_or_policy_config)
    if not route.is_mot:
        return 0
    actions_per_frame = resolve_mot_sequence_actions_per_frame(
        action_horizon=action_horizon,
        frame_chunk_size=frame_chunk_size,
    )
    if route.uses_native_packed_rollout or mot_config_uses_strict_rollout_parity(config_or_policy_config):
        return 0
    return actions_per_frame


def mot_config_uses_strict_rollout_parity(config_or_policy_config: Any) -> bool:
    """Return whether the experiment data config uses strict rollout-parity targets."""

    data_config = getattr(config_or_policy_config, "data", None)
    sample_config = getattr(data_config, "sample_construction", None)
    if sample_config is None:
        return False
    return (
        _enum_value(getattr(sample_config, "target_alignment", None))
        == SampleTargetAlignment.NEXT_AFTER_CONTEXT.value
        and _enum_value(getattr(sample_config, "rollout_context_policy", None))
        == RolloutContextPolicy.ONE_FRAME.value
    )


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

    route = resolve_mot_runtime_route(policy_config)
    if not route.requires_legacy_block_restore:
        return {
            "policy_variant": "mot",
            "backend": "split_cache" if route.uses_split_cache_rollout else "packed_coupling",
            "route": route.to_report(),
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
        "route": route.to_report(),
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


def _policy_config(config_or_policy_config: Any) -> Any:
    return getattr(config_or_policy_config, "policy_variant", config_or_policy_config)


def _looks_like_mot_policy_config(policy_config: Any) -> bool:
    raw_name = getattr(policy_config, "name", None)
    if raw_name is None:
        return hasattr(policy_config, "runtime_mode") or hasattr(policy_config, "current_block_coupling")
    return _enum_value(raw_name) == PolicyVariantName.MOT.value


def _coerce_runtime_mode(
    raw_value: object,
    *,
    explicit_coupling: CurrentBlockCoupling | None,
) -> MoTRuntimeMode:
    if raw_value is None:
        if explicit_coupling is not None:
            return MoTRuntimeMode.NON_JOINT_TWO_STREAM
        return MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE
    return MoTRuntimeMode(_enum_value(raw_value))


def _coerce_current_block_coupling(raw_value: object) -> CurrentBlockCoupling | None:
    if raw_value is None:
        return None
    return CurrentBlockCoupling(_enum_value(raw_value))


def _resolve_current_block_coupling(
    *,
    runtime_mode: MoTRuntimeMode,
    explicit_coupling: CurrentBlockCoupling | None,
) -> CurrentBlockCoupling:
    if explicit_coupling is not None:
        return explicit_coupling
    if runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
        return CurrentBlockCoupling.JOINT
    return CurrentBlockCoupling.VIDEO_THEN_ACTION


def _enum_value(raw_value: object) -> object:
    return getattr(raw_value, "value", raw_value)
