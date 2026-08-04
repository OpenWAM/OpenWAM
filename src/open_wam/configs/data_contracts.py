"""Shared action, sampling, and dataset configuration contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import (
    ActionMappingLossMaskMode,
    ActionMappingMode,
    ActionMappingSamplerMaskMode,
    ActionNormalizationMode,
    ActionTargetReferenceSource,
    ActionTargetRepresentation,
    ActionTargetStateEncoding,
    AnchorPolicy,
    DataSplit,
    GripperRepresentation,
    LatentTemporalLayout,
    LatentWindowProfile,
    PaddedTargetPolicy,
    ReplayStatusPolicy,
    RolloutContextPolicy,
    RotationRepresentation,
    SampleOrderMode,
    SampleStateAnchorMode,
    SampleTargetAlignment,
    SampleWeightMode,
    SegmentContextPolicy,
    TailPaddingPolicy,
    WindowSamplingMode,
    coerce_fields,
)

__all__ = [
    "ActionMappingConfig",
    "ActionNormalizationConfig",
    "ActionSchemaConfig",
    "ActionTargetConfig",
    "CausalPrefixSuffixBucketConfig",
    "DataConfig",
    "GeneralistDynamicsMixtureConfig",
    "SampleConstructionConfig",
    "ViewLayoutConfig",
]


@dataclass(frozen=True)
class ViewLayoutConfig:
    """Placement of one source camera inside the canonical RGB canvas."""

    source_name: str
    canonical_name: str
    top: int
    left: int
    height: int
    width: int


@dataclass(frozen=True)
class ActionSchemaConfig:
    """Dataset-level action and state schema.

    Attributes:
        action_dim:
            Final action dimension exposed to all head variants.
        action_horizon:
            Number of action steps predicted for one model call.
        state_dim:
            Current state feature dimension.
        state_horizon:
            Number of state steps attached to one model call.
    """

    action_dim: int
    action_horizon: int
    state_dim: int
    state_horizon: int = 1


@dataclass(frozen=True)
class ActionNormalizationConfig:
    """Optional numeric normalization for action targets before or after mapping."""

    mode: ActionNormalizationMode = ActionNormalizationMode.NONE
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    q01: tuple[float, ...] = ()
    q99: tuple[float, ...] = ()
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={"mode": ActionNormalizationMode},
            transforms={
                "mean": _float_tuple,
                "std": _float_tuple,
                "q01": _float_tuple,
                "q99": _float_tuple,
                "lower": _float_tuple,
                "upper": _float_tuple,
            },
        )
        if self.mode == ActionNormalizationMode.QUANTILES and len(self.q01) != len(self.q99):
            raise ValueError("Quantile action normalization requires `q01` and `q99` to have the same length.")
        if self.mode == ActionNormalizationMode.GAUSSIAN:
            if not self.mean or not self.std:
                raise ValueError("Gaussian action normalization requires non-empty `mean` and `std` values.")
            if len(self.mean) != len(self.std):
                raise ValueError("Gaussian action normalization requires `mean` and `std` to match.")
            for index, value in enumerate(self.std):
                if float(value) <= 0.0:
                    raise ValueError(f"Gaussian action normalization std must be positive at index {index}.")
        if self.mode == ActionNormalizationMode.JOINT_LIMITS:
            if not self.lower or not self.upper:
                raise ValueError("Joint-limit action normalization requires non-empty `lower` and `upper` values.")
            if len(self.lower) != len(self.upper):
                raise ValueError("Joint-limit action normalization requires `lower` and `upper` to match.")
            for index, (lower, upper) in enumerate(zip(self.lower, self.upper, strict=True)):
                if float(upper) <= float(lower):
                    raise ValueError(f"Joint limit upper bound must exceed lower bound at index {index}.")


def _float_tuple(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _coerce_action_normalization_config(value: ActionNormalizationConfig | dict[str, object]) -> ActionNormalizationConfig:
    if isinstance(value, ActionNormalizationConfig):
        return value
    if not isinstance(value, dict):
        raise ValueError("Expected action normalization config to be a mapping.")
    return ActionNormalizationConfig(
        mode=value.get("mode", ActionNormalizationMode.NONE),
        mean=tuple(float(item) for item in value.get("mean", ())),
        std=tuple(float(item) for item in value.get("std", ())),
        q01=tuple(float(item) for item in value.get("q01", ())),
        q99=tuple(float(item) for item in value.get("q99", ())),
        lower=tuple(float(item) for item in value.get("lower", ())),
        upper=tuple(float(item) for item in value.get("upper", ())),
        clip_min=value.get("clip_min"),
        clip_max=value.get("clip_max"),
    )


@dataclass(frozen=True)
class ActionMappingConfig:
    """Map dataset-native action vectors into model-facing action dimensions.

    `mode=none` preserves the existing data contract. `sparse_canvas` and
    `pad_and_reorder` build a target vector whose active channels are selected
    by `source_to_target_indices`; the returned action mask marks only those
    active target dimensions as valid.
    """

    mode: ActionMappingMode = ActionMappingMode.NONE
    source_dim: int | None = None
    target_dim: int | None = None
    source_to_target_indices: tuple[int, ...] = ()
    active_target_indices: tuple[int, ...] = ()
    inactive_value: float = 0.0
    loss_mask_mode: ActionMappingLossMaskMode = ActionMappingLossMaskMode.SOURCE_MASK
    sampler_mask_mode: ActionMappingSamplerMaskMode = ActionMappingSamplerMaskMode.NONE
    normalization: ActionNormalizationConfig = field(default_factory=ActionNormalizationConfig)

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "mode": ActionMappingMode,
                "loss_mask_mode": ActionMappingLossMaskMode,
                "sampler_mask_mode": ActionMappingSamplerMaskMode,
            },
        )
        if self.mode == ActionMappingMode.NONE:
            return
        if self.source_dim is None or self.source_dim <= 0:
            raise ValueError("Action mapping requires a positive `source_dim`.")
        if self.target_dim is None or self.target_dim <= 0:
            raise ValueError("Action mapping requires a positive `target_dim`.")
        if len(self.source_to_target_indices) != self.source_dim:
            raise ValueError(
                "Action mapping requires exactly one target index per source channel, "
                f"got source_dim={self.source_dim}, indices={len(self.source_to_target_indices)}."
            )
        if len(set(self.source_to_target_indices)) != len(self.source_to_target_indices):
            raise ValueError("Action mapping target indices must be unique.")
        for target_index in self.source_to_target_indices:
            if target_index < 0 or target_index >= self.target_dim:
                raise ValueError(
                    f"Action mapping target index {target_index} is outside target_dim={self.target_dim}."
                )
        if self.active_target_indices:
            for target_index in self.active_target_indices:
                if target_index < 0 or target_index >= self.target_dim:
                    raise ValueError(
                        f"Active target index {target_index} is outside target_dim={self.target_dim}."
                    )


@dataclass(frozen=True)
class ActionTargetConfig:
    """How raw dataset supervision is exposed as action targets.

    Attributes:
        representation:
            Target representation consumed by action heads. `raw` forwards the
            dataset-provided action tensor unchanged. Other modes may derive the
            target from state, proprio, or decoded video as the project grows.
        source_key:
            Row key used when `representation == "raw"`.
        pose_source_key:
            Row key used when the target is derived from pose state rather than
            from the dataset action tensor itself.
        state_encoding:
            How the pose source tensor should be unpacked. The current LIBERO
            path uses `eef_pos_axisangle_gripper_2d`, i.e. `[xyz, axisangle, gripper]`.
        reference_source:
            Which observed state anchors the relative pose target. The default
            and currently supported value is `anchor_state`.
        rotation_representation:
            Rotation parameterization exposed in the action target. The current
            WM default is `axis_angle`, yielding a target such as
            `[xyz, axis_angle, gripper]` when `include_gripper` is enabled.
            `continuous_6d` exposes the first two rotation-matrix columns.
        include_gripper:
            Whether to append gripper state to pose-derived targets.
        gripper_representation:
            How multi-channel gripper state should be exposed when
            `include_gripper` is enabled. `first_channel` and `all_channels`
            expose measured state, while `action_command` copies the scalar
            gripper command directly from the raw dataset action tensor.
        gripper_action_index:
            Channel index used when `gripper_representation == action_command`.
            The default `-1` means "take the last action dimension".
        gripper_position_source_key:
            Row key used when absolute joint-position targets expose measured
            gripper qpos with `gripper_representation=first_channel` or
            `all_channels`.
        joint_position_source_key:
            Row key used when `representation == absolute_joint_position`.
            This should expose measured joint positions, e.g. LIBERO
            `robot0_joint_pos`, not relative action deltas.
        joint_position_normalization:
            Optional normalization applied to joint-position channels before
            the configured gripper target is appended.
        normalization:
            Optional normalization applied to the final model-facing target
            vector for representations that forward raw target columns. This is
            inverted by rollout adapters before simulator execution.
    """

    representation: ActionTargetRepresentation = ActionTargetRepresentation.RAW
    source_key: str = "actions"
    pose_source_key: str = "state"
    state_encoding: ActionTargetStateEncoding = ActionTargetStateEncoding.IDENTITY
    reference_source: ActionTargetReferenceSource = ActionTargetReferenceSource.ANCHOR_STATE
    rotation_representation: RotationRepresentation = RotationRepresentation.AXIS_ANGLE
    include_gripper: bool = True
    gripper_representation: GripperRepresentation = GripperRepresentation.FIRST_CHANNEL
    gripper_action_index: int = -1
    gripper_position_source_key: str = "robot0_gripper_qpos"
    joint_position_source_key: str = "robot0_joint_pos"
    joint_position_normalization: ActionNormalizationConfig = field(default_factory=ActionNormalizationConfig)
    normalization: ActionNormalizationConfig = field(default_factory=ActionNormalizationConfig)

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "representation": ActionTargetRepresentation,
                "state_encoding": ActionTargetStateEncoding,
                "reference_source": ActionTargetReferenceSource,
                "rotation_representation": RotationRepresentation,
                "gripper_representation": GripperRepresentation,
            },
            transforms={
                "joint_position_normalization": _coerce_action_normalization_config,
                "normalization": _coerce_action_normalization_config,
            },
        )


@dataclass(frozen=True)
class CausalPrefixSuffixBucketConfig:
    """One `(observed_prefix, future_suffix)` bucket for causal video training."""

    observed_frames: int
    future_frames: int

    @property
    def total_frames(self) -> int:
        return int(self.observed_frames) + int(self.future_frames)


@dataclass(frozen=True)
class SampleConstructionConfig:
    """How one latent training sample is constructed from a source segment."""

    mode: WindowSamplingMode = WindowSamplingMode.FULL_SEGMENT
    anchor_policy: AnchorPolicy = AnchorPolicy.RANDOM_VALID
    num_frames: int = 4
    action_horizon: int = 16
    state_horizon: int = 1
    state_anchor_mode: SampleStateAnchorMode = SampleStateAnchorMode.PROPRIO_CONTEXT_FRAME
    frame_stride: int = 1
    chunk_size: int = 1
    window_size: int = 1
    predict_blocks_per_sample: int = 1
    randomize_geometry: bool = True
    # Compatibility gate for strict next-after-context configs that
    # intentionally randomize chunk/window geometry.
    allow_next_after_context_random_geometry: bool = False
    segment_frames: int | None = None
    segment_min_frames: int | None = None
    segment_max_frames: int | None = None
    segment_length_stride: int = 1
    segment_locality_block_size: int = 4
    # When True (uniform_segment mode only): draw segment_length from the
    # candidate list with an unseeded RNG so each __getitem__ call picks
    # a fresh length even for the same virtual index. Default False keeps
    # PR88's deterministic-per-index behavior for reproducibility.
    randomize_segment_length: bool = False
    # When True (uniform_segment mode only): ignore the virtual index's
    # deterministic latent_start and draw a fresh valid start per __getitem__
    # call. This is useful with randomize_segment_length for true segment
    # augmentation while keeping the virtual index as a trajectory sampler.
    randomize_segment_start: bool = False
    # When True (uniform_segment mode only): only sample segments fully inside
    # the source latent span. This disables tail zero-order-hold / action-mask
    # padding for trajectories shorter than the requested segment.
    require_full_segment: bool = False
    # Number of virtual frames to expose before trajectory frame 0 in
    # uniform_segment mode. These frames repeat the first stored latent and are
    # useful for fixed-geometry cold-start training without rewriting latent
    # datasets on disk.
    start_padding_frames: int = 0
    # Expected raw-frame offset used when precomputing `condition_latent`
    # payloads. For single-frame context experiments, -1 means the condition is
    # the raw frame immediately before the sampled latent source span.
    condition_source_frame_offset: int = 0
    # `legacy` preserves historical fixed-segment behavior. `next_after_context`
    # is the strict rollout-parity contract: materialize context before the
    # target horizon, mask it from supervision, and supervise only the next
    # `segment_frames` latent frames.
    target_alignment: SampleTargetAlignment = SampleTargetAlignment.LEGACY
    # Strict rollout-parity context source. `one_frame` matches live rollout
    # bootstrap; `rollout_history` prepends the configured inference history
    # outside the supervised target horizon.
    rollout_context_policy: RolloutContextPolicy = RolloutContextPolicy.ONE_FRAME
    rollout_context_frames: int | None = None
    # Hierarchical fixed-segment context reservation. Prefix frames are
    # prepended outside `segment_frames`, so the configured segment length
    # remains the target horizon. `none` keeps legacy behavior; `fixed` uses
    # `context_prefix_frames`; `rollout_history` derives the prefix from sampled
    # chunk/window geometry.
    context_prefix_policy: SegmentContextPolicy = SegmentContextPolicy.NONE
    context_prefix_frames: int = 0
    tail_padding_policy: TailPaddingPolicy = TailPaddingPolicy.ZERO_ORDER_HOLD
    padded_target_policy: PaddedTargetPolicy = PaddedTargetPolicy.MASK_LOSS
    # Hierarchical fixed-segment sampler factors. `task_start_power=0.5`
    # preserves the historical midpoint between task-uniform and
    # transition-uniform parallel-stream fixed-128 sampling.
    task_start_power: float = 0.5
    demo_count_power: float = 0.0
    trajectory_start_power: float = 1.0
    sample_weight_mode: SampleWeightMode = SampleWeightMode.UNIFORM
    sample_order_mode: SampleOrderMode = SampleOrderMode.EPOCH_ORDER
    # Used by task_virtual_start_count_power: task mass is proportional to the
    # number of eligible virtual starts raised to this power. 0 is task-uniform,
    # 1 is transition-uniform.
    sample_weight_length_power: float = 1.0
    sample_weight_min: float | None = None
    sample_weight_max: float | None = None
    causal_prefix_suffix_buckets: tuple[CausalPrefixSuffixBucketConfig, ...] = field(default_factory=tuple)

    @property
    def effective_causal_prefix_suffix_buckets(self) -> tuple[CausalPrefixSuffixBucketConfig, ...]:
        """Return explicit causal buckets or the dataset fallback bucket."""

        if self.causal_prefix_suffix_buckets:
            return self.causal_prefix_suffix_buckets
        observed_frames = max(1, int(self.num_frames) // 2)
        return (
            CausalPrefixSuffixBucketConfig(
                observed_frames=observed_frames,
                future_frames=int(self.num_frames) - observed_frames,
            ),
        )

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "mode": WindowSamplingMode,
                "anchor_policy": AnchorPolicy,
                "state_anchor_mode": SampleStateAnchorMode,
                "sample_weight_mode": SampleWeightMode,
                "sample_order_mode": SampleOrderMode,
                "target_alignment": SampleTargetAlignment,
                "rollout_context_policy": RolloutContextPolicy,
                "context_prefix_policy": SegmentContextPolicy,
                "tail_padding_policy": TailPaddingPolicy,
                "padded_target_policy": PaddedTargetPolicy,
            },
        )
        if self.segment_frames is not None and self.segment_frames <= 0:
            raise ValueError("`sample_construction.segment_frames` must be positive when set.")
        if self.sample_weight_min is not None and self.sample_weight_min < 0:
            raise ValueError("`sample_construction.sample_weight_min` must be non-negative when set.")
        if self.sample_weight_max is not None and self.sample_weight_max <= 0:
            raise ValueError("`sample_construction.sample_weight_max` must be positive when set.")
        if self.sample_weight_length_power < 0:
            raise ValueError("`sample_construction.sample_weight_length_power` must be non-negative.")
        if not math.isfinite(float(self.task_start_power)) or self.task_start_power < 0:
            raise ValueError("`sample_construction.task_start_power` must be finite and non-negative.")
        if not math.isfinite(float(self.demo_count_power)):
            raise ValueError("`sample_construction.demo_count_power` must be finite.")
        if not math.isfinite(float(self.trajectory_start_power)) or self.trajectory_start_power < 0:
            raise ValueError("`sample_construction.trajectory_start_power` must be finite and non-negative.")
        if (
            self.sample_weight_min is not None
            and self.sample_weight_max is not None
            and self.sample_weight_min > self.sample_weight_max
        ):
            raise ValueError("`sample_construction.sample_weight_min` cannot exceed `sample_weight_max`.")
        if self.segment_min_frames is not None and self.segment_min_frames <= 0:
            raise ValueError("`sample_construction.segment_min_frames` must be positive when set.")
        if self.segment_max_frames is not None and self.segment_max_frames <= 0:
            raise ValueError("`sample_construction.segment_max_frames` must be positive when set.")
        if (
            self.segment_min_frames is not None
            and self.segment_max_frames is not None
            and self.segment_min_frames > self.segment_max_frames
        ):
            raise ValueError("`sample_construction.segment_min_frames` cannot exceed `segment_max_frames`.")
        if self.segment_length_stride <= 0:
            raise ValueError("`sample_construction.segment_length_stride` must be positive.")
        if self.segment_locality_block_size <= 0:
            raise ValueError("`sample_construction.segment_locality_block_size` must be positive.")
        if self.start_padding_frames < 0:
            raise ValueError("`sample_construction.start_padding_frames` must be non-negative.")
        if not isinstance(self.condition_source_frame_offset, int):
            raise ValueError("`sample_construction.condition_source_frame_offset` must be an integer.")
        if self.rollout_context_frames is not None and int(self.rollout_context_frames) <= 0:
            raise ValueError("`sample_construction.rollout_context_frames` must be positive or null.")
        if self.context_prefix_frames < 0:
            raise ValueError("`sample_construction.context_prefix_frames` must be non-negative.")
        if self.mode == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT:
            if self.segment_frames is None:
                raise ValueError(
                    "`sample_construction.segment_frames` is required when "
                    "`sample_construction.mode=hierarchical_fixed_segment`."
                )
            if self.segment_min_frames is not None or self.segment_max_frames is not None:
                raise ValueError(
                    "`hierarchical_fixed_segment` uses `segment_frames`; do not set "
                    "`segment_min_frames` or `segment_max_frames`."
                )
            if self.randomize_segment_length or self.randomize_segment_start:
                raise ValueError(
                    "`hierarchical_fixed_segment` samples starts through the hierarchical sampler; "
                    "do not set `randomize_segment_length` or `randomize_segment_start`."
                )
            if self.require_full_segment:
                raise ValueError(
                    "`hierarchical_fixed_segment` uses explicit padding policies; "
                    "do not set `require_full_segment`."
                )
            if self.sample_weight_mode != SampleWeightMode.UNIFORM:
                raise ValueError(
                    "`hierarchical_fixed_segment` uses task/trajectory power fields; "
                    "do not set legacy `sample_weight_mode`."
                )
            if self.sample_order_mode != SampleOrderMode.EPOCH_ORDER:
                raise ValueError("`hierarchical_fixed_segment` does not support replacement `sample_order_mode`.")
            if self.tail_padding_policy != TailPaddingPolicy.ZERO_ORDER_HOLD:
                raise ValueError("`hierarchical_fixed_segment` currently supports only zero-order-hold tail padding.")
            if self.padded_target_policy != PaddedTargetPolicy.MASK_LOSS:
                raise ValueError("`hierarchical_fixed_segment` currently supports only masked padded targets.")
            if self.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT:
                if self.chunk_size != 4:
                    raise ValueError(
                        "`target_alignment=next_after_context` currently requires "
                        "`sample_construction.chunk_size=4` to match rollout chunking."
                    )
                if self.randomize_geometry and not self.allow_next_after_context_random_geometry:
                    raise ValueError(
                        "`target_alignment=next_after_context` requires fixed rollout chunking; "
                        "set `sample_construction.randomize_geometry=false` unless "
                        "`allow_next_after_context_random_geometry=true`."
                    )
                if self.start_padding_frames != 0:
                    raise ValueError(
                        "`target_alignment=next_after_context` deprecates virtual head padding; "
                        "set `sample_construction.start_padding_frames=0`."
                    )
                if self.context_prefix_policy != SegmentContextPolicy.NONE or self.context_prefix_frames != 0:
                    raise ValueError(
                        "`target_alignment=next_after_context` uses "
                        "`rollout_context_policy` / `rollout_context_frames`; remove legacy context fields "
                        "`sample_construction.context_prefix_policy` and `sample_construction.context_prefix_frames`."
                    )
        for bucket in self.causal_prefix_suffix_buckets:
            if bucket.observed_frames <= 0 or bucket.future_frames <= 0:
                raise ValueError(
                    "Causal prefix/suffix buckets require positive observed/future lengths, "
                    f"got observed_frames={bucket.observed_frames}, future_frames={bucket.future_frames}."
                )
            if bucket.total_frames > self.num_frames:
                raise ValueError(
                    "Causal prefix/suffix bucket total must not exceed `sample_construction.num_frames`, "
                    f"got bucket_total={bucket.total_frames}, num_frames={self.num_frames}."
                )


@dataclass(frozen=True)
class GeneralistDynamicsMixtureConfig:
    """Optional encoded-dynamics source mixed into generalist training.

    The five weights define the new GJD training paradigm at the data-sample
    level. Dataset wrappers stamp the selected bucket into sample metadata so
    Parallel-stream and dual-expert runtimes can force the corresponding
    joint/FDM/IDM mode.
    """

    train_latent_root: str | None = None
    val_latent_root: str | None = None
    allow_train_latent_root_for_val: bool = False
    real_joint_weight: float = 0.6
    real_action_conditioned_video_weight: float = 0.1
    real_video_conditioned_action_weight: float = 0.1
    counterfactual_action_conditioned_video_weight: float = 0.1
    counterfactual_video_conditioned_action_weight: float = 0.1
    conditional_history_frames: int | None = None
    seed: int = 0
    length_multiplier: float = 1.0

    def __post_init__(self) -> None:
        weights = {
            "real_joint_weight": self.real_joint_weight,
            "real_action_conditioned_video_weight": self.real_action_conditioned_video_weight,
            "real_video_conditioned_action_weight": self.real_video_conditioned_action_weight,
            "counterfactual_action_conditioned_video_weight": self.counterfactual_action_conditioned_video_weight,
            "counterfactual_video_conditioned_action_weight": self.counterfactual_video_conditioned_action_weight,
        }
        for name, value in weights.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"`data.generalist_dynamics_mixture.{name}` must be finite and non-negative.")
        if sum(float(value) for value in weights.values()) <= 0.0:
            raise ValueError("`data.generalist_dynamics_mixture` must contain at least one positive weight.")
        if not isinstance(self.allow_train_latent_root_for_val, bool):
            raise ValueError("`data.generalist_dynamics_mixture.allow_train_latent_root_for_val` must be boolean.")
        if self.conditional_history_frames is not None and int(self.conditional_history_frames) <= 0:
            raise ValueError("`data.generalist_dynamics_mixture.conditional_history_frames` must be positive or null.")
        if not math.isfinite(float(self.length_multiplier)) or float(self.length_multiplier) <= 0.0:
            raise ValueError("`data.generalist_dynamics_mixture.length_multiplier` must be finite and positive.")


@dataclass(frozen=True)
class DataConfig:
    """Shared data-layer config independent from head choice."""

    dataset_name: str
    dataset_type: str
    repo_id: str | None
    local_root: str | None
    val_local_root: str | None
    empty_text_embedding_path: str | None
    latent_root: str | None
    latent_subdir: str
    latent_window_profile: LatentWindowProfile
    split: DataSplit
    cache_dir: str | None
    camera_names: tuple[str, ...]
    latent_camera_names: tuple[str, ...]
    canonical_height: int
    canonical_width: int
    view_layout: tuple[ViewLayoutConfig, ...]
    num_frames: int
    frame_stride: int
    sample_stride: int
    episode_cache_size: int
    train_fraction: float
    split_seed: int
    max_train_episodes: int | None
    max_val_episodes: int | None
    replay_status_path: str | None
    val_replay_status_path: str | None
    replay_status_policy: ReplayStatusPolicy
    require_replay_status: bool
    val_replay_status_policy: ReplayStatusPolicy | None
    val_require_replay_status: bool | None
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    action_schema: ActionSchemaConfig
    action_target: ActionTargetConfig
    action_mapping: ActionMappingConfig
    sample_construction: SampleConstructionConfig
    generalist_dynamics_mixture: GeneralistDynamicsMixtureConfig = field(
        default_factory=GeneralistDynamicsMixtureConfig
    )
    latent_temporal_layout: LatentTemporalLayout = LatentTemporalLayout.WAN_CAUSAL_STRIDE4
    adapter_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "split": DataSplit,
                "latent_window_profile": LatentWindowProfile,
                "latent_temporal_layout": LatentTemporalLayout,
                "replay_status_policy": ReplayStatusPolicy,
            },
            optional_enum_fields={
                "val_replay_status_policy": ReplayStatusPolicy,
            },
        )
        if self.latent_temporal_layout is LatentTemporalLayout.EQUAL_BUCKET_LEGACY:
            raise ValueError(
                "`data.latent_temporal_layout=equal_bucket_legacy` is deprecated and unsupported. "
                "Equal-bucket latent/action alignment silently drops early actions for Wan/LingBot latents. "
                "Use `wan_causal_stride4`, re-encode/rebuild affected metadata if needed, and do not train "
                "or evaluate new runs with the legacy equal-bucket layout."
            )
        if not isinstance(self.adapter_options, Mapping):
            raise TypeError("`data.adapter_options` must be a mapping.")
        object.__setattr__(self, "adapter_options", dict(self.adapter_options))
