from __future__ import annotations

from dataclasses import dataclass, field

from .enums import (
    AnchorPolicy,
    ActionMappingLossMaskMode,
    ActionMappingMode,
    ActionMappingSamplerMaskMode,
    ActionNormalizationMode,
    ActionTargetReferenceSource,
    ActionTargetRepresentation,
    ActionTargetStateEncoding,
    ConsortiumCacheMode,
    ConsortiumChannelSelectionMode,
    ConsortiumCloudCacheBackend,
    ConsortiumFramePackingOrder,
    ConsortiumMissingChannelPolicy,
    ConsortiumRandomMode,
    ConsortiumSplitMode,
    ConsortiumViewPackingMode,
    ConsortiumWeightMode,
    DataSplit,
    GripperRepresentation,
    LatentWindowProfile,
    ReplayStatusPolicy,
    RotationRepresentation,
    TemporalPositionMode,
    WindowSamplingMode,
    coerce_fields,
)


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
    q01: tuple[float, ...] = ()
    q99: tuple[float, ...] = ()
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self) -> None:
        coerce_fields(self, enum_fields={"mode": ActionNormalizationMode})
        if self.mode == ActionNormalizationMode.QUANTILES and len(self.q01) != len(self.q99):
            raise ValueError("Quantile action normalization requires `q01` and `q99` to have the same length.")


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
        )

@dataclass(frozen=True)
class ConsortiumChannelMappingConfig:
    """Map one source visual key to one canonical consortium slot."""

    source_name: str
    target_slot: str


@dataclass(frozen=True)
class ConsortiumMemberConfig:
    """One dataset member included in a consortium experiment."""

    member_id: str | None = None
    repo_id: str | None = None
    local_root: str | None = None
    enabled: bool = True
    source_group: str | None = None
    include_channels: tuple[str, ...] = ()
    channel_mappings: tuple[ConsortiumChannelMappingConfig, ...] = ()
    sampling_weight: float | None = None

    def __post_init__(self) -> None:
        if self.repo_id is None and self.local_root is None:
            raise ValueError("ConsortiumMemberConfig requires either `repo_id` or `local_root`.")


@dataclass(frozen=True)
class ConsortiumEpisodeSelectionConfig:
    """Explicit episode membership for one member when split mode is manifest-driven."""

    member_id: str
    episode_indices: tuple[int, ...]


@dataclass(frozen=True)
class ConsortiumLocalCacheConfig:
    """Optional local-disk cache for consortium source files."""

    mode: ConsortiumCacheMode = ConsortiumCacheMode.DISABLED
    root: str | None = None

    def __post_init__(self) -> None:
        coerce_fields(self, enum_fields={"mode": ConsortiumCacheMode})


@dataclass(frozen=True)
class ConsortiumCloudCacheConfig:
    """Optional cloud-style cache for consortium source files."""

    mode: ConsortiumCacheMode = ConsortiumCacheMode.DISABLED
    backend: ConsortiumCloudCacheBackend = ConsortiumCloudCacheBackend.FILESYSTEM
    root: str | None = None

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "mode": ConsortiumCacheMode,
                "backend": ConsortiumCloudCacheBackend,
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
    frame_stride: int = 1
    chunk_size: int = 1
    window_size: int = 1
    predict_blocks_per_sample: int = 1
    randomize_geometry: bool = True
    causal_prefix_suffix_buckets: tuple[CausalPrefixSuffixBucketConfig, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "mode": WindowSamplingMode,
                "anchor_policy": AnchorPolicy,
            },
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
class DataConfig:
    """Shared data-layer config independent from head choice."""

    dataset_name: str
    dataset_type: str
    repo_id: str | None
    local_root: str | None
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
    replay_status_policy: ReplayStatusPolicy
    require_replay_status: bool
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    action_schema: ActionSchemaConfig
    action_target: ActionTargetConfig
    action_mapping: ActionMappingConfig
    sample_construction: SampleConstructionConfig

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "split": DataSplit,
                "latent_window_profile": LatentWindowProfile,
                "replay_status_policy": ReplayStatusPolicy,
            },
        )


@dataclass(frozen=True)
class GenericDataConfig(DataConfig):
    """Fallback config for arbitrary multiview sources.

    This keeps the ingestion path open for future datasets whose defaults do not
    match RobotWin or LIBERO. Users can override camera names, layouts, action
    schema, and source type entirely from YAML without adding a new subclass.
    """

    dataset_name: str = "custom"
    dataset_type: str = "synthetic_multiview"
    repo_id: str | None = None
    local_root: str | None = None
    empty_text_embedding_path: str | None = None
    latent_root: str | None = None
    latent_subdir: str = "latents"
    latent_window_profile: LatentWindowProfile = LatentWindowProfile.EXACT_CHUNKED_WINDOW
    split: DataSplit = DataSplit.TRAIN
    cache_dir: str | None = None
    camera_names: tuple[str, ...] = ("camera_0",)
    latent_camera_names: tuple[str, ...] = ("camera_0",)
    canonical_height: int = 384
    canonical_width: int = 320
    view_layout: tuple[ViewLayoutConfig, ...] = field(
        default_factory=lambda: (
            ViewLayoutConfig(
                source_name="camera_0",
                canonical_name="camera_0",
                top=0,
                left=0,
                height=384,
                width=320,
            ),
        )
    )
    num_frames: int = 2
    frame_stride: int = 1
    sample_stride: int = 1
    episode_cache_size: int = 2
    train_fraction: float = 1.0
    split_seed: int = 0
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
    replay_status_path: str | None = None
    replay_status_policy: ReplayStatusPolicy = ReplayStatusPolicy.INCLUDE_ALL
    require_replay_status: bool = False
    train_batch_size: int = 2
    val_batch_size: int = 2
    num_workers: int = 0
    action_schema: ActionSchemaConfig = field(
        default_factory=lambda: ActionSchemaConfig(
            action_dim=7,
            action_horizon=4,
            state_dim=8,
            state_horizon=1,
        )
    )
    action_target: ActionTargetConfig = field(default_factory=ActionTargetConfig)
    action_mapping: ActionMappingConfig = field(default_factory=ActionMappingConfig)
    sample_construction: SampleConstructionConfig = field(default_factory=SampleConstructionConfig)


@dataclass(frozen=True)
class RobotWinDataConfig(DataConfig):
    """Default phase-2 data config for the RobotWin stage."""

    dataset_name: str = "robotwin"
    dataset_type: str = "synthetic_robotwin"
    repo_id: str | None = None
    local_root: str | None = None
    empty_text_embedding_path: str | None = None
    latent_root: str | None = None
    latent_subdir: str = "latents"
    latent_window_profile: LatentWindowProfile = LatentWindowProfile.EXACT_CHUNKED_WINDOW
    split: DataSplit = DataSplit.TRAIN
    cache_dir: str | None = None
    camera_names: tuple[str, ...] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    latent_camera_names: tuple[str, ...] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    canonical_height: int = 384
    canonical_width: int = 320
    view_layout: tuple[ViewLayoutConfig, ...] = field(
        default_factory=lambda: (
            ViewLayoutConfig(
                source_name="cam_high",
                canonical_name="cam_high",
                top=0,
                left=0,
                height=256,
                width=320,
            ),
            ViewLayoutConfig(
                source_name="cam_left_wrist",
                canonical_name="cam_left_wrist",
                top=256,
                left=0,
                height=128,
                width=160,
            ),
            ViewLayoutConfig(
                source_name="cam_right_wrist",
                canonical_name="cam_right_wrist",
                top=256,
                left=160,
                height=128,
                width=160,
            ),
        )
    )
    num_frames: int = 2
    frame_stride: int = 1
    sample_stride: int = 1
    episode_cache_size: int = 1
    train_fraction: float = 1.0
    split_seed: int = 0
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
    replay_status_path: str | None = None
    replay_status_policy: ReplayStatusPolicy = ReplayStatusPolicy.INCLUDE_ALL
    require_replay_status: bool = False
    train_batch_size: int = 2
    val_batch_size: int = 2
    num_workers: int = 0
    action_schema: ActionSchemaConfig = field(
        default_factory=lambda: ActionSchemaConfig(
            action_dim=30,
            action_horizon=32,
            state_dim=30,
            state_horizon=1,
        )
    )
    action_target: ActionTargetConfig = field(default_factory=ActionTargetConfig)
    action_mapping: ActionMappingConfig = field(default_factory=ActionMappingConfig)
    sample_construction: SampleConstructionConfig = field(default_factory=SampleConstructionConfig)


@dataclass(frozen=True)
class LiberoDataConfig(DataConfig):
    """LeRobot-v2 LIBERO dataset config.

    The visual backbone must still see the same LingBot-compatible canvas
    geometry. LIBERO has only two views, so the adapter maps:

    - `image` to the full top row at 256x320
    - `wrist_image` to the bottom row at 128x320

    This preserves the canonical 384x320 RGB canvas and therefore the same
    latent grid of 24x20 expected by the shared video backbone.

    The default action target is a 7D reference-relative EEF target
    `[xyz, axis_angle, gripper_1d_command]`. Pose comes from proprio state,
    while the 1D gripper channel comes from the raw LIBERO action command.
    """

    dataset_name: str = "libero"
    dataset_type: str = "lerobot_v2"
    repo_id: str | None = "physical-intelligence/libero"
    local_root: str | None = None
    empty_text_embedding_path: str | None = None
    latent_root: str | None = None
    latent_subdir: str = "latents"
    latent_window_profile: LatentWindowProfile = LatentWindowProfile.EXACT_CHUNKED_WINDOW
    split: DataSplit = DataSplit.TRAIN
    cache_dir: str | None = None
    camera_names: tuple[str, ...] = (
        "image",
        "wrist_image",
    )
    latent_camera_names: tuple[str, ...] = (
        "image",
        "wrist_image",
    )
    canonical_height: int = 384
    canonical_width: int = 320
    view_layout: tuple[ViewLayoutConfig, ...] = field(
        default_factory=lambda: (
            ViewLayoutConfig(
                source_name="image",
                canonical_name="image",
                top=0,
                left=0,
                height=256,
                width=320,
            ),
            ViewLayoutConfig(
                source_name="wrist_image",
                canonical_name="wrist_image",
                top=256,
                left=0,
                height=128,
                width=320,
            ),
        )
    )
    num_frames: int = 4
    frame_stride: int = 1
    sample_stride: int = 1
    episode_cache_size: int = 2
    train_fraction: float = 1.0
    split_seed: int = 0
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
    replay_status_path: str | None = None
    replay_status_policy: ReplayStatusPolicy = ReplayStatusPolicy.SUCCESSFUL_ONLY
    require_replay_status: bool = False
    train_batch_size: int = 2
    val_batch_size: int = 2
    num_workers: int = 0
    action_schema: ActionSchemaConfig = field(
        default_factory=lambda: ActionSchemaConfig(
            action_dim=7,
            action_horizon=6,
            state_dim=8,
            state_horizon=1,
        )
    )
    action_target: ActionTargetConfig = field(
        default_factory=lambda: ActionTargetConfig(
            representation=ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE,
            source_key="actions",
            pose_source_key="state",
            state_encoding=ActionTargetStateEncoding.EEF_POS_AXISANGLE_GRIPPER_2D,
            reference_source=ActionTargetReferenceSource.ANCHOR_STATE,
            rotation_representation=RotationRepresentation.AXIS_ANGLE,
            include_gripper=True,
            gripper_representation=GripperRepresentation.ACTION_COMMAND,
            gripper_action_index=-1,
        )
    )
    action_mapping: ActionMappingConfig = field(default_factory=ActionMappingConfig)
    sample_construction: SampleConstructionConfig = field(default_factory=SampleConstructionConfig)


@dataclass(frozen=True)
class CalvinDataConfig(DataConfig):
    """Native CALVIN numpy dataset config using static and gripper RGB views."""

    dataset_name: str = "calvin"
    dataset_type: str = "calvin_npz"
    repo_id: str | None = None
    local_root: str | None = None
    empty_text_embedding_path: str | None = None
    latent_root: str | None = None
    latent_subdir: str = "latents"
    latent_window_profile: LatentWindowProfile = LatentWindowProfile.EXACT_CHUNKED_WINDOW
    split: DataSplit = DataSplit.TRAIN
    cache_dir: str | None = None
    camera_names: tuple[str, ...] = (
        "rgb_static",
        "rgb_gripper",
    )
    latent_camera_names: tuple[str, ...] = (
        "rgb_static",
        "rgb_gripper",
    )
    canonical_height: int = 384
    canonical_width: int = 320
    view_layout: tuple[ViewLayoutConfig, ...] = field(
        default_factory=lambda: (
            ViewLayoutConfig(
                source_name="rgb_static",
                canonical_name="rgb_static",
                top=0,
                left=0,
                height=256,
                width=320,
            ),
            ViewLayoutConfig(
                source_name="rgb_gripper",
                canonical_name="rgb_gripper",
                top=256,
                left=0,
                height=128,
                width=320,
            ),
        )
    )
    num_frames: int = 4
    frame_stride: int = 1
    sample_stride: int = 1
    episode_cache_size: int = 2
    train_fraction: float = 1.0
    split_seed: int = 0
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
    replay_status_path: str | None = None
    replay_status_policy: ReplayStatusPolicy = ReplayStatusPolicy.INCLUDE_ALL
    require_replay_status: bool = False
    train_batch_size: int = 2
    val_batch_size: int = 2
    num_workers: int = 0
    action_schema: ActionSchemaConfig = field(
        default_factory=lambda: ActionSchemaConfig(
            action_dim=7,
            action_horizon=6,
            state_dim=15,
            state_horizon=1,
        )
    )
    action_target: ActionTargetConfig = field(
        default_factory=lambda: ActionTargetConfig(
            representation=ActionTargetRepresentation.RAW,
            source_key="rel_actions",
            pose_source_key="robot_obs",
            state_encoding=ActionTargetStateEncoding.IDENTITY,
        )
    )
    action_mapping: ActionMappingConfig = field(default_factory=ActionMappingConfig)
    sample_construction: SampleConstructionConfig = field(default_factory=SampleConstructionConfig)


@dataclass(frozen=True)
class LeRobotConsortiumDataConfig(DataConfig):
    """Config for a multi-repo LeRobot consortium loader.

    The consortium loader keeps the public `WAMSample` / `WAMBatch` contract
    unchanged while allowing one experiment to read from many LeRobot-format
    datasets with heterogeneous camera names, resolutions, and fps metadata.
    """

    dataset_name: str = "lerobot_consortium"
    dataset_type: str = "lerobot_consortium"
    repo_id: str | None = None
    local_root: str | None = None
    empty_text_embedding_path: str | None = None
    latent_root: str | None = None
    latent_subdir: str = "latents"
    latent_window_profile: LatentWindowProfile = LatentWindowProfile.EXACT_CHUNKED_WINDOW
    split: DataSplit = DataSplit.TRAIN
    cache_dir: str | None = None
    camera_names: tuple[str, ...] = (
        "observation.images.slot0",
        "observation.images.slot1",
        "observation.images.slot2",
    )
    latent_camera_names: tuple[str, ...] = (
        "observation.images.slot0",
        "observation.images.slot1",
        "observation.images.slot2",
    )
    canonical_height: int = 384
    canonical_width: int = 320
    view_layout: tuple[ViewLayoutConfig, ...] = field(
        default_factory=lambda: (
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=256,
                width=320,
            ),
            ViewLayoutConfig(
                source_name="observation.images.slot1",
                canonical_name="observation.images.slot1",
                top=256,
                left=0,
                height=128,
                width=160,
            ),
            ViewLayoutConfig(
                source_name="observation.images.slot2",
                canonical_name="observation.images.slot2",
                top=256,
                left=160,
                height=128,
                width=160,
            ),
        )
    )
    num_frames: int = 2
    frame_stride: int = 1
    sample_stride: int = 1
    episode_cache_size: int = 2
    train_fraction: float = 1.0
    split_seed: int = 0
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
    replay_status_path: str | None = None
    replay_status_policy: ReplayStatusPolicy = ReplayStatusPolicy.INCLUDE_ALL
    require_replay_status: bool = False
    train_batch_size: int = 2
    val_batch_size: int = 2
    num_workers: int = 0
    action_schema: ActionSchemaConfig = field(
        default_factory=lambda: ActionSchemaConfig(
            action_dim=7,
            action_horizon=4,
            state_dim=8,
            state_horizon=1,
        )
    )
    action_target: ActionTargetConfig = field(default_factory=ActionTargetConfig)
    action_mapping: ActionMappingConfig = field(default_factory=ActionMappingConfig)
    sample_construction: SampleConstructionConfig = field(default_factory=SampleConstructionConfig)
    consortium_members: tuple[ConsortiumMemberConfig, ...] = ()
    channel_selection_mode: ConsortiumChannelSelectionMode = ConsortiumChannelSelectionMode.ALL_AVAILABLE
    required_channels: tuple[str, ...] = ()
    channel_mappings: tuple[ConsortiumChannelMappingConfig, ...] = ()
    view_packing_mode: ConsortiumViewPackingMode = ConsortiumViewPackingMode.MULTICAM_AS_SLOTS
    frame_packing_order: ConsortiumFramePackingOrder = ConsortiumFramePackingOrder.CAMERA_MAJOR
    missing_channel_policy: ConsortiumMissingChannelPolicy = ConsortiumMissingChannelPolicy.ZERO_FILL
    random_mode: ConsortiumRandomMode = ConsortiumRandomMode.NONE
    weight_mode: ConsortiumWeightMode = ConsortiumWeightMode.PROPORTIONAL_TO_SIZE
    sampling_seed: int = 0
    split_mode: ConsortiumSplitMode = ConsortiumSplitMode.HASH_BY_EPISODE
    explicit_train_episodes: tuple[ConsortiumEpisodeSelectionConfig, ...] = ()
    explicit_val_episodes: tuple[ConsortiumEpisodeSelectionConfig, ...] = ()
    local_cache: ConsortiumLocalCacheConfig = field(default_factory=ConsortiumLocalCacheConfig)
    cloud_cache: ConsortiumCloudCacheConfig = field(default_factory=ConsortiumCloudCacheConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={
                "channel_selection_mode": ConsortiumChannelSelectionMode,
                "view_packing_mode": ConsortiumViewPackingMode,
                "frame_packing_order": ConsortiumFramePackingOrder,
                "missing_channel_policy": ConsortiumMissingChannelPolicy,
                "random_mode": ConsortiumRandomMode,
                "weight_mode": ConsortiumWeightMode,
                "split_mode": ConsortiumSplitMode,
            },
        )
        if self.view_packing_mode == ConsortiumViewPackingMode.MULTICAM_AS_FRAMES:
            if len(self.camera_names) != 1:
                raise ValueError(
                    "`view_packing_mode=multicam_as_frames` requires exactly one `camera_names` slot."
                )
            if len(self.latent_camera_names) != 1:
                raise ValueError(
                    "`view_packing_mode=multicam_as_frames` requires exactly one `latent_camera_names` slot."
                )
            if len(self.view_layout) != 1:
                raise ValueError(
                    "`view_packing_mode=multicam_as_frames` requires exactly one `view_layout` entry."
                )
