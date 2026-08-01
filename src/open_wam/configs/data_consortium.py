"""Configuration contracts for heterogeneous LeRobot dataset consortiums."""

from __future__ import annotations

from dataclasses import dataclass, field

from .data_contracts import (
    ActionMappingConfig,
    ActionSchemaConfig,
    ActionTargetConfig,
    DataConfig,
    SampleConstructionConfig,
    ViewLayoutConfig,
)
from .enums import (
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
    LatentWindowProfile,
    ReplayStatusPolicy,
    coerce_fields,
)

__all__ = [
    "ConsortiumChannelMappingConfig",
    "ConsortiumCloudCacheConfig",
    "ConsortiumEpisodeSelectionConfig",
    "ConsortiumLocalCacheConfig",
    "ConsortiumMemberConfig",
    "LeRobotConsortiumDataConfig",
]


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
    val_local_root: str | None = None
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
    val_replay_status_path: str | None = None
    replay_status_policy: ReplayStatusPolicy = ReplayStatusPolicy.INCLUDE_ALL
    require_replay_status: bool = False
    val_replay_status_policy: ReplayStatusPolicy | None = None
    val_require_replay_status: bool | None = None
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
