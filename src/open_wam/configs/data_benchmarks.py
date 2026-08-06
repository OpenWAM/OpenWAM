"""Typed presets for directly supported single-dataset adapters."""

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
    ActionTargetReferenceSource,
    ActionTargetRepresentation,
    ActionTargetStateEncoding,
    DataSplit,
    GripperRepresentation,
    LegacyPicklePolicy,
    LatentWindowProfile,
    ReplayStatusPolicy,
    RotationRepresentation,
)

__all__ = [
    "CalvinDataConfig",
    "GenericDataConfig",
    "LiberoDataConfig",
    "RobotWinDataConfig",
]


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
    val_local_root: str | None = None
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


@dataclass(frozen=True)
class RobotWinDataConfig(DataConfig):
    """Default phase-2 data config for the RobotWin stage."""

    dataset_name: str = "robotwin"
    dataset_type: str = "synthetic_robotwin"
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
    val_local_root: str | None = None
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
    val_replay_status_path: str | None = None
    replay_status_policy: ReplayStatusPolicy = ReplayStatusPolicy.SUCCESSFUL_ONLY
    require_replay_status: bool = False
    val_replay_status_policy: ReplayStatusPolicy | None = None
    val_require_replay_status: bool | None = None
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
    val_local_root: str | None = None
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
    language_annotation_pickle_policy: LegacyPicklePolicy = LegacyPicklePolicy.SAFE_ONLY

    def __post_init__(self) -> None:
        super().__post_init__()
        from .enums import coerce_fields

        coerce_fields(
            self,
            enum_fields={
                "language_annotation_pickle_policy": LegacyPicklePolicy,
            },
        )
