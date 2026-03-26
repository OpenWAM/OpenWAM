from __future__ import annotations

from dataclasses import dataclass, field


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

    representation: str = "raw"
    source_key: str = "actions"
    pose_source_key: str = "state"
    state_encoding: str = "identity"
    reference_source: str = "anchor_state"
    rotation_representation: str = "axis_angle"
    include_gripper: bool = True
    gripper_representation: str = "first_channel"
    gripper_action_index: int = -1


@dataclass(frozen=True)
class DataConfig:
    """Shared data-layer config independent from head choice."""

    dataset_name: str
    dataset_type: str
    repo_id: str | None
    local_root: str | None
    split: str
    cache_dir: str | None
    camera_names: tuple[str, ...]
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
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    action_schema: ActionSchemaConfig
    action_target: ActionTargetConfig


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
    split: str = "train"
    cache_dir: str | None = None
    camera_names: tuple[str, ...] = ("camera_0",)
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
    train_fraction: float = 0.95
    split_seed: int = 0
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
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


@dataclass(frozen=True)
class RobotWinDataConfig(DataConfig):
    """Default phase-2 data config for the RobotWin stage."""

    dataset_name: str = "robotwin"
    dataset_type: str = "synthetic_robotwin"
    repo_id: str | None = None
    local_root: str | None = None
    split: str = "train"
    cache_dir: str | None = None
    camera_names: tuple[str, ...] = (
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
    train_fraction: float = 0.95
    split_seed: int = 0
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
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
    split: str = "train"
    cache_dir: str | None = None
    camera_names: tuple[str, ...] = (
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
    train_fraction: float = 0.95
    split_seed: int = 0
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
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
            representation="eef_pose_relative_to_reference",
            source_key="actions",
            pose_source_key="state",
            state_encoding="eef_pos_axisangle_gripper_2d",
            reference_source="anchor_state",
            rotation_representation="axis_angle",
            include_gripper=True,
            gripper_representation="action_command",
            gripper_action_index=-1,
        )
    )
