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
class DataConfig:
    """Shared data-layer config independent from head choice."""

    dataset_name: str
    dataset_type: str
    repo_id: str | None
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


@dataclass(frozen=True)
class RobotWinDataConfig(DataConfig):
    """Default phase-2 data config for the RobotWin stage."""

    dataset_name: str = "robotwin"
    dataset_type: str = "synthetic_robotwin"
    repo_id: str | None = None
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


@dataclass(frozen=True)
class LiberoDataConfig(DataConfig):
    """LeRobot-v2 LIBERO dataset config.

    The visual backbone must still see the same LingBot-compatible canvas
    geometry. LIBERO has only two views, so the adapter maps:

    - `image` to the full top row at 256x320
    - `wrist_image` to the bottom row at 128x320

    This preserves the canonical 384x320 RGB canvas and therefore the same
    latent grid of 24x20 expected by the shared video backbone.
    """

    dataset_name: str = "libero"
    dataset_type: str = "lerobot_v2"
    repo_id: str | None = "physical-intelligence/libero"
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
