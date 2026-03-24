"""Raw-video data handling for the new WAM framework."""

from .contracts import WAMBatch, WAMSample, collate_wam_samples, move_wam_batch_to_device
from .factory import build_train_val_datasets, register_dataset_builder
from .lerobot_v2 import LeRobotV2WindowDataset, build_lerobot_train_val_episode_split, load_lerobot_v2_metadata
from .raw_video import (
    CanonicalVideoBatch,
    ConfiguredCanonicalVideoPreprocessor,
    RobotWinCanonicalVideoPreprocessor,
    build_canonical_video_preprocessor,
)
from .synthetic import SyntheticWindowDataset, build_synthetic_batch, build_synthetic_views

__all__ = [
    "CanonicalVideoBatch",
    "ConfiguredCanonicalVideoPreprocessor",
    "LeRobotV2WindowDataset",
    "RobotWinCanonicalVideoPreprocessor",
    "SyntheticWindowDataset",
    "WAMBatch",
    "WAMSample",
    "build_canonical_video_preprocessor",
    "build_lerobot_train_val_episode_split",
    "build_synthetic_batch",
    "build_synthetic_views",
    "build_train_val_datasets",
    "collate_wam_samples",
    "load_lerobot_v2_metadata",
    "move_wam_batch_to_device",
    "register_dataset_builder",
]
