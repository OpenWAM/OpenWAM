"""Raw-video data handling for the new WAM framework."""

from .action_transforms import (
    PoseSequence,
    build_relative_pose_targets,
    expected_pose_target_dim,
    reconstruct_absolute_pose_targets,
    state_sequence_to_pose_sequence,
)
from .contracts import WAMBatch, WAMSample, collate_wam_samples, move_wam_batch_to_device
from .factory import build_train_val_datasets, register_dataset_builder
from .libero_hdf5 import LiberoOfflineWindowDataset, build_libero_offline_train_val_episode_split, load_libero_offline_metadata
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
    "expected_pose_target_dim",
    "LiberoOfflineWindowDataset",
    "LeRobotV2WindowDataset",
    "PoseSequence",
    "RobotWinCanonicalVideoPreprocessor",
    "SyntheticWindowDataset",
    "WAMBatch",
    "WAMSample",
    "build_canonical_video_preprocessor",
    "build_libero_offline_train_val_episode_split",
    "build_lerobot_train_val_episode_split",
    "build_relative_pose_targets",
    "build_synthetic_batch",
    "build_synthetic_views",
    "build_train_val_datasets",
    "collate_wam_samples",
    "load_libero_offline_metadata",
    "load_lerobot_v2_metadata",
    "move_wam_batch_to_device",
    "reconstruct_absolute_pose_targets",
    "register_dataset_builder",
    "state_sequence_to_pose_sequence",
]
