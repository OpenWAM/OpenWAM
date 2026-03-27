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
from .latent_contracts import (
    LatentWAMBatch,
    LatentWAMSample,
    collate_latent_wam_samples,
    move_latent_wam_batch_to_device,
)
from .latent_factory import build_train_val_latent_datasets, register_latent_dataset_builder
from .lerobot_v2_latent import (
    LocalLeRobotLatentWindowDataset,
    build_local_lerobot_latent_train_val_datasets,
    discover_local_lerobot_repo_bundles,
)
from .latent_synthetic import SyntheticLatentWindowDataset, build_synthetic_latent_batch
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
    "LatentWAMBatch",
    "LatentWAMSample",
    "LeRobotV2WindowDataset",
    "LocalLeRobotLatentWindowDataset",
    "PoseSequence",
    "RobotWinCanonicalVideoPreprocessor",
    "SyntheticLatentWindowDataset",
    "SyntheticWindowDataset",
    "WAMBatch",
    "WAMSample",
    "build_canonical_video_preprocessor",
    "build_local_lerobot_latent_train_val_datasets",
    "build_libero_offline_train_val_episode_split",
    "build_lerobot_train_val_episode_split",
    "build_relative_pose_targets",
    "build_synthetic_batch",
    "build_synthetic_latent_batch",
    "build_synthetic_views",
    "build_train_val_latent_datasets",
    "build_train_val_datasets",
    "collate_latent_wam_samples",
    "collate_wam_samples",
    "discover_local_lerobot_repo_bundles",
    "load_libero_offline_metadata",
    "load_lerobot_v2_metadata",
    "move_latent_wam_batch_to_device",
    "move_wam_batch_to_device",
    "register_latent_dataset_builder",
    "reconstruct_absolute_pose_targets",
    "register_dataset_builder",
    "state_sequence_to_pose_sequence",
]
