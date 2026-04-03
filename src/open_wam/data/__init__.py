"""Raw-video data handling for the new WAM framework."""

from .action_transforms import (
    PoseSequence,
    build_relative_pose_targets,
    expected_pose_target_dim,
    reconstruct_absolute_pose_targets,
    state_sequence_to_pose_sequence,
)
from .contracts import WAMBatch, WAMSample, collate_wam_samples, move_wam_batch_to_device
from .factory import DatasetLoaderSpec, build_train_val_datasets, register_dataset_builder, resolve_dataset_loader_spec
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
from .lerobot_consortium import (
    LeRobotConsortiumWindowDataset,
    build_lerobot_consortium_catalog,
    build_lerobot_consortium_train_val_datasets,
    discover_local_lerobot_consortium_members,
    resolve_lerobot_consortium_train_val_split,
)
from .lerobot_consortium_report import (
    build_lerobot_consortium_report,
    format_lerobot_consortium_report,
)
from .lerobot_consortium_index import (
    LeRobotConsortiumInventoryRow,
    LeRobotConsortiumRepoTarget,
    build_lerobot_consortium_inventory,
    build_lerobot_consortium_inventory_row,
    infer_lerobot_consortium_source_group,
    load_lerobot_consortium_inventory_rows,
    load_lerobot_consortium_repo_targets,
    render_lerobot_consortium_inventory_markdown,
    write_lerobot_consortium_inventory_csv,
    write_lerobot_consortium_inventory_json,
    write_lerobot_consortium_inventory_markdown,
    write_lerobot_consortium_repo_targets,
)
from .lerobot_consortium_contracts import (
    build_lerobot_consortium_contract_catalog,
    build_lerobot_consortium_contract_catalog_from_inventory_rows,
    write_lerobot_consortium_contract_catalog,
)
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
    "DatasetLoaderSpec",
    "expected_pose_target_dim",
    "LiberoOfflineWindowDataset",
    "LatentWAMBatch",
    "LatentWAMSample",
    "LeRobotConsortiumWindowDataset",
    "LeRobotConsortiumInventoryRow",
    "LeRobotConsortiumRepoTarget",
    "LeRobotV2WindowDataset",
    "LocalLeRobotLatentWindowDataset",
    "PoseSequence",
    "RobotWinCanonicalVideoPreprocessor",
    "SyntheticLatentWindowDataset",
    "SyntheticWindowDataset",
    "WAMBatch",
    "WAMSample",
    "build_canonical_video_preprocessor",
    "build_lerobot_consortium_catalog",
    "build_lerobot_consortium_contract_catalog",
    "build_lerobot_consortium_contract_catalog_from_inventory_rows",
    "build_lerobot_consortium_inventory",
    "build_lerobot_consortium_inventory_row",
    "build_lerobot_consortium_report",
    "build_lerobot_consortium_train_val_datasets",
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
    "discover_local_lerobot_consortium_members",
    "format_lerobot_consortium_report",
    "infer_lerobot_consortium_source_group",
    "discover_local_lerobot_repo_bundles",
    "load_lerobot_consortium_inventory_rows",
    "load_lerobot_consortium_repo_targets",
    "load_libero_offline_metadata",
    "load_lerobot_v2_metadata",
    "move_latent_wam_batch_to_device",
    "move_wam_batch_to_device",
    "register_latent_dataset_builder",
    "reconstruct_absolute_pose_targets",
    "register_dataset_builder",
    "resolve_dataset_loader_spec",
    "resolve_lerobot_consortium_train_val_split",
    "render_lerobot_consortium_inventory_markdown",
    "state_sequence_to_pose_sequence",
    "write_lerobot_consortium_contract_catalog",
    "write_lerobot_consortium_inventory_csv",
    "write_lerobot_consortium_inventory_json",
    "write_lerobot_consortium_inventory_markdown",
    "write_lerobot_consortium_repo_targets",
]
