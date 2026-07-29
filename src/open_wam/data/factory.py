from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset, Sampler

from open_wam.configs import DataConfig

from .calvin_npz import build_calvin_npz_train_val_datasets
from .contracts import WAMSample
from .lerobot_consortium import build_lerobot_consortium_train_val_datasets
from .lerobot_v2 import LeRobotV2WindowDataset, build_lerobot_train_val_episode_split
from .lerobot_video import build_lerobot_v2_video_train_val_datasets
from .libero_hdf5 import (
    LiberoOfflineWindowDataset,
    build_libero_offline_train_val_episode_split,
)
from .mixed_video import build_mixed_video_train_val_datasets
from .registries import (
    DATASET_ADAPTERS,
    register_dataset_builder,
)
from .synthetic import SyntheticWindowDataset


@dataclass(frozen=True)
class DatasetLoaderSpec:
    """Optional dataset-provided loader behavior for one split."""

    sampler: Sampler[int] | None
    shuffle: bool


def build_train_val_datasets(data_config: DataConfig) -> tuple[Dataset[WAMSample], Dataset[WAMSample]]:
    """Build train/val datasets from the config-defined source type."""

    builder = DATASET_ADAPTERS.require_raw_builder(data_config.dataset_type)
    return builder(data_config)


def resolve_dataset_loader_spec(
    dataset: Dataset[WAMSample],
    *,
    split: str,
    world_size: int = 1,
    rank: int = 0,
) -> DatasetLoaderSpec:
    if split == "train":
        build_train_sampler = getattr(dataset, "build_train_sampler", None)
        if callable(build_train_sampler):
            sampler = build_train_sampler(world_size=world_size, rank=rank)
            if sampler is None:
                return DatasetLoaderSpec(sampler=None, shuffle=True)
            return DatasetLoaderSpec(
                sampler=sampler,
                shuffle=False,
            )
        return DatasetLoaderSpec(sampler=None, shuffle=True)
    return DatasetLoaderSpec(sampler=None, shuffle=False)


def _build_synthetic_datasets(data_config: DataConfig) -> tuple[Dataset[WAMSample], Dataset[WAMSample]]:
    return (
        SyntheticWindowDataset(data_config, length=8),
        SyntheticWindowDataset(data_config, length=2),
    )


def _build_lerobot_v2_datasets(data_config: DataConfig) -> tuple[Dataset[WAMSample], Dataset[WAMSample]]:
    # LeRobot-v2 repos typically expose only a train split at the repository
    # level, so we split by episode index locally to keep train/val behavior
    # consistent with the rest of the framework.
    train_episodes, val_episodes = build_lerobot_train_val_episode_split(data_config)
    return (
        LeRobotV2WindowDataset(data_config=data_config, episodes=train_episodes),
        LeRobotV2WindowDataset(data_config=data_config, episodes=val_episodes),
    )


def _build_libero_hdf5_datasets(data_config: DataConfig) -> tuple[Dataset[WAMSample], Dataset[WAMSample]]:
    train_episodes, val_episodes = build_libero_offline_train_val_episode_split(data_config)
    return (
        LiberoOfflineWindowDataset(data_config=data_config, episodes=train_episodes),
        LiberoOfflineWindowDataset(data_config=data_config, episodes=val_episodes),
    )


register_dataset_builder("synthetic_robotwin", _build_synthetic_datasets)
register_dataset_builder("synthetic_multiview", _build_synthetic_datasets)
register_dataset_builder("lerobot_v2", _build_lerobot_v2_datasets)
register_dataset_builder("libero_hdf5", _build_libero_hdf5_datasets)
register_dataset_builder("lerobot_consortium", build_lerobot_consortium_train_val_datasets)
register_dataset_builder("calvin_npz", build_calvin_npz_train_val_datasets)
register_dataset_builder("lerobot_v2_video", build_lerobot_v2_video_train_val_datasets)
register_dataset_builder("mixed_video", build_mixed_video_train_val_datasets)
