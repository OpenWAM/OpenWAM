from __future__ import annotations

from collections.abc import Callable

from torch.utils.data import Dataset

from open_wam.configs import DataConfig

from .contracts import WAMSample
from .libero_hdf5 import LiberoOfflineWindowDataset, build_libero_offline_train_val_episode_split
from .lerobot_v2 import LeRobotV2WindowDataset, build_lerobot_train_val_episode_split
from .synthetic import SyntheticWindowDataset


DatasetPairBuilder = Callable[[DataConfig], tuple[Dataset[WAMSample], Dataset[WAMSample]]]

_DATASET_BUILDERS: dict[str, DatasetPairBuilder] = {}


def register_dataset_builder(dataset_type: str, builder: DatasetPairBuilder) -> None:
    """Register one train/val dataset builder for a source type.

    The registry is the extension point collaborators should use when adding a
    new source. The Lightning datamodule depends only on `dataset_type` and does
    not need source-specific conditionals once the builder is registered here.
    """

    _DATASET_BUILDERS[dataset_type] = builder


def build_train_val_datasets(data_config: DataConfig) -> tuple[Dataset[WAMSample], Dataset[WAMSample]]:
    """Build train/val datasets from the config-defined source type."""

    try:
        builder = _DATASET_BUILDERS[data_config.dataset_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_DATASET_BUILDERS))
        raise ValueError(
            f"Unsupported dataset_type '{data_config.dataset_type}'. "
            f"Registered dataset types: {supported}"
        ) from exc
    return builder(data_config)


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
