"""Dataset selection for local LeRobot latent repositories."""

from __future__ import annotations

from dataclasses import replace

from torch.utils.data import Dataset

from open_wam.configs import BatchingMode, DataConfig, DataSplit, WindowSamplingMode

from .latent_contracts import LatentWAMSample
from .lerobot_v2_latent_base_dataset import FullSegmentLocalLeRobotLatentDataset
from .lerobot_v2_latent_causal_dataset import (
    CausalPrefixSuffixLocalLeRobotLatentDataset,
)
from .lerobot_v2_latent_hierarchical_dataset import (
    HierarchicalFixedSegmentLocalLeRobotLatentDataset,
)
from .lerobot_v2_latent_split import LocalLatentTrainValWindowPlanner
from .lerobot_v2_latent_uniform_dataset import (
    UniformSegmentLocalLeRobotLatentDataset,
)

__all__ = ["build_local_lerobot_latent_train_val_datasets"]


def build_local_lerobot_latent_train_val_datasets(
    data_config: DataConfig,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    window_plan = LocalLatentTrainValWindowPlanner(data_config).plan()
    train_windows = list(window_plan.train_windows)
    val_windows = list(window_plan.val_windows)

    dataset_cls: type[Dataset[LatentWAMSample]]
    if data_config.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT:
        dataset_cls = FullSegmentLocalLeRobotLatentDataset
    elif data_config.sample_construction.mode == WindowSamplingMode.UNIFORM_SEGMENT:
        segment_min_frames = int(
            data_config.sample_construction.segment_min_frames or data_config.num_frames
        )
        segment_max_frames = int(
            data_config.sample_construction.segment_max_frames or segment_min_frames
        )
        if data_config.batching.mode is BatchingMode.STRICT and segment_min_frames != segment_max_frames and (
            data_config.train_batch_size != 1 or data_config.val_batch_size != 1
        ):
            raise ValueError(
                "Uniform segment sampling with variable segment lengths requires train/val batch size 1 because "
                "latent/action tensor lengths vary across examples."
            )
        dataset_cls = UniformSegmentLocalLeRobotLatentDataset
    elif (
        data_config.sample_construction.mode
        == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT
    ):
        dataset_cls = HierarchicalFixedSegmentLocalLeRobotLatentDataset
    elif (
        data_config.sample_construction.mode == WindowSamplingMode.CAUSAL_PREFIX_SUFFIX
    ):
        dataset_cls = CausalPrefixSuffixLocalLeRobotLatentDataset
    else:
        raise ValueError(
            f"Unsupported sample_construction.mode for local latent datasets: "
            f"{data_config.sample_construction.mode!r}"
        )

    val_data_config = replace(data_config, split=DataSplit.VAL)
    return (
        dataset_cls(data_config=data_config, windows=train_windows),
        dataset_cls(data_config=val_data_config, windows=val_windows),
    )
