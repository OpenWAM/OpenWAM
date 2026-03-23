from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl  # type: ignore
    except ModuleNotFoundError:
        pl = None  # type: ignore

from open_wam.configs.data import DataConfig


class _RandomRobotWinDataset(Dataset):
    """Synthetic RobotWin-like dataset used for Lightning smoke experiments."""

    def __init__(self, data_config: DataConfig, length: int) -> None:
        self.data_config = data_config
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        num_frames = self.data_config.num_frames
        action_schema = self.data_config.action_schema
        return {
            "cam_high": torch.randint(0, 255, (num_frames, 300, 400, 3), dtype=torch.uint8),
            "cam_left_wrist": torch.randint(0, 255, (num_frames, 160, 200, 3), dtype=torch.uint8),
            "cam_right_wrist": torch.randint(0, 255, (num_frames, 160, 200, 3), dtype=torch.uint8),
            "actions": torch.randn(action_schema.action_horizon, action_schema.action_dim),
            "action_mask": torch.ones(action_schema.action_horizon, action_schema.action_dim),
            "state": torch.randn(action_schema.state_horizon, action_schema.state_dim),
        }


def _collate_robotwin(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


if pl is None:
    class RandomRobotWinDataModule:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Lightning is required to use RandomRobotWinDataModule.")
else:
    class RandomRobotWinDataModule(pl.LightningDataModule):
        """Minimal DataModule matching the new experiment config split."""

        def __init__(
            self,
            data_config: DataConfig,
            train_size: int = 8,
            val_size: int = 2,
            batch_size: int = 2,
            num_workers: int = 0,
        ) -> None:
            super().__init__()
            self.data_config = data_config
            self.train_size = train_size
            self.val_size = val_size
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.train_dataset: _RandomRobotWinDataset | None = None
            self.val_dataset: _RandomRobotWinDataset | None = None

        def setup(self, stage: str | None = None) -> None:
            self.train_dataset = _RandomRobotWinDataset(self.data_config, self.train_size)
            self.val_dataset = _RandomRobotWinDataset(self.data_config, self.val_size)

        def train_dataloader(self) -> DataLoader:
            assert self.train_dataset is not None
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=_collate_robotwin,
            )

        def val_dataloader(self) -> DataLoader:
            assert self.val_dataset is not None
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=_collate_robotwin,
            )

