from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl  # type: ignore
    except ModuleNotFoundError:
        pl = None  # type: ignore

from open_wam.configs.data import DataConfig
from open_wam.data import (
    WAMSample,
    SyntheticWindowDataset,
    build_train_val_datasets,
    collate_wam_samples,
    resolve_dataset_loader_spec,
)


if pl is None:
    class OpenWAMDataModule:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Lightning is required to use OpenWAMDataModule.")


    class RandomRobotWinDataModule:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Lightning is required to use RandomRobotWinDataModule.")
else:
    class OpenWAMDataModule(pl.LightningDataModule):
        """Lightning datamodule that emits the uniform `WAMBatch` contract.

        This wrapper deliberately stays ignorant of source-specific fields such
        as camera names or parquet columns. All of that belongs in the dataset
        adapter selected by `data.dataset_type`.
        """

        def __init__(self, data_config: DataConfig) -> None:
            super().__init__()
            self.data_config = data_config
            self.train_dataset: Dataset[WAMSample] | None = None
            self.val_dataset: Dataset[WAMSample] | None = None

        def setup(self, stage: str | None = None) -> None:
            self.train_dataset, self.val_dataset = build_train_val_datasets(self.data_config)

        def _resolve_loader_spec(self, dataset: Dataset[WAMSample], *, split: str):
            trainer = getattr(self, "_trainer", None)
            world_size = int(getattr(trainer, "world_size", 1)) if trainer is not None else 1
            rank = int(getattr(trainer, "global_rank", 0)) if trainer is not None else 0
            return resolve_dataset_loader_spec(
                dataset,
                split=split,
                world_size=world_size,
                rank=rank,
            )

        def train_dataloader(self) -> DataLoader:
            assert self.train_dataset is not None
            loader_spec = self._resolve_loader_spec(self.train_dataset, split="train")
            return DataLoader(
                self.train_dataset,
                batch_size=self.data_config.train_batch_size,
                shuffle=loader_spec.shuffle,
                sampler=loader_spec.sampler,
                num_workers=self.data_config.num_workers,
                collate_fn=collate_wam_samples,
            )

        def val_dataloader(self) -> DataLoader:
            assert self.val_dataset is not None
            loader_spec = self._resolve_loader_spec(self.val_dataset, split="val")
            return DataLoader(
                self.val_dataset,
                batch_size=self.data_config.val_batch_size,
                shuffle=loader_spec.shuffle,
                sampler=loader_spec.sampler,
                num_workers=self.data_config.num_workers,
                collate_fn=collate_wam_samples,
            )


    class RandomRobotWinDataModule(OpenWAMDataModule):
        """Backward-compatible alias for the earlier synthetic smoke datamodule."""

        def __init__(
            self,
            data_config: DataConfig,
            train_size: int = 8,
            val_size: int = 2,
            batch_size: int = 2,
            num_workers: int = 0,
        ) -> None:
            super().__init__(data_config)
            self._train_size = train_size
            self._val_size = val_size
            self._batch_size = batch_size
            self._num_workers = num_workers

        def setup(self, stage: str | None = None) -> None:
            self.train_dataset = SyntheticWindowDataset(self.data_config, self._train_size)
            self.val_dataset = SyntheticWindowDataset(self.data_config, self._val_size)

        def train_dataloader(self) -> DataLoader:
            assert self.train_dataset is not None
            return DataLoader(
                self.train_dataset,
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=self._num_workers,
                collate_fn=collate_wam_samples,
            )

        def val_dataloader(self) -> DataLoader:
            assert self.val_dataset is not None
            return DataLoader(
                self.val_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                collate_fn=collate_wam_samples,
            )
