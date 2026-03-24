"""Lightning entrypoints for the new WAM framework."""

from .datamodule import OpenWAMDataModule, RandomRobotWinDataModule
from .module import OpenWAMLightningModule

__all__ = [
    "OpenWAMDataModule",
    "OpenWAMLightningModule",
    "RandomRobotWinDataModule",
]
