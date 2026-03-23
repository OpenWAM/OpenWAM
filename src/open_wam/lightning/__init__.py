"""Lightning entrypoints for the new WAM framework."""

from .datamodule import RandomRobotWinDataModule
from .module import OpenWAMLightningModule

__all__ = [
    "OpenWAMLightningModule",
    "RandomRobotWinDataModule",
]

