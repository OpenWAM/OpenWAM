from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Lightning is not installed. Install dependencies with `uv sync` first."
        ) from exc

from open_wam.lightning import OpenWAMLightningModule, RandomRobotWinDataModule
from open_wam.utils import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "--config", dest="config", type=str, required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    module = OpenWAMLightningModule(config)
    datamodule = RandomRobotWinDataModule(
        data_config=config.data,
        batch_size=2,
        train_size=8,
        val_size=2,
        num_workers=0,
    )
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        log_every_n_steps=config.trainer.log_every_n_steps,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        enable_checkpointing=config.trainer.enable_checkpointing,
        enable_model_summary=config.trainer.enable_model_summary,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

