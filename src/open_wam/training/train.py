from __future__ import annotations

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

from open_wam.lightning import OpenWAMDataModule, OpenWAMLightningModule
from open_wam.training import TrainingRuntime, load_training_cli_config, parse_train_cli, should_use_composable_runtime


def main() -> None:
    try:
        overrides = parse_train_cli()
        config = load_training_cli_config(overrides)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if should_use_composable_runtime(config):
        runtime = TrainingRuntime.from_config(config)
        runtime.run()
        return
    module = OpenWAMLightningModule(config)
    datamodule = OpenWAMDataModule(data_config=config.data)
    trainer_kwargs = dict(
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
    if config.trainer.default_root_dir is not None:
        trainer_kwargs["default_root_dir"] = config.trainer.default_root_dir
    trainer = pl.Trainer(
        **trainer_kwargs,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
