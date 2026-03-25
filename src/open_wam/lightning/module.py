from __future__ import annotations

import torch
from torch import nn

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl  # type: ignore
    except ModuleNotFoundError:
        pl = None  # type: ignore

from open_wam.configs import (
    ExperimentConfig,
)
from open_wam.data import WAMBatch, move_wam_batch_to_device
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.pipelines import build_variant_pipeline_from_config


if pl is None:
    class OpenWAMLightningModule(nn.Module):  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Lightning is required to use OpenWAMLightningModule.")
else:
    class OpenWAMLightningModule(pl.LightningModule):
        """Lightning wrapper around the unified WAM pipeline.

        The module intentionally keeps orchestration outside the backbone and
        action-head implementations. The shared pipeline remains the single
        place where raw views become backbone outputs and then head outputs.
        """

        def __init__(self, config: ExperimentConfig) -> None:
            super().__init__()
            self.config = config
            self.pipeline = build_variant_pipeline_from_config(config)
            self.save_hyperparameters(ignore=["pipeline"])

        def _policy_batch_from_batch(self, batch: WAMBatch) -> PolicyTrainBatch:
            return PolicyTrainBatch(
                actions=batch.actions,
                action_mask=batch.action_mask,
                state=batch.state,
                extra={
                    "task_text": batch.task_text,
                    "metadata": batch.metadata,
                    "state_mask": batch.state_mask,
                },
            )

        def training_step(self, batch: WAMBatch, batch_idx: int) -> torch.Tensor:
            output = self.pipeline.forward_train(
                views=batch.views,
                batch=self._policy_batch_from_batch(batch),
            )
            self.log("train/loss", output.decoder_output.loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/action_mse", output.decoder_output.metrics["action_mse"], on_step=True, on_epoch=True)
            return output.decoder_output.loss

        def validation_step(self, batch: WAMBatch, batch_idx: int) -> None:
            output = self.pipeline.forward_train(
                views=batch.views,
                batch=self._policy_batch_from_batch(batch),
            )
            self.log("val/loss", output.decoder_output.loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/action_mse", output.decoder_output.metrics["action_mse"], on_step=False, on_epoch=True)

        def transfer_batch_to_device(self, batch: WAMBatch, device: torch.device, dataloader_idx: int):
            return move_wam_batch_to_device(batch, device)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-4)
