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

from open_wam.configs import ExperimentConfig
from open_wam.data import WAMBatch, build_canonical_video_preprocessor, move_wam_batch_to_device
from open_wam.models.action_heads import ActionHeadTrainingBatch, ContractOnlyActionHead, ContractOnlyActionHeadConfig
from open_wam.pipelines import UnifiedWAMPipeline


def _build_action_head(config: ExperimentConfig) -> ContractOnlyActionHead:
    # Phase 2/3 still uses a single placeholder head. Keep the builder small but
    # explicit so future variants have one obvious integration point.
    if config.action_head.name != "contract_only":
        raise ValueError(
            f"Unsupported action head '{config.action_head.name}' in phase-2 Lightning module."
        )
    return ContractOnlyActionHead(
        ContractOnlyActionHeadConfig(
            action_dim=config.action_head.action_dim,
            action_horizon=config.action_head.action_horizon,
            state_dim=config.action_head.state_dim,
            hidden_size=config.action_head.hidden_size,
        )
    )


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
            self.pipeline = UnifiedWAMPipeline(
                action_head=_build_action_head(config),
                backbone_config=config.backbone,
                preprocessor=build_canonical_video_preprocessor(config.data),
            )
            self.save_hyperparameters(ignore=["pipeline"])

        def _head_batch_from_batch(self, batch: WAMBatch) -> ActionHeadTrainingBatch:
            # The Lightning boundary converts the generic data artifact into the
            # action-head contract. From this point on, head implementations
            # should not depend on source-specific dataset objects.
            return ActionHeadTrainingBatch(
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
                batch=self._head_batch_from_batch(batch),
            )
            self.log("train/loss", output.head_output.loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/action_mse", output.head_output.metrics["action_mse"], on_step=True, on_epoch=True)
            return output.head_output.loss

        def validation_step(self, batch: WAMBatch, batch_idx: int) -> None:
            output = self.pipeline.forward_train(
                views=batch.views,
                batch=self._head_batch_from_batch(batch),
            )
            self.log("val/loss", output.head_output.loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/action_mse", output.head_output.metrics["action_mse"], on_step=False, on_epoch=True)

        def transfer_batch_to_device(self, batch: WAMBatch, device: torch.device, dataloader_idx: int):
            return move_wam_batch_to_device(batch, device)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-4)
