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
from open_wam.models.action_heads import ActionHeadTrainingBatch, ContractOnlyActionHead, ContractOnlyActionHeadConfig
from open_wam.pipelines import UnifiedWAMPipeline


def _build_action_head(config: ExperimentConfig) -> ContractOnlyActionHead:
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
            )
            self.save_hyperparameters(ignore=["pipeline"])

        def _views_from_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {
                "cam_high": batch["cam_high"],
                "cam_left_wrist": batch["cam_left_wrist"],
                "cam_right_wrist": batch["cam_right_wrist"],
            }

        def _head_batch_from_batch(self, batch: dict[str, torch.Tensor]) -> ActionHeadTrainingBatch:
            return ActionHeadTrainingBatch(
                actions=batch["actions"],
                action_mask=batch["action_mask"],
                state=batch["state"],
            )

        def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            output = self.pipeline.forward_train(
                views=self._views_from_batch(batch),
                batch=self._head_batch_from_batch(batch),
            )
            self.log("train/loss", output.head_output.loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/action_mse", output.head_output.metrics["action_mse"], on_step=True, on_epoch=True)
            return output.head_output.loss

        def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
            output = self.pipeline.forward_train(
                views=self._views_from_batch(batch),
                batch=self._head_batch_from_batch(batch),
            )
            self.log("val/loss", output.head_output.loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/action_mse", output.head_output.metrics["action_mse"], on_step=False, on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-4)

