from __future__ import annotations

from dataclasses import asdict
from typing import Any

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
    PolicyVariantName,
    SampleLossWeightMode,
)
from open_wam.configs.enums import serialize_enum_values
from open_wam.data import WAMBatch, move_wam_batch_to_device
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.models.policy_variants.mot.runtime_routing import (
    mot_policy_requires_legacy_split_cache_inference,
)
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training.controls import apply_training_component_controls
from open_wam.training.optim import build_optimizer, build_scheduler


def policy_variant_requires_module_mutating_infer_backend(policy_variant: Any) -> bool:
    """Return whether validation inference would mutate module ownership."""

    policy_name = getattr(policy_variant, "name", None)
    if policy_name not in {PolicyVariantName.MOT, PolicyVariantName.MOT.value}:
        return False
    return mot_policy_requires_legacy_split_cache_inference(policy_variant)


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
            if config.training.sample_loss_weight_mode != SampleLossWeightMode.NONE:
                raise ValueError(
                    "sample_loss_weight_mode is only supported by the composable runtime because the Lightning "
                    "training path receives a scalar reduced decoder loss. Set trainer.runtime=composable or disable "
                    "sample_loss_weight_mode."
                )
            self.pipeline = build_variant_pipeline_from_config(config)
            self.trainability_report = apply_training_component_controls(self.pipeline, config.training)
            self.save_hyperparameters({"config": serialize_enum_values(asdict(config))})

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

        def _select_validation_action_prediction(self, batch: WAMBatch, output) -> torch.Tensor:
            if output.decoder_output.action_pred.shape == batch.actions.shape:
                return output.decoder_output.action_pred
            raw_chunk_action_pred = output.policy_output.aux.get("raw_chunk_action_pred")
            if isinstance(raw_chunk_action_pred, torch.Tensor) and raw_chunk_action_pred.shape == batch.actions.shape:
                return raw_chunk_action_pred
            return output.decoder_output.action_pred

        def _select_validation_video_prediction(self, output) -> torch.Tensor | None:
            target_video_latents = output.visual_outputs.frontend.video_latents
            for source_name in ("predicted_latents", "predicted_video_latents"):
                candidate = output.decoder_output.aux.get(source_name)
                if isinstance(candidate, torch.Tensor) and candidate.shape == target_video_latents.shape:
                    return candidate
            for source_name in ("predicted_latents", "predicted_video_latents"):
                candidate = output.policy_output.aux.get(source_name)
                if isinstance(candidate, torch.Tensor) and candidate.shape == target_video_latents.shape:
                    return candidate
            return None

        def _skip_infer_validation_for_module_mutating_backend(self) -> bool:
            return policy_variant_requires_module_mutating_infer_backend(self.config.policy_variant)

        def training_step(self, batch: WAMBatch, batch_idx: int) -> torch.Tensor:
            output = self.pipeline.forward_train(
                views=batch.views,
                batch=self._policy_batch_from_batch(batch),
            )
            self.log("train/loss", output.decoder_output.loss, on_step=True, on_epoch=True, prog_bar=True)
            for metric_name, metric_value in output.decoder_output.metrics.items():
                self.log(f"train/{metric_name}", metric_value, on_step=True, on_epoch=True, prog_bar=(metric_name == "action_mse"))
            return output.decoder_output.loss

        def validation_step(self, batch: WAMBatch, batch_idx: int) -> None:
            train_output = self.pipeline.forward_train(
                views=batch.views,
                batch=self._policy_batch_from_batch(batch),
            )
            self.log("val/loss", train_output.decoder_output.loss, on_step=False, on_epoch=True, prog_bar=True)
            for metric_name, metric_value in train_output.decoder_output.metrics.items():
                self.log(f"val/{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=(metric_name == "action_mse"))

            if self._skip_infer_validation_for_module_mutating_backend():
                self.log(
                    "val/infer_skipped_module_mutating_backend",
                    batch.actions.new_tensor(1.0),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                return

            infer_output = self.pipeline.forward_infer_step(
                batch.views,
                PolicyInferContext(
                    state=batch.state,
                    extra={
                        "task_text": batch.task_text,
                        "metadata": batch.metadata,
                        "allow_mot_legacy_backend_restore": False,
                    },
                ),
            )
            action_prediction = self._select_validation_action_prediction(batch, infer_output)
            if action_prediction.shape == batch.actions.shape:
                action_mse = torch.nn.functional.mse_loss(
                    action_prediction.float(),
                    batch.actions.float(),
                    reduction="none",
                )
                if batch.action_mask is not None:
                    action_mse = action_mse * batch.action_mask.float()
                    action_denom = batch.action_mask.float().sum().clamp_min(1.0)
                else:
                    action_denom = torch.tensor(float(action_mse.numel()), device=action_mse.device)
                self.log(
                    "val/infer_action_mse",
                    action_mse.sum() / action_denom,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

            video_prediction = self._select_validation_video_prediction(infer_output)
            if video_prediction is not None:
                target_video_latents = infer_output.visual_outputs.frontend.video_latents
                self.log(
                    "val/infer_video_latent_mse",
                    torch.nn.functional.mse_loss(video_prediction.float(), target_video_latents.float()),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        def transfer_batch_to_device(self, batch: WAMBatch, device: torch.device, dataloader_idx: int):
            return move_wam_batch_to_device(batch, device)

        def configure_optimizers(self):
            optimizer = build_optimizer(self, self.config.training)
            scheduler = build_scheduler(optimizer, self.config.training)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
