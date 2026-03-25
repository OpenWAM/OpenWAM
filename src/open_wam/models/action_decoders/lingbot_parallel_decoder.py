from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.models.policy_variants.contracts import PolicyInferOutput, PolicyTrainBatch, PolicyTrainOutput

from .base import ActionDecoder, ActionDecoderInferOutput, ActionDecoderTrainOutput, align_policy_features
from open_wam.models.policy_variants.parallel_stream.reference_runtime import data_seq_to_patch


class LingbotParallelActionDecoder(ActionDecoder):
    """Pass-through decoder and exact LingBot joint loss for the parallel-stream runtime."""

    def __init__(self, hidden_size: int, action_dim: int, action_horizon: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.dropout = dropout

    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        latent_pred = policy_output.aux["latent_pred"]
        train_artifacts = policy_output.aux["lingbot_train_artifacts"]
        latent_scheduler = train_artifacts.latent_scheduler
        action_scheduler = train_artifacts.action_scheduler
        input_dict = train_artifacts.input_dict
        action_pred = policy_output.policy_features

        action_pred_5d = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=input_dict["action_dict"]["targets"].shape[-3],
        )
        latent_pred_5d = data_seq_to_patch(
            policy_output.aux["patch_size"],
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )

        batch_frames, num_frames = input_dict["latent_dict"]["timesteps"].shape
        latent_loss_weight = latent_scheduler.training_weight(input_dict["latent_dict"]["timesteps"].flatten()).reshape(
            batch_frames,
            num_frames,
        )
        action_loss_weight = action_scheduler.training_weight(input_dict["action_dict"]["timesteps"].flatten()).reshape(
            batch_frames,
            num_frames,
        )

        latent_loss = F.mse_loss(
            latent_pred_5d.float(),
            input_dict["latent_dict"]["targets"].float().detach(),
            reduction="none",
        )
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        action_loss = F.mse_loss(
            action_pred_5d.float(),
            input_dict["action_dict"]["targets"].float().detach(),
            reduction="none",
        )
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict["action_dict"]["actions_mask"].float()
        action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_mask = input_dict["action_dict"]["actions_mask"].float().permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        loss = latent_loss + action_loss
        return ActionDecoderTrainOutput(
            action_pred=action_pred,
            loss=loss,
            metrics={
                "action_mse": action_loss.detach(),
                "latent_mse": latent_loss.detach(),
                "joint_loss": loss.detach(),
            },
            aux={"decoder": self.__class__.__name__},
        )

    def forward_infer(self, policy_output: PolicyInferOutput) -> ActionDecoderInferOutput:
        action_pred = align_policy_features(policy_output.policy_features, self.action_horizon)
        return ActionDecoderInferOutput(
            action_pred=action_pred,
            aux={"decoder": self.__class__.__name__},
        )
