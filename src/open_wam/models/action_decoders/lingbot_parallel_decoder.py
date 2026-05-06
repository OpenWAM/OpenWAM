from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.configs import JointDenoiseTrainingMode
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
        loss_weights = policy_output.aux.get("loss_weights", {})
        latent_scheduler = train_artifacts.latent_scheduler
        action_scheduler = train_artifacts.action_scheduler
        input_dict = train_artifacts.input_dict
        action_pred = policy_output.policy_features
        configured_latent_loss_weight = float(loss_weights.get("latent", 1.0))
        configured_action_loss_weight = float(loss_weights.get("action", 1.0))

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
        latent_scheduler_weight = latent_scheduler.training_weight(input_dict["latent_dict"]["timesteps"].flatten()).reshape(
            batch_frames,
            num_frames,
        )
        action_scheduler_weight = action_scheduler.training_weight(input_dict["action_dict"]["timesteps"].flatten()).reshape(
            batch_frames,
            num_frames,
        )

        latent_loss = F.mse_loss(
            latent_pred_5d.float(),
            input_dict["latent_dict"]["targets"].float().detach(),
            reduction="none",
        )
        latent_loss = latent_loss * latent_scheduler_weight[:, None, :, None, None]
        latent_loss_mask = input_dict["latent_dict"].get("loss_mask")
        if latent_loss_mask is None:
            latent_loss_mask = torch.ones_like(input_dict["latent_dict"]["targets"])
        latent_loss = latent_loss * latent_loss_mask.float()
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = (
            latent_loss_mask.float().permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).sum(dim=1)
        )
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        action_loss = F.mse_loss(
            action_pred_5d.float(),
            input_dict["action_dict"]["targets"].float().detach(),
            reduction="none",
        )
        action_loss = action_loss * action_scheduler_weight[:, None, :, None, None]
        action_loss_mask = input_dict["action_dict"].get("loss_mask")
        if action_loss_mask is None:
            action_loss_mask = torch.ones_like(input_dict["action_dict"]["targets"])
        effective_action_mask = input_dict["action_dict"]["actions_mask"].float() * action_loss_mask.float()
        action_loss = action_loss * effective_action_mask
        action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_mask = effective_action_mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        weighted_latent_loss = latent_loss * configured_latent_loss_weight
        weighted_action_loss = action_loss * configured_action_loss_weight
        loss = weighted_latent_loss + weighted_action_loss
        metrics = {
            "action_mse": action_loss.detach(),
            "latent_mse": latent_loss.detach(),
            "weighted_action_loss": weighted_action_loss.detach(),
            "weighted_latent_loss": weighted_latent_loss.detach(),
            "joint_loss": loss.detach(),
        }
        joint_denoise_mode = input_dict.get("joint_denoise_training_mode")
        if joint_denoise_mode is not None:
            mode_value = str(joint_denoise_mode)
            metric_device = loss.device
            one = torch.ones((), device=metric_device)
            zero = torch.zeros((), device=metric_device)
            for mode in JointDenoiseTrainingMode:
                active = one if mode_value == mode.value else zero
                prefix = f"joint_denoise/{mode.value}"
                metrics[f"{prefix}/count"] = active.detach()
                metrics[f"{prefix}/action_mse_sum"] = (action_loss * active).detach()
                metrics[f"{prefix}/latent_mse_sum"] = (latent_loss * active).detach()
            metrics["joint_denoise/action_loss_active"] = (
                effective_action_mask.float().sum() > 0
            ).to(dtype=torch.float32).detach()
            metrics["joint_denoise/latent_loss_active"] = (
                latent_loss_mask.float().sum() > 0
            ).to(dtype=torch.float32).detach()
        return ActionDecoderTrainOutput(
            action_pred=action_pred,
            loss=loss,
            metrics=metrics,
            aux={"decoder": self.__class__.__name__},
        )

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: object | None = None,
    ) -> ActionDecoderInferOutput:
        del previous_state
        action_pred = self._apply_action_sampler_mask(align_policy_features(policy_output.policy_features, self.action_horizon))
        return ActionDecoderInferOutput(
            action_pred=action_pred,
            aux={"decoder": self.__class__.__name__},
        )
