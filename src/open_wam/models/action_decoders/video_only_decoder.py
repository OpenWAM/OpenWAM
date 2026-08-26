from __future__ import annotations

from typing import Any

import torch

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.models.action_decoders.base import (
    ActionDecoder,
    ActionDecoderInferOutput,
    ActionDecoderTrainOutput,
    require_decoder_artifact_payload,
)
from open_wam.models.policy_variants.contracts import PolicyInferOutput, PolicyTrainBatch, PolicyTrainOutput
from open_wam.models.policy_variants.video_flow_artifacts import (
    VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
    VideoFlowInferArtifacts,
    VideoFlowTrainArtifacts,
)


def _masked_video_flow_match_loss(
    *,
    flow_pred: torch.Tensor,
    targets: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    future_loss_mask: torch.Tensor,
) -> torch.Tensor:
    per_token_loss = torch.nn.functional.mse_loss(flow_pred.float(), targets.float().detach(), reduction="none")
    timestep_weight = scheduler.training_weight(timesteps.flatten()).reshape(timesteps.shape)
    per_token_loss = per_token_loss * timestep_weight[:, None, :, None, None]
    per_token_loss = per_token_loss * future_loss_mask.float()
    denom = future_loss_mask.float().sum().clamp_min(1.0) * float(flow_pred.shape[1] * flow_pred.shape[3] * flow_pred.shape[4])
    return per_token_loss.sum() / denom


def _masked_video_latent_mse(
    *,
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    future_loss_mask: torch.Tensor,
) -> torch.Tensor:
    per_token = torch.nn.functional.mse_loss(
        predicted_latents.float(),
        target_latents.float(),
        reduction="none",
    )
    per_token = per_token * future_loss_mask.float()
    denom = future_loss_mask.float().sum().clamp_min(1.0) * float(
        predicted_latents.shape[1] * predicted_latents.shape[3] * predicted_latents.shape[4]
    )
    return per_token.sum() / denom


class VideoOnlyActionDecoder(ActionDecoder):
    """Decoder contract adapter for pure video-latent supervision."""

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        action_horizon: int,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del hidden_size, dropout
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.training_config = training_config
        self.inference_config = inference_config

    @property
    def decoder_artifact_contract(self) -> str:
        return VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT

    def _empty_action_prediction(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.action_horizon, self.action_dim, device=device, dtype=dtype)

    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        del batch
        artifacts = require_decoder_artifact_payload(
            policy_output,
            contract=VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
            payload_type=VideoFlowTrainArtifacts,
        )

        latent_loss = _masked_video_flow_match_loss(
            flow_pred=artifacts.flow_pred,
            targets=artifacts.targets,
            timesteps=artifacts.timesteps,
            scheduler=artifacts.scheduler,
            future_loss_mask=artifacts.future_loss_mask,
        )
        latent_mse = _masked_video_latent_mse(
            predicted_latents=artifacts.predicted_latents,
            target_latents=artifacts.target_latents,
            future_loss_mask=artifacts.future_loss_mask,
        )
        weighted_latent_loss = latent_loss * self.training_config.objective_weight("latent")
        return ActionDecoderTrainOutput(
            action_pred=self._empty_action_prediction(
                artifacts.predicted_latents.shape[0],
                device=artifacts.predicted_latents.device,
                dtype=artifacts.predicted_latents.dtype,
            ),
            loss=weighted_latent_loss,
            metrics={
                "latent_mse": latent_mse.detach(),
                "video_only_flow_loss": latent_loss.detach(),
                "weighted_latent_loss": weighted_latent_loss.detach(),
            },
            aux={
                "predicted_latents": artifacts.predicted_latents.detach(),
                "predicted_video_latents": artifacts.predicted_latents.detach(),
            },
        )

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: Any | None = None,
    ) -> ActionDecoderInferOutput:
        del previous_state
        artifacts = require_decoder_artifact_payload(
            policy_output,
            contract=VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
            payload_type=VideoFlowInferArtifacts,
        )
        predicted_latents = artifacts.predicted_latents
        return ActionDecoderInferOutput(
            action_pred=self._empty_action_prediction(
                predicted_latents.shape[0],
                device=predicted_latents.device,
                dtype=predicted_latents.dtype,
            ),
            next_state=None,
            aux={
                "predicted_latents": predicted_latents,
                "predicted_video_latents": predicted_latents,
            },
        )
