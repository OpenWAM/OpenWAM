from __future__ import annotations

from typing import Any

import torch

from open_wam.configs import DynamicsObjective, InferenceConfig, TrainingConfig
from open_wam.models.action_decoders.base import (
    ActionDecoder,
    ActionDecoderInferOutput,
    ActionDecoderTrainOutput,
    require_decoder_artifact_payload,
)
from open_wam.models.common.metric_rollups import add_dynamics_objective_metrics
from open_wam.models.policy_variants.contracts import (
    PolicyInferOutput,
    PolicyTrainBatch,
    PolicyTrainOutput,
)
from open_wam.models.policy_variants.dual_expert.decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
    DualExpertInferArtifacts,
    DualExpertTrainArtifacts,
)


def _masked_action_flow_match_loss(
    *,
    flow_pred: torch.Tensor,
    targets: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: Any,
    action_mask: torch.Tensor | None,
    action_dim: int,
) -> torch.Tensor:
    timestep_weight = scheduler.training_weight(timesteps.flatten()).reshape(timesteps.shape)
    per_token_loss = torch.nn.functional.mse_loss(flow_pred.float(), targets.float().detach(), reduction="none")
    per_token_loss = per_token_loss * timestep_weight[:, :, None]
    if action_mask is not None:
        per_token_loss = per_token_loss * action_mask.float()
        denom = action_mask.float().sum(dim=-1).clamp_min(1.0)
    else:
        denom = torch.full(
            timesteps.shape,
            fill_value=float(action_dim),
            device=per_token_loss.device,
        )
    return (per_token_loss.sum(dim=-1) / denom).mean()


def _masked_action_mse(
    *,
    action_pred: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> torch.Tensor:
    action_mse = torch.nn.functional.mse_loss(
        action_pred.float(),
        target_actions.float(),
        reduction="none",
    )
    if action_mask is not None:
        action_mse = action_mse * action_mask.float()
        action_denom = action_mask.float().sum().clamp_min(1.0)
    else:
        action_denom = torch.tensor(float(action_mse.numel()), device=action_mse.device)
    return action_mse.sum() / action_denom


_POSE_BLOCK_DIMS = 10
_BLOCK_GROUPS = (("xyz", 0, 3), ("rot6", 3, 9), ("grip", 9, 10))


def _masked_action_mse_by_group(
    *,
    action_pred: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    action_dim: int,
) -> dict[str, torch.Tensor]:
    """Report per-pose-group MSE without changing the loss.

    The repeated [xyz(3), rot6(6), gripper(1)] convention is used only when
    the action width is a positive multiple of ten; other widths are untouched.
    """
    if action_dim <= 0 or action_dim % _POSE_BLOCK_DIMS:
        return {}
    squared = torch.nn.functional.mse_loss(
        action_pred.float(), target_actions.float(), reduction="none"
    )
    if action_mask is not None:
        mask = action_mask.float()
        if mask.shape != squared.shape:
            mask = mask.expand_as(squared)
        squared = squared * mask
    else:
        mask = None
    blocks = action_dim // _POSE_BLOCK_DIMS
    out: dict[str, torch.Tensor] = {}
    for name, lo, hi in (*_BLOCK_GROUPS, ("pose", 0, 9)):
        cols = [block * _POSE_BLOCK_DIMS + index for block in range(blocks) for index in range(lo, hi)]
        part = squared[..., cols]
        denom = (
            mask[..., cols].sum().clamp_min(1.0)
            if mask is not None
            else torch.tensor(float(part.numel()), device=part.device)
        )
        out[f"action_mse_{name}"] = part.sum() / denom
    return out


def _masked_video_flow_match_loss(
    *,
    flow_pred: torch.Tensor,
    targets: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: Any,
    future_loss_mask: torch.Tensor,
) -> torch.Tensor:
    per_token_loss = torch.nn.functional.mse_loss(flow_pred.float(), targets.float().detach(), reduction="none")
    timestep_weight = scheduler.training_weight(timesteps.flatten()).reshape(timesteps.shape)
    per_token_loss = per_token_loss * timestep_weight[:, None, :, None, None]
    per_token_loss = per_token_loss * future_loss_mask.float()
    denom = future_loss_mask.float().sum().clamp_min(1.0) * float(
        flow_pred.shape[1] * flow_pred.shape[3] * flow_pred.shape[4]
    )
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


class DualExpertActionDecoder(ActionDecoder):
    """DualExpert-specific decoder/loss adapter.

    The DualExpert policy variant owns action/video runtime orchestration, while this
    decoder owns final supervised outputs and loss accounting.
    """

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
        del hidden_size, inference_config, dropout
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.training_config = training_config

    @property
    def dynamics_metric_namespace(self) -> str:
        return "dual_expert_generalist"

    @property
    def decoder_artifact_contract(self) -> str:
        return DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT

    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        train_artifacts = require_decoder_artifact_payload(
            policy_output,
            contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
            payload_type=DualExpertTrainArtifacts,
        )

        diffusion_loss = _masked_action_flow_match_loss(
            flow_pred=train_artifacts.action.flow_pred,
            targets=train_artifacts.action.targets,
            timesteps=train_artifacts.action.timesteps,
            scheduler=train_artifacts.action.scheduler,
            action_mask=train_artifacts.action.action_mask,
            action_dim=self.action_dim,
        )
        weighted_action_loss = diffusion_loss * self.training_config.objective_weight("action")
        action_mse = _masked_action_mse(
            action_pred=train_artifacts.action.denoised_actions,
            target_actions=batch.actions,
            action_mask=train_artifacts.action.action_mask,
        )

        if train_artifacts.video is not None:
            latent_loss = _masked_video_flow_match_loss(
                flow_pred=train_artifacts.video.flow_pred,
                targets=train_artifacts.video.targets,
                timesteps=train_artifacts.video.timesteps,
                scheduler=train_artifacts.video.scheduler,
                future_loss_mask=train_artifacts.video.future_loss_mask,
            )
            latent_mse = _masked_video_latent_mse(
                predicted_latents=train_artifacts.video.predicted_latents,
                target_latents=train_artifacts.video.target_latents,
                future_loss_mask=train_artifacts.video.future_loss_mask,
            )
            weighted_video_loss = latent_loss * self.training_config.objective_weight("latent")
        else:
            latent_loss = diffusion_loss.new_zeros(())
            latent_mse = diffusion_loss.new_zeros(())
            weighted_video_loss = diffusion_loss.new_zeros(())

        total_loss = weighted_action_loss + weighted_video_loss
        aux: dict[str, Any] = {
            "flow_pred": train_artifacts.action.flow_pred.detach(),
        }
        if train_artifacts.video is not None:
            aux.update(
                {
                    "predicted_latents": train_artifacts.video.predicted_latents.detach(),
                    "predicted_video_latents": train_artifacts.video.predicted_latents.detach(),
                    "future_video_flow_pred": train_artifacts.video.flow_pred.detach(),
                }
            )
        group_mse = _masked_action_mse_by_group(
            action_pred=train_artifacts.action.denoised_actions,
            target_actions=batch.actions,
            action_mask=train_artifacts.action.action_mask,
            action_dim=self.action_dim,
        )
        metrics = {
            "action_mse": action_mse.detach(),
            **{key: value.detach() for key, value in group_mse.items()},
            "action_diffusion_loss": diffusion_loss.detach(),
            "weighted_action_diffusion_loss": weighted_action_loss.detach(),
            "latent_mse": latent_mse.detach(),
            "video_diffusion_loss": latent_loss.detach(),
            "weighted_video_diffusion_loss": weighted_video_loss.detach(),
            "joint_loss": total_loss.detach(),
        }
        # A1 generalist per-mode metrics. Only populated when the variant ran
        # the dual-expert generalist sampler this segment. Metric names intentionally
        # spell out denoised MSE semantics because parallel-stream logs flow-loss sums.
        dynamics_objective = (
            None
            if policy_output.decoder_artifacts is None
            else policy_output.decoder_artifacts.dynamics_objective
        )
        if dynamics_objective is not None:
            action_mask = train_artifacts.action.action_mask
            action_active = (
                (action_mask.float().sum() > 0).to(dtype=torch.float32)
                if action_mask is not None
                else torch.ones((), device=total_loss.device)
            )
            if train_artifacts.video is not None:
                latent_active = (
                    train_artifacts.video.future_loss_mask.float().sum() > 0
                ).to(dtype=torch.float32)
            else:
                latent_active = torch.zeros((), device=total_loss.device)
            add_dynamics_objective_metrics(
                metrics,
                namespace="dual_expert_generalist",
                mode_value=dynamics_objective.value,
                modes=DynamicsObjective,
                action_loss=action_mse,
                latent_loss=latent_mse,
                action_loss_active=action_active,
                latent_loss_active=latent_active,
                action_metric_name="action_denoised_mse_sum",
                latent_metric_name="latent_denoised_mse_sum",
                action_metric_aliases=("action_mse_sum",),
                latent_metric_aliases=("latent_mse_sum",),
            )
        return ActionDecoderTrainOutput(
            action_pred=train_artifacts.action.denoised_actions,
            loss=total_loss,
            metrics=metrics,
            aux=aux,
        )

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: Any | None = None,
    ) -> ActionDecoderInferOutput:
        del previous_state
        infer_artifacts = require_decoder_artifact_payload(
            policy_output,
            contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
            payload_type=DualExpertInferArtifacts,
        )

        aux: dict[str, Any] = {}
        if infer_artifacts.predicted_latents is not None:
            aux["predicted_latents"] = infer_artifacts.predicted_latents
            aux["predicted_video_latents"] = infer_artifacts.predicted_latents
        return ActionDecoderInferOutput(
            action_pred=self._apply_action_sampler_mask(infer_artifacts.action_pred),
            next_state=None,
            aux=aux,
        )
