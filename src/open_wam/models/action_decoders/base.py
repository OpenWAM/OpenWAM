from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.models.common.flow_matching import (
    ActionFlowMatchTrainArtifacts,
    build_action_flow_match_inference_scheduler,
    build_action_flow_match_train_artifacts,
)
from open_wam.models.policy_variants.contracts import PolicyInferOutput, PolicyTrainBatch, PolicyTrainOutput


@dataclass
class ActionDecoderTrainOutput:
    """Common train-time action-decoder outputs."""

    action_pred: torch.Tensor
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecoderInferOutput:
    """Common inference-time action-decoder outputs."""

    action_pred: torch.Tensor
    next_state: Any | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderRolloutState:
    """Reusable decoder-owned inference state.

    Sequence-native decoders such as VPP-style action models may cache a full
    predicted action chunk and only refresh it every few environment steps.
    Keeping this state generic lets the pipeline support that behavior without
    turning decoders into hidden stateful singletons.
    """

    action_chunk: torch.Tensor | None = None
    chunk_index: int = 0
    step_within_chunk: int = 0
    cached_sequence_context: dict[str, Any] = field(default_factory=dict)
    goal_context: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectActionDecoderTrainInputs:
    """Optional train-only decoder inputs that bypass the shared visual core."""

    current_frame: torch.Tensor
    input_space: str
    current_action_index: int = 0
    state: torch.Tensor | None = None
    text_context: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def align_policy_features(policy_features: torch.Tensor, target_length: int) -> torch.Tensor:
    """Interpolate `[B, T, D]` features to the action horizon."""

    if policy_features.shape[1] == target_length:
        return policy_features
    return F.interpolate(
        policy_features.transpose(1, 2),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


class ActionDecoder(nn.Module, ABC):
    """Action decoder interface shared across policy variants."""

    def configure_action_sampler_mask(
        self,
        sampler_mask: torch.Tensor | None,
        *,
        inactive_value: float = 0.0,
    ) -> None:
        """Configure optional inference-time channel pinning for mapped action spaces."""

        if sampler_mask is None:
            self._buffers.pop("_action_sampler_mask", None)
            self._action_sampler_inactive_value = float(inactive_value)
            return
        if sampler_mask.ndim != 2:
            raise ValueError(f"Action sampler mask must have shape [H, D], got {tuple(sampler_mask.shape)}.")
        mask = sampler_mask.detach().to(dtype=torch.float32).unsqueeze(0)
        if "_action_sampler_mask" in self._buffers:
            self._buffers["_action_sampler_mask"] = mask
        else:
            self.register_buffer("_action_sampler_mask", mask, persistent=False)
        self._action_sampler_inactive_value = float(inactive_value)

    def _apply_action_sampler_mask(self, actions: torch.Tensor, *, start_index: int = 0) -> torch.Tensor:
        sampler_mask = getattr(self, "_action_sampler_mask", None)
        if sampler_mask is None:
            return actions
        if actions.ndim not in {2, 3}:
            raise ValueError(f"Action sampler mask supports [B, D] or [B, H, D], got {tuple(actions.shape)}.")
        if actions.shape[-1] != sampler_mask.shape[-1]:
            raise ValueError(
                f"Action sampler mask dim {sampler_mask.shape[-1]} does not match action dim {actions.shape[-1]}."
            )
        horizon = actions.shape[-2] if actions.ndim == 3 else 1
        end_index = int(start_index) + int(horizon)
        if end_index > sampler_mask.shape[1]:
            raise ValueError(
                "Action sampler mask horizon is shorter than the requested action slice, "
                f"got mask_horizon={sampler_mask.shape[1]}, start_index={start_index}, horizon={horizon}."
            )
        mask = sampler_mask[:, int(start_index) : end_index].to(device=actions.device, dtype=actions.dtype)
        if actions.ndim == 2:
            mask = mask[:, 0]
        inactive = actions.new_full((), float(getattr(self, "_action_sampler_inactive_value", 0.0)))
        return actions * mask + inactive * (1.0 - mask)

    @abstractmethod
    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        """Decode actions and compute loss."""

    @abstractmethod
    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: Any | None = None,
    ) -> ActionDecoderInferOutput:
        """Decode actions for one inference step."""

    def supports_direct_train_inputs(self) -> bool:
        """Whether this decoder can train directly from dataset visual inputs."""

        return False

    def forward_train_direct(
        self,
        direct_inputs: DirectActionDecoderTrainInputs,
        batch: PolicyTrainBatch,
    ) -> ActionDecoderTrainOutput:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement direct train-time conditioning inputs."
        )


class LinearActionDecoder(ActionDecoder):
    """Reusable flow-matching action decoder for horizon-aligned policy features.

    Unlike the earlier direct-regression version, this decoder now follows the
    same training pattern as LingBot:
    - sample one action timestep per horizon slot
    - corrupt clean actions into `noisy_actions`
    - predict the flow target `noise - action`
    - apply scheduler-derived timestep weights in the loss

    Shapes:
    - `policy_features`: `[B, T_policy, H]`
    - `noisy_actions`: `[B, H_action, D_action]`
    - `timesteps`: `[B, H_action]`
    - predicted flow / clean actions: `[B, H_action, D_action]`
    """

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        action_horizon: int,
        *,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.training_config = training_config
        self.inference_config = inference_config
        self.noisy_action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.timestep_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_dim),
        )

    def _timestep_embedding(
        self,
        timesteps: torch.Tensor,
        *,
        dim: int,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(float(max_period), device=timesteps.device, dtype=torch.float32))
            * torch.arange(start=0, end=half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = timesteps.float().unsqueeze(-1) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def _predict_flow(
        self,
        policy_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        aligned_features = align_policy_features(policy_features, self.action_horizon)
        noisy_action_hidden = self.noisy_action_proj(noisy_actions)
        timestep_hidden = self.timestep_proj(self._timestep_embedding(timesteps, dim=self.hidden_size))
        fused_hidden = aligned_features + noisy_action_hidden + timestep_hidden
        return self.flow_head(fused_hidden)

    def _denoised_actions_from_flow(
        self,
        *,
        noisy_actions: torch.Tensor,
        flow_pred: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler,
    ) -> torch.Tensor:
        sigma = scheduler.sigma_for_timesteps(timesteps.flatten()).reshape(timesteps.shape)
        return noisy_actions - sigma[..., None].to(noisy_actions.dtype) * flow_pred

    def _resolve_train_artifacts(
        self,
        policy_output: PolicyTrainOutput,
        batch: PolicyTrainBatch,
    ) -> ActionFlowMatchTrainArtifacts:
        train_artifacts = policy_output.aux.get("action_flow_match_train_artifacts")
        if train_artifacts is not None:
            return train_artifacts
        return build_action_flow_match_train_artifacts(
            batch.actions,
            batch.action_mask,
            training_config=self.training_config,
        )

    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        train_artifacts = self._resolve_train_artifacts(policy_output, batch)
        flow_pred = self._predict_flow(
            policy_output.policy_features,
            train_artifacts.noisy_actions,
            train_artifacts.timesteps,
        )
        denoised_actions = self._denoised_actions_from_flow(
            noisy_actions=train_artifacts.noisy_actions,
            flow_pred=flow_pred,
            timesteps=train_artifacts.timesteps,
            scheduler=train_artifacts.scheduler,
        )
        timestep_weight = train_artifacts.scheduler.training_weight(train_artifacts.timesteps.flatten()).reshape(
            train_artifacts.timesteps.shape
        )
        per_token_loss = F.mse_loss(flow_pred.float(), train_artifacts.targets.float().detach(), reduction="none")
        per_token_loss = per_token_loss * timestep_weight[:, :, None]
        if train_artifacts.action_mask is not None:
            per_token_loss = per_token_loss * train_artifacts.action_mask.float()
            denom = train_artifacts.action_mask.float().sum(dim=-1).clamp_min(1.0)
        else:
            denom = torch.full(
                train_artifacts.timesteps.shape,
                fill_value=float(self.action_dim),
                device=per_token_loss.device,
            )
        per_horizon_loss = per_token_loss.sum(dim=-1) / denom
        loss = per_horizon_loss.mean()
        action_mse = F.mse_loss(denoised_actions.float(), batch.actions.float(), reduction="none")
        if batch.action_mask is not None:
            action_mse = action_mse * batch.action_mask.float()
            action_denom = batch.action_mask.float().sum().clamp_min(1.0)
        else:
            action_denom = torch.tensor(float(action_mse.numel()), device=action_mse.device)
        action_mse_value = action_mse.sum() / action_denom
        weighted_loss = loss * self.training_config.objective_weight("action")
        return ActionDecoderTrainOutput(
            action_pred=denoised_actions,
            loss=weighted_loss,
            metrics={
                "action_mse": action_mse_value.detach(),
                "action_diffusion_loss": loss.detach(),
                "weighted_action_diffusion_loss": weighted_loss.detach(),
            },
            aux={
                "decoder": self.__class__.__name__,
                "flow_pred": flow_pred.detach(),
            },
        )

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: Any | None = None,
    ) -> ActionDecoderInferOutput:
        del previous_state
        scheduler = build_action_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        sample = torch.randn(
            policy_output.policy_features.shape[0],
            self.action_horizon,
            self.action_dim,
            device=policy_output.policy_features.device,
            dtype=policy_output.policy_features.dtype,
        )
        sample = self._apply_action_sampler_mask(sample)
        for timestep_idx, timestep in enumerate(scheduler.timesteps.to(device=sample.device)):
            timestep_values = torch.full(
                (sample.shape[0], self.action_horizon),
                fill_value=float(timestep),
                device=sample.device,
                dtype=torch.float32,
            )
            flow_pred = self._predict_flow(policy_output.policy_features, sample, timestep_values)
            sample = scheduler.step(
                flow_pred,
                timestep,
                sample,
                to_final=timestep_idx == len(scheduler.timesteps) - 1,
            )
            sample = self._apply_action_sampler_mask(sample)
        return ActionDecoderInferOutput(
            action_pred=sample,
            aux={
                "decoder": self.__class__.__name__,
                "num_inference_steps": torch.tensor(float(len(scheduler.timesteps)), device=sample.device),
            },
        )
