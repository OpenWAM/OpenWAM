from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.models.action_decoders.base import (
    ActionDecoderInferOutput,
    ActionDecoderTrainOutput,
    DecoderRolloutState,
    DirectActionDecoderTrainInputs,
)
from open_wam.models.action_decoders.sequence_base import SequenceActionDecoder
from open_wam.models.common.flow_matching import (
    build_action_flow_match_inference_scheduler,
    build_action_flow_match_train_artifacts,
)
from open_wam.models.policy_variants.contracts import (
    DecoderSequenceContext,
    PolicyInferOutput,
    PolicyTrainBatch,
    PolicyTrainOutput,
    VideoConditionWindowContext,
)

from .video_conditioned_expert import (
    VideoConditionedActionExpert,
    init_conditioned_action_expert_from_video_core,
)


class VideoConditionedActionDecoder(SequenceActionDecoder):
    """Current-action decoder over a typed local video-conditioning window."""

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        action_horizon: int,
        context_dim: int,
        text_context_dim: int,
        state_dim: int,
        freq_dim: int,
        num_layers: int,
        num_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        cross_attn_norm: bool,
        eps: float,
        input_space: str,
        train_mode: str,
        action_chunk_anchor_mode: str,
        action_expert_init_mode: str,
        rollout_chunk_steps: int,
        direct_latent_channels: int,
        direct_rgb_patch_size: int,
        use_text_conditioning: bool,
        use_state_conditioning: bool,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.context_dim = int(context_dim)
        self.text_context_dim = int(text_context_dim)
        self.state_dim = int(state_dim)
        self.freq_dim = int(freq_dim)
        self.input_space = str(input_space)
        self.train_mode = str(train_mode)
        self.action_chunk_anchor_mode = str(action_chunk_anchor_mode)
        self.action_expert_init_mode = str(action_expert_init_mode)
        self.rollout_chunk_steps = int(rollout_chunk_steps)
        self.direct_latent_channels = int(direct_latent_channels)
        self.direct_rgb_patch_size = int(direct_rgb_patch_size)
        self.use_text_conditioning = bool(use_text_conditioning)
        self.use_state_conditioning = bool(use_state_conditioning)
        self.training_config = training_config
        self.inference_config = inference_config
        self._action_expert_initialized = False
        self.context_dropout = nn.Dropout(float(dropout))
        self.action_expert = VideoConditionedActionExpert(
            hidden_size=self.hidden_size,
            action_dim=self.action_dim,
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            attention_head_dim=int(attention_head_dim),
            ffn_dim=int(ffn_dim),
            freq_dim=self.freq_dim,
            context_dim=self.context_dim,
            cross_attn_norm=bool(cross_attn_norm),
            eps=float(eps),
        )
        self.goal_proj = (
            nn.Linear(self.text_context_dim, self.context_dim)
            if self.use_text_conditioning and self.text_context_dim > 0
            else None
        )
        self.state_proj = (
            nn.Linear(self.state_dim, self.context_dim)
            if self.use_state_conditioning and self.state_dim > 0
            else None
        )
        self.direct_latent_proj = nn.Conv2d(self.direct_latent_channels, self.context_dim, kernel_size=1)
        self.direct_rgb_proj = nn.Conv2d(
            3,
            self.context_dim,
            kernel_size=self.direct_rgb_patch_size,
            stride=self.direct_rgb_patch_size,
        )
        self.direct_current_action_head = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.action_dim),
        )

    def initialize_from_video_core(self, video_core) -> None:
        if self._action_expert_initialized:
            return
        init_conditioned_action_expert_from_video_core(
            action_expert=self.action_expert,
            video_core=video_core,
            mode=self.action_expert_init_mode,
        )
        self._action_expert_initialized = True

    def trainable_adapter_modules(self) -> list[nn.Module]:
        """Return the lightweight trainable surface for frozen-backbone warm starts."""

        modules: list[nn.Module] = [
            self.action_expert.action_embedder,
            self.action_expert.context_proj,
            self.action_expert.action_proj_out,
            self.direct_latent_proj,
            self.direct_rgb_proj,
            self.direct_current_action_head,
        ]
        if self.goal_proj is not None:
            modules.append(self.goal_proj)
        if self.state_proj is not None:
            modules.append(self.state_proj)
        return modules

    @staticmethod
    def _require_video_condition_window(sequence_context: DecoderSequenceContext) -> VideoConditionWindowContext:
        if sequence_context.video_condition_window is None:
            raise ValueError(
                "VideoConditionedActionDecoder requires `decoder_sequence_context.video_condition_window`."
            )
        return sequence_context.video_condition_window

    @staticmethod
    def _flatten_condition_tokens(window: VideoConditionWindowContext) -> torch.Tensor:
        local_tokens = window.local_window_tokens
        if local_tokens.ndim == 4:
            return local_tokens.flatten(1, 2)
        if local_tokens.ndim == 3:
            return local_tokens
        raise ValueError(
            "VideoConditionedActionDecoder expects local window tokens with shape [B, T, N, D] or [B, T, D], "
            f"got {tuple(local_tokens.shape)}."
        )

    @staticmethod
    def _pool_optional_context(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Expected optional conditioning tensor rank 2 or 3, got {tuple(features.shape)}.")

    def _build_condition_context(self, sequence_context: DecoderSequenceContext) -> tuple[torch.Tensor, VideoConditionWindowContext]:
        window = self._require_video_condition_window(sequence_context)
        context_tokens = self._flatten_condition_tokens(window)
        if context_tokens.shape[-1] != self.context_dim:
            raise ValueError(
                "VideoConditionedActionDecoder requires local video condition tokens to match `context_dim`, "
                f"got token_dim={context_tokens.shape[-1]}, context_dim={self.context_dim}."
            )
        context_parts = [context_tokens]
        if self.goal_proj is not None and sequence_context.goal_features is not None:
            pooled_goal = self._pool_optional_context(sequence_context.goal_features)
            context_parts.append(self.goal_proj(pooled_goal)[:, None, :])
        if self.state_proj is not None and sequence_context.state_sequence is not None:
            pooled_state = self._pool_optional_context(sequence_context.state_sequence)
            context_parts.append(self.state_proj(pooled_state)[:, None, :])
        return self.context_dropout(torch.cat(context_parts, dim=1)), window

    def _predict_flow(
        self,
        sequence_context: DecoderSequenceContext,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, VideoConditionWindowContext]:
        condition_context, window = self._build_condition_context(sequence_context)
        flow_pred = self.action_expert.forward_conditioned(
            action_tokens=noisy_actions,
            timestep=timesteps,
            context=condition_context,
        )
        return flow_pred, window

    def supports_direct_train_inputs(self) -> bool:
        return self.train_mode == "current_frame_regression"

    def uses_video_condition_window(self) -> bool:
        return not self.supports_direct_train_inputs()

    def _encode_direct_current_frame(self, direct_inputs: DirectActionDecoderTrainInputs) -> torch.Tensor:
        current_frame = direct_inputs.current_frame
        if direct_inputs.input_space == "video_latent":
            if current_frame.ndim != 4:
                raise ValueError(
                    "Direct latent current-frame regression expects current_frame with shape [B, C, H, W], "
                    f"got {tuple(current_frame.shape)}."
                )
            if current_frame.shape[1] != self.direct_latent_channels:
                raise ValueError(
                    "Direct latent current-frame regression requires the frame channel count to match "
                    f"`direct_latent_channels`, got channels={current_frame.shape[1]}, "
                    f"direct_latent_channels={self.direct_latent_channels}."
                )
            tokens = self.direct_latent_proj(current_frame).flatten(2).transpose(1, 2)
            return tokens
        if direct_inputs.input_space == "rgb_video":
            if current_frame.ndim != 4:
                raise ValueError(
                    "Direct RGB current-frame regression expects current_frame with shape [B, 3, H, W], "
                    f"got {tuple(current_frame.shape)}."
                )
            if current_frame.shape[1] != 3:
                raise ValueError(
                    "Direct RGB current-frame regression requires exactly 3 channels, "
                    f"got channels={current_frame.shape[1]}."
                )
            tokens = self.direct_rgb_proj(current_frame).flatten(2).transpose(1, 2)
            return tokens
        raise ValueError(f"Unsupported direct-train input space {direct_inputs.input_space!r}.")

    def _build_direct_context(self, direct_inputs: DirectActionDecoderTrainInputs) -> torch.Tensor:
        context_parts = [self._encode_direct_current_frame(direct_inputs)]
        if self.goal_proj is not None and direct_inputs.text_context is not None:
            pooled_goal = self._pool_optional_context(direct_inputs.text_context)
            context_parts.append(self.goal_proj(pooled_goal)[:, None, :])
        if self.state_proj is not None and direct_inputs.state is not None:
            pooled_state = self._pool_optional_context(direct_inputs.state)
            context_parts.append(self.state_proj(pooled_state)[:, None, :])
        return self.context_dropout(torch.cat(context_parts, dim=1))

    def forward_train_direct(
        self,
        direct_inputs: DirectActionDecoderTrainInputs,
        batch: PolicyTrainBatch,
    ) -> ActionDecoderTrainOutput:
        if not self.supports_direct_train_inputs():
            raise ValueError(
                "VideoConditionedActionDecoder direct training is available only when "
                "`train_mode = current_frame_regression`."
            )
        current_action_index = int(direct_inputs.current_action_index)
        if not (0 <= current_action_index < batch.actions.shape[1]):
            raise ValueError(
                "Direct current-frame regression requires a valid current action index, "
                f"got current_action_index={current_action_index}, action_horizon={batch.actions.shape[1]}."
            )
        context = self._build_direct_context(direct_inputs)
        pooled_context = self.context_dropout(context.mean(dim=1))
        current_action_pred = self.direct_current_action_head(pooled_context)
        target_actions = batch.actions[:, current_action_index]
        per_dim_loss = F.mse_loss(current_action_pred.float(), target_actions.float(), reduction="none")
        current_action_mask = None if batch.action_mask is None else batch.action_mask[:, current_action_index]
        if current_action_mask is not None:
            per_dim_loss = per_dim_loss * current_action_mask.float()
            denom = current_action_mask.float().sum().clamp_min(1.0)
        else:
            denom = torch.tensor(float(per_dim_loss.numel()), device=per_dim_loss.device)
        loss = per_dim_loss.sum() / denom
        weighted_loss = loss * self.training_config.objective_weight("action")
        return ActionDecoderTrainOutput(
            action_pred=current_action_pred[:, None, :],
            loss=weighted_loss,
            metrics={
                "action_mse": loss.detach(),
                "current_action_mse": loss.detach(),
                "weighted_current_action_mse": weighted_loss.detach(),
            },
            aux={
                "decoder": self.__class__.__name__,
                "train_mode": self.train_mode,
                "video_condition_input_space": direct_inputs.input_space,
                "current_action_index": torch.tensor(float(current_action_index), device=current_action_pred.device),
                "direct_condition_token_count": torch.tensor(float(context.shape[1]), device=current_action_pred.device),
            },
        )

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

    def _resolve_train_artifacts(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch):
        train_artifacts = policy_output.aux.get("action_flow_match_train_artifacts")
        if train_artifacts is not None:
            return train_artifacts
        return build_action_flow_match_train_artifacts(
            batch.actions,
            batch.action_mask,
            training_config=self.training_config,
        )

    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        if self.supports_direct_train_inputs():
            raise ValueError(
                "VideoConditionedActionDecoder with `train_mode = current_frame_regression` must be trained "
                "through the pipeline's direct-train path."
            )
        sequence_context = self.require_train_sequence_context(policy_output)
        train_artifacts = self._resolve_train_artifacts(policy_output, batch)
        flow_pred, window = self._predict_flow(
            sequence_context,
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
        loss = (per_token_loss.sum(dim=-1) / denom).mean()
        weighted_loss = loss * self.training_config.objective_weight("action")
        action_mse = F.mse_loss(denoised_actions.float(), batch.actions.float(), reduction="none")
        if batch.action_mask is not None:
            action_mse = action_mse * batch.action_mask.float()
            action_denom = batch.action_mask.float().sum().clamp_min(1.0)
        else:
            action_denom = torch.tensor(float(action_mse.numel()), device=action_mse.device)
        action_mse_value = action_mse.sum() / action_denom
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
                "video_condition_input_space": self.input_space,
                "action_chunk_anchor_mode": self.action_chunk_anchor_mode,
                "current_frame_index": torch.tensor(float(window.current_frame_index), device=denoised_actions.device),
                "current_action_index": torch.tensor(float(window.current_action_index), device=denoised_actions.device),
                "local_video_window_frames": torch.tensor(
                    float(window.local_window_frames or 0),
                    device=denoised_actions.device,
                ),
            },
        )

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: DecoderRolloutState | None = None,
    ) -> ActionDecoderInferOutput:
        if self.supports_direct_train_inputs():
            raise ValueError(
                "VideoConditionedActionDecoder with `train_mode = current_frame_regression` is a train-only "
                "mode. It does not currently define rollout-time inference semantics."
            )
        sequence_context = self.require_infer_sequence_context(policy_output)
        if previous_state is not None and not isinstance(previous_state, DecoderRolloutState):
            raise TypeError(
                "VideoConditionedActionDecoder expected `DecoderRolloutState` or None, "
                f"got {type(previous_state).__name__}."
            )
        effective_chunk_steps = min(self.rollout_chunk_steps, self.action_horizon)
        if previous_state is not None and previous_state.action_chunk is not None:
            step_within_chunk = int(previous_state.step_within_chunk)
            if step_within_chunk < effective_chunk_steps and step_within_chunk < int(previous_state.action_chunk.shape[1]):
                cached_chunk = self._apply_action_sampler_mask(previous_state.action_chunk)
                next_state = DecoderRolloutState(
                    action_chunk=cached_chunk,
                    chunk_index=previous_state.chunk_index,
                    step_within_chunk=step_within_chunk + 1,
                    cached_sequence_context=dict(previous_state.cached_sequence_context),
                    goal_context=previous_state.goal_context,
                    aux=dict(previous_state.aux),
                )
                return ActionDecoderInferOutput(
                    action_pred=cached_chunk,
                    next_state=next_state,
                    aux={
                        "decoder": self.__class__.__name__,
                        "sampled_new_chunk": False,
                        "current_action": cached_chunk[:, step_within_chunk],
                        "video_condition_input_space": self.input_space,
                        "action_chunk_anchor_mode": self.action_chunk_anchor_mode,
                        "current_frame_index": torch.tensor(
                            float(previous_state.aux.get("current_frame_index", 0)),
                            device=cached_chunk.device,
                        ),
                        "current_action_index": torch.tensor(float(step_within_chunk), device=cached_chunk.device),
                        "local_video_window_frames": torch.tensor(
                            float(previous_state.aux.get("local_video_window_frames", 0)),
                            device=cached_chunk.device,
                        ),
                    },
                )
        scheduler = build_action_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        sample = torch.randn(
            sequence_context.sequence_tokens.shape[0],
            self.action_horizon,
            self.action_dim,
            device=sequence_context.sequence_tokens.device,
            dtype=sequence_context.sequence_tokens.dtype,
        )
        sample = self._apply_action_sampler_mask(sample)
        resolved_window: VideoConditionWindowContext | None = None
        for timestep in scheduler.timesteps.to(device=sample.device):
            dense_timestep = torch.full(
                (sample.shape[0], self.action_horizon),
                fill_value=float(timestep),
                device=sample.device,
                dtype=torch.float32,
            )
            flow_pred, resolved_window = self._predict_flow(sequence_context, sample, dense_timestep)
            sample = scheduler.step(flow_pred, timestep, sample)
            sample = self._apply_action_sampler_mask(sample)
        next_state = DecoderRolloutState(
            action_chunk=sample.detach(),
            chunk_index=0 if previous_state is None else int(previous_state.chunk_index) + 1,
            step_within_chunk=1,
            cached_sequence_context={
                "source_stage": sequence_context.source_stage,
                "frame_count": sequence_context.frame_count,
                "layout_family": sequence_context.sequence_layout.get("family"),
            },
            goal_context=sequence_context.goal_features.detach() if sequence_context.goal_features is not None else None,
            aux={
                "rollout_chunk_steps": effective_chunk_steps,
                "current_frame_index": 0 if resolved_window is None else int(resolved_window.current_frame_index),
                "local_video_window_frames": 0 if resolved_window is None else int(resolved_window.local_window_frames or 0),
            },
        )
        return ActionDecoderInferOutput(
            action_pred=sample,
            next_state=next_state,
            aux={
                "decoder": self.__class__.__name__,
                "num_inference_steps": torch.tensor(float(len(scheduler.timesteps)), device=sample.device),
                "sampled_new_chunk": True,
                "current_action": sample[:, 0],
                "video_condition_input_space": self.input_space,
                "action_chunk_anchor_mode": self.action_chunk_anchor_mode,
                "current_frame_index": torch.tensor(
                    float(0 if resolved_window is None else resolved_window.current_frame_index),
                    device=sample.device,
                ),
                "current_action_index": torch.tensor(0.0, device=sample.device),
                "local_video_window_frames": torch.tensor(
                    float(resolved_window.local_window_frames or 0) if resolved_window is not None else 0.0,
                    device=sample.device,
                ),
            },
        )
