from __future__ import annotations

import torch
import torch.nn.functional as F

from open_wam.configs import (
    ActionGenerationBackendFamily,
    InferenceConfig,
    TrainingConfig,
    VPPActionDecoderConfig,
)
from open_wam.models.policy_variants.contracts import PolicyInferOutput, PolicyTrainBatch, PolicyTrainOutput

from .action_generation import EDMActionGenerationBackend
from .base import ActionDecoderInferOutput, ActionDecoderTrainOutput, DecoderRolloutState
from .goal_conditioning import build_goal_conditioning_adapter
from .sequence_base import SequenceActionDecoder
from .sequence_denoisers import build_sequence_denoiser
from .state_sequence import build_state_sequence_adapter
from .temporal_compression import build_temporal_compression_adapter


class VPPSequenceActionDecoder(SequenceActionDecoder):
    """Sequence-native action decoder with semantics close to VPP.

    The implementation keeps the repo's shared contracts while matching the
    core VPP action-head ideas:
    - preserve a visual token sequence
    - compress it with a learned temporal latent resampler
    - build an encoder memory from visual, goal, and state tokens
    - denoise an action chunk with EDM-style preconditioning
    - reuse the sampled chunk across multiple inference steps
    """

    def __init__(
        self,
        config: VPPActionDecoderConfig,
        *,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        state_dim: int,
        observation_token_dim: int,
        goal_feature_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.inference_config = inference_config
        self.state_dim = state_dim
        self.hidden_size = config.hidden_size
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.rollout_chunk_steps = config.rollout_chunk_steps or config.action_horizon
        if config.action_generation_backend != ActionGenerationBackendFamily.EDM_DIFFUSION:
            raise ValueError(
                "VPP sequence decoder currently supports only `action_generation_backend = edm_diffusion`, "
                f"got {config.action_generation_backend!r}."
            )

        self.temporal_compression = build_temporal_compression_adapter(
            config.temporal_compression_adapter_family,
            hidden_size=config.hidden_size,
            input_dim=observation_token_dim,
            compressed_tokens_per_frame=config.compressed_tokens_per_frame,
            depth=config.compression_depth,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_frames=config.temporal_compression_max_frames,
        )
        self.goal_conditioning = build_goal_conditioning_adapter(
            config.goal_conditioning_adapter_family,
            hidden_size=config.hidden_size,
        )
        self.state_sequence_adapter = build_state_sequence_adapter(
            config.state_sequence_adapter_family,
            input_dim=state_dim,
            hidden_size=config.hidden_size,
        )
        self.sequence_denoiser = build_sequence_denoiser(
            config.sequence_denoiser_family,
            hidden_size=config.hidden_size,
            action_dim=config.action_dim,
            goal_input_dim=goal_feature_dim,
            num_heads=config.num_heads,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            dropout=config.dropout,
        )

        num_sampling_steps = config.num_sampling_steps or inference_config.action_num_inference_steps
        self.generation_backend = EDMActionGenerationBackend(
            sigma_data=config.sigma_data,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            noise_schedule=config.diffusion_noise_schedule,
            sampler=config.diffusion_sampler,
            num_sampling_steps=num_sampling_steps,
        )

    def _prepare_sequence_memory(self, sequence_context):
        observation_tokens = self.temporal_compression(sequence_context)
        observation_tokens = self.goal_conditioning(observation_tokens, sequence_context.goal_features)
        state_tokens = self.state_sequence_adapter(
            sequence_context,
            target_length=max(1, sequence_context.frame_count or 1),
            target_hidden_size=self.hidden_size,
        )
        return self.sequence_denoiser.prepare_context(
            observation_tokens=observation_tokens,
            goal_features=sequence_context.goal_features,
            state_tokens=state_tokens,
        )

    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        sequence_context = self.require_train_sequence_context(policy_output)
        prepared_context = self._prepare_sequence_memory(sequence_context)
        action_diffusion_loss, denoised_actions, sigmas, noise = self.generation_backend.compute_training_loss(
            clean_actions=batch.actions,
            denoiser=lambda noised_actions, sigma: self.sequence_denoiser.denoise_actions(
                context=prepared_context,
                noised_actions=noised_actions,
                sigma=sigma,
            ),
        )
        weighted_action_loss = action_diffusion_loss * self.training_config.objective_weight("action")
        action_mse = F.mse_loss(denoised_actions.float(), batch.actions.float())
        predicted_latents = policy_output.aux.get("predicted_latents")
        target_latents = batch.extra.get("video_latents")
        latent_loss = None
        weighted_latent_loss = None
        if isinstance(predicted_latents, torch.Tensor) and isinstance(target_latents, torch.Tensor):
            latent_loss = F.mse_loss(predicted_latents.float(), target_latents.float())
            weighted_latent_loss = latent_loss * self.training_config.objective_weight("latent")
        total_loss = weighted_action_loss
        if weighted_latent_loss is not None:
            total_loss = total_loss + weighted_latent_loss
        return ActionDecoderTrainOutput(
            action_pred=denoised_actions,
            loss=total_loss,
            metrics={
                "action_mse": action_mse.detach(),
                "action_diffusion_loss": action_diffusion_loss.detach(),
                "weighted_action_diffusion_loss": weighted_action_loss.detach(),
                **(
                    {
                        "latent_mse": latent_loss.detach(),
                        "weighted_latent_loss": weighted_latent_loss.detach(),
                    }
                    if latent_loss is not None and weighted_latent_loss is not None
                    else {}
                ),
            },
            aux={
                "decoder": self.__class__.__name__,
                "context_memory_tokens": torch.tensor(
                    float(prepared_context.memory.shape[1]),
                    device=prepared_context.memory.device,
                ),
                "sampled_sigmas": sigmas.detach(),
                "sampled_noise": noise.detach(),
                **(
                    {"predicted_latents": predicted_latents.detach()}
                    if isinstance(predicted_latents, torch.Tensor)
                    else {}
                ),
            },
        )

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: DecoderRolloutState | None = None,
    ) -> ActionDecoderInferOutput:
        if previous_state is not None and not isinstance(previous_state, DecoderRolloutState):
            raise TypeError(
                "VPP decoder expected `DecoderRolloutState` or None, "
                f"got {type(previous_state).__name__}."
            )

        sequence_context = self.require_infer_sequence_context(policy_output)
        step_within_chunk = 0
        cached_chunk = previous_state.action_chunk if previous_state is not None else None
        if cached_chunk is not None and previous_state is not None:
            step_within_chunk = previous_state.step_within_chunk
            if step_within_chunk < self.rollout_chunk_steps:
                cached_chunk = self._apply_action_sampler_mask(cached_chunk)
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
                        **(
                            {"predicted_latents": predicted_latents.detach()}
                            if isinstance((predicted_latents := policy_output.aux.get("predicted_latents")), torch.Tensor)
                            else {}
                        ),
                    },
                )

        prepared_context = self._prepare_sequence_memory(sequence_context)
        sampled_chunk = self.generation_backend.sample(
            batch_size=prepared_context.memory.shape[0],
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            device=prepared_context.memory.device,
            dtype=prepared_context.memory.dtype,
            denoiser=lambda noised_actions, sigma: self.sequence_denoiser.denoise_actions(
                context=prepared_context,
                noised_actions=noised_actions,
                sigma=sigma,
            ),
            sample_transform=self._apply_action_sampler_mask,
        )
        next_state = DecoderRolloutState(
            action_chunk=sampled_chunk.detach(),
            chunk_index=(0 if previous_state is None else previous_state.chunk_index + 1),
            step_within_chunk=1 if self.rollout_chunk_steps > 1 else 0,
            cached_sequence_context={
                "source_stage": sequence_context.source_stage,
                "frame_count": sequence_context.frame_count,
                "layout_family": sequence_context.sequence_layout.get("family"),
            },
            goal_context=sequence_context.goal_features.detach() if sequence_context.goal_features is not None else None,
            aux={"rollout_chunk_steps": self.rollout_chunk_steps},
        )
        return ActionDecoderInferOutput(
            action_pred=sampled_chunk,
            next_state=next_state,
            aux={
                "decoder": self.__class__.__name__,
                "sampled_new_chunk": True,
                "current_action": sampled_chunk[:, 0],
                **(
                    {"predicted_latents": predicted_latents.detach()}
                    if isinstance((predicted_latents := policy_output.aux.get("predicted_latents")), torch.Tensor)
                    else {}
                ),
            },
        )
