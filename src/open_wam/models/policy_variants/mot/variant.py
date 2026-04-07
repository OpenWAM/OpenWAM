from __future__ import annotations

import torch

from open_wam.models.action_decoders import ActionDecoderInferOutput, ActionDecoderTrainOutput
from open_wam.models.common.flow_matching import (
    FlowMatchScheduler,
    build_video_flow_match_train_artifacts,
    build_video_flow_match_inference_scheduler,
    build_action_flow_match_inference_scheduler,
    build_action_flow_match_train_artifacts,
    denoised_actions_from_flow,
    denoised_video_latents_from_flow,
)
from open_wam.configs import InferenceConfig, MoTPolicyConfig, MoTRuntimeMode, TrainingConfig
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig

from ..base import PolicyVariant
from ..contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)
from .modules import MoTActionExpert, init_action_expert_from_video_core
from .runtime import (
    build_mot_attention_mask,
    forward_joint_video_action_denoise,
    forward_action_with_video_cache,
    move_mot_video_cache,
    prefill_video_kv_cache,
    resolve_mot_condition_latents,
)


class MoTPolicyVariant(PolicyVariant):
    """Method-5 scaffold for the future FastWAM-style MoT runtime.

    This class intentionally only wires the config/build surface in the first
    landing. The actual action expert and mixed-attention runtime are added in
    follow-up changes instead of silently degrading into another policy family.
    """

    def __init__(
        self,
        config: MoTPolicyConfig,
        backbone_config: SharedVideoTransformerConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_dim: int,
        action_horizon: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone_config = backbone_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        action_hidden_size = (
            int(config.action_hidden_size)
            if config.action_hidden_size is not None
            else int(backbone_config.hidden_size)
        )
        self.action_expert = MoTActionExpert(
            hidden_size=action_hidden_size,
            action_dim=action_dim,
            num_layers=config.num_action_layers,
            num_heads=backbone_config.num_heads,
            attention_head_dim=backbone_config.attention_head_dim,
            ffn_dim=(
                int(config.action_ffn_dim)
                if config.action_ffn_dim is not None
                else (backbone_config.ffn_dim or (backbone_config.hidden_size * backbone_config.mlp_ratio))
            ),
            text_dim=backbone_config.text_dim,
            freq_dim=backbone_config.freq_dim,
            cross_attn_norm=backbone_config.cross_attn_norm,
            eps=backbone_config.latent_norm_eps,
        )
        self._action_expert_initialized = False

    def _build_video_train_rollout(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
    ) -> dict[str, torch.Tensor | FlowMatchScheduler]:
        video_latents = visual_outputs.frontend.video_latents
        observed_prefix_frames = int(self.config.video_prefix_frames)
        if video_latents.shape[2] <= observed_prefix_frames:
            raise ValueError(
                "MoT joint video training requires at least one future frame after the observed prefix, "
                f"got video_latents.shape={tuple(video_latents.shape)}, video_prefix_frames={observed_prefix_frames}."
            )
        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
        )
        noisy_latents = video_artifacts.noisy_latents.clone()
        timesteps = video_artifacts.timesteps.clone()
        noisy_latents[:, :, :observed_prefix_frames] = video_latents[:, :, :observed_prefix_frames]
        timesteps[:, :observed_prefix_frames] = 0.0
        future_loss_mask = torch.zeros(
            video_latents.shape[0],
            1,
            video_latents.shape[2],
            1,
            1,
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        future_loss_mask[:, :, observed_prefix_frames:] = 1.0
        flow_pred = visual_tower.predict_video_flow(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_context=visual_outputs.frontend.conditioning.text_context,
            frame_start=0,
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=noisy_latents,
            flow_pred=flow_pred,
            timesteps=timesteps,
            scheduler=video_artifacts.scheduler,
        )
        return {
            "flow_pred": flow_pred,
            "flow_targets": video_artifacts.targets,
            "predicted_latents": predicted_latents,
            "target_latents": video_latents,
            "timesteps": timesteps,
            "scheduler": video_artifacts.scheduler,
            "future_loss_mask": future_loss_mask,
        }

    @staticmethod
    def _masked_video_flow_match_loss(
        *,
        flow_pred: torch.Tensor,
        targets: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: FlowMatchScheduler,
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

    @staticmethod
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

    def _maybe_initialize_action_expert(self, visual_tower: VisualTower) -> None:
        if self._action_expert_initialized:
            return
        init_action_expert_from_video_core(
            action_expert=self.action_expert,
            video_core=visual_tower.core,
            mode=str(self.config.action_expert_init_mode),
        )
        self._action_expert_initialized = True

    def initialize_for_training(self, visual_tower: VisualTower) -> None:
        self._maybe_initialize_action_expert(visual_tower)

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend",)

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        return PolicyPreparedInputs(
            batch=batch,
            variant_inputs={
                "video_latents": visual_outputs.frontend.video_latents,
                "text_context": visual_outputs.frontend.conditioning.text_context,
                "video_tokens_per_frame": visual_outputs.frontend.token_grid.tokens_per_frame,
            },
        )

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        self._maybe_initialize_action_expert(visual_tower)
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            return self._forward_train_joint_denoise(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                prepared_inputs=prepared_inputs,
            )
        return self._forward_train_prefill_action_denoise(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
        )

    def _forward_train_prefill_action_denoise(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        self._maybe_initialize_action_expert(visual_tower)
        video_latents = prepared_inputs.variant_inputs["video_latents"]
        text_context = prepared_inputs.variant_inputs["text_context"]
        video_train_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
        )
        condition_latents = resolve_mot_condition_latents(
            video_latents=video_latents,
            condition_mode=self.config.condition_mode,
            video_prefix_frames=self.config.video_prefix_frames,
            teacher_forcing_video_noise_prob=self.config.teacher_forcing_video_noise_prob,
            training=True,
            scheduler=video_train_artifacts.scheduler,
        )
        video_cache = prefill_video_kv_cache(
            visual_tower=visual_tower,
            observed_prefix=condition_latents,
            text_context=text_context,
            frame_start=0,
        )
        train_artifacts = build_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            prepared_inputs.batch.action_mask,
            training_config=self.training_config,
        )
        resolved_text = text_context
        if resolved_text is None:
            resolved_text = condition_latents.new_zeros(
                condition_latents.shape[0],
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
            )
        action_pre = self.action_expert.pre_dit(
            action_tokens=train_artifacts.noisy_actions,
            timestep=train_artifacts.timesteps,
            context=resolved_text,
        )
        action_hidden_states = forward_action_with_video_cache(
            action_expert=self.action_expert,
            action_pre=action_pre,
            video_cache=video_cache,
            attention_mask=build_mot_attention_mask(
                video_seq_len=video_cache.video_seq_len,
                action_seq_len=train_artifacts.noisy_actions.shape[1],
                device=train_artifacts.noisy_actions.device,
                condition_mode=self.config.condition_mode,
                video_tokens_per_frame=prepared_inputs.variant_inputs["video_tokens_per_frame"],
            ),
        )
        flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=train_artifacts.noisy_actions,
            flow_pred=flow_pred,
            timesteps=train_artifacts.timesteps,
            scheduler=train_artifacts.scheduler,
        )
        timestep_weight = train_artifacts.scheduler.training_weight(train_artifacts.timesteps.flatten()).reshape(
            train_artifacts.timesteps.shape
        )
        per_token_loss = torch.nn.functional.mse_loss(
            flow_pred.float(),
            train_artifacts.targets.float().detach(),
            reduction="none",
        )
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
        diffusion_loss = (per_token_loss.sum(dim=-1) / denom).mean()
        weighted_loss = diffusion_loss * self.training_config.objective_weight("action")
        video_rollout = None
        if self.training_config.objective_enabled("latent"):
            video_rollout = self._build_video_train_rollout(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
            )
            latent_loss = self._masked_video_flow_match_loss(
                flow_pred=video_rollout["flow_pred"],
                targets=video_rollout["flow_targets"],
                timesteps=video_rollout["timesteps"],
                scheduler=video_rollout["scheduler"],
                future_loss_mask=video_rollout["future_loss_mask"],
            )
            latent_mse = self._masked_video_latent_mse(
                predicted_latents=video_rollout["predicted_latents"],
                target_latents=video_rollout["target_latents"],
                future_loss_mask=video_rollout["future_loss_mask"],
            )
            weighted_latent_loss = latent_loss * self.training_config.objective_weight("latent")
        else:
            latent_loss = diffusion_loss.new_zeros(())
            latent_mse = diffusion_loss.new_zeros(())
            weighted_latent_loss = diffusion_loss.new_zeros(())
        total_loss = weighted_loss + weighted_latent_loss
        action_mse = torch.nn.functional.mse_loss(
            denoised_actions.float(),
            prepared_inputs.batch.actions.float(),
            reduction="none",
        )
        if prepared_inputs.batch.action_mask is not None:
            action_mse = action_mse * prepared_inputs.batch.action_mask.float()
            action_denom = prepared_inputs.batch.action_mask.float().sum().clamp_min(1.0)
        else:
            action_denom = torch.tensor(float(action_mse.numel()), device=action_mse.device)
        action_mse_value = action_mse.sum() / action_denom
        batch_size = condition_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=condition_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "mot_action_mse": action_mse_value.detach(),
                "mot_action_diffusion_loss": diffusion_loss.detach(),
                "mot_video_diffusion_loss": latent_loss.detach(),
                "mot_video_prefix_frames": condition_latents.new_tensor(float(self.config.video_prefix_frames)),
            },
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "video_cache_seq_len": video_cache.video_seq_len,
                "decoder_output": ActionDecoderTrainOutput(
                    action_pred=denoised_actions,
                    loss=total_loss,
                    metrics={
                        "action_mse": action_mse_value.detach(),
                        "action_diffusion_loss": diffusion_loss.detach(),
                        "weighted_action_diffusion_loss": weighted_loss.detach(),
                        "latent_mse": latent_mse.detach(),
                        "video_diffusion_loss": latent_loss.detach(),
                        "weighted_video_diffusion_loss": weighted_latent_loss.detach(),
                        "joint_loss": total_loss.detach(),
                    },
                    aux={
                        "flow_pred": flow_pred.detach(),
                        **(
                            {
                                "predicted_latents": video_rollout["predicted_latents"].detach(),
                                "predicted_video_latents": video_rollout["predicted_latents"].detach(),
                                "future_video_flow_pred": video_rollout["flow_pred"].detach(),
                            }
                            if video_rollout is not None
                            else {}
                        ),
                    },
                ),
            },
        )

    def _forward_train_joint_denoise(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        video_latents = prepared_inputs.variant_inputs["video_latents"]
        text_context = prepared_inputs.variant_inputs["text_context"]
        observed_prefix_frames = int(self.config.video_prefix_frames)
        if video_latents.shape[2] <= observed_prefix_frames:
            raise ValueError(
                "MoT joint denoise requires at least one future frame after the observed prefix, "
                f"got video_latents.shape={tuple(video_latents.shape)}, video_prefix_frames={observed_prefix_frames}."
            )

        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
        )
        noisy_video_latents = video_artifacts.noisy_latents.clone()
        video_timesteps = video_artifacts.timesteps.clone()
        noisy_video_latents[:, :, :observed_prefix_frames] = video_latents[:, :, :observed_prefix_frames]
        video_timesteps[:, :observed_prefix_frames] = 0.0
        future_loss_mask = torch.zeros(
            video_latents.shape[0],
            1,
            video_latents.shape[2],
            1,
            1,
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        future_loss_mask[:, :, observed_prefix_frames:] = 1.0

        train_artifacts = build_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            prepared_inputs.batch.action_mask,
            training_config=self.training_config,
        )
        resolved_text = text_context
        if resolved_text is None:
            resolved_text = video_latents.new_zeros(
                video_latents.shape[0],
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
            )
        action_pre = self.action_expert.pre_dit(
            action_tokens=train_artifacts.noisy_actions,
            timestep=train_artifacts.timesteps,
            context=resolved_text,
        )
        video_flow_pred, action_hidden_states = forward_joint_video_action_denoise(
            visual_tower=visual_tower,
            noisy_video_latents=noisy_video_latents,
            video_timesteps=video_timesteps,
            action_expert=self.action_expert,
            action_pre=action_pre,
            text_context=resolved_text,
            attention_mask=build_mot_attention_mask(
                video_seq_len=prepared_inputs.variant_inputs["video_tokens_per_frame"] * video_latents.shape[2],
                action_seq_len=train_artifacts.noisy_actions.shape[1],
                device=train_artifacts.noisy_actions.device,
                condition_mode=self.config.condition_mode,
                video_tokens_per_frame=prepared_inputs.variant_inputs["video_tokens_per_frame"],
                video_can_attend_action=self.config.video_can_attend_action,
            ),
        )
        flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=train_artifacts.noisy_actions,
            flow_pred=flow_pred,
            timesteps=train_artifacts.timesteps,
            scheduler=train_artifacts.scheduler,
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=noisy_video_latents,
            flow_pred=video_flow_pred,
            timesteps=video_timesteps,
            scheduler=video_artifacts.scheduler,
        )
        timestep_weight = train_artifacts.scheduler.training_weight(train_artifacts.timesteps.flatten()).reshape(
            train_artifacts.timesteps.shape
        )
        per_token_loss = torch.nn.functional.mse_loss(
            flow_pred.float(),
            train_artifacts.targets.float().detach(),
            reduction="none",
        )
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
        diffusion_loss = (per_token_loss.sum(dim=-1) / denom).mean()
        weighted_loss = diffusion_loss * self.training_config.objective_weight("action")
        latent_loss = self._masked_video_flow_match_loss(
            flow_pred=video_flow_pred,
            targets=video_artifacts.targets,
            timesteps=video_timesteps,
            scheduler=video_artifacts.scheduler,
            future_loss_mask=future_loss_mask,
        )
        latent_mse = self._masked_video_latent_mse(
            predicted_latents=predicted_latents,
            target_latents=video_latents,
            future_loss_mask=future_loss_mask,
        )
        weighted_latent_loss = (
            latent_loss * self.training_config.objective_weight("latent")
            if self.training_config.objective_enabled("latent")
            else diffusion_loss.new_zeros(())
        )
        total_loss = weighted_loss + weighted_latent_loss
        action_mse = torch.nn.functional.mse_loss(
            denoised_actions.float(),
            prepared_inputs.batch.actions.float(),
            reduction="none",
        )
        if prepared_inputs.batch.action_mask is not None:
            action_mse = action_mse * prepared_inputs.batch.action_mask.float()
            action_denom = prepared_inputs.batch.action_mask.float().sum().clamp_min(1.0)
        else:
            action_denom = torch.tensor(float(action_mse.numel()), device=action_mse.device)
        action_mse_value = action_mse.sum() / action_denom
        batch_size = video_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=video_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "mot_action_mse": action_mse_value.detach(),
                "mot_action_diffusion_loss": diffusion_loss.detach(),
                "mot_video_diffusion_loss": latent_loss.detach(),
                "mot_video_prefix_frames": video_latents.new_tensor(float(self.config.video_prefix_frames)),
            },
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "runtime_mode": str(self.config.runtime_mode),
                "decoder_output": ActionDecoderTrainOutput(
                    action_pred=denoised_actions,
                    loss=total_loss,
                    metrics={
                        "action_mse": action_mse_value.detach(),
                        "action_diffusion_loss": diffusion_loss.detach(),
                        "weighted_action_diffusion_loss": weighted_loss.detach(),
                        "latent_mse": latent_mse.detach(),
                        "video_diffusion_loss": latent_loss.detach(),
                        "weighted_video_diffusion_loss": weighted_latent_loss.detach(),
                        "joint_loss": total_loss.detach(),
                    },
                    aux={
                        "flow_pred": flow_pred.detach(),
                        "predicted_latents": predicted_latents.detach(),
                        "predicted_video_latents": predicted_latents.detach(),
                        "future_video_flow_pred": video_flow_pred.detach(),
                    },
                ),
            },
        )

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        self._maybe_initialize_action_expert(visual_tower)
        state = previous_state or PolicyInferState()
        action_device_raw = context.extra.get("action_device")
        action_device = (
            next(self.action_expert.parameters()).device
            if action_device_raw is None
            else torch.device(str(action_device_raw))
        )
        action_dtype = next(self.action_expert.parameters()).dtype
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            runtime_device = next(visual_tower.core.parameters()).device
            if action_device != runtime_device:
                raise ValueError(
                    "MoT joint_denoise inference currently requires video and action to run on the same device, "
                    f"got runtime_device={runtime_device}, action_device={action_device}."
                )
            state.cache["text_context"] = visual_outputs.frontend.conditioning.text_context
            state.cache["action_device"] = str(action_device)
            del context
            return state
        if "video_cache" not in state.cache:
            condition_latents = resolve_mot_condition_latents(
                video_latents=visual_outputs.frontend.video_latents,
                condition_mode=self.config.condition_mode,
                video_prefix_frames=self.config.video_prefix_frames,
                teacher_forcing_video_noise_prob=self.config.teacher_forcing_video_noise_prob,
                training=False,
            )
            prefetched_video_cache = prefill_video_kv_cache(
                visual_tower=visual_tower,
                observed_prefix=condition_latents,
                text_context=visual_outputs.frontend.conditioning.text_context,
                frame_start=int(state.cursor.current_start_frame),
            )
            state.cache["video_cache"] = move_mot_video_cache(
                prefetched_video_cache,
                device=action_device,
                dtype=action_dtype,
            )
            resolved_text_context = visual_outputs.frontend.conditioning.text_context
            state.cache["text_context"] = (
                None
                if resolved_text_context is None
                else resolved_text_context.to(device=action_device, dtype=action_dtype)
            )
            state.cache["video_tokens_per_frame"] = max(
                1,
                visual_outputs.frontend.token_grid.tokens_per_frame * condition_latents.shape[2] // max(1, visual_outputs.frontend.video_latents.shape[2]),
            )
        state.cache["action_device"] = str(action_device)
        del context
        return state

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        self._maybe_initialize_action_expert(visual_tower)
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            device = next(visual_tower.core.parameters()).device
            action_device = next(self.action_expert.parameters()).device
            if action_device != device:
                raise ValueError(
                    "MoT joint_denoise inference currently requires visual tower and action expert on the same device, "
                    f"got visual_device={device}, action_device={action_device}."
                )
            dtype = next(self.action_expert.parameters()).dtype
            batch_size = visual_outputs.frontend.video_latents.shape[0]
            observed_prefix_frames = int(self.config.video_prefix_frames)
            video_latents = visual_outputs.frontend.video_latents.to(device=device, dtype=dtype)
            if video_latents.shape[2] <= observed_prefix_frames:
                raise ValueError(
                    "MoT joint_denoise inference requires at least one future frame after the observed prefix, "
                    f"got video_latents.shape={tuple(video_latents.shape)}, video_prefix_frames={observed_prefix_frames}."
                )
            if self.inference_config.video_num_inference_steps != self.inference_config.action_num_inference_steps:
                raise ValueError(
                    "MoT joint_denoise inference currently requires matching video/action inference step counts, "
                    f"got video_num_inference_steps={self.inference_config.video_num_inference_steps}, "
                    f"action_num_inference_steps={self.inference_config.action_num_inference_steps}."
                )
            observed_prefix = video_latents[:, :, :observed_prefix_frames]
            future_template = video_latents[:, :, observed_prefix_frames:]
            noisy_video_latents = torch.cat(
                [
                    observed_prefix,
                    torch.randn_like(future_template, device=device, dtype=dtype),
                ],
                dim=2,
            )
            action_scheduler = build_action_flow_match_inference_scheduler(
                training_config=self.training_config,
                inference_config=self.inference_config,
            )
            video_scheduler = build_video_flow_match_inference_scheduler(
                training_config=self.training_config,
                inference_config=self.inference_config,
            )
            sample = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=device,
                dtype=dtype,
            )
            text_context = infer_state.cache.get("text_context")
            if text_context is None:
                text_context = torch.zeros(
                    batch_size,
                    visual_tower.config.max_text_tokens,
                    visual_tower.config.text_dim,
                    device=device,
                    dtype=dtype,
                )
            else:
                text_context = text_context.to(device=device, dtype=dtype)
            attention_mask = build_mot_attention_mask(
                video_seq_len=visual_outputs.frontend.token_grid.tokens_per_frame * video_latents.shape[2],
                action_seq_len=self.action_horizon,
                device=device,
                condition_mode=self.config.condition_mode,
                video_tokens_per_frame=visual_outputs.frontend.token_grid.tokens_per_frame,
                video_can_attend_action=self.config.video_can_attend_action,
            )
            for video_timestep, action_timestep in zip(video_scheduler.timesteps, action_scheduler.timesteps, strict=True):
                dense_video_timestep = torch.full(
                    (batch_size, video_latents.shape[2]),
                    float(video_timestep),
                    device=device,
                    dtype=torch.float32,
                )
                dense_video_timestep[:, :observed_prefix_frames] = 0.0
                dense_action_timestep = torch.full(
                    (batch_size, self.action_horizon),
                    float(action_timestep),
                    device=device,
                    dtype=torch.float32,
                )
                action_pre = self.action_expert.pre_dit(
                    action_tokens=sample,
                    timestep=dense_action_timestep,
                    context=text_context,
                )
                video_flow_pred, action_hidden_states = forward_joint_video_action_denoise(
                    visual_tower=visual_tower,
                    noisy_video_latents=noisy_video_latents,
                    video_timesteps=dense_video_timestep,
                    action_expert=self.action_expert,
                    action_pre=action_pre,
                    text_context=text_context,
                    attention_mask=attention_mask,
                    frame_start=int(infer_state.cursor.current_start_frame),
                )
                flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
                noisy_video_latents = video_scheduler.step(video_flow_pred, video_timestep, noisy_video_latents)
                noisy_video_latents[:, :, :observed_prefix_frames] = observed_prefix
                sample = action_scheduler.step(flow_pred, action_timestep, sample)
            predicted_latents = noisy_video_latents[:, :, observed_prefix_frames:].detach()
            next_state = infer_state
            next_state.step_index += 1
            return PolicyInferOutput(
                policy_features=sample.new_zeros(batch_size, 0, self.action_expert.hidden_size),
                next_state=next_state,
                aux={
                    "variant": self.config.name,
                    "method_family": "mot",
                    "condition_mode": str(self.config.condition_mode),
                    "predicted_latents": predicted_latents,
                    "predicted_video_latents": predicted_latents,
                    "decoder_output": ActionDecoderInferOutput(
                        action_pred=sample,
                        next_state=None,
                        aux={
                            "predicted_latents": predicted_latents,
                            "predicted_video_latents": predicted_latents,
                        },
                    ),
                },
            )
        predicted_latents = None
        video_latents = visual_outputs.frontend.video_latents
        prefix_frames = int(self.config.video_prefix_frames)
        if video_latents.shape[2] > prefix_frames:
            observed_prefix = video_latents[:, :, :prefix_frames]
            future_template = torch.zeros_like(video_latents[:, :, prefix_frames:])
            text_context_for_video = visual_outputs.frontend.conditioning.text_context
            if text_context_for_video is None:
                text_context_for_video = torch.zeros(
                    video_latents.shape[0],
                    visual_tower.config.max_text_tokens,
                    visual_tower.config.text_dim,
                    device=video_latents.device,
                    dtype=video_latents.dtype,
                )
            predicted_latents = visual_tower.generate_conditioned_future_latents(
                observed_prefix=observed_prefix,
                future_template=future_template,
                text_context=text_context_for_video,
                negative_text_context=visual_outputs.frontend.conditioning.negative_text_context,
                frame_start=int(infer_state.cursor.current_start_frame),
                num_inference_steps=self.inference_config.video_num_inference_steps,
                num_train_timesteps=self.training_config.video_num_train_timesteps,
                sigma_shift=self.training_config.video_sigma_shift,
                guidance_scale=self.inference_config.guidance_scale,
                cache_name="mot_infer_predicted_video_latents",
            )
        batch_size = 1 if context.previous_action is None else context.previous_action.shape[0]
        device = next(self.action_expert.parameters()).device
        dtype = next(self.action_expert.parameters()).dtype
        scheduler = build_action_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        sample = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        text_context = infer_state.cache.get("text_context")
        if text_context is None:
            text_context = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=device,
                dtype=dtype,
            )
        video_cache = infer_state.cache["video_cache"]
        attention_mask = build_mot_attention_mask(
            video_seq_len=video_cache.video_seq_len,
            action_seq_len=self.action_horizon,
            device=device,
            condition_mode=self.config.condition_mode,
            video_tokens_per_frame=infer_state.cache.get("video_tokens_per_frame"),
        )
        for timestep in scheduler.timesteps:
            dense_timestep = torch.full(
                (batch_size, self.action_horizon),
                float(timestep),
                device=device,
                dtype=torch.float32,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=sample,
                timestep=dense_timestep,
                context=text_context.to(device=device, dtype=dtype),
            )
            action_hidden_states = forward_action_with_video_cache(
                action_expert=self.action_expert,
                action_pre=action_pre,
                video_cache=video_cache,
                attention_mask=attention_mask,
            )
            flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
            sample = scheduler.step(flow_pred, timestep, sample)
        next_state = infer_state
        next_state.step_index += 1
        return PolicyInferOutput(
            policy_features=sample.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            next_state=next_state,
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                **(
                    {"predicted_latents": predicted_latents.detach(), "predicted_video_latents": predicted_latents.detach()}
                    if isinstance(predicted_latents, torch.Tensor)
                    else {}
                ),
                "decoder_output": ActionDecoderInferOutput(
                    action_pred=sample,
                    next_state=None,
                    aux=(
                        {
                            "predicted_latents": predicted_latents.detach(),
                            "predicted_video_latents": predicted_latents.detach(),
                        }
                        if isinstance(predicted_latents, torch.Tensor)
                        else {}
                    ),
                ),
            },
        )
