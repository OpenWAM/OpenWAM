from __future__ import annotations

import torch
from torch import nn

from open_wam.configs import InferenceConfig, RegisterAttachedPolicyConfig, TrainingConfig
from open_wam.models.action_decoders import ActionDecoderInferOutput, ActionDecoderTrainOutput
from open_wam.models.common import (
    build_block_coupled_action_flow_match_train_artifacts,
    build_joint_video_timestep_grid,
    build_video_flow_match_train_artifacts,
    build_joint_runtime_schedulers,
    build_unconditional_conditioning,
    combine_joint_cfg_predictions,
    denoised_actions_from_flow,
    denoised_video_latents_from_flow,
    preserve_joint_observed_video_prefix,
    reduce_slot_aligned_action_flow_match_loss,
    reduce_video_flow_match_loss,
    resolve_runtime_cache_policy,
    resolve_runtime_guidance,
    resolve_runtime_warmup_reference,
    should_update_cache_during_denoise,
)
from open_wam.models.common.video_geometry import unpatchify_video_tokens
from open_wam.models.video_backbone.contracts import CacheState, CacheUpdateMetadata
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..base import PolicyVariant
from ..contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    RolloutCursor,
)
from ..common.rollout import advance_rollout_cursor
from .layout import RegisterSequenceLayout
from .runtime import RegisterAttachedRuntime, RegisterRuntimeSpec


class RegisterAttachedPolicyVariant(PolicyVariant):
    """DreamZero-inspired register-attached policy variant.

    This variant now owns joint video+action diffusion rather than passing
    clean video features into an action-only decoder.

    Training keeps a full clean-video teacher-forcing prefix plus a noisy half:

    - clean video prefix tokens
    - noisy video tokens
    - noisy action-register tokens as denoising targets
    - clean state-register tokens as conditioning context

    Inference drops the clean prefix and instead re-enters the shared core with
    updated noisy video and action samples at every denoising step, which keeps
    the runtime closer to DreamZero than the old "decoder-only action rollout".
    """

    def __init__(
        self,
        config: RegisterAttachedPolicyConfig,
        backbone_config: LingbotCompatibleVideoBackboneConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_dim: int,
        action_horizon: int,
        state_dim: int,
        state_horizon: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone_config = backbone_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.state_horizon = state_horizon
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.register_role_embedding = nn.Embedding(2, config.hidden_size)
        self.runtime = RegisterAttachedRuntime(
            RegisterRuntimeSpec(
                hidden_size=config.hidden_size,
                action_horizon=action_horizon,
                state_horizon=state_horizon,
                num_frame_per_block=config.num_frame_per_block,
                num_action_per_block=config.num_action_per_block,
                num_state_per_block=config.num_state_per_block,
                variant_name=config.name,
            )
        )
        self.video_patch_dim = (
            self.backbone_config.latent_channels
            * self.backbone_config.patch_size_t
            * self.backbone_config.patch_size_h
            * self.backbone_config.patch_size_w
        )
        self.video_flow_head = nn.Linear(config.hidden_size, self.video_patch_dim)
        self.action_flow_head = nn.Linear(config.hidden_size, action_dim)

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend",)

    def _build_layout(self, visual_outputs: VisualStageOutputs) -> RegisterSequenceLayout:
        return self.runtime.build_layout(visual_outputs, include_clean_video_prefix=False)

    def _build_train_layout(self, visual_outputs: VisualStageOutputs) -> RegisterSequenceLayout:
        return self.runtime.build_layout(visual_outputs, include_clean_video_prefix=True)

    def _validate_train_batch(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> None:
        self._build_layout(visual_outputs)
        if visual_outputs.frontend.token_grid.num_frames < 2:
            raise ValueError("Register-attached joint diffusion requires at least two frames.")
        if batch.actions.shape[1] != self.action_horizon or batch.actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected actions with shape [B, {self.action_horizon}, {self.action_dim}], "
                f"got {tuple(batch.actions.shape)}."
            )
        if batch.state is None:
            raise ValueError("Register-attached variant requires state inputs.")
        if batch.state.shape[1] != self.state_horizon or batch.state.shape[2] != self.state_dim:
            raise ValueError(
                f"Expected state with shape [B, {self.state_horizon}, {self.state_dim}], "
                f"got {tuple(batch.state.shape)}."
            )

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        self._validate_train_batch(visual_outputs, batch)
        video_artifacts = build_video_flow_match_train_artifacts(
            visual_outputs.frontend.video_latents,
            training_config=self.training_config,
        )
        if self.config.couple_action_to_video_blocks:
            action_artifacts = build_block_coupled_action_flow_match_train_artifacts(
                batch.actions,
                batch.action_mask,
                training_config=self.training_config,
                future_video_timesteps=video_artifacts.timesteps[:, 1:],
                num_frame_per_block=self.config.num_frame_per_block,
                num_action_per_block=self.config.num_action_per_block,
            )
        else:
            raise ValueError(
                "Register-attached method 2 now defaults to DreamZero-style action/video timestep coupling. "
                "Set `couple_action_to_video_blocks=true`."
            )
        return PolicyPreparedInputs(
            batch=batch,
            variant_inputs={
                "video_flow_match_train_artifacts": video_artifacts,
                "action_flow_match_train_artifacts": action_artifacts,
            },
        )

    def _build_noisy_frontend_outputs(
        self,
        visual_tower: VisualTower,
        *,
        visual_outputs: VisualStageOutputs,
        noisy_video_latents: torch.Tensor,
    ) -> VisualStageOutputs:
        return self.runtime.build_noisy_frontend_outputs(
            visual_tower,
            visual_outputs=visual_outputs,
            noisy_video_latents=noisy_video_latents,
        )

    def _decode_future_video_flow(
        self,
        *,
        hidden_tokens: torch.Tensor,
        token_grid,
    ) -> torch.Tensor:
        patch_tokens = self.video_flow_head(hidden_tokens)
        return unpatchify_video_tokens(
            patch_tokens,
            token_grid=token_grid,
            latent_channels=self.backbone_config.latent_channels,
        )

    def _run_packed_core(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        *,
        noisy_video_tokens: torch.Tensor,
        clean_video_prefix_tokens: torch.Tensor | None,
        action_inputs: torch.Tensor,
        state_inputs: torch.Tensor,
        video_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
        current_start_frame: int,
        cache_state: CacheState | None = None,
        cache_update_metadata: CacheUpdateMetadata | None = None,
        conditioning_override=None,
        include_register_tokens: bool = True,
        cache_reference_token_span: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RegisterSequenceLayout, CacheState, dict[str, object]]:
        runtime_result = self.runtime.run_core(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            clean_video_prefix_tokens=clean_video_prefix_tokens,
            noisy_video_tokens=noisy_video_tokens,
            action_inputs=action_inputs,
            state_inputs=state_inputs,
            action_encoder=self.action_encoder,
            state_encoder=self.state_encoder,
            register_role_embedding=self.register_role_embedding,
            video_timesteps=video_timesteps,
            action_timesteps=action_timesteps,
            current_start_frame=current_start_frame,
            cache_state=cache_state,
            cache_update_metadata=cache_update_metadata,
            conditioning_override=conditioning_override,
            include_register_tokens=include_register_tokens,
            cache_reference_token_span=cache_reference_token_span,
        )
        return (
            runtime_result.video_hidden,
            runtime_result.action_hidden,
            runtime_result.layout,
            runtime_result.cache_state,
            runtime_result.aux,
        )

    def _constant_future_video_timestep_grid(
        self,
        *,
        batch_size: int,
        num_video_frames: int,
        timestep_value: float,
        device: torch.device,
        observed_prefix_frames: int = 0,
    ) -> torch.Tensor:
        return build_joint_video_timestep_grid(
            batch_size=batch_size,
            num_video_frames=num_video_frames,
            timestep_value=float(timestep_value),
            device=device,
            observed_prefix_frames=observed_prefix_frames,
            observed_timestep_value=0.0,
        )

    def _preserve_observed_video_prefix(
        self,
        *,
        rollout_video_latents: torch.Tensor,
        observed_video_latents: torch.Tensor,
        observed_prefix_frames: int,
    ) -> torch.Tensor:
        return preserve_joint_observed_video_prefix(
            rollout_video_latents=rollout_video_latents,
            observed_video_latents=observed_video_latents,
            observed_prefix_frames=observed_prefix_frames,
        )

    def _constant_action_timestep_grid(
        self,
        *,
        batch_size: int,
        timestep_value: float,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full(
            (batch_size, self.action_horizon),
            fill_value=float(timestep_value),
            device=device,
            dtype=torch.float32,
        )

    def _warmup_runtime_cache(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        cache_state: CacheState,
        state_inputs: torch.Tensor,
        guidance_cfg_mode: str,
        current_start_frame: int,
        cache_reference_token_span: tuple[int, int],
    ) -> CacheState:
        """Warm the shared cache with clean reference-video context.

        This mirrors DreamZero's runtime pattern more closely than writing the
        current chunk into cache at the tail of the denoising loop. The warmup
        pass commits the clean reference video to cache first, then the inner
        denoising loop reuses that frozen context.
        """

        batch_size = visual_outputs.frontend.video_tokens.shape[0]
        device = visual_outputs.frontend.video_tokens.device
        zero_action_inputs = torch.zeros(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=visual_outputs.frontend.video_tokens.dtype,
        )
        zero_video_timesteps = torch.zeros(
            batch_size,
            visual_outputs.frontend.token_grid.num_frames,
            device=device,
            dtype=torch.float32,
        )
        zero_action_timesteps = torch.zeros(
            batch_size,
            self.action_horizon,
            device=device,
            dtype=torch.float32,
        )
        _, _, _, warmed_cache_state, _ = self._run_packed_core(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            noisy_video_tokens=visual_outputs.frontend.video_tokens,
            clean_video_prefix_tokens=None,
            action_inputs=zero_action_inputs,
            state_inputs=state_inputs,
            video_timesteps=zero_video_timesteps,
            action_timesteps=zero_action_timesteps,
            current_start_frame=current_start_frame,
            cache_state=cache_state,
            cache_update_metadata=visual_tower.build_runtime_cache_update_metadata(
                cache_state,
                current_start_frame=current_start_frame,
                update_kv_cache=True,
                update_cross_attention_cache=True,
                cfg_mode=guidance_cfg_mode,
            ),
            include_register_tokens=False,
            cache_reference_token_span=cache_reference_token_span,
        )
        return warmed_cache_state

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        batch = prepared_inputs.batch
        if batch.state is None:
            raise ValueError("Register-attached variant requires state inputs.")
        video_artifacts = prepared_inputs.variant_inputs["video_flow_match_train_artifacts"]
        action_artifacts = prepared_inputs.variant_inputs["action_flow_match_train_artifacts"]
        noisy_visual_outputs = self._build_noisy_frontend_outputs(
            visual_tower,
            visual_outputs=visual_outputs,
            noisy_video_latents=video_artifacts.noisy_latents,
        )
        video_hidden, action_hidden, layout, _, core_aux = self._run_packed_core(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            noisy_video_tokens=noisy_visual_outputs.frontend.video_tokens,
            clean_video_prefix_tokens=visual_outputs.frontend.video_tokens,
            action_inputs=action_artifacts.noisy_actions,
            state_inputs=batch.state.to(device=action_artifacts.noisy_actions.device, dtype=action_artifacts.noisy_actions.dtype),
            video_timesteps=video_artifacts.timesteps,
            action_timesteps=action_artifacts.timesteps,
            current_start_frame=0,
        )
        video_flow_pred = self._decode_future_video_flow(
            hidden_tokens=video_hidden,
            token_grid=visual_outputs.frontend.token_grid,
        )
        action_flow_pred = self.action_flow_head(action_hidden)
        denoised_video_latents = denoised_video_latents_from_flow(
            noisy_latents=video_artifacts.noisy_latents,
            flow_pred=video_flow_pred,
            timesteps=video_artifacts.timesteps,
            scheduler=video_artifacts.scheduler,
        )
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=action_artifacts.noisy_actions,
            flow_pred=action_flow_pred,
            timesteps=action_artifacts.timesteps,
            scheduler=action_artifacts.scheduler,
        )
        latent_loss = reduce_video_flow_match_loss(
            flow_pred=video_flow_pred,
            targets=video_artifacts.targets,
            timesteps=video_artifacts.timesteps,
            scheduler=video_artifacts.scheduler,
        )
        action_loss = reduce_slot_aligned_action_flow_match_loss(
            flow_pred=action_flow_pred,
            targets=action_artifacts.targets,
            timesteps=action_artifacts.timesteps,
            scheduler=action_artifacts.scheduler,
            action_mask=action_artifacts.action_mask,
        )
        total_loss = latent_loss + action_loss
        if batch.action_mask is not None:
            action_mse = torch.nn.functional.mse_loss(
                denoised_actions.float(),
                batch.actions.float(),
                reduction="none",
            )
            action_mse = action_mse * batch.action_mask.float()
            action_mse_value = action_mse.sum() / batch.action_mask.float().sum().clamp_min(1.0)
        else:
            action_mse_value = torch.nn.functional.mse_loss(denoised_actions.float(), batch.actions.float())
        decoder_output = ActionDecoderTrainOutput(
            action_pred=denoised_actions,
            loss=total_loss,
            metrics={
                "action_mse": action_mse_value.detach(),
                "video_diffusion_loss": latent_loss.detach(),
                "action_diffusion_loss": action_loss.detach(),
                "joint_loss": total_loss.detach(),
            },
            aux={
                "decoder": "RegisterAttachedJointDiffusion",
                "predicted_video_latents": denoised_video_latents.detach(),
                "future_video_flow_pred": video_flow_pred.detach(),
                "action_flow_pred": action_flow_pred.detach(),
            },
        )
        return PolicyTrainOutput(
            policy_features=action_hidden,
            metrics={"num_image_blocks": torch.tensor(float(layout.num_image_blocks), device=action_hidden.device)},
            aux={
                "variant": self.config.name,
                "layout": layout,
                "core_aux": core_aux,
                "video_flow_match_train_artifacts": video_artifacts,
                "action_flow_match_train_artifacts": action_artifacts,
                "decoder_output": decoder_output,
            },
        )

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        del context
        if previous_state is not None:
            return previous_state
        cursor = RolloutCursor(current_start_frame=0, block_index=0, chunk_size=self.config.num_frame_per_block)
        return PolicyInferState(
            step_index=0,
            cursor=cursor,
            cache=visual_tower.resolve_runtime_cache_state(
                None,
                cursor=cursor,
                stage="register_attached_method2",
                payload={"num_frame_per_block": self.config.num_frame_per_block},
                cfg_mode="joint",
                max_cached_frames=None,
            ),
        )

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        batch_size = visual_outputs.frontend.video_tokens.shape[0]
        dtype = visual_outputs.frontend.video_tokens.dtype
        device = visual_outputs.frontend.video_tokens.device
        if context.state is None:
            state_inputs = torch.zeros(batch_size, self.state_horizon, self.state_dim, device=device, dtype=dtype)
        else:
            state_inputs = context.state.to(device=device, dtype=dtype)
        cache_state = (
            visual_tower.resolve_runtime_cache_state(
                infer_state.cache if isinstance(infer_state.cache, CacheState) else None,
                cursor=infer_state.cursor,
                stage="register_attached_method2",
                payload={"num_frame_per_block": self.config.num_frame_per_block},
                cfg_mode="joint",
                max_cached_frames=None,
            )
        )
        observed_video_latents = visual_outputs.frontend.video_latents
        observed_prefix_frames = max(
            0,
            min(
                int(self.inference_config.joint_observed_video_prefix_frames),
                observed_video_latents.shape[2],
            ),
        )
        # In inference there is no clean teacher-forcing prefix in the packed
        # sequence, so the observed video prefix stays anchored directly in the
        # latent tensor while only the generated suffix is denoised.
        noisy_video_latents = self._preserve_observed_video_prefix(
            rollout_video_latents=torch.randn_like(observed_video_latents),
            observed_video_latents=observed_video_latents,
            observed_prefix_frames=observed_prefix_frames,
        )
        noisy_actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        scheduler_bundle = build_joint_runtime_schedulers(
            training_config=self.training_config,
            inference_config=self.inference_config,
            device=device,
        )
        cache_policy = resolve_runtime_cache_policy(
            inference_config=self.inference_config,
        )
        guidance = resolve_runtime_guidance(
            visual_outputs.frontend.conditioning,
            inference_config=self.inference_config,
        )
        unconditional_conditioning = build_unconditional_conditioning(
            visual_outputs.frontend.conditioning
        )
        video_scheduler = scheduler_bundle.video_scheduler
        action_scheduler = scheduler_bundle.action_scheduler
        use_unipc = scheduler_bundle.use_unipc
        if len(video_scheduler.timesteps) != len(action_scheduler.timesteps):
            raise ValueError(
                "Register-attached joint inference expects video/action schedulers with the same number of steps, "
                f"got video={len(video_scheduler.timesteps)} and action={len(action_scheduler.timesteps)}."
            )
        layout: RegisterSequenceLayout | None = None
        core_aux: dict[str, object] = {}
        latest_core_cache = cache_state
        warmup_reference = resolve_runtime_warmup_reference(
            policy=cache_policy,
            current_start_frame=int(infer_state.cursor.current_start_frame),
            num_video_frames=visual_outputs.frontend.token_grid.num_frames,
            num_frame_per_block=self.config.num_frame_per_block,
        )
        if warmup_reference is not None:
            tokens_per_frame = visual_outputs.frontend.token_grid.tokens_per_frame
            warmup_token_span = (
                warmup_reference.frame_start * tokens_per_frame,
                (warmup_reference.frame_start + warmup_reference.frame_count) * tokens_per_frame,
            )
            latest_core_cache = self._warmup_runtime_cache(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                cache_state=latest_core_cache,
                state_inputs=state_inputs,
                guidance_cfg_mode=guidance.cfg_mode,
                current_start_frame=int(infer_state.cursor.current_start_frame),
                cache_reference_token_span=warmup_token_span,
            )
        for step_index, (video_timestep, action_timestep) in enumerate(
            zip(video_scheduler.timesteps.to(device=device), action_scheduler.timesteps.to(device=device))
        ):
            input_cache_state = latest_core_cache
            noisy_visual_outputs = self._build_noisy_frontend_outputs(
                visual_tower,
                visual_outputs=visual_outputs,
                noisy_video_latents=noisy_video_latents,
            )
            step_cache_update = visual_tower.build_runtime_cache_update_metadata(
                input_cache_state,
                current_start_frame=int(infer_state.cursor.current_start_frame),
                update_kv_cache=should_update_cache_during_denoise(
                    cache_policy,
                    step_index=step_index,
                    num_steps=len(video_scheduler.timesteps),
                ),
                update_cross_attention_cache=cache_policy.update_cross_attention_during_denoise,
                cfg_mode=guidance.cfg_mode,
            )
            video_hidden, action_hidden, layout, latest_core_cache, core_aux = self._run_packed_core(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                noisy_video_tokens=noisy_visual_outputs.frontend.video_tokens,
                clean_video_prefix_tokens=None,
                action_inputs=noisy_actions,
                state_inputs=state_inputs,
                video_timesteps=self._constant_future_video_timestep_grid(
                    batch_size=batch_size,
                    num_video_frames=visual_outputs.frontend.token_grid.num_frames,
                    timestep_value=float(video_timestep),
                    device=device,
                    observed_prefix_frames=observed_prefix_frames,
                ),
                action_timesteps=self._constant_action_timestep_grid(
                    batch_size=batch_size,
                    timestep_value=float(action_timestep),
                    device=device,
                ),
                current_start_frame=int(infer_state.cursor.current_start_frame),
                cache_state=input_cache_state,
                cache_update_metadata=step_cache_update,
            )
            video_flow_pred = self._decode_future_video_flow(
                hidden_tokens=video_hidden,
                token_grid=visual_outputs.frontend.token_grid,
            )
            action_flow_pred = self.action_flow_head(action_hidden)
            if guidance.enabled and unconditional_conditioning is not None:
                uncond_video_hidden, uncond_action_hidden, _, _, _ = self._run_packed_core(
                    visual_tower=visual_tower,
                    visual_outputs=visual_outputs,
                    noisy_video_tokens=noisy_visual_outputs.frontend.video_tokens,
                    clean_video_prefix_tokens=None,
                    action_inputs=noisy_actions,
                    state_inputs=state_inputs,
                    video_timesteps=self._constant_future_video_timestep_grid(
                        batch_size=batch_size,
                        num_video_frames=visual_outputs.frontend.token_grid.num_frames,
                        timestep_value=float(video_timestep),
                        device=device,
                        observed_prefix_frames=observed_prefix_frames,
                    ),
                    action_timesteps=self._constant_action_timestep_grid(
                        batch_size=batch_size,
                        timestep_value=float(action_timestep),
                        device=device,
                    ),
                    current_start_frame=int(infer_state.cursor.current_start_frame),
                    cache_state=input_cache_state,
                    cache_update_metadata=visual_tower.build_runtime_cache_update_metadata(
                        input_cache_state,
                        current_start_frame=int(infer_state.cursor.current_start_frame),
                        update_kv_cache=False,
                        update_cross_attention_cache=False,
                        cfg_mode=guidance.cfg_mode,
                    ),
                    conditioning_override=unconditional_conditioning,
                )
                uncond_video_flow_pred = self._decode_future_video_flow(
                    hidden_tokens=uncond_video_hidden,
                    token_grid=visual_outputs.frontend.token_grid,
                )
                uncond_action_flow_pred = self.action_flow_head(uncond_action_hidden)
                video_flow_pred, action_flow_pred = combine_joint_cfg_predictions(
                    conditioned_video_prediction=video_flow_pred,
                    unconditioned_video_prediction=uncond_video_flow_pred,
                    conditioned_action_prediction=action_flow_pred,
                    unconditioned_action_prediction=uncond_action_flow_pred,
                    guidance=guidance,
                )
            if use_unipc:
                noisy_video_latents = video_scheduler.step(
                    video_flow_pred,
                    video_timestep,
                    noisy_video_latents,
                    step_index=step_index,
                    return_dict=False,
                )[0]
                noisy_video_latents = self._preserve_observed_video_prefix(
                    rollout_video_latents=noisy_video_latents,
                    observed_video_latents=observed_video_latents,
                    observed_prefix_frames=observed_prefix_frames,
                )
                noisy_actions = action_scheduler.step(
                    action_flow_pred,
                    action_timestep,
                    noisy_actions,
                    step_index=step_index,
                    return_dict=False,
                )[0]
            else:
                noisy_video_latents = video_scheduler.step(
                    video_flow_pred,
                    video_timestep,
                    noisy_video_latents,
                    to_final=step_index == len(video_scheduler.timesteps) - 1,
                )
                noisy_video_latents = self._preserve_observed_video_prefix(
                    rollout_video_latents=noisy_video_latents,
                    observed_video_latents=observed_video_latents,
                    observed_prefix_frames=observed_prefix_frames,
                )
                noisy_actions = action_scheduler.step(
                    action_flow_pred,
                    action_timestep,
                    noisy_actions,
                    to_final=step_index == len(action_scheduler.timesteps) - 1,
                )
        next_cursor = advance_rollout_cursor(infer_state.cursor)
        next_cache = visual_tower.advance_runtime_cache_state(
            latest_core_cache,
            next_cursor=next_cursor,
            payload_updates={"num_frame_per_block": self.config.num_frame_per_block},
            tokens_per_frame=layout.tokens_per_frame if layout is not None else None,
        )
        decoder_output = ActionDecoderInferOutput(
            action_pred=noisy_actions,
            aux={
                "decoder": "RegisterAttachedJointDiffusion",
                "predicted_latents": noisy_video_latents.detach(),
                "video_num_inference_steps": torch.tensor(float(len(video_scheduler.timesteps)), device=device),
                "action_num_inference_steps": torch.tensor(float(len(action_scheduler.timesteps)), device=device),
                "joint_sampler": self.inference_config.joint_sampler,
                "joint_cfg_mode": guidance.cfg_mode,
                "joint_cfg_enabled": guidance.enabled,
            },
        )
        return PolicyInferOutput(
            policy_features=noisy_actions,
            next_state=PolicyInferState(
                step_index=infer_state.step_index + 1,
                cursor=next_cursor,
                cache=next_cache,
            ),
            aux={
                "variant": self.config.name,
                "layout": layout,
                "core_aux": core_aux,
                "predicted_latents": noisy_video_latents.detach(),
                "decoder_output": decoder_output,
            },
        )
