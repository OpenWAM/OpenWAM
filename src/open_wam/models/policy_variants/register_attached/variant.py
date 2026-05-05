from __future__ import annotations

import torch

from open_wam.configs import InferenceConfig, RegisterAttachedPolicyConfig, TrainingConfig
from open_wam.models.common import (
    build_block_coupled_action_flow_match_train_artifacts,
    build_joint_video_timestep_grid,
    resolve_joint_train_flow_result,
    run_joint_inference_loop,
    build_video_flow_match_train_artifacts,
    denoised_actions_from_flow,
    denoised_video_latents_from_flow,
    preserve_joint_observed_video_prefix,
    reduce_slot_aligned_action_flow_match_loss,
    reduce_video_flow_match_loss,
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
from .deprecation import raise_register_attached_obsolete
from .layout import RegisterSequenceLayout
from .runtime import RegisterAttachedRuntime, RegisterRuntimeSpec


class RegisterAttachedPolicyVariant(PolicyVariant):
    """OBSOLETE traditional Method 2 register-attached policy variant.

    This class is kept only so historical checkpoints, notes, and tests can
    reference the old structure. Instantiating it raises an explicit obsolete
    warning and error; do not add new runtime behavior here.

    Historical design summary:

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
        raise_register_attached_obsolete(stacklevel=2)

        # Obsolete implementation retained below for archaeology only.
        super().__init__()
        self.config = config
        self.backbone_config = backbone_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.state_horizon = state_horizon
        self.runtime = RegisterAttachedRuntime(
            RegisterRuntimeSpec(
                # Variants pick the rollout program; the shared runtime/backbone
                # own tokenizers, structured attention kernels, cache semantics,
                # and stream output heads.
                hidden_size=config.hidden_size,
                action_horizon=action_horizon,
                state_horizon=state_horizon,
                num_frame_per_block=config.num_frame_per_block,
                num_action_per_block=config.num_action_per_block,
                num_state_per_block=config.num_state_per_block,
                variant_name=config.name,
                structured_block_mode=config.structured_block_mode,
                structured_time_layout=config.structured_time_layout,
                structured_frequency_mode=config.structured_frequency_mode,
                structured_teacher_forcing_layout=config.structured_teacher_forcing_layout,
                structured_attention_kernel=config.structured_attention_kernel,
                structured_cache_kernel=config.structured_cache_kernel,
                stream_input_adapter_family=config.stream_input_adapter_family,
                stream_output_head_family=config.stream_output_head_family,
                use_state_encoder=config.use_state_encoder,
                action_encoder_type=config.action_encoder_type,
                state_encoder_type=config.state_encoder_type,
            )
        )
        self.video_patch_dim = (
            self.backbone_config.latent_channels
            * self.backbone_config.patch_size_t
            * self.backbone_config.patch_size_h
            * self.backbone_config.patch_size_w
        )

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

    def _expand_bootstrap_visual_outputs(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
    ) -> VisualStageOutputs:
        """Expand a single observed frame into a valid first-step bootstrap window.

        DreamZero serving sends one frame on the first call, but our current
        shared register-attached layout still expects the train-time block
        counts implied by `data.num_frames`. We keep the first frame as the only
        observed-prefix frame and repeat its latent/context to synthesize the
        remaining bootstrap slots.
        """

        frontend = visual_outputs.frontend
        if frontend.video_latents.shape[2] != 1:
            return visual_outputs

        target_frames = 1 + self.action_horizon // self.config.num_action_per_block
        repeated_latents = frontend.video_latents.repeat_interleave(target_frames, dim=2)
        repeated_canonical = None
        if frontend.canonical_video is not None:
            repeated_canonical = frontend.canonical_video.repeat_interleave(target_frames, dim=2)
        bootstrap_frontend = visual_tower.run_frontend_from_latents(
            repeated_latents,
            task_text=None,
            text_context=frontend.conditioning.text_context,
            negative_text_context=frontend.conditioning.negative_text_context,
            canonical_video=repeated_canonical,
        )
        return VisualStageOutputs(frontend=bootstrap_frontend)

    def _build_rollout_generation_visual_outputs(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
    ) -> VisualStageOutputs:
        """Build the inference-time denoising window for only the current future block.

        DreamZero rollout only denoises the currently requested future video
        block. The observed prefix should warm up the cache separately rather
        than living inside the denoised tensor itself.
        """

        frontend = visual_outputs.frontend
        generation_frames = self.config.num_frame_per_block
        available_frames = int(frontend.video_latents.shape[2])
        if available_frames < 1:
            raise ValueError("Inference rollout requires at least one observed frame.")
        observed_anchor_latent = frontend.video_latents[:, :, -1:]
        rollout_latents = observed_anchor_latent.repeat_interleave(max(generation_frames, 1), dim=2)

        rollout_canonical = None
        if frontend.canonical_video is not None:
            observed_anchor_canonical = frontend.canonical_video[:, -1:]
            rollout_canonical = observed_anchor_canonical.repeat_interleave(max(generation_frames, 1), dim=1)

        rollout_frontend = visual_tower.run_frontend_from_latents(
            rollout_latents,
            task_text=None,
            text_context=frontend.conditioning.text_context,
            negative_text_context=frontend.conditioning.negative_text_context,
            canonical_video=rollout_canonical,
        )
        return VisualStageOutputs(frontend=rollout_frontend)

    def _build_observed_prefix_visual_outputs(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
    ) -> VisualStageOutputs | None:
        frontend = visual_outputs.frontend
        available_frames = int(frontend.video_latents.shape[2])
        prefix_frames = max(
            0,
            min(
                int(self.inference_config.joint_observed_video_prefix_frames),
                available_frames,
            ),
        )
        if prefix_frames <= 0:
            return None
        prefix_latents = frontend.video_latents[:, :, -prefix_frames:]
        prefix_canonical = None
        if frontend.canonical_video is not None:
            prefix_canonical = frontend.canonical_video[:, -prefix_frames:]
        prefix_frontend = visual_tower.run_frontend_from_latents(
            prefix_latents,
            task_text=None,
            text_context=frontend.conditioning.text_context,
            negative_text_context=frontend.conditioning.negative_text_context,
            canonical_video=prefix_canonical,
        )
        return VisualStageOutputs(frontend=prefix_frontend)

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
        require_matching_block_counts: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], RegisterSequenceLayout, CacheState, dict[str, object]]:
        # Keep method 2 on the shared runtime surface so future within-core
        # variants can reuse the same tokenization, attention, and cache stack.
        runtime_result = self.runtime.run_core(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            clean_video_prefix_tokens=clean_video_prefix_tokens,
            noisy_video_tokens=noisy_video_tokens,
            action_inputs=action_inputs,
            state_inputs=state_inputs,
            video_timesteps=video_timesteps,
            action_timesteps=action_timesteps,
            current_start_frame=current_start_frame,
            cache_state=cache_state,
            cache_update_metadata=cache_update_metadata,
            conditioning_override=conditioning_override,
            include_register_tokens=include_register_tokens,
            cache_reference_token_span=cache_reference_token_span,
            require_matching_block_counts=require_matching_block_counts,
        )
        return (
            runtime_result.video_hidden,
            runtime_result.action_hidden,
            runtime_result.projected_outputs,
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
        cache_branch: str,
        conditioning_override=None,
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
        _, _, _, _, warmed_cache_state, _ = self._run_packed_core(
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
                cache_branch=cache_branch,
            ),
            conditioning_override=conditioning_override,
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
        video_hidden, action_hidden, projected_outputs, layout, _, core_aux = self._run_packed_core(
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
        train_result = resolve_joint_train_flow_result(
            projected_outputs=projected_outputs,
            video_artifacts=video_artifacts,
            action_artifacts=action_artifacts,
            unpatchify_video_prediction=lambda video_patch_flow: unpatchify_video_tokens(
                video_patch_flow,
                token_grid=visual_outputs.frontend.token_grid,
                latent_channels=self.backbone_config.latent_channels,
            ),
            denoised_video_latents_from_flow=denoised_video_latents_from_flow,
            denoised_actions_from_flow=denoised_actions_from_flow,
            reduce_video_flow_match_loss=reduce_video_flow_match_loss,
            reduce_slot_aligned_action_flow_match_loss=reduce_slot_aligned_action_flow_match_loss,
        )
        weighted_latent_loss = (
            train_result.latent_loss * float(self.training_config.objective_weight("latent"))
            if self.training_config.objective_enabled("latent")
            else torch.zeros_like(train_result.latent_loss)
        )
        weighted_action_loss = (
            train_result.action_loss * float(self.training_config.objective_weight("action"))
            if self.training_config.objective_enabled("action")
            else torch.zeros_like(train_result.action_loss)
        )
        total_loss = weighted_latent_loss + weighted_action_loss
        if batch.action_mask is not None:
            action_mse = torch.nn.functional.mse_loss(
                train_result.denoised_actions.float(),
                batch.actions.float(),
                reduction="none",
            )
            action_mse = action_mse * batch.action_mask.float()
            action_mse_value = action_mse.sum() / batch.action_mask.float().sum().clamp_min(1.0)
        else:
            action_mse_value = torch.nn.functional.mse_loss(train_result.denoised_actions.float(), batch.actions.float())
        return PolicyTrainOutput(
            policy_features=train_result.denoised_actions,
            metrics={"num_image_blocks": torch.tensor(float(layout.num_image_blocks), device=action_hidden.device)},
            aux={
                "variant": self.config.name,
                "layout": layout,
                "core_aux": core_aux,
                "video_flow_match_train_artifacts": video_artifacts,
                "action_flow_match_train_artifacts": action_artifacts,
                "predicted_latents": train_result.denoised_video_latents.detach(),
                "joint_train_decoder_artifacts": {
                    "action_pred": train_result.denoised_actions,
                    "loss": total_loss,
                    "metrics": {
                        "action_mse": action_mse_value.detach(),
                        "video_diffusion_loss": train_result.latent_loss.detach(),
                        "action_diffusion_loss": train_result.action_loss.detach(),
                        "weighted_video_diffusion_loss": weighted_latent_loss.detach(),
                        "weighted_action_diffusion_loss": weighted_action_loss.detach(),
                        "joint_loss": total_loss.detach(),
                    },
                    "aux": {
                        "predicted_video_latents": train_result.denoised_video_latents.detach(),
                        "future_video_flow_pred": train_result.video_flow_pred.detach(),
                        "action_flow_pred": train_result.action_flow_pred.detach(),
                    },
                },
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
        if visual_outputs.frontend.token_grid.num_frames == 1:
            visual_outputs = self._expand_bootstrap_visual_outputs(visual_tower, visual_outputs)
        reference_visual_outputs = visual_outputs
        observed_prefix_visual_outputs = self._build_observed_prefix_visual_outputs(
            visual_tower,
            reference_visual_outputs,
        )
        rollout_visual_outputs = self._build_rollout_generation_visual_outputs(
            visual_tower,
            reference_visual_outputs,
        )
        dtype = rollout_visual_outputs.frontend.video_tokens.dtype
        device = rollout_visual_outputs.frontend.video_tokens.device
        if context.state is None:
            state_inputs = torch.zeros(
                rollout_visual_outputs.frontend.video_tokens.shape[0],
                self.state_horizon,
                self.state_dim,
                device=device,
                dtype=dtype,
            )
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
        observed_prefix_frames = int(self.inference_config.joint_observed_video_prefix_frames)
        denoise_start_frame = int(infer_state.cursor.current_start_frame)
        infer_result = run_joint_inference_loop(
            visual_tower=visual_tower,
            visual_outputs=rollout_visual_outputs,
            reference_visual_outputs=(
                observed_prefix_visual_outputs
                if observed_prefix_visual_outputs is not None
                else reference_visual_outputs
            ),
            training_config=self.training_config,
            inference_config=self.inference_config,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            num_frame_per_block=self.config.num_frame_per_block,
            cache_state=cache_state,
            state_inputs=state_inputs,
            current_start_frame=denoise_start_frame,
            warmup_current_start_frame=int(infer_state.cursor.current_start_frame),
            observed_prefix_frames_override=0,
            build_noisy_visual_outputs=lambda noisy_video_latents: self._build_noisy_frontend_outputs(
                visual_tower,
                visual_outputs=rollout_visual_outputs,
                noisy_video_latents=noisy_video_latents,
            ),
            preserve_observed_video_prefix=lambda rollout_video_latents, observed_video_latents, observed_prefix_frames: self._preserve_observed_video_prefix(
                rollout_video_latents=rollout_video_latents,
                observed_video_latents=observed_video_latents,
                observed_prefix_frames=observed_prefix_frames,
            ),
            constant_future_video_timestep_grid=lambda batch_size, num_video_frames, timestep_value, device, observed_prefix_frames: self._constant_future_video_timestep_grid(
                batch_size=batch_size,
                num_video_frames=num_video_frames,
                timestep_value=timestep_value,
                device=device,
                observed_prefix_frames=observed_prefix_frames,
            ),
            constant_action_timestep_grid=lambda batch_size, timestep_value, device: self._constant_action_timestep_grid(
                batch_size=batch_size,
                timestep_value=timestep_value,
                device=device,
            ),
            warmup_runtime_cache=lambda latest_core_cache, warmup_state_inputs, guidance_cfg_mode, cache_reference_token_span, cache_branch, conditioning_override: self._warmup_runtime_cache(
                visual_tower=visual_tower,
                visual_outputs=reference_visual_outputs,
                cache_state=latest_core_cache,
                state_inputs=warmup_state_inputs,
                guidance_cfg_mode=guidance_cfg_mode,
                current_start_frame=int(infer_state.cursor.current_start_frame),
                cache_reference_token_span=cache_reference_token_span,
                cache_branch=cache_branch,
                conditioning_override=conditioning_override,
            ),
            run_conditioned_core=lambda noisy_video_tokens, noisy_actions, video_timestep_grid, action_timestep_grid, input_cache_state, cache_update_metadata: self._run_packed_core(
                visual_tower=visual_tower,
                visual_outputs=rollout_visual_outputs,
                noisy_video_tokens=noisy_video_tokens,
                clean_video_prefix_tokens=None,
                action_inputs=noisy_actions,
                state_inputs=state_inputs,
                video_timesteps=video_timestep_grid,
                action_timesteps=action_timestep_grid,
                current_start_frame=denoise_start_frame,
                cache_state=input_cache_state,
                cache_update_metadata=cache_update_metadata,
                require_matching_block_counts=False,
            ),
            run_unconditioned_core=lambda noisy_video_tokens, noisy_actions, video_timestep_grid, action_timestep_grid, input_cache_state, cache_update_metadata, conditioning_override: self._run_packed_core(
                visual_tower=visual_tower,
                visual_outputs=rollout_visual_outputs,
                noisy_video_tokens=noisy_video_tokens,
                clean_video_prefix_tokens=None,
                action_inputs=noisy_actions,
                state_inputs=state_inputs,
                video_timesteps=video_timestep_grid,
                action_timesteps=action_timestep_grid,
                current_start_frame=denoise_start_frame,
                cache_state=input_cache_state,
                cache_update_metadata=cache_update_metadata,
                conditioning_override=conditioning_override,
                require_matching_block_counts=False,
            ),
            unpatchify_video_prediction=lambda video_patch_flow: unpatchify_video_tokens(
                video_patch_flow,
                token_grid=rollout_visual_outputs.frontend.token_grid,
                latent_channels=self.backbone_config.latent_channels,
            ),
        )
        next_cursor = advance_rollout_cursor(infer_state.cursor)
        next_cache = visual_tower.advance_runtime_cache_state(
            infer_result.latest_core_cache,
            next_cursor=next_cursor,
            payload_updates={"num_frame_per_block": self.config.num_frame_per_block},
            tokens_per_frame=infer_result.layout.tokens_per_frame if infer_result.layout is not None else None,
        )
        return PolicyInferOutput(
            policy_features=infer_result.noisy_actions,
            next_state=PolicyInferState(
                step_index=infer_state.step_index + 1,
                cursor=next_cursor,
                cache=next_cache,
            ),
            aux={
                "variant": self.config.name,
                "layout": infer_result.layout,
                "core_aux": infer_result.core_aux,
                "structured_attention_full_cache_prefix": infer_result.core_aux.get(
                    "structured_attention_full_cache_prefix",
                    False,
                ),
                "predicted_latents": infer_result.noisy_video_latents.detach(),
                "video_num_inference_steps": torch.tensor(float(infer_result.video_num_inference_steps), device=device),
                "action_num_inference_steps": torch.tensor(float(infer_result.action_num_inference_steps), device=device),
                "joint_sampler": self.inference_config.joint_sampler,
                "joint_cfg_mode": infer_result.guidance_cfg_mode,
                "joint_cfg_enabled": infer_result.guidance_enabled,
            },
        )
