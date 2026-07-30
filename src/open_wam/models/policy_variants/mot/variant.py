from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from open_wam.models.common.flow_matching import (
    FlowMatchScheduler,
    VideoFlowMatchTrainArtifacts,
    build_video_flow_match_train_artifacts,
    build_video_flow_match_inference_scheduler,
    build_action_flow_match_inference_scheduler,
    build_action_flow_match_train_artifacts,
    denoised_actions_from_flow,
    denoised_video_latents_from_flow,
    timesteps_matching_sigmas,
)
from open_wam.models.common.video_geometry import unpatchify_video_sequence
from open_wam.models.common.joint_conditioning import (
    resolve_generalist_joint_conditioning_semantics,
)
from open_wam.configs import (
    CurrentBlockCoupling,
    InferenceConfig,
    JointTimestepCoupling,
    MoTGeneralistTrainingMode,
    MoTPolicyConfig,
    MoTRuntimeMode,
    ParallelHistoryStreamVisibility,
    TrainingConfig,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower
from open_wam.models.visual_tower.exact_runtime import (
    clear_exact_prediction_cache,
    initialize_exact_runtime_cache,
    prepare_exact_single_stream_input,
    resolve_runtime_module_dtype,
    run_exact_single_stream_forward,
)
from open_wam.configs.backbone import SharedVideoTransformerConfig

from ..base import PolicyVariant
from ..common.layouts import expand_previous_action
from ..contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)
from .conditioning import MoTConditioning
from .contracts import (
    MoTActionCache,
    MoTActionLayerCache,
    MoTActionTrainArtifacts,
    MoTInferArtifacts,
    MoTRuntimeState,
    MoTTrainArtifacts,
    MoTVideoCache,
    MoTVideoLayerCache,
    MoTVideoTrainArtifacts,
)
from .generalist_modes import (
    apply_generalist_training_mode as _apply_mot_generalist_training_mode,
    generalist_rollout_enabled as _mot_generalist_rollout_enabled,
    generalist_rollout_mode_from_value as _mot_generalist_rollout_mode_from_value,
    is_generalist_conditional_rollout as _is_mot_generalist_conditional_rollout,
    resolve_generalist_rollout_mode as _resolve_mot_generalist_rollout_mode,
    sample_generalist_training_mode as _sample_mot_generalist_training_mode,
)
from .modules import MoTActionExpert, init_action_expert_from_video_core
from .packed_block import MoTPackedBlock, MoTPackedBlockStack
from .packed_inference import MoTPackedInferenceProgram
from .packed_training import MoTPackedTrainingProgram
from .runtime import (
    append_mot_action_cache,
    build_chunk_causal_video_mask,
    build_mot_attention_mask,
    build_mot_inference_action_attention_mask,
    forward_joint_video_action_denoise,
    forward_action_with_video_and_action_cache,
    forward_action_with_video_cache,
    expand_mot_scalar_timestep,
    mot_scheduler_next_sigma,
    move_mot_action_cache,
    move_mot_video_cache,
    prefill_video_kv_cache,
    rewind_mot_runtime_action_cache_to_frame,
    step_mot_flow_with_sigmas,
    trim_mot_action_cache_tail,
    trim_mot_video_cache_tail,
)
from .runtime_routing import (
    MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS,
    ensure_mot_policy_variant_inference_backend,
    is_mot_same_step_coupling,
    resolve_mot_action_only_rollout,
    resolve_mot_current_block_coupling,
    resolve_mot_inference_window_size,
    resolve_mot_joint_timestep_coupling,
    resolve_mot_rollout_frame_chunk_size,
    should_couple_mot_action_to_video_sigmas,
)
from .sequence_layout import MoTTrainingLayout, build_action_grid_ids_for_sequence

# Default LingBot-reference slot-pool window used by both `initialize_exact_runtime_cache`
# and the Method-1-aligned video-cache trim. Rollout callers may override this
# through `PolicyInferContext.extra["mot_inference_window_size"]`. Method 1's
# per-stream effective lookback is `(attn_window // 2) * frame_chunk_size`
# integer frames (60 at attn_window=30, frame_chunk_size=4).
_MOT_SLOT_POOL_ATTN_WINDOW = 30


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
        self.conditioning = MoTConditioning(config)
        self.training_layout = MoTTrainingLayout(config, training_config)
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
            hidden_context_dim=backbone_config.hidden_size,
            freq_dim=backbone_config.freq_dim,
            cross_attn_norm=backbone_config.cross_attn_norm,
            eps=backbone_config.latent_norm_eps,
        )
        self._action_expert_initialized = False
        self._train_video_cache_detach_by_core_id: dict[int, bool] = {}
        # Lazy-initialized at pipeline assembly time when current_block_coupling
        # is set. Owns video_block + action_block pairs after ownership transfer
        # so FSDP can wrap the packed unit cleanly without aliasing.
        self.packed_block_stack: MoTPackedBlockStack | None = None
        self._packed_block_stack_attached = False
        self._legacy_inference_blocks_restored = False

    def attach_visual_tower(self, visual_tower: VisualTower) -> None:
        """Pipeline-time hook: build the packed-coupling block stack.

        Must run AFTER both ``visual_tower`` and ``self.action_expert`` exist
        but BEFORE FSDP sharding. Transfers ownership of video core blocks and
        action expert blocks into ``self.packed_block_stack`` so FSDP only
        sees a single owner per nn.Parameter (no shared-module aliasing).
        Non-packed runtime modes are no-ops.

        ``_maybe_initialize_action_expert`` runs BEFORE the transfer because
        the init helper reads from ``visual_tower.core.blocks`` and writes to
        ``self.action_expert.blocks``; after transfer both ModuleLists are
        empty.
        """
        if self.conditioning.uses_text_proprio_context():
            configure = getattr(visual_tower.core, "configure_proprio_context_encoder", None)
            if not callable(configure):
                raise ValueError(
                    "Deprecated proprio_context_mode=text_context_token requires a core proprio encoder hook."
                )
            configure(enabled=True, state_dim=int(self.state_dim))
        elif self.conditioning.uses_per_chunk_proprio_context():
            configure = getattr(visual_tower.core, "configure_proprio_hidden_context_encoder", None)
            if not callable(configure):
                raise ValueError("proprio_context_mode=per_chunk_additive requires a core proprio hidden encoder hook.")
            configure(enabled=True, state_dim=int(self.state_dim))
        if self._packed_block_stack_attached:
            return
        self._packed_block_stack_attached = True
        if self.config.current_block_coupling is None:
            return
        # Run lazy action-expert init now, while blocks still live under
        # visual_tower.core / self.action_expert.
        self._maybe_initialize_action_expert(visual_tower)
        video_blocks = list(visual_tower.core.blocks)
        action_blocks = list(self.action_expert.blocks)
        # Build the stack first so it owns the children; then drop them from
        # the original ModuleList containers. Param identity is preserved
        # across the move (same nn.Parameter objects, just under a new parent),
        # so any optimizer built from `model.parameters()` after this hook runs
        # sees the same set.
        self.packed_block_stack = MoTPackedBlockStack(video_blocks, action_blocks)
        visual_tower.core.blocks = torch.nn.ModuleList()
        self.action_expert.blocks = torch.nn.ModuleList()

    def restore_packed_blocks_for_legacy_inference(self, visual_tower: VisualTower) -> bool:
        """Move packed-owned blocks back for inference-only legacy cache rollout.

        Packed training transfers block ownership into ``packed_block_stack`` so
        FSDP can shard paired video/action blocks cleanly. Legacy split-cache
        inference needs the pre-packed module lists, so this performs a
        one-way ownership transfer back to ``visual_tower.core.blocks`` and
        ``action_expert.blocks``. ``packed_block_stack`` is cleared afterward
        so the module tree has a single owner for each block.
        """

        if self.packed_block_stack is None:
            return False
        video_blocks = [packed_block.video_block for packed_block in self.packed_block_stack.packed_blocks]
        action_blocks = [packed_block.action_block for packed_block in self.packed_block_stack.packed_blocks]
        if not video_blocks or not action_blocks:
            return False
        visual_tower.core.blocks = torch.nn.ModuleList(video_blocks)
        self.action_expert.blocks = torch.nn.ModuleList(action_blocks)
        self.packed_block_stack = None
        self._legacy_inference_blocks_restored = True
        return True

    def _should_detach_train_video_cache(self, visual_tower: VisualTower) -> bool:
        core_id = id(visual_tower.core)
        detach_cache = self._train_video_cache_detach_by_core_id.get(core_id)
        if detach_cache is None:
            detach_cache = not any(parameter.requires_grad for parameter in visual_tower.core.parameters())
            self._train_video_cache_detach_by_core_id[core_id] = detach_cache
        return bool(detach_cache)

    def _build_video_train_rollout(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
        history_frames: int,
        condition_latents: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> MoTVideoTrainArtifacts:
        video_latents = visual_outputs.frontend.video_latents
        clean_condition_latents, _ = self.conditioning.train_clean_video_condition_latents(
            video_latents=video_latents,
            condition_latents=condition_latents,
            history_frames=history_frames,
        )
        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_condition_latents,
        )
        noisy_latents = video_artifacts.noisy_latents.clone()
        timesteps = video_artifacts.timesteps.clone()
        history_condition_latents = clean_condition_latents if clean_condition_latents is not None else video_latents
        noisy_latents[:, :, :history_frames] = history_condition_latents[:, :, :history_frames]
        timesteps[:, :history_frames] = 0.0
        future_loss_mask = self.training_layout.build_effective_video_loss_mask(
            video_latents=video_latents,
            batch=batch,
            default_history_frames=history_frames,
        )
        flow_pred = visual_tower.predict_video_flow(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_context=visual_outputs.frontend.conditioning.text_context,
            frame_start=0,
            attention_mask=attention_mask,
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=noisy_latents,
            flow_pred=flow_pred,
            timesteps=timesteps,
            scheduler=video_artifacts.scheduler,
        )
        return MoTVideoTrainArtifacts(
            flow_pred=flow_pred,
            targets=video_artifacts.targets,
            timesteps=timesteps,
            scheduler=video_artifacts.scheduler,
            predicted_latents=predicted_latents,
            target_latents=video_latents,
            future_loss_mask=future_loss_mask,
        )

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
        condition_latents = self.conditioning.resolve_train_condition_latents(
            batch,
            video_latents=visual_outputs.frontend.video_latents,
        )
        proprio_state = self.conditioning.resolve_train_proprio_context(batch)
        hidden_proprio_state = self.conditioning.resolve_train_hidden_proprio_context(batch)
        return PolicyPreparedInputs(
            batch=batch,
            variant_inputs={
                "video_latents": visual_outputs.frontend.video_latents,
                "condition_latents": condition_latents,
                "proprio_state": proprio_state,
                "hidden_proprio_state": hidden_proprio_state,
                "text_context": visual_outputs.frontend.conditioning.text_context,
                "video_tokens_per_frame": visual_outputs.frontend.token_grid.tokens_per_frame,
            },
        )

    def _resolve_history_stream_visibility(self) -> ParallelHistoryStreamVisibility:
        return ParallelHistoryStreamVisibility(self.config.history_stream_visibility)

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        self._maybe_initialize_action_expert(visual_tower)
        if self.config.current_block_coupling is not None:
            return self._forward_train_packed_coupling(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                prepared_inputs=prepared_inputs,
            )
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            return self._forward_train_joint_denoise(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                prepared_inputs=prepared_inputs,
            )
        if self.config.runtime_mode == MoTRuntimeMode.NON_JOINT_TWO_STREAM:
            return self._forward_train_non_joint_two_stream(
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
        condition_latents = prepared_inputs.variant_inputs.get("condition_latents")
        text_context = prepared_inputs.variant_inputs["text_context"]
        proprio_state = prepared_inputs.variant_inputs.get("proprio_state")
        hidden_proprio_state = prepared_inputs.variant_inputs.get("hidden_proprio_state")
        history_frames = self.training_layout.resolve_history_frames(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        clean_video_condition_latents, video_condition_source = self.conditioning.train_clean_video_condition_latents(
            video_latents=video_latents,
            condition_latents=condition_latents,
            history_frames=history_frames,
        )
        # This unused draw is part of the legacy prefill training RNG contract:
        # action noise/timesteps are sampled after the video artifacts.
        _ = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_video_condition_latents,
        )
        effective_action_mask = self.training_layout.build_effective_action_mask(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        action_tokens_per_frame = self.training_layout.resolve_action_tokens_per_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        video_tokens_per_frame = int(prepared_inputs.variant_inputs["video_tokens_per_frame"])
        sampled_chunk_size = self.training_layout.resolve_sampled_chunk_size(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        if sampled_chunk_size is None:
            sampled_chunk_size = max(
                1,
                min(int(self.training_config.chunk_size), int(video_latents.shape[2])),
            )
        sampled_window_size = self.training_layout.resolve_sampled_window_size(
            batch=prepared_inputs.batch,
        )
        if sampled_window_size is None:
            sampled_window_size = max(1, int(self.training_config.window_size))
        frame_shift = self.training_layout.resolve_frame_shift(batch=prepared_inputs.batch)
        chunk_origin_frame = self.training_layout.resolve_chunk_origin_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        chunk_causal_video_mask = build_chunk_causal_video_mask(
            video_seq_len=video_tokens_per_frame * int(video_latents.shape[2]),
            video_tokens_per_frame=video_tokens_per_frame,
            action_chunk_size_frames=sampled_chunk_size,
            device=video_latents.device,
            attention_window_size=sampled_window_size,
            chunk_origin_frame=chunk_origin_frame,
        )

        # Method-5 video-prefill action denoise is aligned to method-1 full-seg
        # semantics: the action expert conditions on the full clean video
        # sample, but the video K/V prefill itself stays chunk-causal so future
        # chunks do not leak through the shared video backbone.
        action_condition_latents = (
            clean_video_condition_latents if clean_video_condition_latents is not None else video_latents
        )
        video_text_context = self.conditioning.resolve_text_context(
            visual_tower,
            text_context,
            proprio_state,
            batch_size=int(action_condition_latents.shape[0]),
            device=action_condition_latents.device,
            dtype=action_condition_latents.dtype,
            materialize_if_missing=self.conditioning.uses_proprio_context(),
        )
        video_cross_attention_mask = self.conditioning.build_proprio_cross_attention_mask(
            resolved_text_context=video_text_context,
            proprio_state=proprio_state,
            query_frames_per_copy=int(action_condition_latents.shape[2]),
            tokens_per_frame=video_tokens_per_frame,
            chunk_size_frames=sampled_chunk_size,
            chunk_origin_frame=chunk_origin_frame,
        ) if video_text_context is not None else None
        video_cache = prefill_video_kv_cache(
            visual_tower=visual_tower,
            observed_prefix=action_condition_latents,
            text_context=video_text_context,
            frame_start=0,
            attention_mask=chunk_causal_video_mask,
            cross_attention_mask=video_cross_attention_mask,
            detach_cache=self._should_detach_train_video_cache(visual_tower),
        )
        train_artifacts = build_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            effective_action_mask,
            training_config=self.training_config,
        )
        train_artifacts = self.training_layout.apply_history_action_condition(
            train_artifacts=train_artifacts,
            actions=prepared_inputs.batch.actions,
            observed_num_frames=int(video_latents.shape[2]),
            history_frames=history_frames,
        )
        resolved_text = self.conditioning.resolve_text_context(
            visual_tower,
            text_context,
            proprio_state,
            batch_size=int(action_condition_latents.shape[0]),
            device=action_condition_latents.device,
            dtype=action_condition_latents.dtype,
            materialize_if_missing=True,
        )
        if resolved_text is None:  # pragma: no cover - materialized above
            raise RuntimeError("M5 action text context unexpectedly resolved to None.")
        action_cross_attention_mask = (
            None
            if action_tokens_per_frame is None
            else self.conditioning.build_proprio_cross_attention_mask(
                resolved_text_context=resolved_text,
                proprio_state=proprio_state,
                query_frames_per_copy=int(video_latents.shape[2]),
                tokens_per_frame=int(action_tokens_per_frame),
                chunk_size_frames=sampled_chunk_size,
                chunk_origin_frame=chunk_origin_frame,
            )
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=train_artifacts.noisy_actions,
            timestep=train_artifacts.timesteps,
            context=resolved_text,
            cross_attention_mask=action_cross_attention_mask,
            action_grid_ids=build_action_grid_ids_for_sequence(
                batch_size=train_artifacts.noisy_actions.shape[0],
                seq_len=train_artifacts.noisy_actions.shape[1],
                action_tokens_per_frame=action_tokens_per_frame,
                device=train_artifacts.noisy_actions.device,
                frame_shift=frame_shift,
            ) if action_tokens_per_frame is not None else None,
            hidden_context=(
                None
                if action_tokens_per_frame is None
                else self.conditioning.action_hidden_context_for_tokens(
                    visual_tower,
                    hidden_proprio_state,
                    action_tokens=train_artifacts.noisy_actions,
                    action_tokens_per_frame=int(action_tokens_per_frame),
                    chunk_size_frames=sampled_chunk_size,
                )
            ),
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
                action_tokens_per_frame=action_tokens_per_frame,
                action_chunk_size_frames=sampled_chunk_size,
                clean_video_frames=int(video_latents.shape[2]),
                clean_action_frames=history_frames,
                attention_window_size=sampled_window_size,
            ),
        )
        flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=train_artifacts.noisy_actions,
            flow_pred=flow_pred,
            timesteps=train_artifacts.timesteps,
            scheduler=train_artifacts.scheduler,
        )
        video_rollout: MoTVideoTrainArtifacts | None = None
        if self.training_config.objective_enabled("latent"):
            video_rollout = self._build_video_train_rollout(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                batch=prepared_inputs.batch,
                history_frames=history_frames,
                condition_latents=condition_latents,
                attention_mask=chunk_causal_video_mask,
            )
        batch_size = action_condition_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=action_condition_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "mot_history_frames": action_condition_latents.new_tensor(float(history_frames)),
                "mot_video_prefix_frames": action_condition_latents.new_tensor(float(history_frames)),
            },
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "video_cache_seq_len": video_cache.video_seq_len,
                "runtime_mode": str(self.config.runtime_mode),
                "sampled_chunk_size": sampled_chunk_size,
                "sampled_window_size": sampled_window_size,
                "video_condition_source": video_condition_source,
                "mot_train_artifacts": MoTTrainArtifacts(
                    action=MoTActionTrainArtifacts(
                        flow_pred=flow_pred,
                        targets=train_artifacts.targets,
                        timesteps=train_artifacts.timesteps,
                        scheduler=train_artifacts.scheduler,
                        denoised_actions=denoised_actions,
                        action_mask=train_artifacts.action_mask,
                    ),
                    video=video_rollout,
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                    history_frames=int(history_frames),
                    video_cache_seq_len=video_cache.video_seq_len,
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
        condition_latents = prepared_inputs.variant_inputs.get("condition_latents")
        text_context = prepared_inputs.variant_inputs["text_context"]
        proprio_state = prepared_inputs.variant_inputs.get("proprio_state")
        hidden_proprio_state = prepared_inputs.variant_inputs.get("hidden_proprio_state")
        current_block_coupling = resolve_mot_current_block_coupling(self.config)
        if not is_mot_same_step_coupling(current_block_coupling):
            raise NotImplementedError(
                "M5 joint_denoise train supports same-step couplings only; "
                f"got current_block_coupling={current_block_coupling.value!r}. "
                "Use runtime_mode='non_joint_two_stream' for staged video_then_action."
            )
        history_frames = self.training_layout.resolve_history_frames(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        clean_video_condition_latents, video_condition_source = self.conditioning.train_clean_video_condition_latents(
            video_latents=video_latents,
            condition_latents=condition_latents,
            history_frames=history_frames,
        )

        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_video_condition_latents,
        )
        noisy_video_latents = video_artifacts.noisy_latents.clone()
        video_timesteps = video_artifacts.timesteps.clone()
        history_condition_latents = clean_video_condition_latents if clean_video_condition_latents is not None else video_latents
        noisy_video_latents[:, :, :history_frames] = history_condition_latents[:, :, :history_frames]
        video_timesteps[:, :history_frames] = 0.0
        future_loss_mask = self.training_layout.build_effective_video_loss_mask(
            video_latents=video_latents,
            batch=prepared_inputs.batch,
            default_history_frames=history_frames,
        )
        effective_action_mask = self.training_layout.build_effective_action_mask(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        action_tokens_per_frame = self.training_layout.resolve_action_tokens_per_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        sampled_chunk_size = self.training_layout.resolve_sampled_chunk_size(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        sampled_window_size = self.training_layout.resolve_sampled_window_size(
            batch=prepared_inputs.batch,
        )
        if sampled_chunk_size is None:
            sampled_chunk_size = max(1, int(self.training_config.chunk_size))
        if sampled_window_size is None:
            sampled_window_size = max(1, int(self.training_config.window_size))
        frame_shift = self.training_layout.resolve_frame_shift(batch=prepared_inputs.batch)
        train_artifacts = build_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            effective_action_mask,
            training_config=self.training_config,
        )
        resolved_text = self.conditioning.resolve_text_context(
            visual_tower,
            text_context,
            proprio_state,
            batch_size=int(video_latents.shape[0]),
            device=video_latents.device,
            dtype=video_latents.dtype,
            materialize_if_missing=True,
        )
        if resolved_text is None:  # pragma: no cover - materialized above
            raise RuntimeError("M5 joint action text context unexpectedly resolved to None.")
        action_cross_attention_mask = (
            None
            if action_tokens_per_frame is None
            else self.conditioning.build_proprio_cross_attention_mask(
                resolved_text_context=resolved_text,
                proprio_state=proprio_state,
                query_frames_per_copy=int(video_latents.shape[2]),
                tokens_per_frame=int(action_tokens_per_frame),
                chunk_size_frames=sampled_chunk_size,
            )
        )
        video_cross_attention_mask = self.conditioning.build_proprio_cross_attention_mask(
            resolved_text_context=resolved_text,
            proprio_state=proprio_state,
            query_frames_per_copy=int(video_latents.shape[2]),
            tokens_per_frame=int(prepared_inputs.variant_inputs["video_tokens_per_frame"]),
            chunk_size_frames=sampled_chunk_size,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=train_artifacts.noisy_actions,
            timestep=train_artifacts.timesteps,
            context=resolved_text,
            cross_attention_mask=action_cross_attention_mask,
            action_grid_ids=build_action_grid_ids_for_sequence(
                batch_size=train_artifacts.noisy_actions.shape[0],
                seq_len=train_artifacts.noisy_actions.shape[1],
                action_tokens_per_frame=action_tokens_per_frame,
                device=train_artifacts.noisy_actions.device,
                frame_shift=frame_shift,
            ) if action_tokens_per_frame is not None else None,
            hidden_context=(
                None
                if action_tokens_per_frame is None
                else self.conditioning.action_hidden_context_for_tokens(
                    visual_tower,
                    hidden_proprio_state,
                    action_tokens=train_artifacts.noisy_actions,
                    action_tokens_per_frame=int(action_tokens_per_frame),
                    chunk_size_frames=sampled_chunk_size,
                )
            ),
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
                action_tokens_per_frame=action_tokens_per_frame,
                action_chunk_size_frames=sampled_chunk_size,
                clean_video_frames=history_frames,
                attention_window_size=sampled_window_size,
                current_block_coupling=current_block_coupling,
            ),
            use_activation_checkpointing=self.config.use_activation_checkpointing,
            video_cross_attention_mask=video_cross_attention_mask,
            video_hidden_context=self.conditioning.video_hidden_context_for_tokens(
                visual_tower,
                hidden_proprio_state,
                video_latents=video_latents,
                chunk_size_frames=sampled_chunk_size,
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
        batch_size = video_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=video_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "mot_history_frames": video_latents.new_tensor(float(history_frames)),
                "mot_video_prefix_frames": video_latents.new_tensor(float(history_frames)),
            },
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "runtime_mode": str(self.config.runtime_mode),
                "current_block_coupling": current_block_coupling.value,
                "sampled_chunk_size": sampled_chunk_size,
                "sampled_window_size": sampled_window_size,
                "video_condition_source": video_condition_source,
                "mot_train_artifacts": MoTTrainArtifacts(
                    action=MoTActionTrainArtifacts(
                        flow_pred=flow_pred,
                        targets=train_artifacts.targets,
                        timesteps=train_artifacts.timesteps,
                        scheduler=train_artifacts.scheduler,
                        denoised_actions=denoised_actions,
                        action_mask=train_artifacts.action_mask,
                    ),
                    video=MoTVideoTrainArtifacts(
                        flow_pred=video_flow_pred,
                        targets=video_artifacts.targets,
                        timesteps=video_timesteps,
                        scheduler=video_artifacts.scheduler,
                        predicted_latents=predicted_latents,
                        target_latents=video_latents,
                        future_loss_mask=future_loss_mask,
                    ),
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                    history_frames=int(history_frames),
                ),
            },
        )

    def _forward_train_packed_coupling(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        return MoTPackedTrainingProgram(
            config=self.config,
            training_config=self.training_config,
            conditioning=self.conditioning,
            training_layout=self.training_layout,
            action_expert=self.action_expert,
            packed_block_stack=self.packed_block_stack,
            initialize_action_expert=self._maybe_initialize_action_expert,
        ).run(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
        )

    def _forward_train_non_joint_two_stream(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        return self._forward_train_packed_coupling(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
        )

    def _forward_infer_packed_coupling(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
        runtime_state: MoTRuntimeState,
    ) -> PolicyInferOutput:
        return MoTPackedInferenceProgram(
            config=self.config,
            training_config=self.training_config,
            inference_config=self.inference_config,
            conditioning=self.conditioning,
            action_expert=self.action_expert,
            packed_block_stack=self.packed_block_stack,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        ).run(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            context=context,
            infer_state=infer_state,
            runtime_state=runtime_state,
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
        runtime_state = state.variant_state if isinstance(state.variant_state, MoTRuntimeState) else MoTRuntimeState()
        action_device_raw = context.extra.get("action_device")
        action_device = (
            next(self.action_expert.parameters()).device
            if action_device_raw is None
            else torch.device(str(action_device_raw))
        )
        action_dtype = next(self.action_expert.parameters()).dtype
        proprio_state = self.conditioning.resolve_proprio_state(
            context.state,
            label="M5 inference",
            fallback_state=runtime_state.proprio_state,
        )
        if proprio_state is not None:
            runtime_state.proprio_state = proprio_state.detach().clone()
        hidden_proprio_state = self.conditioning.resolve_infer_hidden_proprio_context(
            context.state,
            fallback_state=runtime_state.hidden_proprio_state,
        )
        if hidden_proprio_state is not None:
            runtime_state.hidden_proprio_state = hidden_proprio_state.detach().clone()
        generalist_rollout_mode = (
            _resolve_mot_generalist_rollout_mode(context)
            if _mot_generalist_rollout_enabled(self.config)
            else MoTGeneralistTrainingMode.JOINT
        )
        generalist_rollout_semantics = resolve_generalist_joint_conditioning_semantics(
            generalist_rollout_mode,
            joint_mode=MoTGeneralistTrainingMode.JOINT,
            action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
            video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
        )
        infer_text_context = visual_outputs.frontend.conditioning.text_context
        if generalist_rollout_semantics.drop_text_conditioning and infer_text_context is not None:
            infer_text_context = torch.zeros_like(infer_text_context)
        resolved_text_context = self.conditioning.resolve_text_context(
            visual_tower,
            infer_text_context,
            proprio_state,
            batch_size=int(visual_outputs.frontend.video_latents.shape[0]),
            device=action_device,
            dtype=action_dtype,
            materialize_if_missing=(
                self.conditioning.uses_proprio_context()
                or bool(getattr(self.config, "generalist_mode_text_token", False))
            ),
        )
        generalist_mode_text_token_count = 0
        if bool(getattr(self.config, "generalist_mode_text_token", False)):
            if resolved_text_context is None:  # pragma: no cover - materialized above
                raise RuntimeError("M5 mode-token rollout expected materialized text context.")
            resolved_text_context, generalist_mode_text_token_count = self.conditioning.append_generalist_mode_text_token(
                visual_tower,
                resolved_text_context,
                generalist_rollout_mode,
            )
        runtime_state.generalist_mode_text_token_count = int(generalist_mode_text_token_count)
        # Only `joint_denoise` stays on the simultaneous video+action denoise
        # path. `non_joint_two_stream` falls through to the method-1-aligned
        # default path below (video fully denoised first, then action attends
        # clean video K/V via `forward_action_with_video_cache`).
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            runtime_device = next(visual_tower.core.parameters()).device
            if action_device != runtime_device:
                raise ValueError(
                    "MoT joint_denoise inference currently requires video and action to run on the same device, "
                    f"got runtime_device={runtime_device}, action_device={action_device}, "
                    f"runtime_mode={self.config.runtime_mode!r}."
                )
            runtime_state.text_context = resolved_text_context
            runtime_state.action_device = str(action_device)
            state.variant_state = runtime_state
            del context
            return state
        condition_latents = visual_outputs.frontend.video_latents
        current_condition_frame_start = int(state.cursor.current_start_frame)
        # Note: `runtime_state.video_cache` is populated inside
        # `forward_infer_step` after the slot-pool warmup + video denoise
        # last-step write, so we don't prefill it here.
        runtime_state.text_context = resolved_text_context
        runtime_state.video_tokens_per_frame = int(visual_outputs.frontend.token_grid.tokens_per_frame)
        runtime_state.chunk_advance_frames = max(1, int(self.inference_config.frame_chunk_size))
        # Only initialize `next_condition_frame_start` on the first chunk of a
        # session. After that, `forward_infer_step` at the end of each chunk
        # sets it to the current chunk's `generation_frame_start` so the NEXT
        # chunk's observation write lands on the same rotary positions as the
        # current chunk's pred entries (overwriting them, keeping the cache
        # contiguous). Without this guard, advancing here by
        # `condition_latents.shape[2]` double-advances alongside
        # `cursor.current_start_frame` and leaves a `chunk_frames`-wide gap
        # of empty rotary slots at every chunk boundary, which desynchronizes
        # the training-time contiguous rotary assumption from the inference
        # cache layout (Method 1 avoids this by using `advance_frame_start=
        # False` inside its denoise rollout and a separate post-rollout
        # `warmup_cache` that writes observations at the same frame_start
        # where the pred just landed).
        if runtime_state.past_clean_latents is None:
            runtime_state.next_condition_frame_start = int(
                current_condition_frame_start + int(condition_latents.shape[2])
            )
        runtime_state.action_device = str(action_device)
        state.variant_state = runtime_state
        del context
        return state

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        runtime_state = (
            infer_state.variant_state if isinstance(infer_state.variant_state, MoTRuntimeState) else MoTRuntimeState()
        )
        self._maybe_initialize_action_expert(visual_tower)
        current_block_coupling_for_infer = resolve_mot_current_block_coupling(self.config)
        mot_inference_backend = ensure_mot_policy_variant_inference_backend(
            policy_variant=self,
            visual_tower=visual_tower,
            policy_config=self.config,
            allow_module_mutation=bool(context.extra.get("allow_mot_legacy_backend_restore", True)),
        )
        use_legacy_cache_infer = (
            self.config.current_block_coupling is not None
            and mot_inference_backend["backend"] == "legacy_split_cache"
            and current_block_coupling_for_infer in MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS
        )
        if self.config.current_block_coupling is not None and not use_legacy_cache_infer:
            return self._forward_infer_packed_coupling(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                context=context,
                infer_state=infer_state,
                runtime_state=runtime_state,
            )
        # Only `joint_denoise` uses the simultaneous video+action denoise
        # branch below. `non_joint_two_stream` falls through to the
        # method-1-aligned default path at the bottom of this function, which
        # denoises video to completion first via
        # `visual_tower.generate_conditioned_future_latents` and then runs the
        # action expert against the resulting all-clean video K/V cache.
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            current_block_coupling = resolve_mot_current_block_coupling(self.config)
            generalist_rollout_mode = (
                _resolve_mot_generalist_rollout_mode(context)
                if _mot_generalist_rollout_enabled(self.config)
                else MoTGeneralistTrainingMode.JOINT
            )
            if _is_mot_generalist_conditional_rollout(generalist_rollout_mode):
                raise ValueError(
                    "M5 GJD conditional FDM/IDM rollout is implemented for native packed joint coupling, "
                    f"not legacy runtime_mode={self.config.runtime_mode!r}."
                )
            if not is_mot_same_step_coupling(current_block_coupling):
                raise NotImplementedError(
                    "M5 joint_denoise inference supports same-step couplings only; "
                    f"got current_block_coupling={current_block_coupling.value!r}."
                )
            device = next(visual_tower.core.parameters()).device
            action_device = next(self.action_expert.parameters()).device
            if action_device != device:
                raise ValueError(
                    "MoT joint_denoise inference currently requires visual tower and action expert on the same device, "
                    f"got visual_device={device}, action_device={action_device}, "
                    f"runtime_mode={self.config.runtime_mode!r}."
                )
            dtype = next(self.action_expert.parameters()).dtype
            batch_size = visual_outputs.frontend.video_latents.shape[0]
            observed_prefix_frames = int(self.config.video_prefix_frames)
            video_latents = visual_outputs.frontend.video_latents.to(device=device, dtype=dtype)
            if video_latents.shape[2] <= observed_prefix_frames:
                raise ValueError(
                    "MoT two-stream inference requires at least one future frame after the observed prefix, "
                    f"got video_latents.shape={tuple(video_latents.shape)}, video_prefix_frames={observed_prefix_frames}, "
                    f"runtime_mode={self.config.runtime_mode!r}."
                )
            if self.inference_config.video_num_inference_steps != self.inference_config.action_num_inference_steps:
                raise ValueError(
                    "MoT two-stream inference currently requires matching video/action inference step counts, "
                    f"got video_num_inference_steps={self.inference_config.video_num_inference_steps}, "
                    f"action_num_inference_steps={self.inference_config.action_num_inference_steps}, "
                    f"runtime_mode={self.config.runtime_mode!r}."
                )
            frame_chunk_size, action_horizon, action_tokens_per_frame = resolve_mot_rollout_frame_chunk_size(
                context,
                default_frame_chunk_size=int(self.inference_config.frame_chunk_size),
                base_action_horizon=int(self.action_horizon),
            )
            observed_prefix = video_latents[:, :, :observed_prefix_frames]
            future_template = video_latents[:, :, observed_prefix_frames:]
            if future_template.shape[2] > frame_chunk_size:
                future_template = future_template[:, :, :frame_chunk_size].contiguous()
            elif future_template.shape[2] < frame_chunk_size:
                missing_frames = int(frame_chunk_size - future_template.shape[2])
                future_template = torch.cat(
                    [
                        future_template,
                        future_template[:, :, -1:].expand(-1, -1, missing_frames, -1, -1),
                    ],
                    dim=2,
                ).contiguous()
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
            couple_action_video_sigmas = should_couple_mot_action_to_video_sigmas(
                self.config,
                current_block_coupling,
            )
            joint_timestep_coupling = resolve_mot_joint_timestep_coupling(
                self.config,
                current_block_coupling,
            )
            action_timestep_lookup_scheduler = None
            if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
                action_timestep_lookup_scheduler = build_action_flow_match_inference_scheduler(
                    training_config=self.training_config,
                    inference_config=self.inference_config,
                    num_inference_steps_override=self.training_config.action_num_train_timesteps,
                )
            sample = torch.randn(
                batch_size,
                action_horizon,
                self.action_dim,
                device=device,
                dtype=dtype,
            )
            text_context = runtime_state.text_context
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
            hidden_proprio_state = runtime_state.hidden_proprio_state
            hidden_proprio_sequence = None
            if hidden_proprio_state is not None:
                hidden_proprio_sequence = hidden_proprio_state.to(device=device, dtype=dtype)[:, None, :].expand(
                    -1,
                    int(video_latents.shape[2]),
                    -1,
                )
            attention_mask = build_mot_attention_mask(
                video_seq_len=visual_outputs.frontend.token_grid.tokens_per_frame * video_latents.shape[2],
                action_seq_len=action_horizon,
                device=device,
                condition_mode=self.config.condition_mode,
                video_tokens_per_frame=visual_outputs.frontend.token_grid.tokens_per_frame,
                video_can_attend_action=self.config.video_can_attend_action,
                action_tokens_per_frame=action_tokens_per_frame,
                action_chunk_size_frames=frame_chunk_size,
                clean_video_frames=observed_prefix_frames,
                current_block_coupling=current_block_coupling,
            )
            for step_index, video_timestep in enumerate(video_scheduler.timesteps):
                action_timestep = action_scheduler.timesteps[step_index]
                shared_sigma = None
                shared_sigma_next = None
                if couple_action_video_sigmas:
                    shared_sigma = video_scheduler.sigmas[step_index].to(device=device, dtype=torch.float32)
                    shared_sigma_next = mot_scheduler_next_sigma(video_scheduler, step_index).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
                        if action_timestep_lookup_scheduler is None:  # pragma: no cover - defensive guard
                            raise RuntimeError("M5 match-sigma joint denoise requires an action timestep lookup scheduler.")
                        action_timestep = timesteps_matching_sigmas(
                            action_timestep_lookup_scheduler,
                            shared_sigma.reshape(1),
                        )[0].to(device=device, dtype=torch.float32)
                    elif joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
                        action_timestep = video_timestep.to(device=device, dtype=torch.float32)
                dense_video_timestep = expand_mot_scalar_timestep(
                    video_timestep,
                    shape=(batch_size, video_latents.shape[2]),
                    device=device,
                )
                dense_video_timestep[:, :observed_prefix_frames] = 0.0
                dense_action_timestep = expand_mot_scalar_timestep(
                    action_timestep,
                    shape=(batch_size, action_horizon),
                    device=device,
                )
                action_pre = self.action_expert.pre_dit(
                    action_tokens=sample,
                    timestep=dense_action_timestep,
                    context=text_context,
                    hidden_context=self.conditioning.action_hidden_context_for_tokens(
                        visual_tower,
                        hidden_proprio_sequence,
                        action_tokens=sample,
                        action_tokens_per_frame=action_tokens_per_frame,
                    ),
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
                    video_hidden_context=self.conditioning.video_hidden_context_for_tokens(
                        visual_tower,
                        hidden_proprio_sequence,
                        video_latents=noisy_video_latents,
                    ),
                )
                flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
                if shared_sigma is None or shared_sigma_next is None:
                    noisy_video_latents = video_scheduler.step(video_flow_pred, video_timestep, noisy_video_latents)
                else:
                    noisy_video_latents = step_mot_flow_with_sigmas(
                        noisy_video_latents,
                        video_flow_pred,
                        sigma=shared_sigma,
                        sigma_next=shared_sigma_next,
                    )
                noisy_video_latents[:, :, :observed_prefix_frames] = observed_prefix
                if shared_sigma is None or shared_sigma_next is None:
                    sample = action_scheduler.step(flow_pred, action_timestep, sample)
                else:
                    sample = step_mot_flow_with_sigmas(
                        sample,
                        flow_pred,
                        sigma=shared_sigma,
                        sigma_next=shared_sigma_next,
                    )
            predicted_latents = noisy_video_latents[:, :, observed_prefix_frames:].detach()
            next_state = infer_state
            next_state.step_index += 1
            next_state.variant_state = runtime_state
            return PolicyInferOutput(
                policy_features=sample.new_zeros(batch_size, 0, self.action_expert.hidden_size),
                next_state=next_state,
                aux={
                    "variant": self.config.name,
                    "method_family": "mot",
                    "condition_mode": str(self.config.condition_mode),
                    "current_block_coupling": current_block_coupling.value,
                    "rollout_frame_chunk_size": int(frame_chunk_size),
                    "rollout_action_horizon": int(action_horizon),
                    "predicted_latents": predicted_latents,
                    "predicted_video_latents": predicted_latents,
                    "mot_infer_artifacts": MoTInferArtifacts(
                        action_pred=sample,
                        predicted_latents=predicted_latents,
                        condition_mode=str(self.config.condition_mode),
                        runtime_mode=str(self.config.runtime_mode),
                    ),
                    "joint_timestep_coupling": joint_timestep_coupling.value,
                    "coupled_action_video_sigmas": bool(couple_action_video_sigmas),
                    "mot_generalist_mode_text_token": (
                        generalist_rollout_mode.value
                        if int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) > 0
                        else None
                    ),
                    "mot_generalist_mode_text_token_count": int(
                        getattr(runtime_state, "generalist_mode_text_token_count", 0)
                    ),
                },
            )
        # True Method-1-aligned NON_JOINT_TWO_STREAM rollout with persistent
        # KV cache on the shared video core. Each chunk:
        #   1) First chunk only -- bootstrap the shared transformer's
        #      `_exact_runtime_caches[cache_name]` with observed env latents
        #      at frame_start=0, all frames clean (timestep=0), update_cache=2.
        #      Matches Method 1's `_write_exact_cache_chunk` bootstrap.
        #   2) Every chunk -- run video denoise with ONLY the
        #      `frame_chunk_size` noisy current-chunk latents as Q. Past
        #      context comes from cache. `update_cache=0` during denoise
        #      steps, `update_cache=1` on the last step writes the new
        #      chunk's clean K/V back into cache, matching Method 1 exactly.
        #   3) Extract a MoTVideoCache view of the updated cache (taking the
        #      cond half when CFG is doubled) so the action expert can
        #      cross-attend the full rollout history.
        #   4) Run action denoise against that MoTVideoCache.
        video_latents = visual_outputs.frontend.video_latents
        batch_size = int(video_latents.shape[0])
        device = next(self.action_expert.parameters()).device
        dtype = next(self.action_expert.parameters()).dtype
        video_device = next(visual_tower.core.parameters()).device
        video_dtype = resolve_runtime_module_dtype(visual_tower.core)

        chunk_frames, action_horizon, action_tokens_per_frame = resolve_mot_rollout_frame_chunk_size(
            context,
            default_frame_chunk_size=int(self.inference_config.frame_chunk_size),
            base_action_horizon=int(self.action_horizon),
        )
        runtime_state.chunk_advance_frames = int(chunk_frames)
        current_block_coupling = resolve_mot_current_block_coupling(self.config)
        action_only_rollout = resolve_mot_action_only_rollout(
            context,
            current_block_coupling=current_block_coupling,
        )
        if current_block_coupling not in MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS:
            raise NotImplementedError(
                "M5 legacy split-cache inference only supports staged video_then_action and decoupled_same_step; "
                f"got current_block_coupling={current_block_coupling.value!r}."
            )
        generalist_rollout_mode = (
            _resolve_mot_generalist_rollout_mode(context)
            if _mot_generalist_rollout_enabled(self.config)
            else MoTGeneralistTrainingMode.JOINT
        )
        if _is_mot_generalist_conditional_rollout(generalist_rollout_mode):
            raise ValueError(
                "M5 GJD conditional FDM/IDM rollout requires packed joint coupling. "
                "Do not use split-cache action-only rollout as an IDM/FDM substitute."
            )
        generalist_rollout_semantics = resolve_generalist_joint_conditioning_semantics(
            generalist_rollout_mode,
            joint_mode=MoTGeneralistTrainingMode.JOINT,
            action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
            video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
        )
        video_commit_before_action = current_block_coupling == CurrentBlockCoupling.VIDEO_THEN_ACTION

        text_context_for_video = runtime_state.text_context
        if text_context_for_video is None:
            text_context_for_video = visual_outputs.frontend.conditioning.text_context
        if generalist_rollout_semantics.drop_text_conditioning and text_context_for_video is not None:
            text_context_for_video = torch.zeros_like(text_context_for_video)
        if text_context_for_video is None:
            text_context_for_video = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=video_device,
                dtype=video_dtype,
            )
        else:
            text_context_for_video = text_context_for_video.to(
                device=video_device, dtype=video_dtype
            )
        if (
            bool(getattr(self.config, "generalist_mode_text_token", False))
            and int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) <= 0
        ):
            text_context_for_video, token_count = self.conditioning.append_generalist_mode_text_token(
                visual_tower,
                text_context_for_video,
                generalist_rollout_mode,
            )
            runtime_state.text_context = text_context_for_video
            runtime_state.generalist_mode_text_token_count = int(token_count)
        # Full Method-1 alignment: cache at 2B with CFG throughout the
        # video path. Bootstrap uses `force_cfg_batch=True` so every
        # subsequent denoise step (with `guidance_scale>1` and
        # `negative_text_emb`) can do CFG batching consistently. The
        # action expert runs at batch=B, so when we extract the
        # MoTVideoCache for action we slice the cond half `[:B]`.
        negative_text_context = visual_outputs.frontend.conditioning.negative_text_context
        negative_text_context = self.conditioning.resolve_text_context(
            visual_tower,
            negative_text_context,
            runtime_state.proprio_state,
            batch_size=batch_size,
            device=video_device,
            dtype=video_dtype,
            materialize_if_missing=False,
        )
        if bool(getattr(self.config, "generalist_mode_text_token", False)) and negative_text_context is not None:
            negative_text_context, _ = self.conditioning.append_generalist_mode_text_token(
                visual_tower,
                negative_text_context,
                generalist_rollout_mode,
            )
        use_cfg = (
            negative_text_context is not None
            and bool(self.inference_config.use_cache)
        )

        cache_name = "mot_non_joint_two_stream_cache"
        latent_channels = int(visual_tower.config.latent_channels)
        latent_height = int(video_latents.shape[-2])
        latent_width = int(video_latents.shape[-1])
        is_first_chunk = runtime_state.past_clean_latents is None
        skip_observation_update = bool(context.extra.get("mot_skip_observation_update", False))
        if skip_observation_update and is_first_chunk:
            raise ValueError("MoT open-loop extension requires an initialized non-joint rollout cache.")
        condition_frame_start_override_raw = context.extra.get("mot_condition_frame_start")
        if skip_observation_update and condition_frame_start_override_raw is not None:
            raise ValueError("MoT condition-frame rewind is only valid for observation-conditioned replans.")
        inference_window_size = resolve_mot_inference_window_size(
            context,
            default_window_size=_MOT_SLOT_POOL_ATTN_WINDOW,
        )
        # Method-1-aligned per-chunk warmup. On chunk 0 we allocate the
        # slot-pool backend via `initialize_exact_runtime_cache` and write the
        # bootstrap obs latents at frame_start=0. On subsequent chunks the
        # driver passes a fresh window of real env observations (encoded
        # into `video_latents`). We:
        #   1) clear the prediction cache (last chunk's denoise last-step
        #      pred K/V),
        #   2) write the real-env observation as a NEW stable chunk at
        #      `frame_start = runtime_state.next_condition_frame_start`.
        # `observed_prefix.shape[2]` is allowed to vary between chunks --
        # chunk 0 may use a 1-frame bootstrap (matching Method 1) while
        # subsequent chunks pass `chunk_frames` real env-observation latents
        # that overwrite the previous chunk's pred slots. The Route-A
        # inference mask removes the old chunk_frames-aligned bootstrap
        # constraint by treating past KV as always-visible.
        current_obs_frame_start = int(runtime_state.next_condition_frame_start)
        if is_first_chunk:
            initialize_exact_runtime_cache(
                visual_tower.core,
                cache_name=cache_name,
                attn_window=inference_window_size,
                batch_size=batch_size,
                frame_chunk_size=chunk_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                device=video_device,
                action_per_frame=action_tokens_per_frame,
                use_cfg=use_cfg,
            )
            current_obs_frame_start = 0
        elif condition_frame_start_override_raw is not None:
            current_obs_frame_start = int(condition_frame_start_override_raw)
        if self.inference_config.use_cache and not skip_observation_update:
            clear_exact_prediction_cache(visual_tower.core, cache_name=cache_name)
        observed_prefix = video_latents.to(device=video_device, dtype=video_dtype)
        if not skip_observation_update:
            boot_video_input = prepare_exact_single_stream_input(
                latents=observed_prefix,
                timestep=0.0,
                text_emb=text_context_for_video,
                frame_st_id=current_obs_frame_start,
                backbone_config=visual_tower.config,
                action_mode=False,
            )
            run_exact_single_stream_forward(
                visual_tower.core,
                input_dict=boot_video_input,
                update_cache=2,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=1.0,
                negative_text_emb=negative_text_context,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )
            runtime_state.past_clean_latents = observed_prefix.detach()
        # Observation-conditioned replans write real observations into the
        # next slots. Async open-loop extensions intentionally skip this
        # write so planning ahead does not leak too-early real frames into a
        # future chunk; they extend from the already generated cache instead.
        generation_frame_start = (
            current_obs_frame_start + chunk_frames
            if skip_observation_update
            else current_obs_frame_start + int(observed_prefix.shape[2])
        )
        if is_first_chunk:
            runtime_state.chunk_origin_frame = int(generation_frame_start) % int(chunk_frames)
        runtime_state.next_condition_frame_start = (
            generation_frame_start + chunk_frames if skip_observation_update else generation_frame_start
        )

        if action_only_rollout:
            predicted_latents = observed_prefix.new_empty(
                batch_size,
                latent_channels,
                0,
                latent_height,
                latent_width,
            )
        else:
            # Cache-aware video denoise on the current noisy chunk only.
            latents = torch.randn(
                batch_size,
                latent_channels,
                chunk_frames,
                latent_height,
                latent_width,
                device=video_device,
                dtype=video_dtype,
            )
            video_scheduler = FlowMatchScheduler(
                shift=self.training_config.video_sigma_shift,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=self.training_config.video_num_train_timesteps,
            )
            video_scheduler.set_timesteps(self.inference_config.video_num_inference_steps)
            video_timesteps = F.pad(
                video_scheduler.timesteps.to(device=video_device),
                (0, 1),
                mode="constant",
                value=0,
            )
            for index, timestep in enumerate(video_timesteps):
                last_step = index == len(video_timesteps) - 1
                video_input = prepare_exact_single_stream_input(
                    latents=latents,
                    timestep=timestep,
                    text_emb=text_context_for_video,
                    frame_st_id=generation_frame_start,
                    backbone_config=visual_tower.config,
                    action_mode=False,
                )
                video_noise_pred = run_exact_single_stream_forward(
                    visual_tower.core,
                    input_dict=video_input,
                    update_cache=1 if (last_step and video_commit_before_action and self.inference_config.use_cache) else 0,
                    cache_name=cache_name,
                    action_mode=False,
                    guidance_scale=self.inference_config.guidance_scale,
                    negative_text_emb=negative_text_context,
                    force_cfg_batch=use_cfg,
                )
                if not last_step:
                    video_noise_pred = unpatchify_video_sequence(
                        visual_tower.core.patch_size,
                        video_noise_pred,
                        chunk_frames,
                        latent_height,
                        latent_width,
                        batch_size=batch_size,
                    ).to(dtype=video_dtype)
                    latents = video_scheduler.step(video_noise_pred, timestep, latents)
            predicted_latents = latents

        # Don't advance `next_condition_frame_start` past the observation
        # write position. Method 1 with `advance_frame_start=False` keeps
        # frame_start at the value warmup set it to, so that the NEXT
        # chunk's warmup writes its real observations at the same rotary
        # positions that the current chunk's pred entries just landed on
        # (teacher-forcing the pred positions with real obs). The pred
        # entries (chunk_frames tokens at rotary [gen_start..gen_start+4))
        # will be cleared + overwritten by the next chunk's stable obs
        # write via `clear_exact_prediction_cache` plus
        # `run_exact_single_stream_forward(update_cache=2)`. The TOTAL number
        # of clean video frames the
        # action expert sees this chunk is observation frames +
        # current-chunk pred frames.
        total_clean_video_frames = generation_frame_start + (0 if action_only_rollout else chunk_frames)
        action_visible_video_end_frame = (
            total_clean_video_frames
            if video_commit_before_action
            else generation_frame_start
        )

        def extract_mot_video_cache_from_exact_cache() -> MoTVideoCache:
            # With CFG active the cache is doubled `[cond, uncond]` on the
            # batch dim; slice the cond half for the action expert (batch=B).
            cache_state = visual_tower.core._resolve_exact_cache_state(cache_name)
            if cache_state is None:
                raise RuntimeError(
                    f"MoT non_joint_two_stream expected cache state at `{cache_name}` "
                    "but the shared transformer returned None."
                )
            extracted_layers: list[MoTVideoLayerCache] = []
            for entry in cache_state.self_attention_kv:
                if entry.key is None or entry.value is None:
                    raise RuntimeError(
                        "MoT non_joint_two_stream cache extraction found an empty layer entry."
                    )
                key = entry.key
                value = entry.value
                if key.shape[0] == 2 * batch_size:
                    key = key[:batch_size]
                    value = value[:batch_size]
                elif key.shape[0] != batch_size:
                    raise RuntimeError(
                        "MoT non_joint_two_stream cache batch dimension must match the current batch "
                        f"(or 2x for CFG), got cache_batch={key.shape[0]}, batch_size={batch_size}."
                    )
                extracted_layers.append(
                    MoTVideoLayerCache(key=key.detach(), value=value.detach())
                )
            return MoTVideoCache(
                layers=tuple(extracted_layers),
                video_seq_len=int(extracted_layers[0].key.shape[2]),
            )

        action_video_cache = extract_mot_video_cache_from_exact_cache()
        def prepare_action_video_cache(cache: MoTVideoCache) -> MoTVideoCache:
            # Method-1 alignment: Method 1's slot pool stores both video and
            # action so video occupies `(attn_window // 2) * latent_token_per_chunk`
            # tokens, which is integer-frame-aligned. Method 5 only writes video so the
            # slot pool fills with `(attn_window // 2) * (latent + action)` tokens
            # (= 67.5 frames here), leaving a partial leading frame after eviction.
            # Trim to Method 1's per-stream cap so the action expert sees the
            # same frame-aligned video lookback Method 1 does.
            method1_video_lookback_frames = (
                (inference_window_size // 2) * int(chunk_frames)
            )
            max_video_tokens_for_action = int(method1_video_lookback_frames) * int(
                runtime_state.video_tokens_per_frame
            ) if runtime_state.video_tokens_per_frame else None
            if (
                max_video_tokens_for_action is not None
                and max_video_tokens_for_action > 0
                and cache.video_seq_len > max_video_tokens_for_action
            ):
                cache = trim_mot_video_cache_tail(
                    cache,
                    max_video_seq_len=max_video_tokens_for_action,
                )
            return move_mot_video_cache(cache, device=device, dtype=dtype)

        action_video_cache = prepare_action_video_cache(action_video_cache)
        runtime_state.video_cache = action_video_cache
        cached_batch_size = int(action_video_cache.layers[0].key.shape[0])
        if cached_batch_size != batch_size:
            raise ValueError(
                "MoT cached-action inference requires the current observation batch to match the cached video batch, "
                f"got current_batch_size={batch_size}, cached_batch_size={cached_batch_size}."
            )
        scheduler = build_action_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        sample = torch.randn(
            batch_size,
            action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        text_context = runtime_state.text_context
        if text_context is None:
            text_context = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=device,
                dtype=dtype,
            )
        # Method-1-aligned action denoise with persistent action K/V
        # cache. Past action chunks' clean K/V live in
        # `runtime_state.action_cache`; fresh `action_horizon` tokens are
        # the only Q this forward recomputes every step. On the final
        # (padded timestep=0) step we capture the fresh per-layer K/V
        # and append to `runtime_state.action_cache`, mirroring Method 1's
        # `update_cache=1` at the last action step.
        action_cache_rewind_frame_start_raw = context.extra.get("mot_action_cache_rewind_frame_start")
        if action_cache_rewind_frame_start_raw is None:
            action_cache_rewind_frame_start_raw = context.extra.get("mot_action_cache_prefix_frames")
        if action_cache_rewind_frame_start_raw is not None:
            rewind_mot_runtime_action_cache_to_frame(
                runtime_state,
                absolute_frame_start=int(action_cache_rewind_frame_start_raw),
                action_tokens_per_frame=action_tokens_per_frame,
            )
        past_action_cache = runtime_state.action_cache
        # Diagnostic: setting OPEN_WAM_MOT_DISABLE_PAST_ACTION_CACHE=1 forces
        # the action expert to see only video + current noisy action per
        # chunk (no past action history). Useful for isolating whether
        # autoregressive drift in the past_action K/V chain is the source
        # of chunk-to-chunk instability.
        if os.environ.get("OPEN_WAM_MOT_DISABLE_PAST_ACTION_CACHE", "0") == "1":
            past_action_cache = None
        past_action_seq_len = int(past_action_cache.action_seq_len) if past_action_cache is not None else 0
        if past_action_seq_len % action_tokens_per_frame != 0:
            raise ValueError(
                "MoT non_joint_two_stream action cache length must be a multiple of action_tokens_per_frame, "
                f"got past_action_seq_len={past_action_seq_len}, action_tokens_per_frame={action_tokens_per_frame}."
            )
        past_action_frames = past_action_seq_len // action_tokens_per_frame
        # Method-1 byte-aligned mask: replicates
        # `build_chunked_temporal_exact_attention_profile` for the inference
        # `[video_cache; past_action_cache; current_action]` layout. Block
        # ids are video=chunk*2 / action=chunk*2+1, the within-window check
        # uses `training_config.window_size` (same value Method 1 passes as
        # `input_dict["window_size"]` at inference), and clean/noise causal
        # rules match Method 1's chunked_temporal_exact profile.
        if runtime_state.video_tokens_per_frame is None or runtime_state.video_tokens_per_frame <= 0:
            raise RuntimeError(
                "MoT inference mask requires `runtime_state.video_tokens_per_frame` to be set, "
                f"got {runtime_state.video_tokens_per_frame!r}."
            )
        video_lookback_frames_for_mask = int(action_video_cache.video_seq_len) // int(
            runtime_state.video_tokens_per_frame
        )
        current_action_frame_start = int(generation_frame_start)
        video_frame_start = int(action_visible_video_end_frame - video_lookback_frames_for_mask)
        past_action_frame_start = (
            int(runtime_state.action_cache_start_frame)
            if past_action_cache is not None
            else int(current_action_frame_start)
        )
        if past_action_cache is not None:
            cached_action_end_frame = int(past_action_frame_start + past_action_frames)
            if cached_action_end_frame != current_action_frame_start:
                raise RuntimeError(
                    "MoT action cache frame span is not contiguous with the current chunk, "
                    f"cache_span=[{past_action_frame_start}, {cached_action_end_frame}), "
                    f"current_action_frame_start={current_action_frame_start}."
                )
        attention_mask = build_mot_inference_action_attention_mask(
            video_seq_len=action_video_cache.video_seq_len,
            past_action_seq_len=past_action_seq_len,
            current_action_seq_len=action_horizon,
            video_tokens_per_frame=int(runtime_state.video_tokens_per_frame),
            action_tokens_per_frame=action_tokens_per_frame,
            chunk_size_frames=max(1, int(self.training_config.chunk_size)),
            window_size_frames=max(1, int(self.training_config.window_size)),
            device=device,
            video_can_attend_action=False,
            video_frame_start=video_frame_start,
            past_action_frame_start=past_action_frame_start,
            current_action_frame_start=current_action_frame_start,
            chunk_origin_frame=int(runtime_state.chunk_origin_frame),
            current_block_coupling=current_block_coupling,
        )
        # Method-1-aligned cache write: run the denoise loop without
        # capturing K/V, then issue a SEPARATE fresh forward at timestep=0
        # with the final denoised sample to capture cache-bound K/V. Mirrors
        # `_write_exact_cache_chunk(update_cache=1)` in
        # `run_parallel_action_conditioned_inference_rollout`, which calls a
        # fresh single-stream forward after all denoise steps complete
        # rather than reusing the loop's last-step K/V.
        fresh_action_kv: MoTActionCache | None = None
        for timestep in scheduler.timesteps.to(device=device):
            dense_timestep = torch.full(
                (batch_size, action_horizon),
                float(timestep),
                device=device,
                dtype=torch.float32,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=sample,
                timestep=dense_timestep,
                context=text_context.to(device=device, dtype=dtype),
                action_grid_ids=build_action_grid_ids_for_sequence(
                    batch_size=batch_size,
                    seq_len=action_horizon,
                    action_tokens_per_frame=action_tokens_per_frame,
                    device=device,
                    frame_shift=int(current_action_frame_start),
                ),
                hidden_context=self.conditioning.action_hidden_context_for_tokens(
                    visual_tower,
                    runtime_state.hidden_proprio_state,
                    action_tokens=sample,
                    action_tokens_per_frame=action_tokens_per_frame,
                    chunk_size_frames=chunk_frames,
                ),
            )
            action_hidden_states, _ = forward_action_with_video_and_action_cache(
                action_expert=self.action_expert,
                action_pre=action_pre,
                video_cache=action_video_cache,
                action_cache=past_action_cache,
                attention_mask=attention_mask,
            )
            flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
            sample = scheduler.step(flow_pred, timestep, sample)
        # Separate cache-write forward at timestep=0 with the final denoised
        # sample. This is the Method-1 parity step.
        cache_write_timestep = torch.zeros(
            (batch_size, action_horizon),
            device=device,
            dtype=torch.float32,
        )
        cache_write_action_pre = self.action_expert.pre_dit(
            action_tokens=sample,
            timestep=cache_write_timestep,
            context=text_context.to(device=device, dtype=dtype),
            action_grid_ids=build_action_grid_ids_for_sequence(
                batch_size=batch_size,
                seq_len=action_horizon,
                action_tokens_per_frame=action_tokens_per_frame,
                device=device,
                frame_shift=int(current_action_frame_start),
            ),
            hidden_context=self.conditioning.action_hidden_context_for_tokens(
                visual_tower,
                runtime_state.hidden_proprio_state,
                action_tokens=sample,
                action_tokens_per_frame=action_tokens_per_frame,
                chunk_size_frames=chunk_frames,
            ),
        )
        _, fresh_action_kv = forward_action_with_video_and_action_cache(
            action_expert=self.action_expert,
            action_pre=cache_write_action_pre,
            video_cache=action_video_cache,
            action_cache=past_action_cache,
            attention_mask=attention_mask,
        )
        if fresh_action_kv is None:
            raise RuntimeError(
                "MoT non_joint_two_stream cache-write forward did not produce fresh K/V."
            )
        # Append fresh action K/V to the persistent action cache for next chunk.
        fresh_action_kv_moved = move_mot_action_cache(
            fresh_action_kv, device=device, dtype=dtype
        )
        if past_action_cache is None:
            runtime_state.action_cache = fresh_action_kv_moved
            runtime_state.action_cache_start_frame = int(current_action_frame_start)
        else:
            runtime_state.action_cache = append_mot_action_cache(
                past_action_cache, fresh_action_kv_moved
            )
        if (
            current_block_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP
            and self.inference_config.use_cache
            and not action_only_rollout
        ):
            deferred_video_input = prepare_exact_single_stream_input(
                latents=predicted_latents.to(device=video_device, dtype=video_dtype),
                timestep=0.0,
                text_emb=text_context_for_video,
                frame_st_id=generation_frame_start,
                backbone_config=visual_tower.config,
                action_mode=False,
            )
            run_exact_single_stream_forward(
                visual_tower.core,
                input_dict=deferred_video_input,
                update_cache=1,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=1.0,
                negative_text_emb=negative_text_context,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )
            action_video_cache = prepare_action_video_cache(
                extract_mot_video_cache_from_exact_cache()
            )
            runtime_state.video_cache = action_video_cache
        # Method-1 alignment: video and action share the same effective
        # lookback. Method 1 stores both streams in the slot pool and
        # `attn_window` evicts them together; to mirror that with Method 5's
        # split caches we trim the action cache to exactly the video cache's
        # current frame count. Asymmetric lookback (action shorter or longer
        # than video) is OOD: training always saw matched lengths, so the
        # action expert hallucinates when past action covers a different
        # frame span than past video.
        video_tokens_per_frame_for_trim = runtime_state.video_tokens_per_frame
        if video_tokens_per_frame_for_trim is not None and video_tokens_per_frame_for_trim > 0:
            video_lookback_frames = int(action_video_cache.video_seq_len) // int(
                video_tokens_per_frame_for_trim
            )
            max_action_seq_len = max(
                action_tokens_per_frame,
                video_lookback_frames * action_tokens_per_frame,
            )
            action_cache_before_trim = runtime_state.action_cache
            if action_cache_before_trim.action_seq_len > max_action_seq_len:
                dropped_action_tokens = int(action_cache_before_trim.action_seq_len - max_action_seq_len)
                runtime_state.action_cache_start_frame += int(dropped_action_tokens // action_tokens_per_frame)
            runtime_state.action_cache = trim_mot_action_cache_tail(
                action_cache_before_trim,
                max_action_seq_len=max_action_seq_len,
            )
        next_state = infer_state
        next_state.step_index += 1
        next_state.cursor.current_start_frame = int(
            infer_state.cursor.current_start_frame + max(1, runtime_state.chunk_advance_frames)
        )
        next_state.variant_state = runtime_state
        return PolicyInferOutput(
            policy_features=sample.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            next_state=next_state,
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "current_block_coupling": current_block_coupling.value,
                "generation_frame_start": int(current_action_frame_start),
                "mot_action_only_rollout": bool(action_only_rollout),
                "mot_generalist_mode_text_token": (
                    generalist_rollout_mode.value
                    if int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) > 0
                    else None
                ),
                "mot_generalist_mode_text_token_count": int(
                    getattr(runtime_state, "generalist_mode_text_token_count", 0)
                ),
                "mot_cache_debug": {
                    "video_cache_seq_len": int(runtime_state.video_cache.video_seq_len) if runtime_state.video_cache is not None else 0,
                    "action_video_cache_seq_len": int(action_video_cache.video_seq_len),
                    "action_cache_seq_len": int(runtime_state.action_cache.action_seq_len) if runtime_state.action_cache is not None else 0,
                    "action_cache_start_frame": int(runtime_state.action_cache_start_frame),
                    "action_cache_frames_before_chunk": int(past_action_frames),
                    "total_clean_video_frames": int(total_clean_video_frames),
                    "is_first_chunk": bool(is_first_chunk),
                    "use_cfg": bool(use_cfg),
                    "current_start_frame": int(next_state.cursor.current_start_frame),
                    "next_condition_frame_start": int(runtime_state.next_condition_frame_start),
                    "chunk_advance_frames": int(runtime_state.chunk_advance_frames),
                    "video_frame_start": int(video_frame_start),
                    "past_action_frame_start": int(past_action_frame_start),
                    "current_action_frame_start": int(current_action_frame_start),
                    "chunk_origin_frame": int(runtime_state.chunk_origin_frame),
                    "skip_observation_update": bool(skip_observation_update),
                    "condition_frame_start_override": (
                        None
                        if condition_frame_start_override_raw is None
                        else int(condition_frame_start_override_raw)
                    ),
                    "current_block_coupling": current_block_coupling.value,
                    "mot_action_only_rollout": bool(action_only_rollout),
                    "video_commit_before_action": bool(video_commit_before_action),
                    "action_visible_video_end_frame": int(action_visible_video_end_frame),
                    "inference_window_size": int(inference_window_size),
                    "rollout_frame_chunk_size": int(chunk_frames),
                    "rollout_action_horizon": int(action_horizon),
                    "action_cache_rewind_frame_start": (
                        None
                        if action_cache_rewind_frame_start_raw is None
                        else int(action_cache_rewind_frame_start_raw)
                    ),
                },
                **(
                    {"predicted_latents": predicted_latents.detach(), "predicted_video_latents": predicted_latents.detach()}
                    if isinstance(predicted_latents, torch.Tensor)
                    else {}
                ),
                "mot_infer_artifacts": MoTInferArtifacts(
                    action_pred=sample,
                    predicted_latents=predicted_latents.detach() if isinstance(predicted_latents, torch.Tensor) else None,
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                ),
            },
        )
