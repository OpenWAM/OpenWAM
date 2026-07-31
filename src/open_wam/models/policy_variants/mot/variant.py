from __future__ import annotations

import torch

from open_wam.models.common.flow_matching import (
    VideoFlowMatchTrainArtifacts,
    build_video_flow_match_train_artifacts,
    build_action_flow_match_train_artifacts,
    denoised_actions_from_flow,
    denoised_video_latents_from_flow,
)
from open_wam.models.common.joint_conditioning import (
    resolve_generalist_joint_conditioning_semantics,
)
from open_wam.configs import (
    InferenceConfig,
    MoTGeneralistTrainingMode,
    MoTPolicyConfig,
    MoTRuntimeMode,
    ParallelHistoryStreamVisibility,
    TrainingConfig,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower
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
from .attention import build_chunk_causal_video_mask, build_mot_attention_mask
from .cache_execution import (
    forward_action_with_video_cache,
    prefill_video_kv_cache,
)
from .conditioning import MoTConditioning
from .contracts import (
    MoTActionLayerCache,
    MoTActionTrainArtifacts,
    MoTRuntimeState,
    MoTTrainArtifacts,
    MoTVideoTrainArtifacts,
)
from .generalist_modes import (
    apply_generalist_training_mode as _apply_mot_generalist_training_mode,
    generalist_rollout_enabled as _mot_generalist_rollout_enabled,
    generalist_rollout_mode_from_value as _mot_generalist_rollout_mode_from_value,
    resolve_generalist_rollout_mode as _resolve_mot_generalist_rollout_mode,
    sample_generalist_training_mode as _sample_mot_generalist_training_mode,
)
from .joint_denoise_inference import MoTJointDenoiseInferenceProgram
from .modules import MoTActionExpert, init_action_expert_from_video_core
from .packed_block import MoTPackedBlock, MoTPackedBlockStack
from .packed_inference import MoTPackedInferenceProgram
from .packed_training import MoTPackedTrainingProgram
from .runtime_routing import (
    MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS,
    ensure_mot_policy_variant_inference_backend,
    resolve_mot_current_block_coupling,
)
from .sequence_layout import MoTTrainingLayout, build_action_grid_ids_for_sequence
from .split_cache_inference import MoTSplitCacheInferenceProgram


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
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            return MoTJointDenoiseInferenceProgram(
                config=self.config,
                training_config=self.training_config,
                inference_config=self.inference_config,
                conditioning=self.conditioning,
                action_expert=self.action_expert,
                action_dim=self.action_dim,
                action_horizon=self.action_horizon,
            ).run(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                context=context,
                infer_state=infer_state,
                runtime_state=runtime_state,
            )
        return MoTSplitCacheInferenceProgram(
            config=self.config,
            training_config=self.training_config,
            inference_config=self.inference_config,
            conditioning=self.conditioning,
            action_expert=self.action_expert,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        ).run(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            context=context,
            infer_state=infer_state,
            runtime_state=runtime_state,
        )
