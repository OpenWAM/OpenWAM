from __future__ import annotations

import torch

from open_wam.configs import (
    InferenceConfig,
    MoTGeneralistTrainingMode,
    MoTPolicyConfig,
    MoTRuntimeMode,
    TrainingConfig,
)
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.models.common.joint_conditioning import (
    resolve_generalist_joint_conditioning_semantics,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..base import PolicyVariant
from ..contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)
from .conditioning import MoTConditioning
from .contracts import MoTRuntimeState
from .generalist_modes import (
    apply_generalist_training_mode as _apply_mot_generalist_training_mode,
    generalist_rollout_enabled as _mot_generalist_rollout_enabled,
    generalist_rollout_mode_from_value as _mot_generalist_rollout_mode_from_value,
    resolve_generalist_rollout_mode as _resolve_mot_generalist_rollout_mode,
    sample_generalist_training_mode as _sample_mot_generalist_training_mode,
)
from .joint_denoise_inference import MoTJointDenoiseInferenceProgram
from .modules import MoTActionExpert, init_action_expert_from_video_core
from .packed_block import MoTPackedBlockStack
from .packed_inference import MoTPackedInferenceProgram
from .packed_training import MoTPackedTrainingProgram
from .runtime_routing import (
    MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS,
    ensure_mot_policy_variant_inference_backend,
    resolve_mot_current_block_coupling,
)
from .sequence_layout import MoTTrainingLayout
from .split_cache_inference import MoTSplitCacheInferenceProgram
from .unpacked_training import MoTUnpackedTrainingProgram


class MoTPolicyVariant(PolicyVariant):
    """Own M5 modules and route execution through policy-local programs."""

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
        unpacked_program = MoTUnpackedTrainingProgram(
            config=self.config,
            training_config=self.training_config,
            conditioning=self.conditioning,
            training_layout=self.training_layout,
            action_expert=self.action_expert,
            initialize_action_expert=self._maybe_initialize_action_expert,
            should_detach_video_cache=self._should_detach_train_video_cache,
        )
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            return unpacked_program.run_joint_denoise(
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
        return unpacked_program.run_prefill_action_denoise(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
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
