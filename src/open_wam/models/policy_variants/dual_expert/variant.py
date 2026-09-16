from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

from open_wam.configs import (
    InferenceConfig,
    TrainingConfig,
)
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import BatchingMode
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.contracts import SampleConstructionMetadata
from open_wam.models.common.dynamics_objectives import (
    resolve_dynamics_rollout_plan,
    resolve_dynamics_sample_plan,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..base import VideoActionPolicyVariant
from ..contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyModuleTopology,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVisualStage,
)
from .conditioning import DualExpertConditioning
from open_wam.models.common.video_action_state import VideoActionRolloutState
from open_wam.models.decoder_artifacts import DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT
from .inference_backend import ensure_dual_expert_policy_variant_inference_backend
from .module_topology import build_dual_expert_module_topology
from .modules import DualExpertActionExpert, init_action_expert_from_video_core
from .packed_block import DualExpertPackedBlockStack
from .inference import DualExpertInferenceProgram
from .packed_training import DualExpertPackedTrainingProgram
from .sequence_layout import DualExpertTrainingLayout


class DualExpertPolicyVariant(VideoActionPolicyVariant):
    """Own dual-expert modules and route execution through policy-local programs."""

    def __init__(
        self,
        config: DualExpertPolicyConfig,
        backbone_config: SharedVideoTransformerConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_dim: int,
        action_horizon: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone_config = backbone_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.conditioning = DualExpertConditioning(config)
        self.training_layout = DualExpertTrainingLayout(config, training_config)
        action_hidden_size = (
            int(config.action_hidden_size)
            if config.action_hidden_size is not None
            else int(backbone_config.hidden_size)
        )
        self.action_expert = DualExpertActionExpert(
            hidden_size=action_hidden_size,
            action_dim=action_dim,
            num_layers=config.num_action_layers,
            num_heads=backbone_config.num_heads,
            attention_head_dim=backbone_config.attention_head_dim,
            ffn_dim=(
                int(config.action_ffn_dim)
                if config.action_ffn_dim is not None
                else (
                    backbone_config.ffn_dim
                    or (backbone_config.hidden_size * backbone_config.mlp_ratio)
                )
            ),
            text_dim=backbone_config.text_dim,
            hidden_context_dim=backbone_config.hidden_size,
            freq_dim=backbone_config.freq_dim,
            cross_attn_norm=backbone_config.cross_attn_norm,
            eps=backbone_config.latent_norm_eps,
        )
        self._action_expert_initialized = False
        # Lazy-initialized at pipeline assembly time. Owns video_block +
        # action_block pairs after ownership transfer so FSDP can wrap the
        # packed unit cleanly without aliasing.
        self.packed_block_stack: DualExpertPackedBlockStack | None = None
        self._packed_block_stack_attached = False

    @property
    def decoder_artifact_contract(self) -> str:
        return DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT

    @property
    def action_tokens_per_frame(self) -> int:
        density, remainder = divmod(
            self.action_horizon, self.inference_config.frame_chunk_size
        )
        if remainder:
            raise ValueError("Action horizon must contain complete model-frame groups.")
        return density

    def attach_visual_tower(self, visual_tower: VisualTower) -> None:
        """Build the packed-coupling block stack after visual weight loading.

        Must run AFTER both ``visual_tower`` and ``self.action_expert`` exist
        but BEFORE FSDP sharding. Transfers ownership of video core blocks and
        action expert blocks into ``self.packed_block_stack`` so FSDP only
        sees a single owner per nn.Parameter (no shared-module aliasing).
        ``_maybe_initialize_action_expert`` runs BEFORE the transfer because
        the init helper reads from ``visual_tower.core.blocks`` and writes to
        ``self.action_expert.blocks``; after transfer both ModuleLists are
        empty. Both stacks retain non-owning execution views for inference.
        """
        if self._packed_block_stack_attached:
            return
        self._packed_block_stack_attached = True
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
        self.packed_block_stack = DualExpertPackedBlockStack(
            video_blocks, action_blocks
        )
        visual_tower.core.blocks = torch.nn.ModuleList()
        self.action_expert.blocks = torch.nn.ModuleList()
        visual_tower.core.bind_execution_blocks(video_blocks)
        self.action_expert.bind_execution_blocks(action_blocks)

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

    def module_topology(self, visual_tower: VisualTower) -> PolicyModuleTopology:
        """Describe action/video ownership after packed-block attachment."""

        return build_dual_expert_module_topology(
            visual_tower=visual_tower,
            action_expert=self.action_expert,
            packed_block_stack=self.packed_block_stack,
        )

    def on_checkpoint_loaded(
        self,
        *,
        loaded_state_keys: frozenset[str],
        missing_state_keys: frozenset[str],
    ) -> None:
        """Record when checkpoint weights fully initialize the lazy expert."""

        prefix = "action_expert."
        if any(key.startswith(prefix) for key in loaded_state_keys) and not any(
            key.startswith(prefix) for key in missing_state_keys
        ):
            self._action_expert_initialized = True

    def required_visual_stages(self) -> tuple[PolicyVisualStage, ...]:
        return (PolicyVisualStage.FRONTEND,)

    def validate_pipeline_assembly(
        self,
        *,
        data_action_dim: int,
        num_frames: int,
        backbone_num_layers: int,
    ) -> None:
        super().validate_pipeline_assembly(
            data_action_dim=data_action_dim,
            num_frames=num_frames,
            backbone_num_layers=backbone_num_layers,
        )
        if self.action_horizon <= 0:
            raise ValueError("Dual Expert requires a positive action horizon.")
        if self.config.video_prefix_frames >= num_frames:
            raise ValueError(
                "Dual Expert requires `video_prefix_frames < data.num_frames`, "
                f"got prefix={self.config.video_prefix_frames}, frames={num_frames}."
            )
        if self.config.num_action_layers != backbone_num_layers:
            raise ValueError(
                "Dual Expert requires one action block per visual backbone block, "
                f"got action_layers={self.config.num_action_layers}, "
                f"backbone_layers={backbone_num_layers}."
            )

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(
            batch.extra.get("metadata")
        )
        dynamics_sample_plan = resolve_dynamics_sample_plan(
            program=self.config.program,
            sample_metadata=sample_metadata,
        )
        condition_latents = self.conditioning.resolve_train_condition_latents(
            batch,
            video_latents=visual_outputs.frontend.video_latents,
            dynamics_sample_plan=dynamics_sample_plan,
        )
        proprio_state = self.conditioning.resolve_train_proprio_context(batch)
        hidden_proprio_context = self.conditioning.resolve_train_hidden_proprio_context(
            batch
        )
        return PolicyPreparedInputs(
            batch=batch,
            variant_inputs={
                "video_latents": visual_outputs.frontend.video_latents,
                "condition_latents": condition_latents,
                "sample_metadata": sample_metadata,
                "dynamics_sample_plan": dynamics_sample_plan,
                "proprio_state": proprio_state,
                "hidden_proprio_context": hidden_proprio_context,
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
        return self._forward_train_packed_coupling(
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
        return DualExpertPackedTrainingProgram(
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

    def forward_train_batch(
        self,
        visual_tower: VisualTower,
        visual_outputs: Sequence[VisualStageOutputs],
        prepared_inputs: Sequence[PolicyPreparedInputs],
        *,
        batching_mode: BatchingMode | str,
    ) -> tuple[PolicyTrainOutput, ...]:
        """Keep per-sample preparation/loss semantics, sharing heavy token execution."""
        return DualExpertPackedTrainingProgram(
            config=self.config,
            training_config=self.training_config,
            conditioning=self.conditioning,
            training_layout=self.training_layout,
            action_expert=self.action_expert,
            packed_block_stack=self.packed_block_stack,
            initialize_action_expert=self._maybe_initialize_action_expert,
        ).run_batch(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
            batching_mode=batching_mode,
        )

    def _forward_infer_sequence(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
        runtime_state: VideoActionRolloutState,
    ) -> PolicyInferOutput:
        return DualExpertInferenceProgram(
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
        state = (
            replace(previous_state, cursor=replace(previous_state.cursor))
            if previous_state is not None
            else PolicyInferState()
        )
        runtime_state = (
            replace(state.variant_state)
            if isinstance(state.variant_state, VideoActionRolloutState)
            else VideoActionRolloutState()
        )
        runtime_state.require_complete_video_history()
        action_device = next(self.action_expert.parameters()).device
        action_dtype = next(self.action_expert.parameters()).dtype
        proprio_state = self.conditioning.resolve_proprio_state(
            context.state,
            label="dual-expert inference",
            fallback_state=runtime_state.proprio_state,
        )
        hidden_proprio_state = self.conditioning.resolve_infer_hidden_proprio_context(
            context.state,
            fallback_state=runtime_state.hidden_proprio_state,
        )
        dynamics_rollout_plan = resolve_dynamics_rollout_plan(
            program=self.config.program,
            request=context.dynamics,
        )
        generalist_rollout_mode = dynamics_rollout_plan.objective
        generalist_rollout_semantics = dynamics_rollout_plan.semantics
        infer_text_context = visual_outputs.frontend.conditioning.text_context
        if (
            generalist_rollout_semantics.drop_text_conditioning
            and infer_text_context is not None
        ):
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
                or self.config.generalist_mode_text_token
            ),
        )
        generalist_mode_text_token_count = 0
        if self.config.generalist_mode_text_token:
            if resolved_text_context is None:  # pragma: no cover - materialized above
                raise RuntimeError(
                    "dual-expert mode-token rollout expected materialized text context."
                )
            resolved_text_context, generalist_mode_text_token_count = (
                self.conditioning.append_generalist_mode_text_token(
                    visual_tower,
                    resolved_text_context,
                    generalist_rollout_mode,
                )
            )
        return replace(
            state,
            variant_state=replace(
                runtime_state,
                text_context=resolved_text_context,
                generalist_mode_text_token_count=int(generalist_mode_text_token_count),
                proprio_state=None
                if proprio_state is None
                else proprio_state.detach().clone(),
                hidden_proprio_state=None
                if hidden_proprio_state is None
                else hidden_proprio_state.detach().clone(),
            ),
        )

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        runtime_state = (
            infer_state.variant_state
            if isinstance(infer_state.variant_state, VideoActionRolloutState)
            else VideoActionRolloutState()
        )
        self._maybe_initialize_action_expert(visual_tower)
        ensure_dual_expert_policy_variant_inference_backend(
            policy_variant=self,
            visual_tower=visual_tower,
            policy_config=self.config,
        )
        return self._forward_infer_sequence(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            context=context,
            infer_state=infer_state,
            runtime_state=runtime_state,
        )
