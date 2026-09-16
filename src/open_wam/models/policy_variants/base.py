from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import nn

from open_wam.configs.policy_video_action import VideoActionPolicyConfig
from open_wam.configs.enums import (
    BatchingMode,
    CurrentBlockCoupling,
    FeatureCacheScope,
    DynamicsObjective,
)
from open_wam.configs.policy_video_action import (
    supports_dynamics_routing,
    supports_video_conditioned_action,
)
from open_wam.models.common.denoising import independently_generated_modalities
from open_wam.models.common.observed_history import (
    commit_video_action_observed_history,
    reconcile_video_action_observed_history,
)
from open_wam.models.common.rollout import RolloutCursor
from open_wam.models.common.video_action_state import VideoActionRolloutState
from open_wam.contracts.action_space import ActionSpaceAdapter
from .output_semantics import video_action_program_output_modalities
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from .contracts import (
    PolicyInferContext,
    PolicyTemporalGeometry,
    PolicyTemporalSpan,
    DynamicsRolloutRequest,
    PolicyCompositionCapability,
    PolicyCompositionRngPolicy,
    PolicyRecurrentHistoryPolicy,
    PolicyInferenceCapabilities,
    PolicyInferenceOutputRequest,
    PolicyInferOutput,
    PolicyInferState,
    PolicyModuleTopology,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
    PolicyOutputModality,
    PolicyPipelineRequirements,
    PolicyPreparedInputs,
    PolicyRolloutContract,
    PolicyObservationWindowSessionPolicy,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVisualStage,
)


class PolicyVariant(nn.Module, ABC):
    """Policy-owned training, inference, and module-topology interface."""

    @property
    def source_action_adapter(self) -> ActionSpaceAdapter | None:
        """Optional pretrained action convention used at the data boundary."""
        return None

    @property
    def decoder_artifact_contract(self) -> str | None:
        """Return the typed decoder payload contract emitted by this policy."""

        return None

    @property
    def rollout_contract(self) -> PolicyRolloutContract:
        """Return lifecycle semantics required by generic rollout orchestration."""

        return PolicyRolloutContract()

    @property
    def inference_capabilities(self) -> PolicyInferenceCapabilities:
        """Declare products emitted by the policy's normal inference path.

        Action-only is the conservative default. Policies that emit video must
        opt in so generic composition cannot infer support from architecture or
        config names.
        """

        return PolicyInferenceCapabilities(
            native_modalities=frozenset({PolicyOutputModality.ACTION})
        )

    def validate_inference_output_request(
        self,
        request: PolicyInferenceOutputRequest | None,
    ) -> None:
        """Require requested products to match declared policy capabilities."""

        if request is None:
            return
        capabilities = self.inference_capabilities
        if request.modalities == capabilities.native_modalities:
            return
        if request in capabilities.selective_requests:
            return
        modalities = ", ".join(sorted(item.value for item in request.modalities))
        raise ValueError(
            f"{type(self).__name__} does not support the requested inference "
            f"outputs ({modalities}); native outputs are "
            f"{sorted(item.value for item in capabilities.native_modalities)}."
        )

    def resolve_inference_output_request(
        self,
        context: PolicyInferContext,
    ) -> PolicyInferenceOutputRequest:
        self.validate_inference_output_request(context.output_request)
        return context.output_request or PolicyInferenceOutputRequest(
            modalities=self.inference_capabilities.native_modalities,
        )

    def validate_inference_context(self, context: PolicyInferContext) -> None:
        """Validate ordinary output selection and transferable artifact inputs."""

        self.validate_inference_output_request(context.output_request)
        request = context.video_conditioned_action
        if request is None:
            return
        if context.dynamics is not None or context.video_generation is not None:
            raise ValueError(
                "Video-conditioned action inference cannot also request dynamics "
                "routing or video generation in the same policy call."
            )
        if context.output_request is not None and not context.output_request.requests(
            PolicyOutputModality.ACTION
        ):
            raise ValueError(
                "Video-conditioned action inference requires an action output."
            )
        if not self.inference_capabilities.supports_composition(request.capability):
            raise ValueError(
                f"{type(self).__name__} does not support generated-video to action "
                "composition."
            )

    def resolve_inference_context(
        self,
        context: PolicyInferContext,
    ) -> PolicyInferContext:
        """Translate generic inference inputs into variant-owned semantics."""

        return context

    def pipeline_requirements(
        self,
        *,
        default_action_dim: int,
        default_action_horizon: int,
        default_state_dim: int,
    ) -> PolicyPipelineRequirements:
        """Declare model-space geometry and shared conditioning adapters."""

        return PolicyPipelineRequirements(
            action_dim=default_action_dim,
            action_horizon=default_action_horizon,
            state_dim=default_state_dim,
        )

    def attach_visual_tower(self, visual_tower: VisualTower) -> None:
        """Finalize cross-module ownership after visual weights are loaded."""

        del visual_tower

    def validate_pipeline_assembly(
        self,
        *,
        data_action_dim: int,
        num_frames: int,
        backbone_num_layers: int,
    ) -> None:
        """Validate architecture-owned constraints before module assembly."""

        del data_action_dim, num_frames, backbone_num_layers

    def module_topology(self, visual_tower: VisualTower) -> PolicyModuleTopology:
        """Describe module ownership without exposing backend internals."""

        return PolicyModuleTopology(
            visual_runtime_modules=(visual_tower.core,),
            visual_components=visual_tower.component_topology(),
            fsdp_block_stacks=(visual_tower.core,),
        )

    def on_checkpoint_loaded(
        self,
        *,
        loaded_state_keys: frozenset[str],
        missing_state_keys: frozenset[str],
    ) -> None:
        """Finalize policy-local lazy state after checkpoint loading."""

        del loaded_state_keys, missing_state_keys

    def initialize_for_training(self, visual_tower: VisualTower) -> None:
        """Optional pre-wrap initialization hook for distributed training."""

        del visual_tower

    @abstractmethod
    def required_visual_stages(self) -> tuple[PolicyVisualStage, ...]:
        """Return the visual stages the pipeline must prepare eagerly."""

    def reconcile_observed_history(
        self,
        history: PolicyObservedHistory,
        infer_state: PolicyInferState | None,
    ) -> PolicyObservedHistoryOutput:
        """Optionally replace speculative rollout history with observations."""

        del history
        return PolicyObservedHistoryOutput(
            next_state=infer_state,
            debug={
                "reconciliation_skipped": True,
                "reason": "policy_does_not_reconcile_observed_history",
            },
        )

    @abstractmethod
    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        """Prepare train-time variant inputs."""

    @abstractmethod
    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        """Run the train-time policy forward pass."""

    def forward_train_batch(
        self,
        visual_tower: VisualTower,
        visual_outputs: Sequence[VisualStageOutputs],
        prepared_inputs: Sequence[PolicyPreparedInputs],
        *,
        batching_mode: BatchingMode,
    ) -> PolicyTrainOutput | tuple[PolicyTrainOutput, ...]:
        """Execute isolated sequences together under the variant's decoder contract.

        Consumers opt in explicitly; a loop of full-model forwards is not a
        substitute for shared heavy-layer execution. Return sample-local outputs
        for equal-sample reduction, or a native batched output when the decoder
        owns a different reduction (for example, supervised video-frame means).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support variable-length batch execution."
        )

    @abstractmethod
    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        """Prepare inference state for the current rollout step."""

    @abstractmethod
    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        """Run one inference step."""


class VideoActionPolicyVariant(PolicyVariant, ABC):
    """Shared semantic boundary for interchangeable video/action backends."""

    config: VideoActionPolicyConfig
    action_dim: int
    action_horizon: int

    @property
    @abstractmethod
    def action_tokens_per_frame(self) -> int:
        """Action density of the model's temporal representation."""

    def reconcile_observed_history(
        self,
        history: PolicyObservedHistory,
        infer_state: PolicyInferState | None,
    ) -> PolicyObservedHistoryOutput:
        density = self.rollout_contract.action_tokens_per_frame
        if infer_state is None and history.execution_commit is None:
            video = history.video_latents
            if video.ndim != 5 or video.shape[2] == 0:
                raise ValueError("History seeds require nonempty BCTHW video latents.")
            span = PolicyTemporalSpan(history.start_frame, video.shape[2])
            actions = history.action_history
            if actions is None:
                actions = video.new_zeros(
                    video.shape[0], span.frame_count * density, self.action_dim
                )
                history = replace(
                    history,
                    action_history=actions,
                    action_mask=actions.new_zeros((*actions.shape[:2], 1)),
                )
            if (
                actions is not None
                and actions.ndim == 3
                and actions.shape[1] == (span.frame_count - 1) * density
            ):
                # An observed seed's anchor has no reaching action in this interval.
                mask = history.action_mask
                if mask is None:
                    mask = actions.new_ones((*actions.shape[:2], 1))
                if mask.shape != (*actions.shape[:2], 1):
                    raise ValueError(
                        "History seed action validity must match its action tokens."
                    )
                history = replace(
                    history,
                    action_history=torch.cat(
                        [
                            actions.new_zeros(
                                actions.shape[0], density, actions.shape[-1]
                            ),
                            actions,
                        ],
                        dim=1,
                    ),
                    action_mask=torch.cat(
                        [mask.new_zeros(mask.shape[0], density, 1), mask], dim=1
                    ),
                )
            proprio = history.proprio_history
            if proprio is not None:
                if proprio.ndim == 2:
                    proprio = proprio.unsqueeze(0)
                if proprio.ndim != 3 or proprio.shape[:2] != (
                    video.shape[0],
                    span.frame_count,
                ):
                    raise ValueError(
                        "History seed proprio must align with every video frame."
                    )
            state = PolicyInferState(
                cursor=RolloutCursor(current_start_frame=span.start_frame),
                temporal_geometry=PolicyTemporalGeometry(
                    frame_chunk_size=self.inference_config.frame_chunk_size,
                    attention_window_size=self.inference_config.attention_window_size,
                ),
                variant_state=VideoActionRolloutState(
                    proprio_state=proprio,
                    hidden_proprio_state=None if proprio is None else proprio[:, -1],
                    past_hidden_proprio_states=None
                    if proprio is None
                    else proprio[:, :0],
                ),
            )
            return commit_video_action_observed_history(
                policy_state=state,
                history=history,
                observed_span=span,
                action_tokens_per_frame=density,
                action_dim=self.action_dim,
            )
        return reconcile_video_action_observed_history(
            policy_state=infer_state,
            history=history,
            action_tokens_per_frame=density,
            action_dim=self.action_dim,
        )

    @property
    def rollout_contract(self) -> PolicyRolloutContract:
        return PolicyRolloutContract(
            observation_window_session_policy=PolicyObservationWindowSessionPolicy.REBUILD_FROM_OBSERVATION_WINDOW,
            action_tokens_per_frame=self.action_tokens_per_frame,
            supports_speculative_continuation=True,
        )

    def resolve_inference_context(
        self, context: PolicyInferContext
    ) -> PolicyInferContext:
        geometry = context.require_temporal_geometry()
        request = context.video_conditioned_action
        count = (
            context.video_generation.frame_count
            if context.video_generation is not None
            else request.generated_video.latents.shape[2]
            if request is not None
            else geometry.frame_chunk_size
        )
        if count > self.inference_config.frame_chunk_size:
            raise ValueError(
                "Requested chunk exceeds the configured inference chunk size."
            )
        context = replace(
            context,
            temporal_geometry=PolicyTemporalGeometry(
                frame_chunk_size=count,
                attention_window_size=geometry.attention_window_size,
            ),
        )
        if (
            request is not None
            and self.config.current_block_coupling
            is not CurrentBlockCoupling.VIDEO_THEN_ACTION
        ):
            context = replace(
                context,
                dynamics=DynamicsRolloutRequest(
                    objective=DynamicsObjective.VIDEO_CONDITIONED_ACTION,
                    clean_video=request.generated_video.latents,
                    frame_chunk_size=count,
                ),
                video_conditioned_action=None,
            )
        return context

    @property
    def inference_capabilities(self) -> PolicyInferenceCapabilities:
        native = video_action_program_output_modalities(self.config.program)
        coupling = self.config.current_block_coupling
        fixed_objective = self.config.fixed_conditioning_mode
        routed = supports_dynamics_routing(self.config.program)
        selective: list[PolicyInferenceOutputRequest] = []
        independent = independently_generated_modalities(coupling) & native
        if PolicyOutputModality.VIDEO in independent:
            selective.append(PolicyInferenceOutputRequest.video_only())
        if PolicyOutputModality.ACTION in independent:
            selective.append(PolicyInferenceOutputRequest.action_only())
        return PolicyInferenceCapabilities(
            native_modalities=native,
            required_training_objective=(
                (fixed_objective or DynamicsObjective.JOINT) if routed else None
            ),
            required_future_modalities=(
                frozenset({PolicyOutputModality.ACTION})
                if fixed_objective is DynamicsObjective.ACTION_CONDITIONED_VIDEO
                else frozenset({PolicyOutputModality.VIDEO})
                if fixed_objective is DynamicsObjective.VIDEO_CONDITIONED_ACTION
                else frozenset()
            ),
            selective_requests=tuple(selective),
            composition_capabilities=(
                (
                    PolicyCompositionCapability.video_to_action(
                        required_training_objective=(
                            DynamicsObjective.VIDEO_CONDITIONED_ACTION
                            if routed
                            else None
                        ),
                        rng_policy=(
                            PolicyCompositionRngPolicy.CALLER_STREAM
                            if coupling is CurrentBlockCoupling.VIDEO_THEN_ACTION
                            else PolicyCompositionRngPolicy.ISOLATED_STEP_SEED
                        )
                    ),
                )
                if supports_video_conditioned_action(self.config.program)
                else ()
            ),
            recurrent_history_policy=PolicyRecurrentHistoryPolicy.EXPLICIT_RECONCILIATION,
            feature_cache_scope=(
                FeatureCacheScope.DENOISING_CALL
                if self.inference_config.use_cache
                else FeatureCacheScope.NONE
            ),
        )

    @property
    def source_action_channel_ids(self) -> tuple[int, ...]:
        """Model-space channels that recover the source action representation."""

        return ()

    @property
    def accepted_source_action_shapes(self) -> tuple[tuple[int, int], ...]:
        """Source action layouts accepted before policy-owned adaptation."""

        return ((self.action_dim, self.action_horizon),)

    def pipeline_requirements(
        self,
        *,
        default_action_dim: int,
        default_action_horizon: int,
        default_state_dim: int,
    ) -> PolicyPipelineRequirements:
        del default_action_dim, default_action_horizon
        conditioning = self.config.conditioning_requirements
        return PolicyPipelineRequirements(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            state_dim=default_state_dim,
            proprio_context_mode=conditioning.proprio_context_mode,
            dynamics_mode_context_enabled=conditioning.dynamics_mode_context_enabled,
            text_conditioning_mode=conditioning.text_conditioning_mode,
            source_action_channel_ids=self.source_action_channel_ids,
            accepted_source_action_shapes=self.accepted_source_action_shapes,
        )
