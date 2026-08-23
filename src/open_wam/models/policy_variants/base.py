from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from open_wam.configs.policy_video_action import VideoActionPolicyConfig
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from .contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyModuleTopology,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
    PolicyPipelineRequirements,
    PolicyPreparedInputs,
    PolicyRolloutContract,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVisualStage,
)


class PolicyVariant(nn.Module, ABC):
    """Policy-owned training, inference, and module-topology interface."""

    @property
    def decoder_artifact_contract(self) -> str | None:
        """Return the typed decoder payload contract emitted by this policy."""

        return None

    @property
    def rollout_contract(self) -> PolicyRolloutContract:
        """Return lifecycle semantics required by generic rollout orchestration."""

        return PolicyRolloutContract()

    def build_rollout_infer_extra(
        self,
        *,
        runtime_device: torch.device | None,
    ) -> dict[str, object]:
        """Return backend-owned fields for one rollout inference context."""

        del runtime_device
        return {}

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
            source_action_channel_ids=self.source_action_channel_ids,
            accepted_source_action_shapes=self.accepted_source_action_shapes,
        )
