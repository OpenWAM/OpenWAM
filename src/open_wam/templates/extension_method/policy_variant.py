"""Minimal runnable application-owned policy variant."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from open_wam.sdk.config import ExtensionPolicyConfig
from open_wam.sdk.policy import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVariant,
    PolicyVisualStage,
    VisualStageOutputs,
    VisualTower,
)


class TemplatePolicyVariant(PolicyVariant):
    def __init__(self, config: ExtensionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_norm = nn.LayerNorm(config.hidden_size)

    def required_visual_stages(self) -> tuple[PolicyVisualStage, ...]:
        return (PolicyVisualStage.FRONTEND, PolicyVisualStage.CORE)

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        return PolicyPreparedInputs(
            batch=batch,
            variant_inputs={"policy_features": self._policy_features(visual_outputs)},
        )

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        del visual_tower, visual_outputs
        policy_features = prepared_inputs.variant_inputs["policy_features"]
        return PolicyTrainOutput(
            policy_features=policy_features,
            metrics={
                "template_feature_rms": policy_features.square().mean().sqrt().detach()
            },
        )

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        del visual_tower, visual_outputs, context
        if previous_state is not None:
            return previous_state
        return PolicyInferState()

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        del visual_tower, context
        return PolicyInferOutput(
            policy_features=self._policy_features(visual_outputs),
            next_state=replace(infer_state, step_index=infer_state.step_index + 1),
        )

    def _policy_features(self, visual_outputs: VisualStageOutputs) -> torch.Tensor:
        if visual_outputs.core is None:
            raise ValueError("Template policy requires the visual core stage.")
        tokens = visual_outputs.core.tokens
        if tokens.ndim < 2 or tokens.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "Template policy expected visual-core tokens ending in hidden "
                f"size {self.config.hidden_size}, got {tuple(tokens.shape)}."
            )
        return self.feature_norm(tokens.reshape(tokens.shape[0], -1, tokens.shape[-1]))
