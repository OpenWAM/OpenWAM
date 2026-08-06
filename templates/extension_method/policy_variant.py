"""Typed skeleton for an application-owned policy variant."""

from __future__ import annotations

from open_wam.sdk.config import ExtensionPolicyConfig
from open_wam.sdk.policy import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVariant,
    VisualStageOutputs,
    VisualTower,
)


class TemplatePolicyVariant(PolicyVariant):
    def __init__(self, config: ExtensionPolicyConfig) -> None:
        super().__init__()
        self.config = config

    def attach_site(self) -> str:
        return self.config.attach_site.value

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend", "core")

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        raise NotImplementedError

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        raise NotImplementedError

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        raise NotImplementedError

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        raise NotImplementedError
