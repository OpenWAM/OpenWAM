"""Typed skeleton for an application-owned action decoder."""

from __future__ import annotations

from typing import Any

from open_wam.sdk.config import ExtensionActionDecoderConfig
from open_wam.sdk.policy import (
    ActionDecoder,
    ActionDecoderInferOutput,
    ActionDecoderTrainOutput,
    PolicyInferOutput,
    PolicyTrainBatch,
    PolicyTrainOutput,
)


class TemplateActionDecoder(ActionDecoder):
    def __init__(self, config: ExtensionActionDecoderConfig) -> None:
        super().__init__()
        self.config = config

    def forward_train(
        self,
        policy_output: PolicyTrainOutput,
        batch: PolicyTrainBatch,
    ) -> ActionDecoderTrainOutput:
        raise NotImplementedError

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: Any | None = None,
    ) -> ActionDecoderInferOutput:
        raise NotImplementedError
