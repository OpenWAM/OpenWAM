"""Action-head interfaces and future head implementations."""

from .base import (
    ActionHead,
    ActionHeadInferContext,
    ActionHeadInferOutput,
    ActionHeadInferState,
    ActionHeadTrainingBatch,
    ActionHeadTrainOutput,
)
from .contract_only import ContractOnlyActionHead, ContractOnlyActionHeadConfig

__all__ = [
    "ActionHead",
    "ActionHeadInferContext",
    "ActionHeadInferOutput",
    "ActionHeadInferState",
    "ActionHeadTrainingBatch",
    "ActionHeadTrainOutput",
    "ContractOnlyActionHead",
    "ContractOnlyActionHeadConfig",
]

