"""Lightweight public contracts for evaluation planning."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CausalVideoPredictionRollout": "video_prediction",
    "decode_canonical_latent_views": "video_prediction",
    "EvaluationRequest": "evaluation_contracts",
    "EvaluationSummary": "evaluation_contracts",
    "resolve_evaluation_request": "evaluation_contracts",
    "run_evaluation": "evaluate",
    "rollout_causal_video_prediction": "video_prediction",
}

__all__ = [
    "CausalVideoPredictionRollout",
    "EvaluationRequest",
    "EvaluationSummary",
    "decode_canonical_latent_views",
    "resolve_evaluation_request",
    "rollout_causal_video_prediction",
    "run_evaluation",
]


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
