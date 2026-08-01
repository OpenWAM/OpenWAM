"""Lightweight public contracts for evaluation planning."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "DatasetEpisode": "sampled_eval_sampling",
    "EvaluationRequest": "evaluation_contracts",
    "EvaluationSummary": "evaluation_contracts",
    "SAMPLED_EVAL_DEFAULT_CONFIG": "sampled_eval_planning",
    "SAMPLED_EVAL_METHODS": "sampled_eval_planning",
    "SAMPLED_EVAL_SCHEDULERS": "sampled_eval_planning",
    "SampledEvalCase": "sampled_eval_planning",
    "SampledEvalCaseOptions": "sampled_eval_planning",
    "SampledEvalCheckpointSpec": "sampled_eval_planning",
    "SampledEvalMethodSpec": "sampled_eval_planning",
    "SampledEvalPreflightOptions": "sampled_eval_planning",
    "SampledEvalSchedulerSpec": "sampled_eval_planning",
    "SampledEvalTargetRequest": "sampled_eval_planning",
    "build_sampled_eval_cases": "sampled_eval_planning",
    "parse_sampled_eval_target_requests": "sampled_eval_planning",
    "preflight_sampled_eval_cases": "sampled_eval_planning",
    "resolve_sampled_eval_checkpoint_specs": "sampled_eval_planning",
    "resolve_evaluation_request": "evaluation_contracts",
    "run_evaluation": "evaluate",
    "sampled_eval_scheduler_flags": "sampled_eval_planning",
    "sampled_eval_scheduler_suffix": "sampled_eval_planning",
    "select_sampled_eval_specs_by_key": "sampled_eval_planning",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
