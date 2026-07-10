from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import CandidateTrajectory


@dataclass(frozen=True)
class ConstantEvaluator:
    """Baseline evaluator that leaves candidate ordering to the policy sampler."""

    value: float = 0.0

    def score(self, candidate: CandidateTrajectory, *, goal: Any | None = None) -> float:
        del candidate, goal
        return float(self.value)


@dataclass(frozen=True)
class ActionMagnitudeEvaluator:
    """Prefer shorter/smoother plans when no visual goal metric is available."""

    action_l2_weight: float = 1.0

    def score(self, candidate: CandidateTrajectory, *, goal: Any | None = None) -> float:
        del goal
        actions = candidate.actions
        if actions.size == 0:
            return 0.0
        return -float(self.action_l2_weight) * float(np.mean(np.linalg.norm(actions, axis=-1)))


@dataclass(frozen=True)
class GoalImageL2Evaluator:
    """Score candidates by final predicted image distance to a goal image."""

    weight: float = 1.0

    def score(self, candidate: CandidateTrajectory, *, goal: Any | None = None) -> float:
        if goal is None or not candidate.predicted_videos:
            return 0.0
        goal_image = np.asarray(goal, dtype=np.float32)
        predicted = np.asarray(candidate.predicted_videos[-1], dtype=np.float32)
        if predicted.ndim == 4:
            predicted = predicted[-1]
        if predicted.shape != goal_image.shape:
            raise ValueError(
                "GoalImageL2Evaluator expects the goal image to match the final predicted frame, "
                f"got predicted={predicted.shape}, goal={goal_image.shape}."
            )
        diff = predicted - goal_image
        return -float(self.weight) * float(np.mean(np.square(diff)))


@dataclass(frozen=True)
class GoalDeltaAlignmentEvaluator:
    """Score whether predicted visual change points toward the goal change.

    The goal object must contain:

    - `current`: current RGB canvas in the same layout as predicted frames
    - `target`: desired RGB canvas in the same layout as predicted frames

    This intentionally scores local predicted delta instead of absolute final
    image distance. Static background pixels are ignored unless
    `background_penalty_weight` is positive.
    """

    alignment_weight: float = 1.0
    background_penalty_weight: float = 0.0
    change_threshold: float = 8.0
    eps: float = 1e-6

    def score(self, candidate: CandidateTrajectory, *, goal: Any | None = None) -> float:
        if goal is None or not candidate.predicted_videos:
            return 0.0
        if not isinstance(goal, dict) or "current" not in goal or "target" not in goal:
            raise ValueError("GoalDeltaAlignmentEvaluator expects goal={'current': ..., 'target': ...}.")
        current = np.asarray(goal["current"], dtype=np.float32)
        target = np.asarray(goal["target"], dtype=np.float32)
        predicted = np.asarray(candidate.predicted_videos[-1], dtype=np.float32)
        if predicted.ndim == 4:
            predicted = predicted[-1]
        if predicted.shape != target.shape or current.shape != target.shape:
            raise ValueError(
                "GoalDeltaAlignmentEvaluator expects current, predicted, and target shapes to match, "
                f"got current={current.shape}, predicted={predicted.shape}, target={target.shape}."
            )

        current = current / 255.0
        target = target / 255.0
        predicted = predicted / 255.0
        target_delta = target - current
        predicted_delta = predicted - current
        change_mask = np.mean(np.abs(target_delta), axis=-1) >= (float(self.change_threshold) / 255.0)
        if not np.any(change_mask):
            change_mask = np.ones(target_delta.shape[:2], dtype=bool)

        target_delta_fg = target_delta[change_mask]
        predicted_delta_fg = predicted_delta[change_mask]
        target_norm = float(np.sqrt(np.mean(np.square(target_delta_fg))) + float(self.eps))
        alignment = float(np.mean(predicted_delta_fg * target_delta_fg)) / target_norm
        score = float(self.alignment_weight) * alignment

        if float(self.background_penalty_weight) > 0.0 and np.any(~change_mask):
            background_delta = predicted_delta[~change_mask]
            score -= float(self.background_penalty_weight) * float(np.mean(np.square(background_delta)))
        return score
