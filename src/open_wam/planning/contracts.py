from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PlanningContext:
    """Policy-facing observation state for one candidate branch."""

    views: Mapping[str, Array]
    state: Array | None
    task_text: str | None
    predicted_video: Array | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_prediction(
        self,
        *,
        predicted_video: Array | None,
        views: Mapping[str, Array] | None = None,
        state: Array | None = None,
        metadata_updates: Mapping[str, Any] | None = None,
    ) -> "PlanningContext":
        metadata = dict(self.metadata)
        if metadata_updates:
            metadata.update(metadata_updates)
        return PlanningContext(
            views=self.views if views is None else views,
            state=self.state if state is None else state,
            task_text=self.task_text,
            predicted_video=predicted_video,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ActionChunk:
    """One policy-sampled action chunk in model/env raw action space."""

    actions: Array
    logprob: float | None = None
    sampler_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"ActionChunk.actions must have shape [T, D], got {actions.shape}.")
        object.__setattr__(self, "actions", actions)


@dataclass(frozen=True)
class DynamicsPrediction:
    """Forward-dynamics output for one candidate action chunk."""

    predicted_video: Array | None
    next_context: PlanningContext
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateTrajectory:
    """One imagined trajectory branch."""

    context: PlanningContext
    action_chunks: tuple[ActionChunk, ...] = ()
    predicted_videos: tuple[Array, ...] = ()
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def actions(self) -> Array:
        if not self.action_chunks:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate([chunk.actions for chunk in self.action_chunks], axis=0)

    @property
    def action_count(self) -> int:
        return int(sum(chunk.actions.shape[0] for chunk in self.action_chunks))

    def append(
        self,
        *,
        action_chunk: ActionChunk,
        prediction: DynamicsPrediction,
        score: float,
        metadata_updates: Mapping[str, Any] | None = None,
    ) -> "CandidateTrajectory":
        videos = self.predicted_videos
        if prediction.predicted_video is not None:
            videos = (*videos, np.asarray(prediction.predicted_video))
        metadata = dict(self.metadata)
        if metadata_updates:
            metadata.update(metadata_updates)
        return CandidateTrajectory(
            context=prediction.next_context,
            action_chunks=(*self.action_chunks, action_chunk),
            predicted_videos=videos,
            score=float(score),
            metadata=metadata,
        )


@dataclass(frozen=True)
class PlanningResult:
    """Planner output for one receding-horizon decision."""

    selected: CandidateTrajectory
    candidates: tuple[CandidateTrajectory, ...]
    first_action_chunk: ActionChunk
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PolicyActionSampler(Protocol):
    """Sample policy action chunks for one imagined context."""

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ) -> Sequence[ActionChunk]:
        """Return candidate action chunks ordered by policy preference if known."""


class ForwardDynamicsModel(Protocol):
    """Predict future observations for a candidate action chunk."""

    def predict(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> DynamicsPrediction:
        """Return predicted video plus the next candidate context."""


class ContextPropagator(Protocol):
    """Update non-visual candidate context after an imagined action chunk.

    Forward-dynamics models usually own the visual prediction and cache update.
    A separate propagator can update policy-facing state, proprio, or other
    cheap deterministic context that is not part of the learned video model.
    """

    def propagate(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> PlanningContext:
        """Return `context` with updated non-visual branch state."""


class TrajectoryEvaluator(Protocol):
    """Score imagined candidate trajectories. Higher is better."""

    def score(self, candidate: CandidateTrajectory, *, goal: Any | None = None) -> float:
        """Return a scalar utility for pruning and final selection."""


class BatchTrajectoryEvaluator(TrajectoryEvaluator, Protocol):
    """Score a candidate set jointly.

    This is needed for evaluators such as VLM rerankers where comparing all
    candidates in one prompt is the actual contract. Scalar evaluators can
    continue to implement only `score(...)`.
    """

    def score_candidates(
        self,
        candidates: Sequence[CandidateTrajectory],
        *,
        goal: Any | None = None,
    ) -> Sequence[float]:
        """Return one score per candidate. Higher is better."""
