from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from .contracts import (
    ActionChunk,
    CandidateTrajectory,
    ContextPropagator,
    DynamicsPrediction,
    ForwardDynamicsModel,
    PlanningContext,
    PlanningResult,
    PolicyActionSampler,
    TrajectoryEvaluator,
)
from .evaluators import ConstantEvaluator


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for FDM-guided receding-horizon planning."""

    num_policy_samples: int = 4
    beam_width: int = 4
    chunk_action_steps: int = 16
    min_plan_action_steps: int = 32
    max_plan_chunks: int = 4
    execute_action_steps: int = 16
    policy_temperature: float = 0.8
    include_policy_prior_candidate: bool = False
    policy_prior_temperature: float = 0.0
    policy_prior_abstain_margin: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_policy_samples < 0:
            raise ValueError("num_policy_samples must be non-negative.")
        if self.num_policy_samples == 0 and not self.include_policy_prior_candidate:
            raise ValueError(
                "num_policy_samples may be zero only when include_policy_prior_candidate is true."
            )
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if self.chunk_action_steps <= 0:
            raise ValueError("chunk_action_steps must be positive.")
        if self.min_plan_action_steps <= 0:
            raise ValueError("min_plan_action_steps must be positive.")
        if self.max_plan_chunks <= 0:
            raise ValueError("max_plan_chunks must be positive.")
        if self.execute_action_steps <= 0:
            raise ValueError("execute_action_steps must be positive.")
        if self.policy_prior_abstain_margin < 0.0:
            raise ValueError("policy_prior_abstain_margin must be non-negative.")


class FdmGuidedRecedingHorizonPlanner:
    """Sample policy branches, imagine them with an FDM, and execute the best prefix."""

    def __init__(
        self,
        *,
        policy: PolicyActionSampler,
        dynamics: ForwardDynamicsModel,
        prior_policy: PolicyActionSampler | None = None,
        evaluator: TrajectoryEvaluator | None = None,
        context_propagator: ContextPropagator | None = None,
        config: PlannerConfig | None = None,
    ) -> None:
        self.policy = policy
        self.prior_policy = prior_policy
        self.dynamics = dynamics
        self.evaluator = evaluator or ConstantEvaluator()
        self.context_propagator = context_propagator
        self.config = config or PlannerConfig()

    def plan(
        self,
        context: PlanningContext,
        *,
        goal: Any | None = None,
        seed: int | None = None,
        include_policy_prior_candidate: bool | None = None,
        policy_prior_abstain_margin: float | None = None,
        policy_prior_chunk: ActionChunk | None = None,
    ) -> PlanningResult:
        config = self.config
        use_policy_prior = (
            bool(config.include_policy_prior_candidate)
            if include_policy_prior_candidate is None
            else bool(include_policy_prior_candidate)
        )
        prior_abstain_margin = (
            float(config.policy_prior_abstain_margin)
            if policy_prior_abstain_margin is None
            else float(policy_prior_abstain_margin)
        )
        if prior_abstain_margin < 0.0:
            raise ValueError("policy_prior_abstain_margin must be non-negative.")
        if policy_prior_chunk is not None and config.max_plan_chunks != 1:
            raise ValueError("policy_prior_chunk currently supports only one-chunk receding-horizon plans.")
        if config.num_policy_samples == 0 and not use_policy_prior:
            raise ValueError(
                "Planner has no candidates: num_policy_samples is zero and policy prior is disabled for this call."
            )
        root = CandidateTrajectory(context=context)
        beam: list[CandidateTrajectory] = [root]
        base_seed = config.seed if seed is None else int(seed)
        chunk_index = 0
        evaluated_candidate_count = 0

        while chunk_index < config.max_plan_chunks:
            expanded: list[CandidateTrajectory] = []
            for branch_index, candidate in enumerate(beam):
                policy_seed = _derive_seed(base_seed, chunk_index, branch_index, 0)
                candidate_chunks: list[Any] = []
                if use_policy_prior:
                    if policy_prior_chunk is not None:
                        prior_chunks = (policy_prior_chunk,)
                    else:
                        prior_sampler = self.prior_policy or self.policy
                        prior_chunks = prior_sampler.sample_action_chunks(
                            candidate.context,
                            num_samples=1,
                            chunk_action_steps=config.chunk_action_steps,
                            temperature=config.policy_prior_temperature,
                            seed=_derive_seed(base_seed, chunk_index, branch_index, -1),
                        )
                    candidate_chunks.extend(
                        _with_action_metadata(chunk, {"candidate_role": "policy_prior"})
                        for chunk in prior_chunks[:1]
                    )
                action_chunks = (
                    ()
                    if config.num_policy_samples == 0
                    else self.policy.sample_action_chunks(
                        candidate.context,
                        num_samples=config.num_policy_samples,
                        chunk_action_steps=config.chunk_action_steps,
                        temperature=config.policy_temperature,
                        seed=policy_seed,
                    )
                )
                candidate_chunks.extend(
                    _with_action_metadata(chunk, {"candidate_role": "policy_sample"})
                    for chunk in action_chunks
                )
                for sample_index, action_chunk in enumerate(candidate_chunks):
                    dynamics_seed = _derive_seed(base_seed, chunk_index, branch_index, sample_index + 1)
                    prediction = self.dynamics.predict(
                        candidate.context,
                        action_chunk,
                        seed=dynamics_seed,
                    )
                    if self.context_propagator is not None:
                        propagated_context = self.context_propagator.propagate(
                            prediction.next_context,
                            action_chunk,
                            seed=dynamics_seed,
                        )
                        prediction = DynamicsPrediction(
                            predicted_video=prediction.predicted_video,
                            next_context=propagated_context,
                            metadata=prediction.metadata,
                        )
                    partial = candidate.append(
                        action_chunk=action_chunk,
                        prediction=prediction,
                        score=0.0,
                        metadata_updates={
                            "last_chunk_index": chunk_index,
                            "last_branch_index": branch_index,
                            "last_sample_index": sample_index,
                        },
                    )
                    expanded.append(partial)
            if not expanded:
                raise RuntimeError("Policy sampler produced no candidate action chunks.")
            scores = _score_candidates(self.evaluator, expanded, goal=goal)
            scores = _apply_policy_prior_abstention(
                expanded,
                scores,
                margin=prior_abstain_margin,
            )
            expanded = [
                CandidateTrajectory(
                    context=candidate.context,
                    action_chunks=candidate.action_chunks,
                    predicted_videos=candidate.predicted_videos,
                    score=float(score),
                    metadata=candidate.metadata,
                )
                for candidate, score in zip(expanded, scores, strict=True)
            ]
            evaluated_candidate_count += len(expanded)
            expanded.sort(key=lambda item: item.score, reverse=True)
            beam = expanded[: config.beam_width]
            chunk_index += 1
            if beam[0].action_count >= config.min_plan_action_steps:
                break

        selected = beam[0]
        if not selected.action_chunks:
            raise RuntimeError("Planner selected an empty trajectory.")
        first_chunk = selected.action_chunks[0]
        return PlanningResult(
            selected=selected,
            candidates=tuple(beam),
            first_action_chunk=first_chunk,
            metadata={
                "planned_chunks": len(selected.action_chunks),
                "planned_action_steps": selected.action_count,
                "execute_action_steps": min(config.execute_action_steps, int(first_chunk.actions.shape[0])),
                "evaluated_candidate_count": int(evaluated_candidate_count),
                "final_beam_count": int(len(beam)),
            },
        )


def _derive_seed(base_seed: int, chunk_index: int, branch_index: int, sample_index: int) -> int:
    return int(base_seed + chunk_index * 1_000_003 + branch_index * 10_007 + sample_index)


def _with_action_metadata(action_chunk: ActionChunk, metadata_updates: dict[str, Any]) -> ActionChunk:
    metadata = dict(getattr(action_chunk, "metadata", {}) or {})
    metadata.update(metadata_updates)
    return ActionChunk(
        actions=action_chunk.actions,
        logprob=getattr(action_chunk, "logprob", None),
        sampler_score=getattr(action_chunk, "sampler_score", None),
        metadata=metadata,
    )


def _score_candidates(evaluator: Any, candidates: Sequence[CandidateTrajectory], *, goal: Any | None) -> list[float]:
    batch_scorer = getattr(evaluator, "score_candidates", None)
    if callable(batch_scorer):
        scores = list(batch_scorer(candidates, goal=goal))
    else:
        scores = [float(evaluator.score(candidate, goal=goal)) for candidate in candidates]
    if len(scores) != len(candidates):
        raise ValueError(
            "Trajectory evaluator returned the wrong number of scores: "
            f"expected {len(candidates)}, got {len(scores)}."
        )
    return [float(score) for score in scores]


def _apply_policy_prior_abstention(
    candidates: Sequence[CandidateTrajectory],
    scores: Sequence[float],
    *,
    margin: float,
) -> list[float]:
    """Prefer the deterministic policy prior when it is close to the best candidate."""

    resolved_scores = [float(score) for score in scores]
    if margin <= 0.0 or not candidates:
        return resolved_scores
    best_score = max(resolved_scores)
    for index, candidate in enumerate(candidates):
        if not candidate.action_chunks:
            continue
        role = candidate.action_chunks[-1].metadata.get("candidate_role")
        if role == "policy_prior" and resolved_scores[index] >= best_score - float(margin):
            resolved_scores[index] = best_score + 1e-6
            break
    return resolved_scores
