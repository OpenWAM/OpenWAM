from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from open_wam.planning import (
    ActionChunk,
    CachedPolicyActionSampler,
    CandidateCacheConfig,
    CandidateTrajectory,
    DynamicsPrediction,
    FdmGuidedRecedingHorizonPlanner,
    PlannerConfig,
    PlanningContext,
)
import open_wam.planning.pi0fast as pi0fast_module
from open_wam.planning.pi0fast import Pi0FastBatchMapping, Pi0FastPolicySampler, build_pi0fast_raw_batch
from open_wam.planning.openwam_fdm import OpenWamActionConditionedFdm
from open_wam.planning.evaluators import GoalDeltaAlignmentEvaluator
from open_wam.planning.vlm_evaluators import (
    GeminiVlmCandidateRerankEvaluator,
    GeminiVlmRerankConfig,
    _scores_from_gemini_result,
    build_gemini_candidate_prompt,
    build_tiled_candidate_video,
    infer_libero_progress_hints,
)
from open_wam.simulators import SimulatorObservation


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakePolicy:
    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ):
        del context, temperature, seed
        return [
            ActionChunk(np.full((chunk_action_steps, 2), sample_index, dtype=np.float32))
            for sample_index in range(num_samples)
        ]


class ConstantPolicy:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ):
        del context, temperature, seed
        return [
            ActionChunk(np.full((chunk_action_steps, 2), self.value, dtype=np.float32))
            for _ in range(num_samples)
        ]


class CountingPolicy(ConstantPolicy):
    def __init__(self, value: float) -> None:
        super().__init__(value)
        self.calls = 0

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ):
        self.calls += 1
        return super().sample_action_chunks(
            context,
            num_samples=num_samples,
            chunk_action_steps=chunk_action_steps,
            temperature=temperature,
            seed=seed,
        )


class IndexedCountingPolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.requested_counts: list[int] = []

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ):
        del context, temperature
        self.calls += 1
        self.requested_counts.append(int(num_samples))
        seed_offset = 0 if seed is None else int(seed) % 1000
        return [
            ActionChunk(np.full((chunk_action_steps, 2), seed_offset + sample_index, dtype=np.float32))
            for sample_index in range(num_samples)
        ]


class FakeDynamics:
    def predict(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> DynamicsPrediction:
        del seed
        value = float(action_chunk.actions[0, 0])
        video = np.full((1, 4, 4, 3), value, dtype=np.float32)
        next_context = context.with_prediction(
            predicted_video=video,
            metadata_updates={"last_action_value": value},
        )
        return DynamicsPrediction(predicted_video=video, next_context=next_context)


class StateRecordingPolicy:
    def __init__(self) -> None:
        self.seen_states: list[float] = []

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ):
        del temperature, seed
        value = float(np.asarray(context.state, dtype=np.float32).reshape(-1)[0])
        self.seen_states.append(value)
        return [
            ActionChunk(np.full((chunk_action_steps, 1), value, dtype=np.float32))
            for _ in range(num_samples)
        ]


class IncrementStatePropagator:
    def propagate(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> PlanningContext:
        del action_chunk, seed
        state = np.asarray(context.state, dtype=np.float32) + 1.0
        return context.with_prediction(
            predicted_video=context.predicted_video,
            state=state,
            metadata_updates={"propagated_state": float(state.reshape(-1)[0])},
        )


class CountingPropagator:
    def __init__(self) -> None:
        self.calls = 0

    def propagate(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> PlanningContext:
        del action_chunk, seed
        self.calls += 1
        return context


class FakeOpenWamRollout:
    frame_chunk_size = 2
    action_per_frame = 2

    def __init__(self) -> None:
        self.runner = type("Runner", (), {"pipeline": type("Pipeline", (), {"visual_tower": torch.nn.Linear(1, 1)})()})()
        self.infer_grad_enabled = None
        self.last_infer_kwargs = None

    def infer_chunk(self, **kwargs):
        self.last_infer_kwargs = kwargs
        self.infer_grad_enabled = torch.is_grad_enabled()
        return type(
            "Chunk",
            (),
            {
                "session": "next-session",
                "predicted_latents": torch.zeros(1, 1, 2, 2, 2),
                "debug": {"ok": True},
            },
        )()


class FakeOpenWamWarmupRollout:
    frame_chunk_size = 4
    action_per_frame = 4

    def __init__(self) -> None:
        visual_tower = torch.nn.Linear(1, 1)
        self.encode_grad_enabled = None
        self.reset_grad_enabled = None

        def _encode_video(_self, video, placements, reset_reference_cache):
            del placements, reset_reference_cache
            self.encode_grad_enabled = torch.is_grad_enabled()
            return torch.zeros(
                int(video.shape[0]),
                1,
                int(video.shape[2]),
                2,
                2,
            )

        visual_tower.frontend = type(
            "Frontend",
            (),
            {"encode_video": _encode_video},
        )()
        self.runner = type(
            "Runner",
            (),
            {
                "pipeline": type(
                    "Pipeline",
                    (),
                    {
                        "visual_tower": visual_tower,
                        "policy_variant": type("Variant", (), {"action_dim": 7})(),
                        "canonicalize": self._canonicalize,
                    },
                )()
            },
        )()
        self.captured = {}
        self.canonicalize_views = None

    def _canonicalize(self, views):
        assert "observation.images.agentview_rgb" in views
        self.canonicalize_views = {key: value.detach().cpu().clone() for key, value in views.items()}
        return SimpleNamespace(
            video=torch.zeros(1, 3, 1, 4, 4),
            placements=(),
        )

    def reset_and_warmup(self, **kwargs):
        self.reset_grad_enabled = torch.is_grad_enabled()
        self.captured = kwargs
        return "warmed-session"


class MutatingOpenWamRollout(FakeOpenWamRollout):
    def infer_chunk(self, **kwargs):
        session = kwargs["session"]
        session.policy_state.cache["mutated"] = True
        return type(
            "Chunk",
            (),
            {
                "session": session,
                "predicted_latents": torch.zeros(1, 1, 1, 2, 2),
                "debug": {},
            },
        )()


class PreferLargestFinalVideo:
    def score(self, candidate: CandidateTrajectory, *, goal=None) -> float:
        del goal
        return float(candidate.predicted_videos[-1].mean())


class BatchPreferSmallestFinalVideo:
    def score(self, candidate: CandidateTrajectory, *, goal=None) -> float:
        raise AssertionError("Planner should use score_candidates when available.")

    def score_candidates(self, candidates, *, goal=None):
        del goal
        return [-float(candidate.predicted_videos[-1].mean()) for candidate in candidates]


class BatchScoresActionValue:
    def score(self, candidate: CandidateTrajectory, *, goal=None) -> float:
        raise AssertionError("Planner should use score_candidates when available.")

    def score_candidates(self, candidates, *, goal=None):
        del goal
        return [float(candidate.action_chunks[-1].actions[0, 0]) for candidate in candidates]


class BatchScoresActionCount:
    def score(self, candidate: CandidateTrajectory, *, goal=None) -> float:
        raise AssertionError("Planner should use score_candidates when available.")

    def score_candidates(self, candidates, *, goal=None):
        del goal
        return [float(candidate.action_count) for candidate in candidates]


class ShortHorizonPaddedDynamics:
    crop_decoded_video_to_action_steps = True

    def __init__(self, *, internal_steps: int) -> None:
        self.internal_steps = int(internal_steps)
        self.seen_action_steps: list[int] = []

    def expected_action_steps_per_chunk(self) -> int:
        return self.internal_steps

    def predict(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> DynamicsPrediction:
        del seed
        action_steps = int(action_chunk.actions.shape[0])
        self.seen_action_steps.append(action_steps)
        video = np.full((action_steps, 4, 4, 3), action_steps, dtype=np.uint8)
        next_context = context.with_prediction(
            predicted_video=video,
            metadata_updates={
                "fdm_internal_action_steps": self.internal_steps,
                "fdm_policy_action_steps": action_steps,
            },
        )
        return DynamicsPrediction(
            predicted_video=video,
            next_context=next_context,
            metadata={
                "internal_action_steps": self.internal_steps,
                "policy_action_steps": action_steps,
            },
        )


class MetadataRecordingPolicy:
    def __init__(self) -> None:
        self.seen_rewarm_flags: list[bool] = []
        self.seen_temperatures: list[float] = []

    def reset(self) -> None:
        pass

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ):
        del num_samples, seed
        self.seen_temperatures.append(float(temperature))
        self.seen_rewarm_flags.append(bool(context.metadata.get("short_horizon_fdm_rewarm")))
        return [ActionChunk(np.zeros((chunk_action_steps, 1), dtype=np.float32))]


class FailingPlanningPolicy:
    def reset(self) -> None:
        pass

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ):
        del context, num_samples, chunk_action_steps, temperature, seed
        raise AssertionError("bad policy token stream")


class FakeGeminiClient:
    def __init__(self) -> None:
        self.uploaded_paths = []
        self.generated = []

    def upload_file(self, file_path, *, mime_type=None, display_name=None):
        del mime_type, display_name
        path = Path(file_path)
        self.uploaded_paths.append(path)
        if path.suffix == ".mp4":
            mime = "video/mp4"
        elif path.suffix == ".png":
            mime = "image/png"
        else:
            mime = "application/octet-stream"
        return {"uri": f"file://{path.name}", "mimeType": mime, "name": f"files/{path.stem}"}

    def generate_json(self, *, model, prompt, files, response_schema):
        self.generated.append(
            {
                "model": model,
                "prompt": prompt,
                "files": files,
                "response_schema": response_schema,
            }
        )
        return {
            "winner": "B",
            "ranking": ["B", "A"],
            "scores": {"A": 0.2, "B": 0.9},
            "risks": {"A": "less progress", "B": "closer to target"},
            "confidence": 0.75,
        }


def _context() -> PlanningContext:
    return PlanningContext(
        views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(2, dtype=np.float32),
        task_text="pick up the object",
    )


def _load_planning_runner():
    script_path = REPO_ROOT / "scripts/run_libero_fdm_guided_planning.py"
    spec = importlib.util.spec_from_file_location("run_libero_fdm_guided_planning", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_task8_offline_reranking_runner():
    script_path = REPO_ROOT / "scripts/run_libero_task8_offline_reranking.py"
    spec = importlib.util.spec_from_file_location("run_libero_task8_offline_reranking", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_task8_intervention_runner():
    script_path = REPO_ROOT / "scripts/run_libero_task8_intervention_poc.py"
    spec = importlib.util.spec_from_file_location("run_libero_task8_intervention_poc", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakePi0Policy:
    def __init__(self) -> None:
        self.config = type("Config", (), {"temperature": 0.0, "n_action_steps": 3})()
        self.seen_batches = []
        self.selected_count = 0

    def predict_action_chunk(self, batch):
        self.seen_batches.append(batch)
        return torch.ones(1, 6, 2)

    def select_action(self, batch):
        self.seen_batches.append(batch)
        self.selected_count += 1
        return torch.full((1, 2), float(self.selected_count))


class MarkingPreprocessor:
    def __call__(self, batch):
        batch = dict(batch)
        batch["preprocessed"] = True
        return batch


class ScalingPostprocessor:
    def __call__(self, actions):
        return actions * 2.0


def test_candidate_cache_reuses_larger_sample_prefixes(tmp_path: Path) -> None:
    policy = IndexedCountingPolicy()
    cached = CachedPolicyActionSampler(
        policy,
        CandidateCacheConfig(
            cache_dir=tmp_path,
            namespace="unit-test",
            sample_count=7,
        ),
    )
    context = PlanningContext(
        views={"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(8, dtype=np.float32),
        task_text="put both moka pots on the stove",
    )

    first = cached.sample_action_chunks(
        context,
        num_samples=3,
        chunk_action_steps=10,
        temperature=0.45,
        seed=123,
    )
    second = cached.sample_action_chunks(
        context,
        num_samples=7,
        chunk_action_steps=10,
        temperature=0.45,
        seed=123,
    )

    assert policy.calls == 1
    assert policy.requested_counts == [7]
    np.testing.assert_allclose(first[0].actions, second[0].actions)
    np.testing.assert_allclose(first[2].actions, second[2].actions)
    assert [float(chunk.actions[0, 0]) for chunk in second] == [123.0 + idx for idx in range(7)]
    assert first[0].metadata["candidate_cache_hit"] is False
    assert second[0].metadata["candidate_cache_hit"] is True


def test_candidate_cache_keys_policy_facing_context(tmp_path: Path) -> None:
    policy = IndexedCountingPolicy()
    cached = CachedPolicyActionSampler(
        policy,
        CandidateCacheConfig(cache_dir=tmp_path, namespace="unit-test", sample_count=4),
    )
    base_context = PlanningContext(
        views={"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(8, dtype=np.float32),
        task_text="put both moka pots on the stove",
    )
    changed_context = PlanningContext(
        views={"agentview_image": np.ones((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(8, dtype=np.float32),
        task_text="put both moka pots on the stove",
    )

    cached.sample_action_chunks(
        base_context,
        num_samples=3,
        chunk_action_steps=10,
        temperature=0.45,
        seed=123,
    )
    cached.sample_action_chunks(
        changed_context,
        num_samples=3,
        chunk_action_steps=10,
        temperature=0.45,
        seed=123,
    )

    assert policy.calls == 2
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_planner_returns_first_chunk_from_best_fdm_scored_branch() -> None:
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=FakePolicy(),
        dynamics=FakeDynamics(),
        evaluator=PreferLargestFinalVideo(),
        config=PlannerConfig(
            num_policy_samples=3,
            beam_width=2,
            chunk_action_steps=4,
            min_plan_action_steps=8,
            max_plan_chunks=2,
            execute_action_steps=4,
            policy_temperature=1.2,
        ),
    )

    result = planner.plan(_context(), seed=123)

    assert result.metadata["planned_chunks"] == 2
    assert result.metadata["planned_action_steps"] == 8
    assert result.metadata["evaluated_candidate_count"] == 9
    assert result.metadata["final_beam_count"] == 2
    assert result.first_action_chunk.actions.shape == (4, 2)
    assert np.all(result.first_action_chunk.actions == 2.0)
    assert len(result.candidates) == 2
    assert result.candidates[0].score >= result.candidates[1].score


def test_planner_uses_batch_evaluator_when_available() -> None:
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=FakePolicy(),
        dynamics=FakeDynamics(),
        evaluator=BatchPreferSmallestFinalVideo(),
        config=PlannerConfig(
            num_policy_samples=3,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
        ),
    )

    result = planner.plan(_context(), seed=123)

    assert result.first_action_chunk.actions.shape == (4, 2)
    assert np.all(result.first_action_chunk.actions == 0.0)
    assert result.selected.score == pytest.approx(-0.0)


def test_planner_can_include_policy_prior_candidate() -> None:
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=ConstantPolicy(5.0),
        prior_policy=ConstantPolicy(0.0),
        dynamics=FakeDynamics(),
        evaluator=BatchPreferSmallestFinalVideo(),
        config=PlannerConfig(
            num_policy_samples=2,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
            policy_prior_temperature=0.0,
        ),
    )

    result = planner.plan(_context(), seed=123)

    assert result.metadata["evaluated_candidate_count"] == 3
    assert result.first_action_chunk.metadata["candidate_role"] == "policy_prior"
    assert np.all(result.first_action_chunk.actions == 0.0)


def test_planner_can_run_prior_only_ablation() -> None:
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=CountingPolicy(5.0),
        prior_policy=ConstantPolicy(0.0),
        dynamics=FakeDynamics(),
        evaluator=BatchPreferSmallestFinalVideo(),
        config=PlannerConfig(
            num_policy_samples=0,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
        ),
    )

    result = planner.plan(_context(), seed=123)

    assert planner.policy.calls == 0
    assert result.metadata["evaluated_candidate_count"] == 1
    assert result.metadata["final_beam_count"] == 1
    assert result.first_action_chunk.metadata["candidate_role"] == "policy_prior"
    assert np.all(result.first_action_chunk.actions == 0.0)


def test_prior_only_planner_errors_if_prior_is_disabled_per_call() -> None:
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=ConstantPolicy(5.0),
        prior_policy=ConstantPolicy(0.0),
        dynamics=FakeDynamics(),
        evaluator=BatchPreferSmallestFinalVideo(),
        config=PlannerConfig(
            num_policy_samples=0,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
        ),
    )

    with pytest.raises(ValueError, match="no candidates"):
        planner.plan(_context(), seed=123, include_policy_prior_candidate=False)


def test_planner_uses_precomputed_policy_prior_without_sampling_prior_policy() -> None:
    prior_policy = CountingPolicy(0.0)
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=ConstantPolicy(5.0),
        prior_policy=prior_policy,
        dynamics=FakeDynamics(),
        evaluator=BatchPreferSmallestFinalVideo(),
        config=PlannerConfig(
            num_policy_samples=2,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
        ),
    )
    prior_chunk = ActionChunk(np.full((4, 2), -1.0, dtype=np.float32))

    result = planner.plan(_context(), seed=123, policy_prior_chunk=prior_chunk)

    assert prior_policy.calls == 0
    assert result.metadata["evaluated_candidate_count"] == 3
    assert result.first_action_chunk.metadata["candidate_role"] == "policy_prior"
    assert np.all(result.first_action_chunk.actions == -1.0)


def test_planner_can_suppress_policy_prior_per_call() -> None:
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=ConstantPolicy(5.0),
        prior_policy=ConstantPolicy(0.0),
        dynamics=FakeDynamics(),
        evaluator=BatchPreferSmallestFinalVideo(),
        config=PlannerConfig(
            num_policy_samples=2,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
        ),
    )

    result = planner.plan(_context(), seed=123, include_policy_prior_candidate=False)

    assert result.metadata["evaluated_candidate_count"] == 2
    assert result.first_action_chunk.metadata["candidate_role"] == "policy_sample"
    assert np.all(result.first_action_chunk.actions == 5.0)


def test_planner_policy_prior_margin_prefers_close_baseline_candidate() -> None:
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=ConstantPolicy(0.03),
        prior_policy=ConstantPolicy(0.0),
        dynamics=FakeDynamics(),
        evaluator=BatchScoresActionValue(),
        config=PlannerConfig(
            num_policy_samples=1,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
            policy_prior_abstain_margin=0.05,
        ),
    )

    result = planner.plan(_context(), seed=123)

    assert result.first_action_chunk.metadata["candidate_role"] == "policy_prior"
    assert np.all(result.first_action_chunk.actions == 0.0)


def test_planner_policy_prior_margin_can_be_overridden_per_call() -> None:
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=ConstantPolicy(0.03),
        prior_policy=ConstantPolicy(0.0),
        dynamics=FakeDynamics(),
        evaluator=BatchScoresActionValue(),
        config=PlannerConfig(
            num_policy_samples=1,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
            policy_prior_abstain_margin=0.0,
        ),
    )

    result = planner.plan(_context(), seed=123, policy_prior_abstain_margin=0.05)

    assert result.first_action_chunk.metadata["candidate_role"] == "policy_prior"
    assert np.all(result.first_action_chunk.actions == 0.0)


def test_full_imagined_planner_truncates_failed_future_branch() -> None:
    runner = _load_planning_runner()

    class FailsOnSecondChunkForFirstCandidate:
        def __init__(self) -> None:
            self.candidate_index = -1
            self.call_index = 0

        def reset(self) -> None:
            self.candidate_index += 1
            self.call_index = 0

        def sample_action_chunks(
            self,
            context: PlanningContext,
            *,
            num_samples: int,
            chunk_action_steps: int,
            temperature: float,
            seed: int | None = None,
        ):
            del context, num_samples, temperature, seed
            if self.candidate_index == 0 and self.call_index == 1:
                raise AssertionError("bad imagined-context token stream")
            value = float(self.candidate_index)
            self.call_index += 1
            return [ActionChunk(np.full((chunk_action_steps, 1), value, dtype=np.float32))]

    result = runner._plan_full_imagined_trajectories(
        policy=FailsOnSecondChunkForFirstCandidate(),
        dynamics=FakeDynamics(),
        evaluator=BatchScoresActionCount(),
        context_propagator=None,
        context=_context(),
        goal=None,
        config=PlannerConfig(
            num_policy_samples=2,
            beam_width=2,
            chunk_action_steps=4,
            min_plan_action_steps=8,
            max_plan_chunks=2,
            execute_action_steps=4,
        ),
        seed=123,
    )

    assert len(result.candidates) == 2
    truncated = next(candidate for candidate in result.candidates if candidate.metadata.get("truncated"))
    assert truncated.action_count == 4
    assert truncated.metadata["truncation_chunk_index"] == 1
    assert "policy_error" in truncated.metadata["truncation_reason"]
    assert result.selected.action_count == 8
    assert result.first_action_chunk.actions.shape == (4, 1)


def test_full_imagined_short_horizon_rewarms_after_cropped_fdm_prediction() -> None:
    runner = _load_planning_runner()
    policy = MetadataRecordingPolicy()
    dynamics = ShortHorizonPaddedDynamics(internal_steps=4)
    rewarm_count = 0

    def _rewarm(context: PlanningContext) -> PlanningContext:
        nonlocal rewarm_count
        rewarm_count += 1
        return context.with_prediction(
            predicted_video=context.predicted_video,
            metadata_updates={"rewarm_count": rewarm_count},
        )

    result = runner._plan_full_imagined_trajectories(
        policy=policy,
        dynamics=dynamics,
        evaluator=BatchScoresActionCount(),
        context_propagator=IncrementStatePropagator(),
        context_rewarmer=_rewarm,
        context=PlanningContext(
            views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
            state=np.asarray([0.0], dtype=np.float32),
            task_text="task",
        ),
        goal=None,
        config=PlannerConfig(
            num_policy_samples=1,
            beam_width=1,
            chunk_action_steps=3,
            min_plan_action_steps=6,
            max_plan_chunks=2,
            execute_action_steps=3,
        ),
        seed=123,
    )

    assert result.metadata["planned_chunks"] == 2
    assert result.metadata["planned_action_steps"] == 6
    assert dynamics.seen_action_steps == [3, 3]
    assert rewarm_count == 1
    assert policy.seen_rewarm_flags == [False, True]
    assert result.selected.actions.shape == (6, 1)
    assert [video.shape[0] for video in result.selected.predicted_videos] == [3, 3]


def test_full_imagined_planner_can_include_policy_prior_candidate() -> None:
    runner = _load_planning_runner()
    policy = MetadataRecordingPolicy()

    result = runner._plan_full_imagined_trajectories(
        policy=policy,
        dynamics=FakeDynamics(),
        evaluator=BatchScoresActionCount(),
        context_propagator=None,
        context=_context(),
        goal=None,
        config=PlannerConfig(
            num_policy_samples=1,
            beam_width=2,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
            policy_prior_temperature=0.0,
            policy_temperature=0.8,
        ),
        seed=123,
    )

    assert len(result.candidates) == 2
    roles = {candidate.action_chunks[0].metadata["candidate_role"] for candidate in result.candidates}
    assert roles == {"policy_prior", "full_imagined_policy_sample"}
    assert policy.seen_temperatures == pytest.approx([0.0, 0.8])


def test_full_imagined_prior_only_one_chunk_preserves_live_policy_state() -> None:
    runner = _load_planning_runner()

    class ResetSensitivePolicy:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def sample_action_chunks(
            self,
            context: PlanningContext,
            *,
            num_samples: int,
            chunk_action_steps: int,
            temperature: float,
            seed: int | None = None,
        ):
            del context, num_samples, temperature, seed
            value = 1.0 if self.reset_count == 0 else 99.0
            return [ActionChunk(np.full((chunk_action_steps, 1), value, dtype=np.float32))]

    policy = ResetSensitivePolicy()

    result = runner._plan_full_imagined_trajectories(
        policy=policy,
        dynamics=FakeDynamics(),
        evaluator=BatchScoresActionCount(),
        context_propagator=None,
        context=_context(),
        goal=None,
        config=PlannerConfig(
            num_policy_samples=0,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
            policy_prior_temperature=0.0,
        ),
        seed=123,
    )

    assert policy.reset_count == 0
    assert result.first_action_chunk.metadata["candidate_role"] == "policy_prior"
    np.testing.assert_allclose(result.first_action_chunk.actions, 1.0)


def test_full_imagined_one_chunk_skips_unused_context_propagation() -> None:
    runner = _load_planning_runner()
    propagator = CountingPropagator()

    result = runner._plan_full_imagined_trajectories(
        policy=ConstantPolicy(1.0),
        dynamics=FakeDynamics(),
        evaluator=BatchScoresActionCount(),
        context_propagator=propagator,
        context=_context(),
        goal=None,
        config=PlannerConfig(
            num_policy_samples=0,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
            include_policy_prior_candidate=True,
            policy_prior_temperature=0.0,
        ),
        seed=123,
    )

    assert propagator.calls == 0
    assert result.metadata["planned_chunks"] == 1
    assert result.first_action_chunk.metadata["candidate_role"] == "policy_prior"


def test_full_imagined_can_use_fixed_action_trace_candidate() -> None:
    runner = _load_planning_runner()
    fixed_trace = np.arange(20, dtype=np.float32).reshape(10, 2)

    result = runner._plan_full_imagined_trajectories(
        policy=ConstantPolicy(1.0),
        dynamics=FakeDynamics(),
        evaluator=BatchScoresActionValue(),
        context_propagator=None,
        context=_context(),
        goal=None,
        config=PlannerConfig(
            num_policy_samples=1,
            beam_width=1,
            chunk_action_steps=3,
            min_plan_action_steps=3,
            max_plan_chunks=1,
            execute_action_steps=3,
        ),
        seed=123,
        fixed_candidate_actions=fixed_trace,
        fixed_candidate_start_step=4,
    )

    assert len(result.candidates) == 2
    assert result.first_action_chunk.metadata["candidate_role"] == "fixed_action_trace"
    assert result.first_action_chunk.metadata["fixed_trace_start_step"] == 4
    np.testing.assert_allclose(result.first_action_chunk.actions, fixed_trace[4:7])


def test_full_imagined_speculative_policy_sampling_restores_live_state() -> None:
    runner = _load_planning_runner()

    class StatefulPolicy:
        def __init__(self) -> None:
            self.live_state = ["live"]
            self.reset_count = 0
            self.restore_count = 0

        def snapshot_state(self):
            return list(self.live_state)

        def restore_state(self, snapshot):
            self.live_state = list(snapshot)
            self.restore_count += 1

        def reset(self) -> None:
            self.reset_count += 1
            self.live_state = ["candidate"]

        def sample_action_chunks(
            self,
            context: PlanningContext,
            *,
            num_samples: int,
            chunk_action_steps: int,
            temperature: float,
            seed: int | None = None,
        ):
            del context, num_samples, temperature, seed
            assert self.live_state == ["candidate"]
            return [ActionChunk(np.full((chunk_action_steps, 1), self.reset_count, dtype=np.float32))]

    policy = StatefulPolicy()

    result = runner._plan_full_imagined_trajectories(
        policy=policy,
        dynamics=FakeDynamics(),
        evaluator=BatchScoresActionValue(),
        context_propagator=None,
        context=_context(),
        goal=None,
        config=PlannerConfig(
            num_policy_samples=2,
            beam_width=1,
            chunk_action_steps=4,
            min_plan_action_steps=4,
            max_plan_chunks=1,
            execute_action_steps=4,
        ),
        seed=123,
    )

    assert policy.reset_count == 2
    assert policy.restore_count == 2
    assert policy.live_state == ["live"]
    np.testing.assert_allclose(result.first_action_chunk.actions, 2.0)


def test_action_trace_helpers_roundtrip(tmp_path: Path) -> None:
    runner = _load_planning_runner()
    actions = [np.asarray([1.0, 2.0], dtype=np.float32), np.asarray([3.0, 4.0], dtype=np.float32)]

    path = runner._save_executed_action_trace(
        output_dir=tmp_path,
        rollout_index=0,
        actions=actions,
        task_id=8,
        episode_idx=1,
    )

    assert path is not None
    loaded = runner._load_optional_action_trace(str(path))
    np.testing.assert_allclose(loaded, np.asarray(actions, dtype=np.float32))


def test_full_imagined_failure_fallback_uses_noop_if_policy_prior_fails() -> None:
    runner = _load_planning_runner()
    args = SimpleNamespace(
        full_imagined_failure_fallback="policy_prior",
        chunk_action_steps=4,
        execute_action_steps=3,
        policy_prior_temperature=0.0,
        action_dim=7,
    )

    action_chunk, record = runner._full_imagined_planning_failure_fallback(
        args,
        policy=FailingPlanningPolicy(),
        context=_context(),
        seed=123,
        reason="all branches failed",
    )

    assert record["mode"] == "noop"
    assert record["action_steps"] == 3
    assert action_chunk.actions.shape == (3, 7)
    assert np.all(action_chunk.actions[:, 6] == -1.0)
    assert action_chunk.metadata["candidate_role"] == "noop_fallback"
    assert "policy_prior_fallback_failed" in record["reason"]


def test_full_imagined_receding_validator_allows_shorter_branch_horizon() -> None:
    runner = _load_planning_runner()
    args = SimpleNamespace(
        planner="full_imagined_vlm_receding",
        min_plan_action_steps=30,
        max_env_steps=300,
        max_plan_chunks=3,
        chunk_action_steps=10,
        execute_action_steps=10,
    )

    runner._validate_planner_runtime_contract(args, FakeDynamics())


def test_open_loop_imagined_validator_still_requires_full_horizon() -> None:
    runner = _load_planning_runner()
    args = SimpleNamespace(
        planner="imagined_open_loop",
        min_plan_action_steps=30,
        max_env_steps=300,
        max_plan_chunks=3,
        chunk_action_steps=10,
        execute_action_steps=10,
    )

    with pytest.raises(ValueError, match="covers the requested horizon"):
        runner._validate_planner_runtime_contract(args, FakeDynamics())


def test_task8_offline_capture_manifest_writes_snapshot_sidecar(tmp_path: Path) -> None:
    runner = _load_task8_offline_reranking_runner()
    capture = runner.CapturedStageState(
        episode_idx=4,
        target_stage="pick_up_pot_1",
        env_step=230,
        simulator_state=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        observation=None,
        status=SimpleNamespace(stage_name="pick_up_pot_1"),
        metrics={"distance": 0.4},
        capture_reason="no_progress_stall",
        stage_elapsed_steps=30,
        stage_no_progress_steps=30,
        stage_best_progress=0.45,
    )

    row = runner._write_capture_snapshot(tmp_path, capture, index=2)
    manifest_path = tmp_path / "captured_states.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = runner._load_capture_manifest(manifest_path)

    assert loaded == [row]
    assert row["snapshot_path"] == "captured_states/000002_ep04_pick_up_pot_1_step0230.npz"
    with np.load(tmp_path / row["snapshot_path"]) as data:
        assert np.array_equal(data["simulator_state"], np.asarray([1.0, 2.0, 3.0], dtype=np.float64))

    malformed_path = tmp_path / "malformed.jsonl"
    malformed_path.write_text(json.dumps({"episode_idx": 4}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        runner._load_capture_manifest(malformed_path)


def test_task8_reference_subtask_goal_uses_putdown_release_frame() -> None:
    runner = _load_task8_offline_reranking_runner()
    gripper_actions = np.asarray(
        [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
        dtype=np.float32,
    )

    release_edges = runner._gripper_release_edges(gripper_actions)

    assert release_edges == [4, 8]
    putdown_order = ["put_down_pot_2", "put_down_pot_1"]
    assert runner._reference_putdown_frame_from_releases(
        target_stage="put_down_pot_2",
        putdown_stages=putdown_order,
        release_edges=release_edges,
        action_count=len(gripper_actions),
        settle_frames=1,
    ) == 5
    assert runner._reference_putdown_frame_from_releases(
        target_stage="put_down_pot_1",
        putdown_stages=putdown_order,
        release_edges=release_edges,
        action_count=len(gripper_actions),
        settle_frames=8,
    ) == len(gripper_actions) - 1
    with pytest.raises(RuntimeError, match="not contain enough gripper releases"):
        runner._reference_putdown_frame_from_releases(
            target_stage="put_down_pot_1",
            putdown_stages=putdown_order,
            release_edges=release_edges[:1],
            action_count=len(gripper_actions),
            settle_frames=0,
        )


def test_task8_runner_args_preserve_configured_baseline_action_selection(tmp_path: Path) -> None:
    runner = _load_task8_intervention_runner()
    args = SimpleNamespace(
        policy="pi0",
        benchmark="libero_10",
        task_id=8,
        episode_idx=0,
        max_env_steps=520,
        startup_noop_steps=10,
        seed=0,
        num_policy_samples=7,
        beam_width=8,
        chunk_action_steps=10,
        execute_action_steps=10,
        planner_temperature=0.6,
        action_dim=7,
        action_clip=None,
        gemini_api_key_env="GEMINI_API_KEY",
        gemini_model="gemini-test",
        gemini_demo_video=None,
        gemini_candidate_video_fps=4.0,
        gemini_timeout_seconds=120.0,
        gemini_max_candidates=8,
        gemini_delete_uploaded_files=False,
        gemini_prior_hint_mode="conservative",
        pi0fast_model="lerobot/pi0fast-libero",
        pi0_model="lerobot/pi0_libero_finetuned_v044",
        policy_device="cuda:0",
        policy_compile_model=False,
        baseline_action_selection_mode="predict_chunk",
        dataset_root="/tmp/libero",
        goal_image_camera_name="observation.images.agentview_rgb",
        goal_image_reference_policy="next_episode",
        goal_image_reference_episode_offset=1,
        openwam_config="configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml",
        openwam_checkpoint="/tmp/checkpoint",
        openwam_fdm_mode="forced_action_joint_fdm",
        openwam_predicted_view_key="agentview_image",
        runtime_device="cuda:1",
        decode_device="cuda:1",
        runtime_dtype="bfloat16",
        video_num_inference_steps=8,
        action_num_inference_steps=8,
        camera_height=360,
        camera_width=360,
        video_fps=15.0,
    )

    runner_args = runner._build_runner_args(args, output_dir=tmp_path)

    assert runner_args.pi0fast_action_selection_mode == "predict_chunk"
    assert runner_args.policy_compile_model is False
    assert runner_args.goal_image_reference_policy == "next_episode"
    assert runner_args.goal_image_reference_episode_offset == 1


def test_planner_stops_when_minimum_action_horizon_is_reached() -> None:
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=FakePolicy(),
        dynamics=FakeDynamics(),
        evaluator=PreferLargestFinalVideo(),
        config=PlannerConfig(
            num_policy_samples=2,
            beam_width=1,
            chunk_action_steps=5,
            min_plan_action_steps=5,
            max_plan_chunks=4,
            execute_action_steps=5,
        ),
    )

    result = planner.plan(_context())

    assert result.metadata["planned_chunks"] == 1
    assert result.metadata["planned_action_steps"] == 5


def test_planner_context_propagator_updates_next_policy_context() -> None:
    policy = StateRecordingPolicy()
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=policy,
        dynamics=FakeDynamics(),
        context_propagator=IncrementStatePropagator(),
        config=PlannerConfig(
            num_policy_samples=1,
            beam_width=1,
            chunk_action_steps=1,
            min_plan_action_steps=3,
            max_plan_chunks=3,
            execute_action_steps=1,
        ),
    )
    context = PlanningContext(
        views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.asarray([0.0], dtype=np.float32),
        task_text="task",
    )

    result = planner.plan(context)

    assert policy.seen_states == pytest.approx([0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.selected.context.state, np.asarray([3.0], dtype=np.float32))
    assert result.selected.context.metadata["propagated_state"] == pytest.approx(3.0)


def test_planning_runner_reports_action_stats_and_candidate_diversity(tmp_path: Path) -> None:
    runner = _load_planning_runner()
    actions_a = np.zeros((3, 2), dtype=np.float32)
    actions_b = np.ones((3, 2), dtype=np.float32)
    candidates = (
        CandidateTrajectory(context=_context(), action_chunks=(ActionChunk(actions_a),), score=0.1),
        CandidateTrajectory(context=_context(), action_chunks=(ActionChunk(actions_b),), score=0.2),
    )
    result = SimpleNamespace(
        candidates=candidates,
        first_action_chunk=ActionChunk(actions_b),
    )

    selected_stats = runner._action_array_stats(actions_b)
    diversity = runner._candidate_first_chunk_action_diversity(candidates)
    dump_path = runner._dump_planner_candidate_actions(
        output_dir=tmp_path,
        rollout_index=0,
        replan_index=1,
        result=result,
    )

    assert selected_stats["steps"] == 3
    assert selected_stats["dim"] == 2
    assert selected_stats["mean_l2"] == pytest.approx(np.sqrt(2.0))
    assert selected_stats["temporal_delta_mean_l2"] == 0.0
    assert diversity["candidate_count"] == 2
    assert diversity["action_shape"] == [3, 2]
    assert diversity["mean_pairwise_step_l2"] == pytest.approx(np.sqrt(2.0))
    assert dump_path is not None
    with np.load(dump_path) as data:
        assert data["first_action_chunks"].shape == (2, 3, 2)
        assert data["scores"].tolist() == pytest.approx([0.1, 0.2])


def test_gemini_vlm_reranker_saves_tiled_video_and_uses_structured_scores(tmp_path: Path) -> None:
    candidates = (
        CandidateTrajectory(
            context=_context(),
            action_chunks=(ActionChunk(np.zeros((2, 2), dtype=np.float32), metadata={"candidate_role": "policy_prior"}),),
            predicted_videos=(np.zeros((3, 8, 8, 3), dtype=np.uint8),),
        ),
        CandidateTrajectory(
            context=_context(),
            action_chunks=(ActionChunk(np.ones((2, 2), dtype=np.float32)),),
            predicted_videos=(np.full((3, 8, 8, 3), 128, dtype=np.uint8),),
        ),
    )
    fake_client = FakeGeminiClient()
    evaluator = GeminiVlmCandidateRerankEvaluator(
        GeminiVlmRerankConfig(output_dir=tmp_path, model="gemini-test", video_fps=2.0),
        client=fake_client,
    )

    scores = evaluator.score_candidates(
        candidates,
        goal={
            "task_text": "put the object in the basket",
            "current": np.zeros((8, 8, 3), dtype=np.uint8),
            "target": np.ones((8, 8, 3), dtype=np.uint8) * 255,
        },
    )

    assert scores == pytest.approx([0.2, 0.9])
    assert any(path.name == "candidate_futures_tiled.mp4" for path in fake_client.uploaded_paths)
    assert any(path.name == "current_observation.png" for path in fake_client.uploaded_paths)
    assert any(path.name == "target_reference.png" for path in fake_client.uploaded_paths)
    assert fake_client.generated[0]["model"] == "gemini-test"
    assert "response_schema" in fake_client.generated[0]
    manifest_path = tmp_path / "vlm_replan_0000" / "gemini_rerank_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_debug"]["A"]["action_steps"] == 2
    assert manifest["candidate_debug"]["A"]["candidate_role"] == "policy_prior"
    assert manifest["candidate_debug"]["B"]["predicted_video_frames"] == 3
    assert "motion that only changes camera/background appearance" in manifest["prompt"]


def test_gemini_scores_are_clamped_and_ranking_fallback_is_unit_scaled() -> None:
    labels = ("A", "B", "C")

    assert _scores_from_gemini_result({"scores": {"A": 6.8, "B": -1.0, "C": "bad"}}, labels) == pytest.approx([
        0.68,
        0.0,
        0.0,
    ])
    assert _scores_from_gemini_result({"scores": {"A": 4.2, "B": 6.8}}, ("A", "B")) == pytest.approx(
        [0.42, 0.68]
    )
    assert _scores_from_gemini_result({"ranking": ["C", "A", "B"]}, labels) == pytest.approx(
        [2.0 / 3.0, 1.0 / 3.0, 1.0]
    )


def test_gemini_prompt_decomposes_pick_place_progress() -> None:
    prompt = build_gemini_candidate_prompt(
        labels=("A", "B"),
        goal={"task_text": "put both the alphabet soup and the tomato sauce in the basket"},
    )

    assert "next short control chunk" in prompt
    assert "not judging final task success" in prompt
    assert "remaining object" in prompt
    assert "basket" in prompt
    assert "Do not reward large arm motion" in prompt


def test_gemini_prompt_marks_baseline_policy_candidate() -> None:
    prompt = build_gemini_candidate_prompt(
        labels=("A", "B"),
        goal={"task_text": "put both moka pots on the stove"},
        candidate_roles={"A": "policy_prior", "B": "policy_sample"},
    )

    assert "Candidate A is the deterministic baseline-policy proposal" in prompt
    assert "prefer it unless another candidate clearly improves" in prompt


def test_gemini_prompt_does_not_privilege_baseline_after_stall() -> None:
    prompt = build_gemini_candidate_prompt(
        labels=("A", "B"),
        goal={"task_text": "put both moka pots on the stove", "intervention_was_stall": True},
        candidate_roles={"A": "policy_prior", "B": "policy_sample"},
    )

    assert "Candidate A is the deterministic baseline-policy proposal" in prompt
    assert "has just stalled" in prompt
    assert "Do not prefer it by default" in prompt


def test_gemini_prompt_can_hide_candidate_roles_for_offline_selector() -> None:
    prompt = build_gemini_candidate_prompt(
        labels=("A", "B"),
        goal={"task_text": "put both moka pots on the stove"},
        candidate_roles={"A": "policy_prior", "B": "policy_sample"},
        policy_prior_hint_mode="blind",
    )

    assert "Candidate A is the deterministic baseline-policy proposal" not in prompt
    assert "Candidate roles are hidden" in prompt
    assert "Rank only the visible imagined futures" in prompt


def test_gemini_prompt_supports_full_trajectory_ranking() -> None:
    prompt = build_gemini_candidate_prompt(
        labels=("A", "B"),
        goal={"task_text": "put both moka pots on the stove", "planning_scope": "full_trajectory"},
        policy_prior_hint_mode="blind",
    )

    assert "complete imagined futures" in prompt
    assert "task completion by the end of each video" in prompt
    assert "execute only the first control chunk" in prompt
    assert "next short control chunk" not in prompt


def test_libero_progress_hints_handle_pickup_tasks() -> None:
    hints = infer_libero_progress_hints("pick up the black bowl")

    assert any("black bowl" in hint for hint in hints)
    assert any("lift" in hint for hint in hints)


def test_tiled_candidate_video_pads_shorter_future_with_last_frame() -> None:
    candidates = (
        CandidateTrajectory(
            context=_context(),
            predicted_videos=(np.zeros((2, 4, 4, 3), dtype=np.uint8),),
        ),
        CandidateTrajectory(
            context=_context(),
            predicted_videos=(np.ones((3, 4, 4, 3), dtype=np.uint8) * 50,),
        ),
    )

    tiled = build_tiled_candidate_video(candidates)

    assert tiled.shape == (3, 4, 8, 3)
    assert tiled.dtype == np.uint8


def test_tiled_candidate_video_flips_vertical_orientation_for_display() -> None:
    video = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    video[:, 0, :, :] = 10
    video[:, -1, :, :] = 200
    candidates = (
        CandidateTrajectory(
            context=_context(),
            predicted_videos=(video,),
        ),
    )

    tiled = build_tiled_candidate_video(candidates, border_px=0)
    unflipped = build_tiled_candidate_video(candidates, border_px=0, flip_vertical_for_display=False)

    assert int(tiled[0, 0, 0, 0]) == 200
    assert int(tiled[0, -1, 0, 0]) == 10
    assert int(unflipped[0, 0, 0, 0]) == 10
    assert int(unflipped[0, -1, 0, 0]) == 200


def test_tiled_candidate_video_concatenates_predicted_chunks() -> None:
    candidates = (
        CandidateTrajectory(
            context=_context(),
            predicted_videos=(
                np.zeros((2, 16, 16, 3), dtype=np.uint8),
                np.ones((2, 16, 16, 3), dtype=np.uint8) * 40,
            ),
        ),
        CandidateTrajectory(
            context=_context(),
            predicted_videos=(
                np.ones((2, 16, 16, 3), dtype=np.uint8) * 80,
                np.ones((2, 16, 16, 3), dtype=np.uint8) * 120,
            ),
        ),
    )

    tiled = build_tiled_candidate_video(candidates)

    assert tiled.shape == (4, 16, 32, 3)
    assert int(tiled[0, 8, 8].mean()) == 0
    assert int(tiled[3, 8, 8].mean()) == 40
    assert int(tiled[0, 8, 24].mean()) == 80
    assert int(tiled[3, 8, 24].mean()) == 120


def test_pi0fast_raw_batch_uses_checkpoint_key_mapping_and_pads_state() -> None:
    context = PlanningContext(
        views={
            "agentview_image": np.full((8, 6, 3), 255, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 6, 3), dtype=np.uint8),
        },
        state=np.arange(7, dtype=np.float32),
        task_text="put the bowl in the basket",
    )

    batch = build_pi0fast_raw_batch(
        context,
        Pi0FastBatchMapping(
            image_keys={
                "observation.images.image": "agentview_image",
                "observation.images.image2": "robot0_eye_in_hand_image",
            },
            device="cpu",
            state_dim=32,
        ),
    )

    assert batch["observation.images.image"].shape == (3, 8, 6)
    assert batch["observation.images.image"].dtype == torch.float32
    assert batch["observation.state"].shape == (32,)
    assert torch.equal(batch["observation.state"][:7], torch.arange(7, dtype=torch.float32))
    assert torch.count_nonzero(batch["observation.state"][7:]) == 0
    assert batch["task"] == "put the bowl in the basket"


def test_pi0fast_default_state_dim_matches_libero_preprocessor_raw_state() -> None:
    context = PlanningContext(
        views={"agentview_image": np.zeros((8, 6, 3), dtype=np.uint8)},
        state=np.arange(7, dtype=np.float32),
        task_text="task",
    )

    batch = build_pi0fast_raw_batch(
        context,
        Pi0FastBatchMapping(image_keys={"observation.images.image": "agentview_image"}, device="cpu"),
    )

    assert batch["observation.state"].shape == (8,)
    assert torch.equal(batch["observation.state"][:7], torch.arange(7, dtype=torch.float32))
    assert batch["observation.state"][7].item() == 0.0


def test_pi0fast_raw_batch_applies_lerobot_libero_image_flip_by_default() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    context = PlanningContext(
        views={"agentview_image": image},
        state=np.zeros(8, dtype=np.float32),
        task_text="task",
    )

    batch = build_pi0fast_raw_batch(
        context,
        Pi0FastBatchMapping(image_keys={"observation.images.image": "agentview_image"}, device="cpu"),
    )

    expected = np.moveaxis(image[::-1, ::-1].astype(np.float32) / 255.0, -1, 0)
    np.testing.assert_allclose(batch["observation.images.image"].numpy(), expected)


def test_pi0fast_raw_batch_can_disable_image_flip() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    context = PlanningContext(
        views={"agentview_image": image},
        state=np.zeros(8, dtype=np.float32),
        task_text="task",
    )

    batch = build_pi0fast_raw_batch(
        context,
        Pi0FastBatchMapping(
            image_keys={"observation.images.image": "agentview_image"},
            image_transform="none",
            device="cpu",
        ),
    )

    expected = np.moveaxis(image.astype(np.float32) / 255.0, -1, 0)
    np.testing.assert_allclose(batch["observation.images.image"].numpy(), expected)


def test_pi0fast_sampler_applies_processors_before_clipping() -> None:
    policy = FakePi0Policy()
    sampler = Pi0FastPolicySampler(
        policy=policy,
        batch_mapping=Pi0FastBatchMapping(image_keys={"observation.images.image": "agentview"}, device="cpu"),
        preprocessor=MarkingPreprocessor(),
        postprocessor=ScalingPostprocessor(),
        action_clip=1.5,
        action_selection_mode="predict_chunk",
    )

    chunks = sampler.sample_action_chunks(
        _context(),
        num_samples=1,
        chunk_action_steps=4,
        temperature=0.7,
        seed=3,
    )

    assert policy.config.temperature == 0.7
    assert policy.seen_batches[0]["preprocessed"] is True
    assert chunks[0].actions.shape == (4, 2)
    assert np.all(chunks[0].actions == 1.5)


def test_pi0fast_sampler_repeats_last_action_for_longer_planner_horizon() -> None:
    policy = FakePi0Policy()
    sampler = Pi0FastPolicySampler(
        policy=policy,
        batch_mapping=Pi0FastBatchMapping(image_keys={"observation.images.image": "agentview"}, device="cpu"),
        action_clip=None,
        short_horizon_strategy="repeat_last",
        action_selection_mode="predict_chunk",
    )

    chunks = sampler.sample_action_chunks(
        _context(),
        num_samples=1,
        chunk_action_steps=8,
        temperature=0.0,
    )

    assert chunks[0].actions.shape == (8, 2)
    assert chunks[0].metadata["native_action_steps"] == 6
    assert chunks[0].metadata["requested_action_steps"] == 8
    np.testing.assert_allclose(chunks[0].actions[6:], np.repeat(chunks[0].actions[5:6], 2, axis=0))


def test_pi0_family_sampler_records_native_policy_family() -> None:
    sampler = Pi0FastPolicySampler(
        policy=FakePi0Policy(),
        policy_family="pi0",
        batch_mapping=Pi0FastBatchMapping(image_keys={"observation.images.image": "agentview"}, device="cpu"),
        action_clip=None,
        action_selection_mode="predict_chunk",
    )

    chunks = sampler.sample_action_chunks(
        _context(),
        num_samples=1,
        chunk_action_steps=4,
        temperature=0.0,
    )

    assert chunks[0].metadata["policy"] == "pi0"


def test_pi0_family_sampler_can_override_compile_model(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(compile_model=True)
    seen: dict[str, object] = {}

    class FakePolicyClass:
        @classmethod
        def from_pretrained(cls, model_path, *, config=None, **kwargs):
            seen["model_path"] = model_path
            seen["config"] = config
            seen["kwargs"] = kwargs
            return FakePi0Policy()

    monkeypatch.setattr(pi0fast_module, "_load_lerobot_policy_class", lambda policy_family: FakePolicyClass)
    monkeypatch.setattr(pi0fast_module, "_load_lerobot_policy_config", lambda model_path, **kwargs: config)

    sampler = Pi0FastPolicySampler.from_pretrained(
        "fake/pi0",
        batch_mapping=Pi0FastBatchMapping(image_keys={"observation.images.image": "agentview"}, device="cpu"),
        policy_family="pi0",
        compile_model=False,
        use_saved_processors=False,
    )

    assert sampler.policy_family == "pi0"
    assert seen["model_path"] == "fake/pi0"
    assert seen["config"] is config
    assert config.compile_model is False


def test_pi0fast_sampler_select_action_matches_native_horizon() -> None:
    policy = FakePi0Policy()
    sampler = Pi0FastPolicySampler(
        policy=policy,
        batch_mapping=Pi0FastBatchMapping(image_keys={"observation.images.image": "agentview"}, device="cpu"),
        postprocessor=ScalingPostprocessor(),
        action_clip=None,
        action_selection_mode="select_action",
    )

    chunks = sampler.sample_action_chunks(
        _context(),
        num_samples=1,
        chunk_action_steps=3,
        temperature=0.0,
    )

    assert chunks[0].actions.shape == (3, 2)
    np.testing.assert_allclose(chunks[0].actions, np.asarray([[2, 2], [4, 4], [6, 6]], dtype=np.float32))
    assert chunks[0].metadata["action_selection_mode"] == "select_action"


def test_pi0fast_sampler_select_action_rejects_non_native_horizon() -> None:
    sampler = Pi0FastPolicySampler(
        policy=FakePi0Policy(),
        batch_mapping=Pi0FastBatchMapping(image_keys={"observation.images.image": "agentview"}, device="cpu"),
        action_selection_mode="select_action",
    )

    with pytest.raises(ValueError, match="native action horizon"):
        sampler.sample_action_chunks(
            _context(),
            num_samples=1,
            chunk_action_steps=4,
            temperature=0.0,
        )


def test_planning_context_derives_lerobot_policy_state_from_raw_libero_fields() -> None:
    runner = _load_planning_runner()
    observation = SimulatorObservation(
        views={"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.arange(7, dtype=np.float32),
        task_text="task",
        raw={
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.4, -0.4], dtype=np.float32),
        },
    )

    context = runner._planning_context_from_observation(observation)

    proprio = context.metadata["openwam_proprio_state"]
    assert proprio.shape == (8,)
    np.testing.assert_allclose(context.state, proprio)
    np.testing.assert_allclose(proprio[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(proprio[3:6], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(proprio[6:], [0.4, -0.4])
    np.testing.assert_allclose(context.metadata["sim_proprio_step_history"][0], proprio)


def test_planning_context_falls_back_to_simulator_state_without_raw_libero_fields() -> None:
    runner = _load_planning_runner()
    observation = SimulatorObservation(
        views={"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.arange(7, dtype=np.float32),
        task_text="task",
        raw={},
    )

    context = runner._planning_context_from_observation(observation)

    np.testing.assert_allclose(context.state, np.arange(7, dtype=np.float32))
    np.testing.assert_allclose(context.metadata["openwam_proprio_state"], np.arange(7, dtype=np.float32))


def test_planning_context_can_capture_libero_simulator_snapshot() -> None:
    runner = _load_planning_runner()

    class FakeEnv:
        def get_sim_state(self):
            return np.asarray([1.0, 2.0], dtype=np.float64)

    fake_adapter = SimpleNamespace(_env=FakeEnv())
    observation = SimulatorObservation(
        views={"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.arange(7, dtype=np.float32),
        task_text="task",
        raw={},
    )

    context = runner._planning_context_from_observation(
        observation,
        adapter=fake_adapter,
        task_id=3,
        episode_idx=4,
        seed=5,
    )

    np.testing.assert_allclose(context.metadata["libero_simulator_state"], np.asarray([1.0, 2.0]))
    assert context.metadata["libero_task_id"] == 3
    assert context.metadata["libero_episode_idx"] == 4
    assert context.metadata["libero_seed"] == 5


def test_libero_simulator_propagator_updates_proprio_without_rendered_views() -> None:
    runner = _load_planning_runner()

    def raw_for_state(value: float) -> dict[str, np.ndarray]:
        return {
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eef_pos": np.asarray([value, 0.0, 0.0], dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.1, -0.1], dtype=np.float32),
        }

    class FakeEnv:
        def __init__(self) -> None:
            self.value = 0.0

        def get_sim_state(self):
            return np.asarray([self.value], dtype=np.float64)

        def regenerate_obs_from_state(self, state):
            self.value = float(np.asarray(state).reshape(-1)[0])
            return raw_for_state(self.value)

    class FakeAdapter:
        def __init__(self) -> None:
            self._env = FakeEnv()
            self._last_obs = None
            self.closed = False

        def reset(self, spec):
            self.last_spec = spec
            raw = self._env.regenerate_obs_from_state(np.asarray([0.0], dtype=np.float64))
            self._last_obs = raw
            return self._normalize_observation(raw)

        def step(self, action):
            self._env.value += float(np.asarray(action).reshape(-1)[0])
            raw = raw_for_state(self._env.value)
            self._last_obs = raw
            return SimpleNamespace(observation=self._normalize_observation(raw))

        def close(self):
            self.closed = True

        def _normalize_observation(self, raw, *, init_state_index=None):
            del init_state_index
            return SimulatorObservation(
                views={"agentview_image": np.asarray(raw["agentview_image"], dtype=np.uint8)},
                state=np.asarray([999.0], dtype=np.float32),
                task_text="task",
                raw=raw,
            )

    context = PlanningContext(
        views={"agentview_image": np.full((4, 4, 3), 7, dtype=np.uint8)},
        state=np.asarray([123.0], dtype=np.float32),
        task_text="task",
        predicted_video=np.full((2, 4, 4, 3), 9, dtype=np.uint8),
        metadata={
            "libero_simulator_state": np.asarray([2.0], dtype=np.float64),
            "libero_task_id": 1,
            "libero_episode_idx": 3,
            "libero_seed": 11,
            "openwam_proprio_state": np.zeros(8, dtype=np.float32),
            "sim_proprio_step_history": [np.zeros(8, dtype=np.float32)],
        },
    )
    propagator = runner._LiberoSimulatorProprioPropagator(
        adapter_factory=FakeAdapter,
        action_dim=7,
    )

    output = propagator.propagate(context, ActionChunk(np.ones((2, 7), dtype=np.float32)))

    np.testing.assert_allclose(output.metadata["libero_simulator_state"], np.asarray([4.0]))
    np.testing.assert_allclose(
        output.state[:3],
        np.asarray([4.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        output.metadata["openwam_proprio_state"][:3],
        np.asarray([4.0, 0.0, 0.0], dtype=np.float32),
    )
    assert len(output.metadata["sim_proprio_step_history"]) == 3
    assert output.metadata["sim_propagated_action_steps"] == 2
    assert np.all(output.views["agentview_image"] == 7)
    assert np.all(output.predicted_video == 9)


def test_libero_noop_action_matches_lerobot_reset_settling_convention() -> None:
    runner = _load_planning_runner()

    action = runner._libero_noop_action(action_dim=7)

    np.testing.assert_allclose(action, np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))


def test_rollout_schedule_preserves_legacy_and_supports_explicit_sweeps() -> None:
    runner = _load_planning_runner()
    legacy_args = SimpleNamespace(
        task_id=2,
        episode_idx=5,
        episodes=3,
        task_ids=None,
        episode_indices=None,
    )
    explicit_args = SimpleNamespace(
        task_id=0,
        episode_idx=0,
        episodes=1,
        task_ids="1,3",
        episode_indices="7,8",
    )

    assert runner._resolve_rollout_schedule(legacy_args) == [(2, 5), (2, 6), (2, 7)]
    assert runner._resolve_rollout_schedule(explicit_args) == [(1, 7), (3, 7), (1, 8), (3, 8)]


def test_replay_init_state_lookup_uses_task_local_episode_index(tmp_path: Path) -> None:
    runner = _load_planning_runner()
    replay_path = tmp_path / "replay_status.jsonl"
    rows = [
        {
            "upstream_task_id": 8,
            "task_local_episode_idx": 0,
            "dataset_episode_index": 150,
            "resolved_init_state_index": 15,
        },
        {
            "upstream_task_id": 8,
            "task_local_episode_idx": 1,
            "dataset_episode_index": 151,
            "primary_init_state_index": 22,
        },
    ]
    replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = SimpleNamespace(use_replay_init_state=True, replay_status_path=str(replay_path))

    first = runner._resolve_replay_init_state(args, task_id=8, episode_idx=0)
    second = runner._resolve_replay_init_state(args, task_id=8, episode_idx=1)

    assert first.init_state_index == 15
    assert first.dataset_episode_index == 150
    assert second.init_state_index == 22
    assert second.dataset_episode_index == 151


def test_lerobot_goal_provider_resolves_task_local_final_frame(tmp_path: Path) -> None:
    runner = _load_planning_runner()
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    agentview_dir = root / "videos" / "chunk-000" / "observation.images.agentview_rgb"
    wrist_dir = root / "videos" / "chunk-000" / "observation.images.eye_in_hand_rgb"
    agentview_dir.mkdir(parents=True)
    wrist_dir.mkdir(parents=True)
    episodes = [
        {"episode_index": 10, "tasks": ["task a"], "length": 2},
        {"episode_index": 11, "tasks": ["task b"], "length": 2},
        {"episode_index": 12, "tasks": ["task a"], "length": 2},
    ]
    with (root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in episodes:
            handle.write(json.dumps(record) + "\n")
    first = np.zeros((16, 16, 3), dtype=np.uint8)
    second = np.full((16, 16, 3), 200, dtype=np.uint8)
    wrist_second = np.full((16, 16, 3), 50, dtype=np.uint8)
    imageio.mimsave(agentview_dir / "episode_000012.mp4", [first, second], fps=15)
    imageio.mimsave(wrist_dir / "episode_000012.mp4", [first, wrist_second], fps=15)
    provider = runner._LeRobotFinalFrameGoalProvider(
        dataset_root=root,
        camera_name="observation.images.agentview_rgb,observation.images.eye_in_hand_rgb",
        frame_index=-1,
        output_dir=tmp_path,
    )

    goal = provider(task_id=0, episode_idx=1, task_text="task a", rollout_index=3)

    assert goal.shape == (16, 32, 3)
    assert int(goal[:, :16].mean()) > 190
    assert 40 < int(goal[:, 16:].mean()) < 60
    assert (tmp_path / "goal_rollout_003_task0_ep1.png").is_file()

    start_goal = provider(task_id=0, episode_idx=1, task_text="task a", rollout_index=3, frame_index=0)
    assert int(start_goal.mean()) == 0

    clamped_goal = provider(task_id=0, episode_idx=1, task_text="task a", rollout_index=3, frame_index=999)
    assert int(clamped_goal[:, :16].mean()) > 190

    next_ref_provider = runner._LeRobotFinalFrameGoalProvider(
        dataset_root=root,
        camera_name="observation.images.agentview_rgb,observation.images.eye_in_hand_rgb",
        frame_index=-1,
        output_dir=tmp_path,
        reference_policy="next_episode",
    )
    next_ref_goal = next_ref_provider(task_id=0, episode_idx=0, task_text="task a", rollout_index=4)
    assert int(next_ref_goal[:, :16].mean()) > 190
    assert (tmp_path / "goal_rollout_004_task0_ep0_refep1.png").is_file()


def test_current_canvas_from_observation_resizes_and_concatenates_views() -> None:
    runner = _load_planning_runner()
    observation = SimulatorObservation(
        views={
            "agentview_image": np.full((4, 4, 3), 10, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.full((2, 2, 3), 200, dtype=np.uint8),
        },
        state=None,
        task_text="task",
    )

    canvas = runner._current_canvas_from_observation(
        observation,
        target_shape=(8, 16, 3),
        view_keys=("agentview_image", "robot0_eye_in_hand_image"),
    )

    assert canvas.shape == (8, 16, 3)
    assert int(canvas[:, :8].mean()) == 10
    assert int(canvas[:, 8:].mean()) == 200


def test_goal_delta_alignment_prefers_goal_directed_change() -> None:
    evaluator = GoalDeltaAlignmentEvaluator(background_penalty_weight=0.0, change_threshold=1.0)
    current = np.zeros((4, 4, 3), dtype=np.uint8)
    target = np.zeros((4, 4, 3), dtype=np.uint8)
    target[:, :, 0] = 100
    toward = np.zeros((4, 4, 3), dtype=np.uint8)
    toward[:, :, 0] = 50
    away = np.zeros((4, 4, 3), dtype=np.uint8)
    away[:, :, 1] = 50
    toward_candidate = CandidateTrajectory(
        context=_context(),
        predicted_videos=(toward[None],),
    )
    away_candidate = CandidateTrajectory(
        context=_context(),
        predicted_videos=(away[None],),
    )
    goal = {"current": current, "target": target}

    assert evaluator.score(toward_candidate, goal=goal) > evaluator.score(away_candidate, goal=goal)


def test_openwam_fdm_prediction_updates_policy_view() -> None:
    rollout = FakeOpenWamRollout()
    context = PlanningContext(
        views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    predicted = np.full((2, 4, 4, 3), 123, dtype=np.uint8)
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: predicted,
        predicted_view_key="agentview",
    )

    output = fdm.predict(context, ActionChunk(np.zeros((4, 7), dtype=np.float32)))

    assert rollout.infer_grad_enabled is False
    assert output.next_context.metadata["openwam_fdm_session"] == "next-session"
    assert output.next_context.metadata["openwam_current_frame"] == 2
    assert np.all(output.next_context.views["agentview"] == 123)
    assert np.all(output.predicted_video == predicted)


def test_openwam_fdm_prediction_metadata_retains_latents_for_stitched_artifacts() -> None:
    rollout = FakeOpenWamRollout()
    context = PlanningContext(
        views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: np.zeros((2, 4, 4, 3), dtype=np.uint8),
        predicted_view_key="agentview",
    )

    output = fdm.predict(context, ActionChunk(np.zeros((4, 7), dtype=np.float32)))

    latents = output.metadata[fdm.keys.predicted_latents]
    assert isinstance(latents, torch.Tensor)
    assert tuple(latents.shape) == (1, 1, 2, 2, 2)
    assert latents.device.type == "cpu"


def test_openwam_fdm_scales_normalized_float_prediction_to_uint8() -> None:
    rollout = FakeOpenWamRollout()
    context = PlanningContext(
        views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    predicted = np.full((2, 4, 4, 3), 0.5, dtype=np.float32)
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: predicted,
        predicted_view_key="agentview",
    )

    output = fdm.predict(context, ActionChunk(np.zeros((4, 7), dtype=np.float32)))

    assert output.predicted_video.dtype == np.uint8
    assert output.next_context.views["agentview"].dtype == np.uint8
    assert int(output.predicted_video.mean()) == 127
    assert int(output.next_context.views["agentview"].mean()) == 127


def test_openwam_fdm_projects_prediction_to_context_view_for_scoring() -> None:
    rollout = FakeOpenWamRollout()
    rollout.runner.pipeline.preprocessor = SimpleNamespace(
        placements=(
            SimpleNamespace(source_name="observation.images.agentview_rgb", top=0, left=0, height=4, width=4),
            SimpleNamespace(source_name="observation.images.eye_in_hand_rgb", top=0, left=4, height=4, width=4),
        )
    )
    context = PlanningContext(
        views={
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        },
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    predicted = np.zeros((2, 4, 8, 3), dtype=np.uint8)
    predicted[:, :, :4] = 50
    predicted[:, :, 4:] = 200
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: predicted,
        predicted_view_key="agentview_image",
        project_prediction_to_context_view=True,
        predicted_view_aliases={
            "observation.images.agentview_rgb": "agentview_image",
            "observation.images.eye_in_hand_rgb": "robot0_eye_in_hand_image",
        },
    )

    output = fdm.predict(context, ActionChunk(np.zeros((4, 7), dtype=np.float32)))

    assert output.predicted_video.shape == (2, 4, 4, 3)
    assert int(output.predicted_video.mean()) == 50
    assert int(output.next_context.views["agentview_image"].mean()) == 50
    assert int(output.next_context.views["robot0_eye_in_hand_image"].mean()) == 200


def test_openwam_fdm_projected_prediction_video_uses_requested_view_key() -> None:
    rollout = FakeOpenWamRollout()
    context = PlanningContext(
        views={
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        },
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    predicted = {
        "agentview_image": np.full((2, 4, 4, 3), 50, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((2, 4, 4, 3), 200, dtype=np.uint8),
    }
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: predicted,
        predicted_view_key="robot0_eye_in_hand_image",
        project_prediction_to_context_view=True,
    )

    output = fdm.predict(context, ActionChunk(np.zeros((4, 7), dtype=np.float32)))

    assert output.predicted_video.shape == (2, 4, 4, 3)
    assert int(output.predicted_video.mean()) == 200
    assert int(output.next_context.views["agentview_image"].mean()) == 50
    assert int(output.next_context.views["robot0_eye_in_hand_image"].mean()) == 200


def test_openwam_fdm_can_return_canonical_video_while_projecting_context() -> None:
    rollout = FakeOpenWamRollout()
    rollout.runner.pipeline.preprocessor = SimpleNamespace(
        placements=(
            SimpleNamespace(source_name="observation.images.agentview_rgb", top=0, left=0, height=4, width=4),
            SimpleNamespace(source_name="observation.images.eye_in_hand_rgb", top=0, left=4, height=4, width=4),
        )
    )
    context = PlanningContext(
        views={
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        },
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    predicted = np.zeros((2, 4, 8, 3), dtype=np.uint8)
    predicted[:, :, :4] = 50
    predicted[:, :, 4:] = 200
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: predicted,
        predicted_view_key="agentview_image",
        project_prediction_to_context_view=True,
        return_canonical_prediction_video=True,
        predicted_view_aliases={
            "observation.images.agentview_rgb": "agentview_image",
            "observation.images.eye_in_hand_rgb": "robot0_eye_in_hand_image",
        },
    )

    output = fdm.predict(context, ActionChunk(np.zeros((4, 7), dtype=np.float32)))

    assert output.predicted_video.shape == (2, 4, 8, 3)
    assert int(output.predicted_video[:, :, :4].mean()) == 50
    assert int(output.predicted_video[:, :, 4:].mean()) == 200
    assert int(output.next_context.views["agentview_image"].mean()) == 50
    assert int(output.next_context.views["robot0_eye_in_hand_image"].mean()) == 200


def test_openwam_fdm_can_return_context_canvas_video_while_projecting_context() -> None:
    rollout = FakeOpenWamRollout()
    rollout.runner.pipeline.preprocessor = SimpleNamespace(
        placements=(
            SimpleNamespace(source_name="observation.images.agentview_rgb", top=0, left=0, height=4, width=4),
            SimpleNamespace(source_name="observation.images.eye_in_hand_rgb", top=0, left=4, height=4, width=4),
        )
    )
    context = PlanningContext(
        views={
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    predicted = np.zeros((2, 4, 8, 3), dtype=np.uint8)
    predicted[:, :, :4] = 50
    predicted[:, :, 4:] = 200
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: predicted,
        predicted_view_key="agentview_image",
        project_prediction_to_context_view=True,
        return_context_canvas_prediction_video=True,
        context_canvas_view_keys=("agentview_image", "robot0_eye_in_hand_image"),
        predicted_view_aliases={
            "observation.images.agentview_rgb": "agentview_image",
            "observation.images.eye_in_hand_rgb": "robot0_eye_in_hand_image",
        },
    )

    output = fdm.predict(context, ActionChunk(np.zeros((4, 7), dtype=np.float32)))

    assert output.predicted_video.shape == (2, 2, 4, 3)
    assert int(output.predicted_video[:, :, :2].mean()) == 50
    assert int(output.predicted_video[:, :, 2:].mean()) == 200
    assert int(output.next_context.views["agentview_image"].mean()) == 50
    assert int(output.next_context.views["robot0_eye_in_hand_image"].mean()) == 200


def test_openwam_fdm_rejects_multiple_prediction_video_return_spaces() -> None:
    with pytest.raises(ValueError, match="cannot be both canonical and context-canvas"):
        OpenWamActionConditionedFdm(
            rollout=FakeOpenWamRollout(),
            mode="forced_action_joint_fdm",
            return_canonical_prediction_video=True,
            return_context_canvas_prediction_video=True,
        )


def test_openwam_fdm_pads_short_policy_horizon_and_crops_decoded_video() -> None:
    rollout = FakeOpenWamRollout()
    context = PlanningContext(
        views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    predicted = np.stack(
        [np.full((4, 4, 3), frame, dtype=np.uint8) for frame in range(6)],
        axis=0,
    )
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: predicted,
        predicted_view_key="agentview",
    )

    output = fdm.predict(context, ActionChunk(np.ones((3, 7), dtype=np.float32)))

    raw_action_chunk = rollout.last_infer_kwargs["raw_action_chunk"].detach().cpu().numpy()
    assert raw_action_chunk.shape == (1, 4, 7)
    np.testing.assert_allclose(raw_action_chunk[:, :3], 1.0)
    np.testing.assert_allclose(raw_action_chunk[:, 3:], 1.0)
    assert output.predicted_video.shape == (3, 4, 4, 3)
    assert np.all(output.next_context.views["agentview"] == 2)
    assert output.metadata["policy_action_steps"] == 3
    assert output.metadata["internal_action_steps"] == 4
    assert output.metadata["padded_action_steps"] == 1


def test_planning_runner_prefers_stitched_openwam_latent_debug_decode() -> None:
    runner = _load_planning_runner()
    rollout = FakeOpenWamRollout()

    def _decode(latents: torch.Tensor) -> np.ndarray:
        frame_count = int(latents.shape[2])
        return np.stack(
            [np.full((4, 4, 3), frame, dtype=np.uint8) for frame in range(frame_count)],
            axis=0,
        )

    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=_decode,
        predicted_view_key="agentview",
    )
    candidate = CandidateTrajectory(
        context=PlanningContext(
            views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
            state=np.zeros(2, dtype=np.float32),
            task_text="task",
        ),
        predicted_videos=(
            np.full((2, 4, 4, 3), 100, dtype=np.uint8),
            np.full((3, 4, 4, 3), 200, dtype=np.uint8),
        ),
        metadata={
            fdm.keys.predicted_latents: (
                torch.zeros(1, 1, 2, 2, 2),
                torch.zeros(1, 1, 3, 2, 2),
            )
        },
    )

    video = runner._candidate_debug_video(candidate, dynamics=fdm, max_frames=4)

    assert video.shape == (4, 4, 4, 3)
    assert [int(video[index].mean()) for index in range(4)] == [0, 1, 2, 3]


def test_openwam_fdm_rejects_longer_than_internal_action_horizon() -> None:
    context = PlanningContext(
        views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    fdm = OpenWamActionConditionedFdm(
        rollout=FakeOpenWamRollout(),
        mode="forced_action_joint_fdm",
    )

    with pytest.raises(ValueError, match="longer horizons would silently truncate"):
        fdm.predict(context, ActionChunk(np.zeros((5, 7), dtype=np.float32)))


def test_runner_preflight_accepts_one_chunk_short_policy_horizon() -> None:
    runner = _load_planning_runner()
    fdm = OpenWamActionConditionedFdm(
        rollout=FakeOpenWamRollout(),
        mode="forced_action_joint_fdm",
    )

    runner._validate_planner_runtime_contract(
        SimpleNamespace(chunk_action_steps=3, execute_action_steps=3, max_plan_chunks=1, min_plan_action_steps=3),
        fdm,
    )


def test_runner_preflight_rejects_multi_chunk_short_policy_horizon() -> None:
    runner = _load_planning_runner()
    fdm = OpenWamActionConditionedFdm(
        rollout=FakeOpenWamRollout(),
        mode="forced_action_joint_fdm",
    )

    with pytest.raises(ValueError, match="cannot score policy chunks longer"):
        runner._validate_planner_runtime_contract(
            SimpleNamespace(chunk_action_steps=5, execute_action_steps=4, max_plan_chunks=1, min_plan_action_steps=4),
            fdm,
        )
    with pytest.raises(ValueError, match="Set --max-plan-chunks 1"):
        runner._validate_planner_runtime_contract(
            SimpleNamespace(chunk_action_steps=3, execute_action_steps=3, max_plan_chunks=2, min_plan_action_steps=3),
            fdm,
        )


def test_openwam_fdm_warms_from_views_and_repeats_hidden_proprio() -> None:
    rollout = FakeOpenWamWarmupRollout()
    fdm = OpenWamActionConditionedFdm(rollout=rollout, mode="forced_action_joint_fdm")
    context = PlanningContext(
        views={"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.arange(8, dtype=np.float32),
        task_text="task",
        metadata={"openwam_proprio_state": np.arange(8, dtype=np.float32)},
    )

    warmed = fdm.warm_context_from_views(
        context,
        view_aliases={"observation.images.agentview_rgb": "agentview_image"},
        action_dim=7,
    )

    assert warmed.metadata["openwam_fdm_session"] == "warmed-session"
    assert rollout.encode_grad_enabled is False
    assert rollout.reset_grad_enabled is False
    assert rollout.captured["action_context"].shape == (1, 4, 7)
    assert rollout.captured["hidden_proprio_history"].shape == (1, 1, 8)
    assert torch.equal(rollout.captured["proprio_state"][0], torch.arange(8, dtype=torch.float32))


def test_openwam_fdm_applies_input_view_transform_before_encoding() -> None:
    rollout = FakeOpenWamWarmupRollout()
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        input_view_transform="vertical_flip",
    )
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    context = PlanningContext(
        views={"agentview_image": image},
        state=np.arange(8, dtype=np.float32),
        task_text="task",
        metadata={"openwam_proprio_state": np.arange(8, dtype=np.float32)},
    )

    fdm.warm_context_from_views(
        context,
        view_aliases={"observation.images.agentview_rgb": "agentview_image"},
        action_dim=7,
    )

    encoded = rollout.canonicalize_views["observation.images.agentview_rgb"].numpy()[0]
    np.testing.assert_array_equal(encoded, image[::-1])


def test_openwam_fdm_inverts_input_transform_for_policy_context_views() -> None:
    rollout = FakeOpenWamRollout()
    context = PlanningContext(
        views={
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        },
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": "session"},
    )
    predicted = {
        "agentview_image": np.stack(
            [np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)],
            axis=0,
        )
    }
    fdm = OpenWamActionConditionedFdm(
        rollout=rollout,
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: predicted,
        predicted_view_key="agentview_image",
        project_prediction_to_context_view=True,
        input_view_transform="vertical_flip",
    )

    output = fdm.predict(context, ActionChunk(np.zeros((4, 7), dtype=np.float32)))

    np.testing.assert_array_equal(
        output.next_context.views["agentview_image"],
        predicted["agentview_image"][0, ::-1],
    )
    np.testing.assert_array_equal(output.predicted_video, predicted["agentview_image"][:, ::-1])


def test_openwam_fdm_forks_cached_session_per_candidate() -> None:
    original_session = SimpleNamespace(policy_state=SimpleNamespace(cache={}))
    context = PlanningContext(
        views={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=np.zeros(2, dtype=np.float32),
        task_text="task",
        metadata={"openwam_fdm_session": original_session},
    )
    fdm = OpenWamActionConditionedFdm(
        rollout=MutatingOpenWamRollout(),
        mode="forced_action_joint_fdm",
        decode_video_fn=lambda _latents: np.zeros((1, 4, 4, 3), dtype=np.uint8),
        predicted_view_key="agentview",
    )

    output = fdm.predict(context, ActionChunk(np.zeros((4, 7), dtype=np.float32)))

    assert "mutated" not in original_session.policy_state.cache
    assert output.next_context.metadata["openwam_fdm_session"].policy_state.cache["mutated"] is True
