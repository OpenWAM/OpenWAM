from __future__ import annotations

import json
from pathlib import Path

from open_wam.configs import DataSplit, EvalMode, EvalPredictionSource
from open_wam.evals.evaluation_contracts import EvaluationRequest, EvaluationSummary
from open_wam.evals.evaluation_reporting import build_evaluation_result
from open_wam.runtime.results import write_result_json


def test_evaluation_result_uses_versioned_envelope_and_atomic_writer(
    tmp_path: Path,
) -> None:
    request = EvaluationRequest(
        experiment_config_path=tmp_path / "experiment.yaml",
        mode=EvalMode.BATCH,
        split=DataSplit.VAL,
        max_batches=1,
        max_trajectories=None,
        max_steps_per_trajectory=None,
        batch_size=2,
        checkpoint_path=None,
        device="cpu",
        seed=7,
    )
    summary = EvaluationSummary(
        experiment_name="fixture",
        mode=EvalMode.BATCH,
        split=DataSplit.VAL,
        num_batches=1,
        num_trajectories=0,
        device="cpu",
        video_num_inference_steps=2,
        action_num_inference_steps=2,
        joint_num_inference_steps=None,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        action_prediction_source=EvalPredictionSource.DECODER_ACTION_PRED,
        action_prediction_shape=(2, 4, 7),
        target_action_shape=(2, 4, 7),
        video_prediction_source=EvalPredictionSource.UNAVAILABLE,
        video_prediction_shape=(),
        target_video_shape=(2, 4, 8, 8),
        mean_action_mse=0.25,
        mean_trajectory_action_mse=None,
        mean_video_latent_mse=None,
        mean_trajectory_video_latent_mse=None,
        checkpoint_path=None,
    )
    output_path = tmp_path / "nested" / "result.json"
    result = build_evaluation_result(
        request=request,
        summary=summary,
        provenance={"schema_version": "open_wam.provenance.v1"},
        result_path=str(output_path),
        benchmark="fixture",
    )

    resolved_output = write_result_json(output_path, result)

    assert resolved_output == output_path.resolve()
    assert json.loads(output_path.read_text(encoding="utf-8")) == json.loads(
        json.dumps(result)
    )
    assert result["schema_version"] == "open_wam.result.v1"
    assert result["metrics"]["experiment_name"] == "fixture"
    assert result["metrics"]["mean_action_mse"] == 0.25
    assert result["metrics"]["action_prediction_shape"] == (2, 4, 7)
    assert result["metrics"]["checkpoint_compatibility"] == "strict"
    assert result["metrics"]["checkpoint_missing_keys"] == ()
    assert result["metrics"]["checkpoint_unexpected_keys"] == ()
    assert not tuple(output_path.parent.glob(f".{output_path.name}.*.tmp"))
