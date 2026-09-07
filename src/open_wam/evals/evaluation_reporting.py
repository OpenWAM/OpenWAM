"""Structured reporting for the generic evaluation runtime."""

from __future__ import annotations

from typing import Any, Mapping

from open_wam.evals.evaluation_contracts import EvaluationRequest, EvaluationSummary
from open_wam.runtime.results import build_result_envelope


def build_evaluation_result(
    *,
    request: EvaluationRequest,
    summary: EvaluationSummary,
    provenance: Mapping[str, Any],
    result_path: str,
    benchmark: str | None,
) -> dict[str, Any]:
    """Convert an evaluation summary into the stable public result schema."""

    return build_result_envelope(
        command="openwam-eval",
        config=str(request.source_config_path or request.experiment_config_path),
        checkpoint=summary.checkpoint_path,
        benchmark=benchmark,
        device=summary.device,
        seed=request.seed,
        metrics={
            "experiment_name": summary.experiment_name,
            "mode": summary.mode.value,
            "split": summary.split.value,
            "num_batches": summary.num_batches,
            "num_trajectories": summary.num_trajectories,
            "video_num_inference_steps": summary.video_num_inference_steps,
            "action_num_inference_steps": summary.action_num_inference_steps,
            "joint_num_inference_steps": summary.joint_num_inference_steps,
            "guidance_scale": summary.guidance_scale,
            "action_guidance_scale": summary.action_guidance_scale,
            "mean_action_mse": summary.mean_action_mse,
            "mean_trajectory_action_mse": summary.mean_trajectory_action_mse,
            "mean_video_latent_mse": summary.mean_video_latent_mse,
            "mean_trajectory_video_latent_mse": (
                summary.mean_trajectory_video_latent_mse
            ),
            "action_prediction_source": summary.action_prediction_source.value,
            "action_prediction_shape": summary.action_prediction_shape,
            "target_action_shape": summary.target_action_shape,
            "video_prediction_source": summary.video_prediction_source.value,
            "video_prediction_shape": summary.video_prediction_shape,
            "target_video_shape": summary.target_video_shape,
            "checkpoint_compatibility": summary.checkpoint_compatibility,
            "checkpoint_missing_keys": summary.checkpoint_missing_keys,
            "checkpoint_unexpected_keys": summary.checkpoint_unexpected_keys,
        },
        artifacts={"result_path": result_path},
        provenance=provenance,
    )


__all__ = ["build_evaluation_result"]
