"""Dedicated one-GPU sanity checks for methods 1 through 5.

These tests are intentionally more rigorous than the default CPU smoke suite.
They are opt-in and require exactly the kind of environment a developer would
use for local CUDA validation:

- set `OPEN_WAM_RUN_GPU_SANITY=1`
- run on a host with at least one CUDA GPU

When either prerequisite is missing, the module exits immediately instead of
quietly falling back to CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import gc
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

import open_wam.training.runtime as training_runtime_module
from open_wam.configs import (
    BatchAdapterName,
    MoTConditionMode,
    MoTRuntimeMode,
    JointTimestepCoupling,
    ParallelActionAttentionScope,
    ParallelActionConditionSource,
    ParallelRuntimeMode,
    TrainerAccelerator,
    TrainerPrecision,
    VideoConditionInputSpace,
    VideoConditionTrainMode,
    VisualReadoutConfig,
    VisualReadoutFusionMode,
    VisualReadoutSourceFamily,
)
from open_wam.data import (
    SyntheticLatentWindowDataset,
    build_synthetic_batch,
    build_synthetic_latent_batch,
    move_latent_wam_batch_to_device,
    move_wam_batch_to_device,
)
from open_wam.evals.evaluate import resolve_evaluation_request, run_evaluation
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training import TrainingRuntime
from open_wam.utils.config_loader import load_experiment_config


RUN_GPU_SANITY = os.getenv("OPEN_WAM_RUN_GPU_SANITY") == "1"
if not RUN_GPU_SANITY:
    pytest.skip(
        "Dedicated method-family GPU sanity checks are opt-in. "
        "Set OPEN_WAM_RUN_GPU_SANITY=1 to run them on a CUDA host.",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip(
        "Dedicated method-family GPU sanity checks require at least one CUDA GPU.",
        allow_module_level=True,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_DEVICE = torch.device("cuda:0")


def _write_temp_config(
    tmp_path: Path,
    *,
    source_name: str,
    output_name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    source_path = REPO_ROOT / "configs/experiments" / source_name
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = output_name
    mutate(raw)
    config_path = tmp_path / f"{output_name}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return config_path


def _cap_inference_steps(raw: dict[str, Any], *, steps: int = 2) -> None:
    inference = raw.setdefault("inference", {})
    if "video_num_inference_steps" in inference:
        inference["video_num_inference_steps"] = min(int(inference["video_num_inference_steps"]), steps)
    else:
        inference["video_num_inference_steps"] = steps
    if "action_num_inference_steps" in inference:
        inference["action_num_inference_steps"] = min(int(inference["action_num_inference_steps"]), steps)
    else:
        inference["action_num_inference_steps"] = steps
    if inference.get("joint_num_inference_steps") is not None:
        inference["joint_num_inference_steps"] = min(int(inference["joint_num_inference_steps"]), steps)


def _pipeline_case_path(case_name: str, tmp_path: Path) -> Path:
    if case_name == "parallel_exact":
        return REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    if case_name == "parallel_action_conditioned":
        return _write_temp_config(
            tmp_path,
            source_name="parallel_stream_robotwin_smoke.yaml",
            output_name="parallel_stream_robotwin_action_conditioned_gpu",
            mutate=_mutate_parallel_action_conditioned,
        )
    if case_name == "register_attached":
        return REPO_ROOT / "configs/experiments/register_attached_robotwin_smoke.yaml"
    if case_name == "video_sequence_default":
        return REPO_ROOT / "configs/experiments/video_sequence_policy_robotwin_smoke.yaml"
    if case_name == "video_sequence_core_layer":
        return _write_temp_config(
            tmp_path,
            source_name="video_sequence_policy_robotwin_smoke.yaml",
            output_name="video_sequence_policy_robotwin_core_layer_gpu",
            mutate=_mutate_video_sequence_core_layer,
        )
    if case_name == "video_sequence_core_multi_layer":
        return _write_temp_config(
            tmp_path,
            source_name="video_sequence_policy_robotwin_smoke.yaml",
            output_name="video_sequence_policy_robotwin_multi_layer_gpu",
            mutate=_mutate_video_sequence_core_multi_layer,
        )
    if case_name == "post_latent_legacy":
        return REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    if case_name == "post_decoded_legacy":
        return REPO_ROOT / "configs/experiments/post_decoded_robotwin.yaml"
    if case_name == "post_latent_video_conditioned":
        return REPO_ROOT / "configs/experiments/post_latent_robotwin_video_conditioned.yaml"
    if case_name == "post_decoded_video_conditioned":
        return REPO_ROOT / "configs/experiments/post_decoded_robotwin_video_conditioned.yaml"
    if case_name == "mot_prefill":
        return REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml"
    if case_name == "mot_joint":
        return _write_temp_config(
            tmp_path,
            source_name="mot_robotwin_smoke.yaml",
            output_name="mot_robotwin_joint_gpu",
            mutate=_mutate_mot_joint_denoise,
        )
    if case_name == "mot_non_joint":
        return _write_temp_config(
            tmp_path,
            source_name="mot_robotwin_smoke.yaml",
            output_name="mot_robotwin_non_joint_gpu",
            mutate=_mutate_mot_non_joint_two_stream,
        )
    raise ValueError(f"Unsupported GPU sanity pipeline case {case_name!r}.")


def _train_only_case_path(case_name: str, tmp_path: Path) -> Path:
    if case_name == "method4_current_frame_rgb":
        return REPO_ROOT / "configs/experiments/post_decoded_robotwin_current_frame_regression.yaml"
    if case_name == "method4_current_frame_latent":
        return _write_temp_config(
            tmp_path,
            source_name="post_latent_robotwin_video_conditioned.yaml",
            output_name="post_latent_robotwin_current_frame_regression_gpu",
            mutate=_mutate_post_latent_current_frame_regression,
        )
    raise ValueError(f"Unsupported GPU sanity train-only case {case_name!r}.")


def _mutate_parallel_action_conditioned(raw: dict[str, Any]) -> None:
    policy_variant = raw.setdefault("policy_variant", {})
    policy_variant["runtime_mode"] = ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED.value
    policy_variant["video_condition_on_action"] = True
    policy_variant["video_action_condition_source"] = ParallelActionConditionSource.NOISY_ACTION.value
    policy_variant["video_action_attention_scope"] = ParallelActionAttentionScope.BLOCK_LOCAL.value
    policy_variant["joint_timestep_coupling"] = JointTimestepCoupling.MATCH_SIGMA.value
    _cap_inference_steps(raw, steps=2)
    raw.setdefault("inference", {})["use_cache"] = False


def _mutate_video_sequence_core_layer(raw: dict[str, Any]) -> None:
    raw.setdefault("backbone", {})["num_layers"] = 2
    raw.setdefault("policy_variant", {})["visual_readout"] = {
        "source_family": VisualReadoutSourceFamily.CORE_LAYER_TOKENS.value,
        "layer_index": 0,
    }


def _mutate_video_sequence_core_multi_layer(raw: dict[str, Any]) -> None:
    raw.setdefault("backbone", {})["num_layers"] = 2
    raw.setdefault("policy_variant", {})["visual_readout"] = {
        "source_family": VisualReadoutSourceFamily.CORE_MULTI_LAYER_TOKENS.value,
        "layer_indices": [0, 1],
        "fusion_mode": VisualReadoutFusionMode.CONCAT_PROJECT.value,
    }


def _mutate_mot_joint_denoise(raw: dict[str, Any]) -> None:
    policy_variant = raw.setdefault("policy_variant", {})
    policy_variant["runtime_mode"] = MoTRuntimeMode.JOINT_DENOISE.value
    policy_variant["condition_mode"] = MoTConditionMode.FULL_VIDEO.value
    _cap_inference_steps(raw, steps=2)
    inference = raw.setdefault("inference", {})
    inference["video_num_inference_steps"] = 2
    inference["action_num_inference_steps"] = 2
    inference["use_cache"] = False


def _mutate_mot_non_joint_two_stream(raw: dict[str, Any]) -> None:
    policy_variant = raw.setdefault("policy_variant", {})
    policy_variant["runtime_mode"] = MoTRuntimeMode.NON_JOINT_TWO_STREAM.value
    policy_variant["condition_mode"] = MoTConditionMode.TEACHER_FORCING_COND_VIDEO.value
    policy_variant["video_can_attend_action"] = False
    _cap_inference_steps(raw, steps=2)
    inference = raw.setdefault("inference", {})
    inference["video_num_inference_steps"] = 2
    inference["action_num_inference_steps"] = 2
    inference["use_cache"] = False


def _mutate_post_latent_current_frame_regression(raw: dict[str, Any]) -> None:
    raw.setdefault("data", {})["dataset_type"] = "lerobot_v2_latent_local"
    raw.setdefault("trainer", {})["batch_adapter"] = BatchAdapterName.LATENTS.value
    policy_variant = raw.setdefault("policy_variant", {})
    policy_variant["video_condition_input_space"] = VideoConditionInputSpace.VIDEO_LATENT.value
    raw["action_decoder"] = {
        "name": "video_conditioned_action_decoder",
        "input_space": VideoConditionInputSpace.VIDEO_LATENT.value,
        "train_mode": VideoConditionTrainMode.CURRENT_FRAME_REGRESSION.value,
        "use_text_conditioning": False,
        "use_state_conditioning": True,
    }


def _prepare_pipeline_config(config_path: Path) -> Any:
    config = load_experiment_config(config_path)
    return replace(
        config,
        inference=replace(
            config.inference,
            video_num_inference_steps=min(int(config.inference.video_num_inference_steps), 2),
            action_num_inference_steps=min(int(config.inference.action_num_inference_steps), 2),
            joint_num_inference_steps=(
                min(int(config.inference.joint_num_inference_steps), 2)
                if config.inference.joint_num_inference_steps is not None
                else None
            ),
        ),
        trainer=replace(
            config.trainer,
            accelerator=TrainerAccelerator.GPU,
            devices=1,
            precision=TrainerPrecision.FP32,
        ),
    )


def _prepare_runtime_config(config_path: Path, *, tmp_path: Path) -> Any:
    config = _prepare_pipeline_config(config_path)
    return replace(
        config,
        data=replace(
            config.data,
            train_batch_size=1,
            val_batch_size=1,
            num_workers=0,
            max_train_episodes=1,
            max_val_episodes=1,
        ),
        training=replace(
            config.training,
            num_steps=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
        ),
        trainer=replace(
            config.trainer,
            runtime="composable",
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path),
            limit_train_batches=1,
            limit_val_batches=1,
            enable_checkpointing=False,
            save_interval=None,
            enable_jsonl_logging=False,
            enable_wandb=False,
        ),
    )


def _view_train_batch(batch) -> PolicyTrainBatch:
    extra = {
        "task_text": batch.task_text,
        "metadata": batch.metadata,
    }
    if batch.state_mask is not None:
        extra["state_mask"] = batch.state_mask
    return PolicyTrainBatch(
        actions=batch.actions,
        action_mask=batch.action_mask,
        state=batch.state,
        extra=extra,
    )


def _infer_context(batch) -> PolicyInferContext:
    return PolicyInferContext(
        state=batch.state,
        extra={
            "task_text": batch.task_text,
            "metadata": batch.metadata,
        },
    )


def _synthetic_latent_train_val_pair(data_config) -> tuple[SyntheticLatentWindowDataset, SyntheticLatentWindowDataset]:
    return (
        SyntheticLatentWindowDataset(data_config, length=4),
        SyntheticLatentWindowDataset(data_config, length=2),
    )


@pytest.fixture(autouse=True)
def _clear_cuda_between_tests():
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    ("case_name", "expected_horizon"),
    [
        ("parallel_exact", 8),
        ("parallel_action_conditioned", 8),
        ("video_sequence_default", 6),
        ("video_sequence_core_layer", 6),
        ("video_sequence_core_multi_layer", 6),
        ("post_latent_legacy", 6),
        ("post_decoded_legacy", 6),
        ("post_latent_video_conditioned", 6),
        ("post_decoded_video_conditioned", 6),
        ("mot_prefill", 8),
        ("mot_joint", 8),
        ("mot_non_joint", 8),
    ],
)
def test_gpu_method_family_pipeline_train_and_infer_matrix(
    tmp_path: Path,
    case_name: str,
    expected_horizon: int,
) -> None:
    config_path = _pipeline_case_path(case_name, tmp_path)
    config = _prepare_pipeline_config(config_path)
    pipeline = build_variant_pipeline_from_config(config).to(CUDA_DEVICE)

    batch = move_wam_batch_to_device(build_synthetic_batch(config.data, batch_size=1), CUDA_DEVICE)
    train_batch = _view_train_batch(batch)

    train_output = pipeline.forward_train(batch.views, train_batch)
    infer_output = pipeline.forward_infer_step(batch.views, _infer_context(batch))

    assert train_output.decoder_output.action_pred.shape == (1, expected_horizon, config.action_decoder.action_dim)
    assert infer_output.decoder_output.action_pred.shape == (1, expected_horizon, config.action_decoder.action_dim)

    if case_name == "video_sequence_core_layer":
        assert train_output.policy_output.decoder_sequence_context is not None
        assert infer_output.policy_output.decoder_sequence_context is not None
        assert train_output.policy_output.decoder_sequence_context.source_stage == "core_layer_0"
        assert infer_output.policy_output.decoder_sequence_context.source_stage == "core_layer_0"
    if case_name == "video_sequence_core_multi_layer":
        assert train_output.policy_output.decoder_sequence_context is not None
        assert infer_output.policy_output.decoder_sequence_context is not None
        assert train_output.policy_output.decoder_sequence_context.source_stage == "core_multi_layer"
        assert infer_output.policy_output.decoder_sequence_context.source_stage == "core_multi_layer"


@pytest.mark.parametrize(
    "case_name",
    [
        "method4_current_frame_rgb",
        "method4_current_frame_latent",
    ],
)
def test_gpu_method4_current_frame_regression_modes_train_only(
    tmp_path: Path,
    case_name: str,
) -> None:
    config_path = _train_only_case_path(case_name, tmp_path)
    config = _prepare_pipeline_config(config_path)
    pipeline = build_variant_pipeline_from_config(config).to(CUDA_DEVICE)

    if case_name == "method4_current_frame_latent":
        latent_batch = move_latent_wam_batch_to_device(build_synthetic_latent_batch(config.data, batch_size=1), CUDA_DEVICE)
        train_batch = PolicyTrainBatch(
            actions=latent_batch.actions,
            action_mask=latent_batch.action_mask,
            state=latent_batch.state,
            extra={
                "task_text": latent_batch.task_text,
                "metadata": latent_batch.metadata,
                "state_mask": latent_batch.state_mask,
            },
        )
        train_output = pipeline.forward_train_from_latents(
            latent_batch.video_latents,
            train_batch,
            canonical_video=latent_batch.canonical_video,
            text_context=latent_batch.text_context,
            negative_text_context=latent_batch.negative_text_context,
        )
        assert train_output.decoder_output.action_pred.shape == (1, 1, config.action_decoder.action_dim)
        with pytest.raises(ValueError, match="train-only"):
            pipeline.forward_infer_step_from_latents(
                latent_batch.video_latents,
                _infer_context(latent_batch),
                canonical_video=latent_batch.canonical_video,
                text_context=latent_batch.text_context,
                negative_text_context=latent_batch.negative_text_context,
            )
        return

    batch = move_wam_batch_to_device(build_synthetic_batch(config.data, batch_size=1), CUDA_DEVICE)
    train_batch = _view_train_batch(batch)
    train_output = pipeline.forward_train(batch.views, train_batch)
    assert train_output.decoder_output.action_pred.shape == (1, 1, config.action_decoder.action_dim)
    with pytest.raises(ValueError, match="train-only"):
        pipeline.forward_infer_step(batch.views, _infer_context(batch))


@pytest.mark.parametrize(
    "case_name",
    [
        "parallel_exact",
        "parallel_action_conditioned",
        "register_attached",
        "video_sequence_default",
        "video_sequence_core_layer",
        "post_latent_legacy",
        "post_decoded_legacy",
        "post_latent_video_conditioned",
        "post_decoded_video_conditioned",
        "method4_current_frame_rgb",
        "method4_current_frame_latent",
        "mot_prefill",
        "mot_joint",
        "mot_non_joint",
    ],
)
def test_gpu_method_family_runtime_train_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
) -> None:
    if case_name.startswith("method4_current_frame"):
        config_path = _train_only_case_path(case_name, tmp_path)
    else:
        config_path = _pipeline_case_path(case_name, tmp_path)
    config = _prepare_runtime_config(config_path, tmp_path=tmp_path)

    if case_name == "method4_current_frame_latent":
        monkeypatch.setattr(
            training_runtime_module,
            "build_train_val_latent_datasets",
            _synthetic_latent_train_val_pair,
        )

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


@pytest.mark.parametrize(
    ("case_name", "expected_name"),
    [
        ("parallel_exact", "parallel_stream_robotwin_smoke"),
        ("parallel_action_conditioned", "parallel_stream_robotwin_action_conditioned_gpu"),
        ("video_sequence_default", "video_sequence_policy_robotwin_smoke"),
        ("video_sequence_core_layer", "video_sequence_policy_robotwin_core_layer_gpu"),
        ("post_latent_legacy", "post_latent_robotwin"),
        ("post_decoded_legacy", "post_decoded_robotwin"),
        ("post_latent_video_conditioned", "post_latent_robotwin_video_conditioned"),
        ("post_decoded_video_conditioned", "post_decoded_robotwin_video_conditioned"),
        ("mot_prefill", "mot_robotwin_smoke"),
        ("mot_joint", "mot_robotwin_joint_gpu"),
        ("mot_non_joint", "mot_robotwin_non_joint_gpu"),
    ],
)
def test_gpu_method_family_eval_matrix(
    tmp_path: Path,
    case_name: str,
    expected_name: str,
) -> None:
    config_path = _pipeline_case_path(case_name, tmp_path)
    request = resolve_evaluation_request(
        config_path,
        max_batches_override=1,
        device_override="cuda:0",
    )

    summary = run_evaluation(request)

    assert summary.experiment_name == expected_name
    assert summary.num_batches == 1
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None
