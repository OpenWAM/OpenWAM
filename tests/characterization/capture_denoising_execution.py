"""Source-pinned reduced-model outputs and every training gradient.

Run this script twice with the reference/candidate source roots on PYTHONPATH.
It uses existing fixtures, does not import candidate-only execution internals,
and never updates checked-in goldens. Trained-checkpoint gates remain separate.
"""

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from open_wam.configs import (
    ActionSchemaConfig,
    BatchingMode,
    DynamicsObjective,
    ExperimentConfig,
    InferenceConfig,
    ParallelStreamActionDecoderConfig,
    ParallelStreamPolicyConfig,
    ProprioContextMode,
    RobotWinDataConfig,
    TrainingConfig,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.data.latent_contracts import collate_latent_wam_samples
from open_wam.models.common.dynamics_contracts import DynamicsRolloutRequest
from open_wam.models.policy_variants.contracts import PolicyInferContext
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training.step_executor import (
    LatentBatchAdapter,
    PipelineTrainStepExecutor,
)
from tests.test_causal_video_prediction import _tiny_chunked_conditioned_video_pipeline
from tests.test_sequence_batch_dynamics import dynamics_samples
from tests.test_variable_batch_pipeline import (
    PROGRAMS,
    _collate,
    _samples,
    _tiny_pipeline,
)


def _parallel_pipeline(program, *, token: bool, head_dim: int):
    hidden_size = 4 * head_dim
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=8, state_dim=4, state_horizon=1
            ),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=hidden_size,
            num_layers=2,
            num_heads=4,
            attention_head_dim=head_dim,
            ffn_dim=hidden_size * 2,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
            load_text_conditioning=False,
            load_wan_vae_frontend=False,
        ),
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=hidden_size,
            program=program,
            frame_chunk_size=2,
            action_per_frame=2,
            generalist_mode_text_token=token,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
        action_decoder=ParallelStreamActionDecoderConfig(
            hidden_size=hidden_size, action_dim=4, action_horizon=8
        ),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            enabled_objectives=("action", "latent"),
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
        ),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    return pipeline, PipelineTrainStepExecutor(
        pipeline=pipeline,
        batch_adapter=LatentBatchAdapter(),
        training_config=config.training,
    )


def capture(output_root: Path, device: str, architecture: str) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    cases = [(program, False, None) for program in PROGRAMS]
    cases += [
        (VideoActionProgram.GENERALIST_JOINT_DENOISING, token, objective)
        for token in (False, True)
        for objective in DynamicsObjective
    ]
    cases += [
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            False,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            False,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
    ]
    for program, token, objective in cases:
        if device.startswith("cuda"):
            torch.compiler.reset()
        name = f"{program.value}_{token}_{'default' if objective is None else objective.value}"
        torch.manual_seed(719)
        random.seed(719)
        head_dim = 32 if device.startswith("cuda") else 8
        if architecture == "dual_expert":
            pipeline, executor = _tiny_pipeline(
                program,
                generalist_mode_text_token=token,
                attention_head_dim=head_dim,
                sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
            )
        else:
            pipeline, executor = _parallel_pipeline(
                program, token=token, head_dim=head_dim
            )
        pipeline.to(device)
        variant = pipeline.policy_variant
        variant.inference_config = replace(
            variant.inference_config,
            video_num_inference_steps=3,
            action_num_inference_steps=3,
            attention_window_size=4,
        )
        samples = (
            _samples(program)
            if objective is None
            else dynamics_samples((objective, objective))
        )
        if architecture == "parallel_stream":
            samples = [
                replace(sample, condition_latents=sample.video_latents.clone())
                for sample in samples
            ]
            batch = collate_latent_wam_samples(samples[:1])
        else:
            batch = _collate(samples, BatchingMode.PADDED)
        batch = executor.batch_adapter.move_to_device(batch, torch.device(device))
        torch.manual_seed(911)
        random.seed(911)
        result = executor.forward_train(batch)
        result.loss.backward()
        tensors = {"loss": result.loss.detach().cpu().clone()}
        for key, parameter in pipeline.named_parameters():
            if parameter.grad is not None:
                tensors[f"gradient.{key}"] = parameter.grad.detach().cpu().clone()
        pipeline.zero_grad(set_to_none=True)
        pipeline.eval()
        state = None
        cursors = []
        with torch.no_grad():
            for chunk in range(4):
                torch.manual_seed(1000 + chunk)
                observed = torch.randn(
                    1,
                    48,
                    1 if chunk == 0 and architecture == "dual_expert" else 2,
                    4,
                    4,
                    device=device,
                )
                context = PolicyInferContext(
                    state=torch.randn(1, 1, 4, device=device),

                    dynamics=None
                    if objective is None
                    else DynamicsRolloutRequest(
                        objective=objective,
                        clean_action=(
                            torch.randn(1, 4, 4, device=device)
                            if objective is DynamicsObjective.ACTION_CONDITIONED_VIDEO
                            else None
                        ),
                        clean_video=(
                            torch.randn(1, 48, 1, 4, 4, device=device)
                            if objective is DynamicsObjective.VIDEO_CONDITIONED_ACTION
                            else None
                        ),
                        frame_chunk_size=1 if objective.is_conditional else 2,
                    ),
                )
                output = pipeline.forward_infer_step_from_latents(
                    observed,
                    context,
                    infer_state=state,
                    text_context=torch.randn(1, 3, 16, device=device),
                )
                action = output.decoder_output.action_pred
                if action is not None:
                    tensors[f"chunk{chunk}.action"] = action.cpu().clone()
                video = output.policy_output.generated_video
                if video is not None:
                    tensors[f"chunk{chunk}.video"] = video.latents.cpu().clone()
                elif architecture == "parallel_stream":
                    tensors[f"chunk{chunk}.video"] = (
                        output.policy_output.aux["predicted_latents"].cpu().clone()
                    )
                state = output.policy_output.next_state
                cursors.append(dict(vars(state.cursor)))
        if not all(torch.isfinite(value).all().item() for value in tensors.values()):
            raise AssertionError(f"Non-finite capture for {name}")
        save_file(
            {key: value.contiguous() for key, value in tensors.items()},
            output_root / f"{name}.safetensors",
        )
        (output_root / f"{name}.json").write_text(
            json.dumps(cursors, sort_keys=True) + "\n"
        )
        print(f"{name}: {len(tensors)} tensors", flush=True)

    if architecture == "parallel_stream":
        return
    _, pipeline = _tiny_chunked_conditioned_video_pipeline()
    tower = pipeline.visual_tower.to(device).eval()
    tensors = {}
    with torch.no_grad():
        for frames in (1, 3, 7, 35):
            torch.manual_seed(43)
            tensors[f"history{frames}"] = (
                tower.generate_chunked_conditioned_video_latents(
                    observed_history=torch.randn(1, 48, frames, 2, 4, device=device),
                    future_template=torch.zeros(1, 48, 5, 2, 4, device=device),
                    text_context=torch.randn(1, 3, 8, device=device),
                    negative_text_context=torch.randn(1, 3, 8, device=device),
                    history_frame_start=0,
                    chunk_size=4,
                    window_size=4,
                    chunk_origin_frame=0,
                    num_inference_steps=3,
                    num_train_timesteps=8,
                    sigma_shift=5.0,
                    guidance_scale=5.0,
                    sample_seed=91,
                )
                .cpu()
                .clone()
            )
    save_file(tensors, output_root / "chunked_video.safetensors")
    print("chunked video: CFG, partial chunks and window eviction", flush=True)


def verify(reference_root: Path, actual_root: Path) -> int:
    """Require identical cases, recurrent metadata, tensor values and dtypes."""
    expected_files = {p.name for p in reference_root.iterdir() if p.is_file()}
    actual_files = {p.name for p in actual_root.iterdir() if p.is_file()}
    if not expected_files or expected_files != actual_files:
        raise AssertionError(f"Capture files differ: {expected_files ^ actual_files}")
    count = 0
    for name in sorted(expected_files):
        if name.endswith(".json"):
            if json.loads((reference_root / name).read_text()) != json.loads(
                (actual_root / name).read_text()
            ):
                raise AssertionError(f"Recurrent metadata differs: {name}")
            continue
        expected, actual = (
            load_file(reference_root / name),
            load_file(actual_root / name),
        )
        if expected.keys() != actual.keys():
            raise AssertionError(f"Tensor keys differ: {name}")
        for key in expected:
            torch.testing.assert_close(
                actual[key],
                expected[key],
                rtol=0,
                atol=0,
                equal_nan=False,
                msg=f"{name}/{key} changed",
            )
            count += 1
    if not count:
        raise AssertionError("Capture contains no tensors.")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("--output-root", type=Path, required=True)
    record.add_argument("--device", default="cpu")
    record.add_argument(
        "--architecture",
        choices=("dual_expert", "parallel_stream"),
        default="dual_expert",
    )
    compare = commands.add_parser("verify")
    compare.add_argument("--reference-root", type=Path, required=True)
    compare.add_argument("--actual-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record":
        capture(args.output_root, args.device, args.architecture)
    else:
        print(f"{verify(args.reference_root, args.actual_root)} tensors exactly equal")
