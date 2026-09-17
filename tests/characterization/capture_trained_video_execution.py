"""Capture trained chunk-conditioned video at the frozen latent boundary.

Use the same script and fixture with each source checkout on PYTHONPATH.
No RGB frontend, optimizer, dataset sampler, or checkpoint is modified.
"""

import argparse
import json
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from open_wam.configs import CausalVideoProgram, load_experiment_config
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training.step_executor import (
    LatentBatchAdapter,
    PipelineTrainStepExecutor,
)
from open_wam.training.strategies import SingleDeviceStrategy
from tests.characterization.dual_expert_refactor_artifacts import (
    load_latent_batch_fixture,
)


def capture(args):
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.benchmark_output is not None and args.benchmark_output.exists():
        raise FileExistsError(args.benchmark_output)
    config = load_experiment_config(
        args.config or args.checkpoint / "resolved_config.yaml",
        checkpoint_runtime_compat=args.config is None,
    )
    if (
        config.policy_variant.program
        is not CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO
    ):
        raise ValueError("This gate requires a chunk-conditioned video checkpoint.")
    config = replace(
        config,
        backbone=replace(
            config.backbone,
            pretrained_model_name_or_path=str(args.reference_assets_root),
            runtime_backbone_artifact_path=str(args.checkpoint / "transformer"),
        ),
    )
    torch.manual_seed(89)
    random.seed(89)
    pipeline = build_variant_pipeline_from_config(config).to(args.device)
    executor = PipelineTrainStepExecutor(
        pipeline=pipeline,
        batch_adapter=LatentBatchAdapter(),
        training_config=config.training,
    )
    batch = executor.batch_adapter.move_to_device(
        load_latent_batch_fixture(args.fixture), torch.device(args.device)
    )
    tensors = {}
    if args.training:
        strategy = SingleDeviceStrategy(
            accelerator=config.trainer.accelerator, precision=config.trainer.precision
        )
        torch.manual_seed(91)
        random.seed(91)
        with strategy.autocast_context():
            result = executor.forward_train(batch)
        result.loss.backward()
        tensors["loss"] = result.loss.detach().cpu()
        for name, parameter in pipeline.named_parameters():
            if parameter.grad is not None:
                tensors[f"gradient.{name}"] = parameter.grad.detach().cpu().contiguous()
        pipeline.zero_grad(set_to_none=True)
    pipeline.eval()
    history = torch.cat([batch.condition_latents[:, :, :1], batch.video_latents], dim=2)
    if history.shape[2] < 35:
        raise ValueError("The frozen fixture must contain at least 35 frames.")

    def generate(frames, *, use_cache: bool | None = None):
        return pipeline.visual_tower.generate_chunked_conditioned_video_latents(
            observed_history=history[:, :, :frames],
            future_template=torch.zeros_like(history[:, :, :5]),
            text_context=batch.text_context,
            negative_text_context=batch.negative_text_context,
            history_frame_start=0,
            chunk_size=config.inference.frame_chunk_size,
            window_size=30,
            chunk_origin_frame=0,
            num_inference_steps=config.inference.video_num_inference_steps,
            num_train_timesteps=config.training.video_num_train_timesteps,
            sigma_shift=config.training.video_sigma_shift,
            guidance_scale=config.inference.guidance_scale,
            sample_seed=93,
            use_cache=config.inference.use_cache if use_cache is None else use_cache,
        )

    with torch.inference_mode():
        for frames in (1, 3, 35):
            tensors[f"history{frames}"] = generate(frames).cpu().contiguous()
            print(f"captured history={frames}", flush=True)
    if not all(torch.isfinite(value).all().item() for value in tensors.values()):
        raise AssertionError("Non-finite trained video capture.")
    args.output_root.mkdir(parents=True)
    save_file(tensors, args.output_root / "trained_video.safetensors")
    print(f"captured {len(tensors)} tensors", flush=True)
    if args.benchmark_output is not None:
        benchmark_cache(
            pipeline.visual_tower, generate, tensors["history35"],
            device=torch.device(args.device), output=args.benchmark_output,
        )


@torch.inference_mode()
def benchmark_cache(tower, generate, expected, *, device, output):
    """Measure full trained calls, excluding compilation but including cache binding."""
    for enabled in (False, True):
        generate(35, use_cache=enabled)
    measurements = {False: [], True: []}
    evaluations = 0

    def count_evaluation(module, inputs):
        nonlocal evaluations
        evaluations += 1

    handle = tower.core.proj_out.register_forward_pre_hook(count_evaluation)
    try:
        for repeat in range(3):
            for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                evaluations = 0
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                actual = generate(35, use_cache=enabled)
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - start) * 1000
                peak = torch.cuda.max_memory_allocated(device)
                actual_cpu = actual.cpu().contiguous()
                try:
                    torch.testing.assert_close(actual_cpu, expected, rtol=0, atol=0)
                except AssertionError:
                    delta = actual_cpu.float() - expected.float()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    save_file(
                        {"expected": expected.contiguous(), "actual": actual_cpu},
                        output.with_suffix(".safetensors"),
                    )
                    output.write_text(json.dumps({
                        "exact_outputs": False,
                        "cache_enabled": enabled,
                        "repeat": repeat,
                        "history_frames": 35,
                        "max_absolute_difference": delta.abs().max().item(),
                        "mean_absolute_difference": delta.abs().mean().item(),
                        "rms_difference": delta.square().mean().sqrt().item(),
                        "reference_rms": expected.float().square().mean().sqrt().item(),
                        "ms": elapsed_ms,
                        "peak_bytes": peak,
                    }, indent=2) + "\n")
                    raise
                measurements[enabled].append({
                    "ms": elapsed_ms, "peak_bytes": peak,
                    "model_evaluations": evaluations,
                })
                del actual
    finally:
        handle.remove()
    report = {
        "history_frames": 35, "returned_future_frames": 5, "window_size": 30,
        "exact_outputs": True, "device": torch.cuda.get_device_name(device),
        "samples": {str(k): v for k, v in measurements.items()},
        "median_ms": {
            str(k): statistics.median(row["ms"] for row in rows)
            for k, rows in measurements.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Explicit experiment semantics for the trained transformer.")
    parser.add_argument("--reference-assets-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--benchmark-output", type=Path)
    capture(parser.parse_args())
