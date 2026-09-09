"""Seeded video-batch characterization, independent of batch execution internals."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from open_wam.configs import BatchingConfig
from open_wam.data.latent_batching import LatentBatchCollator
from open_wam.data.latent_contracts import LatentWAMSample
from open_wam.training.step_executor import LatentBatchAdapter, PipelineTrainStepExecutor
from tests.test_causal_video_prediction import _deterministic_cpu_math, _tiny_causal_video_pipeline


def capture(mode: str, *, pad_to_multiple_of: int = 1) -> dict[str, torch.Tensor]:
    with _deterministic_cpu_math():
        config, pipeline, _ = _tiny_causal_video_pipeline()
        torch.manual_seed(61)
        samples = [
            LatentWAMSample(
                video_latents=torch.randn(48, length, 2, 4),
                actions=torch.zeros(0, 7), action_mask=torch.zeros(0, 7),
                state=torch.zeros(0, 8), state_mask=torch.zeros(0, 8),
                text_context=torch.randn(3, 8), task_text="move object",
                metadata={"dataset_type": "mixed_video", "observed_prefix_frames": observed,
                          "future_suffix_frames": length - observed, "valid_video_frames": length},
            )
            for length, observed in ((3, 1), (6, 2))
        ]
        batch = LatentBatchCollator(BatchingConfig(mode=mode, pad_to_multiple_of=pad_to_multiple_of))(samples)
        tensors = {"input.clean_video": batch.video_latents.clone()}
        decode = pipeline.resolve_train_decoder_output

        def decode_and_record(policy_output, policy_batch):
            artifacts = policy_output.decoder_artifacts.payload
            for key in ("targets", "timesteps", "predicted_latents", "target_latents", "future_loss_mask"):
                tensors[f"artifact.{key}"] = getattr(artifacts, key).detach().clone()
            return decode(policy_output, policy_batch)

        pipeline.resolve_train_decoder_output = decode_and_record

        def record_forward(_module, _args, kwargs):
            # VisualTower's public prediction boundary is stable across executors.
            for key in ("noisy_latents", "timesteps", "text_context"):
                value = kwargs[key]
                tensors[f"input.{key}"] = value.detach().cpu().contiguous().clone()

        original = pipeline.visual_tower.predict_video_flow

        def predict(**kwargs):
            record_forward(None, None, kwargs)
            result = original(**kwargs)
            tensors["output.flow"] = result.detach().cpu().contiguous().clone()
            return result

        pipeline.visual_tower.predict_video_flow = predict
        optimizer = torch.optim.AdamW([p for p in pipeline.parameters() if p.requires_grad], lr=1e-4)
        executor = PipelineTrainStepExecutor(pipeline=pipeline, batch_adapter=LatentBatchAdapter(), training_config=config.training)
        torch.manual_seed(79)
        result = executor.forward_train(batch)
        tensors["loss"] = result.loss.detach().clone()
        tensors.update({f"metric.{key}": value.detach().clone() for key, value in result.metrics.items()})
        result.loss.backward()
        for name, parameter in pipeline.named_parameters():
            if parameter.grad is not None:
                tensors[f"grad.{name}"] = parameter.grad.detach().clone()
        optimizer.step()
        for name, parameter in pipeline.named_parameters():
            if parameter.requires_grad:
                tensors[f"updated.{name}"] = parameter.detach().clone()
        tensors["rng"] = torch.get_rng_state().clone()
        return {name: tensor.cpu().contiguous() for name, tensor in tensors.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--out", type=Path, help="Record into a new directory")
    operation.add_argument("--compare", type=Path, help="Compare exactly; never update fixtures")
    args = parser.parse_args()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
    for mode in ("padded", "bucket", "packed"):
        for multiple in (1, 4):
            path = (args.out or args.compare) / f"{mode}_pad{multiple}.safetensors"
            tensors = capture(mode, pad_to_multiple_of=multiple)
            if args.compare:
                expected = load_file(path)
                assert tensors.keys() == expected.keys()
                for key, value in expected.items():
                    torch.testing.assert_close(tensors[key], value, rtol=0, atol=0, msg=key)
            else:
                save_file(tensors, path)
            print(f"{path.name}: {len(tensors)} tensors", flush=True)
