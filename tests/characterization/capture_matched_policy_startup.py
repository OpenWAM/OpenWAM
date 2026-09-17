"""Read-only source-pinned startup comparison with matched noise and CFG=1."""

import argparse
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import open_wam
from open_wam.models.policy_variants import PolicyInferContext
from tests.test_unified_policy_inference import CASES, pipeline_for, request, tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_default_device(args.device)
    original_randn = torch.randn

    def noise(*shape, **kwargs):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        generator = torch.Generator(device=kwargs.get("device", args.device))
        # Modality-specific, repeatable inputs independent of call order.
        generator.manual_seed(910 if len(shape) == 5 else 911)
        return original_randn(*shape, **kwargs, generator=generator)

    def noise_like(tensor, **kwargs):
        return noise(
            *tensor.shape,
            device=kwargs.get("device", tensor.device),
            dtype=kwargs.get("dtype", tensor.dtype),
        )

    print("runtime", open_wam.__file__, flush=True)
    captured = {}
    for program, token, objective in CASES:
        pipeline = pipeline_for("dual_expert", program, token)
        pipeline.policy_variant.inference_config = replace(
            pipeline.policy_variant.inference_config, guidance_scale=1.0
        )
        with (
            torch.no_grad(),
            patch.object(torch, "randn", noise),
            patch.object(torch, "randn_like", noise_like),
        ):
            output = pipeline.forward_infer_step_from_latents(
                torch.ones(1, 48, 1, 4, 4),
                PolicyInferContext(
                    state=torch.ones(1, 1, 4), dynamics=request(objective)
                ),
                text_context=torch.ones(1, 3, 16),
            )
        name = f"{program.value}_{token}_{objective}"
        for modality, tensor in zip(("action", "video"), tensors(output), strict=True):
            assert torch.isfinite(tensor).all()
            captured[f"{name}.{modality}"] = tensor.detach().cpu().contiguous()
        print(name, flush=True)
        del pipeline
    save_file(captured, args.output)


if __name__ == "__main__":
    main()
