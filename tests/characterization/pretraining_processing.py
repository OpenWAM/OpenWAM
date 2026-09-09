"""Frozen RGB preparation and encoding-input numerics for PR #65."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file


def capture(frames_module, layout_module, vae_module):
    generator = np.random.default_rng(274)
    rgb = generator.integers(0, 256, size=(9, 24, 32, 3), dtype=np.uint8)
    tensors = {}
    for mode in ("letterbox_pad", "center_crop"):
        tensors[f"fit.{mode}"] = frames_module.fit_frames(rgb, 32, 32, mode)
    for count, src, dst in ((18, 30, 15), (30, 20, 15), (11, 10, 15)):
        tensors[f"indices.{count}.{src}.{dst}"] = torch.tensor(frames_module.resample_indices(count, src, dst))
    layouts = {}
    for views in (2, 3):
        sizes = [(str(i), 32, 24) for i in range(views)]
        layout = layout_module.make_layout(sizes)
        layouts[str(views)] = layout.to_dict()
        images = {name: rgb[int(name)] for name, _, _ in sizes}
        tensors[f"mosaic.{views}"] = torch.from_numpy(layout_module.render_frame(images, layout))

    class EncoderProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.config = SimpleNamespace(latents_mean=[.1, .2, .3], latents_std=[.7, .8, .9])

        def encode(self, batch):
            tensors["vae.input"] = batch.detach().float().clone()
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: batch[:, :, ::4]))

    videos = [torch.from_numpy(rgb[:n]).permute(0, 3, 1, 2).float() / 255 for n in (5, 9)]
    encoded = vae_module.encode_batch(EncoderProbe(), videos, normalize=True)
    tensors.update({f"vae.output.{i}": value for i, value in enumerate(encoded)})
    return {name: value.contiguous() for name, value in tensors.items()}, layouts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.reference / "scripts/pretraining/encoding"), str(args.reference / "scripts/pretraining/multiview")]
    encoder = importlib.import_module("encode_latents")
    tensors, layouts = capture(encoder, importlib.import_module("layout"), encoder)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("processing.safetensors", "layouts.json"):
        if (args.out / name).exists():
            raise FileExistsError(args.out / name)
    save_file(tensors, args.out / "processing.safetensors", metadata={"source_commit": "ea0b07993c4a2dfe63b03e318fe19c09e45a954e"})
    (args.out / "layouts.json").write_text(json.dumps(layouts, sort_keys=True, indent=2) + "\n")
    print(f"Captured {len(tensors)} tensors and {len(layouts)} layouts", flush=True)
