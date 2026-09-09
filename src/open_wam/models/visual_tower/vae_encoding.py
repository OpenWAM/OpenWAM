"""Vae encoding."""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLWan


def load_vae(path: str, device: torch.device, dtype: torch.dtype) -> AutoencoderKLWan:
    vae = AutoencoderKLWan.from_pretrained(str(path), torch_dtype=dtype)
    vae = vae.to(device=device, dtype=dtype).eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_clip(
    vae: AutoencoderKLWan, video: torch.Tensor, *, normalize: bool
) -> torch.Tensor:
    """float32 [T, 3, H, W] in [0, 1] -> float32 [T', H', W', C] latent."""

    return encode_batch(vae, [video], normalize=normalize)[0]


@torch.no_grad()
def encode_batch(
    vae: AutoencoderKLWan, videos: list[torch.Tensor], *, normalize: bool
) -> list[torch.Tensor]:
    """Encode several equally-shaped clips in one pass.

    The VAE walks frames four at a time through a Python loop, so a lone clip
    leaves the card idle between tiny launches. Batching gives each launch more
    to do without changing the number of iterations.

    Clips of different lengths still batch: the VAE is causal, so frames appended
    after a clip ends cannot influence the latents before it. Padding up to the
    batch's longest clip and trimming each result back is exact.

    The [0, 1] -> [-1, 1] rescale happens in float32 before the cast to the VAE
    dtype; doing that arithmetic in bf16 measurably perturbs the result.
    """

    if not videos:
        return []
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    lengths = [int(v.shape[0]) for v in videos]
    longest = max(lengths)
    padded = [
        v
        if v.shape[0] == longest
        else torch.cat([v, v[-1:].expand(longest - v.shape[0], -1, -1, -1)])
        for v in videos
    ]

    batch = torch.stack(padded).to(device=device, dtype=torch.float32)
    batch = (
        (batch * 2.0 - 1.0).to(dtype=dtype).permute(0, 2, 1, 3, 4)
    )  # [B, 3, T, H, W]

    latents = vae.encode(batch).latent_dist.mode()  # [B, C, T', H', W']

    if normalize:
        mean = torch.tensor(vae.config.latents_mean, device=latents.device).view(
            1, -1, 1, 1, 1
        )
        std = torch.tensor(vae.config.latents_std, device=latents.device).view(
            1, -1, 1, 1, 1
        )
        latents = (latents.float() - mean) * (1.0 / std)

    latents = latents.float().permute(0, 2, 3, 4, 1).contiguous()  # [B, T', H', W', C]
    return [latents[i, : 1 + (length - 1) // 4] for i, length in enumerate(lengths)]
