from __future__ import annotations

import torch

from open_wam.models.policy_variants.common.timesteps import build_scalar_timestep_embedding, expand_token_timestep_context

from .packing import ParallelPackedSequenceLayout


def build_parallel_timestep_context(
    batch_size: int,
    layout: ParallelPackedSequenceLayout,
    hidden_size: int,
    device: torch.device,
    video_scalar: torch.Tensor,
    action_scalar: torch.Tensor,
) -> torch.Tensor:
    contexts = []
    video_embed = build_scalar_timestep_embedding(video_scalar.to(device=device), hidden_size)
    action_embed = build_scalar_timestep_embedding(action_scalar.to(device=device), hidden_size)
    zero_embed = torch.zeros_like(video_embed)
    for name, (start, end) in layout.spans.items():
        length = end - start
        if name == "video_noisy":
            contexts.append(expand_token_timestep_context(video_embed, length))
        elif name == "video_condition":
            contexts.append(expand_token_timestep_context(zero_embed, length))
        elif name == "action_noisy":
            contexts.append(expand_token_timestep_context(action_embed, length))
        else:
            contexts.append(expand_token_timestep_context(zero_embed, length))
    return torch.cat(contexts, dim=1)
