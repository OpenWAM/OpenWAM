"""Pack independent video sequences for the shared single-stream transformer.

Embedding and output transport retain the batch shape. Only real video tokens
enter the heavy layers; self- and cross-attention never cross sample boundaries.
No learned parameters or checkpoint keys are introduced by this execution path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.checkpoint import checkpoint

from open_wam.models.common.sequence_batch_attention import build_sequence_id_attention
from open_wam.models.common.attention_contracts import (
    AttentionProfileSpec,
    PreparedAttentionProfile,
)
from open_wam.models.common.video_geometry import unpatchify_video_sequence

from .shared_transformer_layout import select_chunk_slices


def packed_video_tokens(
    core,
    inputs: dict[str, torch.Tensor],
    *,
    sequence_lengths: Sequence[int],
    use_activation_checkpointing: bool = False,
) -> torch.Tensor:
    """Run one transformer call per layer and restore zero-filled padding.

    Each sample's video positions and text context keep their original indices.
    Prefix/future visibility and loss masks remain owned by the calling policy.
    This path supports the existing full-attention prefix/suffix video recipe.
    """
    latents = inputs["noisy_latents"]
    batch, _, frames, height, width = latents.shape
    if core.patch_size[0] != 1:
        raise ValueError("Packed video execution currently requires temporal patch size 1.")
    lengths = tuple(sequence_lengths)
    if len(lengths) != batch or any(
        isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= frames
        for n in lengths
    ):
        raise ValueError("Packed video requires one valid latent length per sample.")
    if height % core.patch_size[1] or width % core.patch_size[2]:
        raise ValueError("Video spatial dimensions must align with the patch size.")
    if getattr(core, "_runtime_block_devices", None):
        raise ValueError("Packed training requires rank-local transformer blocks.")
    prepared = core.prepare_exact_single_stream_inputs(inputs, action_mode=False)
    tokens_per_frame = (height // core.patch_size[1]) * (width // core.patch_size[2])
    capacity = frames * tokens_per_frame
    counts = tuple(n * tokens_per_frame for n in lengths)
    index = torch.cat([
        torch.arange(n, device=latents.device) + i * capacity
        for i, n in enumerate(counts)
    ])

    def pack(value: torch.Tensor) -> torch.Tensor:
        if value.shape[:2] != (batch, capacity):
            raise ValueError("Prepared video embeddings disagree with token geometry.")
        return value.flatten(0, 1).index_select(0, index).unsqueeze(0)

    hidden = pack(prepared["hidden_states"])
    temb = pack(prepared["temb"])
    timestep_proj = pack(prepared["timestep_proj"])
    rotary = pack(prepared["rotary_emb"])
    text = prepared["text_hidden_states"]
    sample_ids = torch.arange(batch, device=latents.device)
    video_ids = torch.repeat_interleave(
        sample_ids, torch.tensor(counts, device=latents.device), output_size=sum(counts)
    )
    text_ids = sample_ids.repeat_interleave(text.shape[1])
    self_dense, self_sparse = build_sequence_id_attention(video_ids, video_ids)
    cross_dense, cross_sparse = build_sequence_id_attention(video_ids, text_ids)
    profile = PreparedAttentionProfile(
        spec=AttentionProfileSpec("video_sequence_batch", "independent_sequences", "sdpa_or_flex"),
        self_attention_mask=self_dense,
        cross_attention_mask=cross_dense,
        self_attention_block_mask=self_sparse,
        cross_attention_block_mask=cross_sparse,
    )
    kwargs = dict(
        encoder_hidden_states=text.flatten(0, 1).unsqueeze(0),
        temb=timestep_proj,
        rotary_emb=rotary,
        attention_profile=profile,
        # Training does not produce text KV caches. A cache request would also
        # disable sparse cross-attention and break sequence isolation on CUDA.
        cache_text_context=False,
    )
    for block in core.blocks:
        if use_activation_checkpointing and torch.is_grad_enabled():
            hidden, _, _ = checkpoint(block, hidden, use_reentrant=False, **kwargs)
        else:
            hidden, _, _ = block(hidden, **kwargs)
    shift, scale = select_chunk_slices(core.scale_shift_table[None] + temb[:, :, None], 2)
    hidden = (core.norm_out(hidden.float()) * (1.0 + scale) + shift).type_as(hidden)
    projected = core.proj_out(hidden).squeeze(0)
    restored = projected.new_zeros(batch * capacity, projected.shape[-1])
    restored = restored.index_copy(0, index, projected).reshape(batch, capacity, -1)
    return unpatchify_video_sequence(
        core.patch_size, restored, frames, height, width, batch_size=batch
    ).to(latents.dtype)
