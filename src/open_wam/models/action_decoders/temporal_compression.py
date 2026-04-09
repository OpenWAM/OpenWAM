from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.configs import TemporalCompressionAdapterFamily
from open_wam.models.policy_variants.contracts import DecoderSequenceContext
from .vpp_replicas import VideoFormer3DReplica


class TemporalCompressionAdapter(nn.Module, ABC):
    """Shared temporal/token compression interface for sequence decoders.

    The output contract is a generic sequence `[B, S, D]`, where `S` may be a
    frame count, a compressed latent count, or any other decoder-facing token
    length.
    """

    @abstractmethod
    def forward(self, sequence_context: DecoderSequenceContext) -> torch.Tensor:
        """Compress a rich visual sequence context into `[B, S, D]` features."""


class IdentityTemporalCompressionAdapter(TemporalCompressionAdapter):
    """Pass through already-collapsed frame sequences unchanged."""

    def forward(self, sequence_context: DecoderSequenceContext) -> torch.Tensor:
        sequence_tokens = sequence_context.sequence_tokens
        if sequence_tokens.ndim != 3:
            raise ValueError(
                "Identity temporal compression expects `[B, T, D]` tokens, "
                f"got {tuple(sequence_tokens.shape)}"
            )
        return sequence_tokens


class FrameMeanPoolTemporalCompressionAdapter(TemporalCompressionAdapter):
    """Collapse per-frame token grids with a simple mean pool."""

    def forward(self, sequence_context: DecoderSequenceContext) -> torch.Tensor:
        sequence_tokens = sequence_context.sequence_tokens
        if sequence_tokens.ndim == 4:
            return sequence_tokens.mean(dim=2)
        if sequence_tokens.ndim == 3:
            return sequence_tokens
        raise ValueError(
            "Frame-mean temporal compression expects `[B, T, N, D]` or `[B, T, D]`, "
            f"got {tuple(sequence_tokens.shape)}"
        )


class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, latents: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(latents),
            self.kv_norm(features),
            self.kv_norm(features),
            need_weights=False,
        )
        latents = latents + attn_out
        return latents + self.ff(self.ff_norm(latents))


class _TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm(sequence), self.norm(sequence), self.norm(sequence), need_weights=False)
        sequence = sequence + attn_out
        return sequence + self.ff(self.ff_norm(sequence))


class _VideoFormerPerceiverAttention(nn.Module):
    """Closer port of VPP's Perceiver-style latent resampler attention."""

    def __init__(self, hidden_size: int, *, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Video-Former perceiver attention requires hidden_size divisible by num_heads, "
                f"got hidden_size={hidden_size}, num_heads={num_heads}."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.media_norm = nn.LayerNorm(hidden_size)
        self.latent_norm = nn.LayerNorm(hidden_size)
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, features: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        features = self.media_norm(features)
        latents = self.latent_norm(latents)
        q = self._reshape_heads(self.to_q(latents))
        kv_input = torch.cat((features, latents), dim=1)
        k = self._reshape_heads(self.to_k(kv_input))
        v = self._reshape_heads(self.to_v(kv_input))
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attended = attended.permute(0, 2, 1, 3).reshape(latents.shape[0], latents.shape[1], self.hidden_size)
        return self.to_out(attended)


class _VideoFormerAttention(nn.Module):
    """Shared self-attention block used by the closer Video-Former adapter."""

    def __init__(self, hidden_size: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Video-Former attention requires hidden_size divisible by num_heads, "
                f"got hidden_size={hidden_size}, num_heads={num_heads}."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(sequence)
        q = self._reshape_heads(self.q_proj(sequence))
        k = self._reshape_heads(self.k_proj(sequence))
        v = self._reshape_heads(self.v_proj(sequence))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(sequence.shape[0], sequence.shape[1], self.hidden_size)
        return self.out_proj(attended)


class _VideoFormerFeedForward(nn.Module):
    def __init__(self, hidden_size: int, *, dropout: float, mult: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mult, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(self.norm(x))


class TemporalLatentResampler3D(TemporalCompressionAdapter):
    """Per-frame latent resampler with temporal mixing, close to VPP's Video_Former.

    The input is expected to be a frame-major token grid `[B, T, N, D_in]`.
    The adapter learns a fixed number of latent tokens per frame, cross-attends
    those latents to each frame's visual token grid, then performs temporal
    mixing across frames for each latent slot.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        input_dim: int | None = None,
        compressed_tokens_per_frame: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.0,
        max_frames: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.compressed_tokens_per_frame = compressed_tokens_per_frame
        self.max_frames = max_frames
        self.input_proj = nn.Linear(input_dim or hidden_size, hidden_size)
        self.time_pos_emb = nn.Parameter(torch.randn(max_frames, 1, hidden_size) * 0.02)
        self.latents = nn.Parameter(torch.randn(max_frames, compressed_tokens_per_frame, hidden_size) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [_CrossAttentionBlock(hidden_size, num_heads=num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList(
            [_TemporalSelfAttentionBlock(hidden_size, num_heads=num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, sequence_context: DecoderSequenceContext) -> torch.Tensor:
        sequence_tokens = sequence_context.sequence_tokens
        if sequence_tokens.ndim != 4:
            raise ValueError(
                "Temporal latent resampler expects `[B, T, N, D]` visual token grids, "
                f"got {tuple(sequence_tokens.shape)}"
            )
        batch_size, frame_count, token_count, _ = sequence_tokens.shape
        if frame_count > self.max_frames:
            raise ValueError(
                f"Temporal latent resampler only supports up to {self.max_frames} frames, got {frame_count}."
            )
        features = self.input_proj(sequence_tokens)
        time_pos_emb = self.time_pos_emb[:frame_count].unsqueeze(0)
        features = features + time_pos_emb
        frame_features = features.reshape(batch_size * frame_count, token_count, self.hidden_size)
        latents = self.latents[:frame_count].unsqueeze(0).expand(batch_size, -1, -1, -1)
        latents = latents.reshape(batch_size * frame_count, self.compressed_tokens_per_frame, self.hidden_size)

        for cross_block, temporal_block in zip(self.cross_blocks, self.temporal_blocks):
            latents = cross_block(latents, frame_features)
            temporal_input = latents.reshape(batch_size, frame_count, self.compressed_tokens_per_frame, self.hidden_size)
            temporal_input = temporal_input.permute(0, 2, 1, 3).reshape(
                batch_size * self.compressed_tokens_per_frame,
                frame_count,
                self.hidden_size,
            )
            temporal_input = temporal_block(temporal_input)
            latents = temporal_input.reshape(
                batch_size,
                self.compressed_tokens_per_frame,
                frame_count,
                self.hidden_size,
            ).permute(0, 2, 1, 3).reshape(
                batch_size * frame_count,
                self.compressed_tokens_per_frame,
                self.hidden_size,
            )

        latents = latents.reshape(batch_size, frame_count * self.compressed_tokens_per_frame, self.hidden_size)
        return self.output_norm(latents)


class VideoFormer3DTemporalCompressionAdapter(TemporalCompressionAdapter):
    """Local `Video_Former_3D` replica behind the generic compression interface."""

    def __init__(
        self,
        *,
        hidden_size: int,
        compressed_tokens_per_frame: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.0,
        max_frames: int = 32,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.replica = VideoFormer3DReplica(
            hidden_size=hidden_size,
            depth=depth,
            compressed_tokens_per_frame=compressed_tokens_per_frame,
            max_frames=max_frames,
            dim_head=max(1, hidden_size // max(1, num_heads)),
            heads=num_heads,
            dropout=dropout,
        )

    def forward(self, sequence_context: DecoderSequenceContext) -> torch.Tensor:
        sequence_tokens = sequence_context.sequence_tokens
        if sequence_tokens.ndim != 4:
            raise ValueError(
                "Video-Former temporal compression expects `[B, T, N, D]` visual token grids, "
                f"got {tuple(sequence_tokens.shape)}"
            )
        _, frame_count, _, _ = sequence_tokens.shape
        if frame_count > self.max_frames:
            raise ValueError(
                f"Video-Former temporal compression supports up to {self.max_frames} frames, got {frame_count}."
            )
        return self.replica(sequence_tokens)


def build_temporal_compression_adapter(
    family: TemporalCompressionAdapterFamily | str,
    *,
    hidden_size: int | None = None,
    input_dim: int | None = None,
    compressed_tokens_per_frame: int = 2,
    depth: int = 2,
    num_heads: int = 8,
    dropout: float = 0.0,
    max_frames: int = 32,
) -> TemporalCompressionAdapter:
    resolved = TemporalCompressionAdapterFamily(family)
    if resolved == TemporalCompressionAdapterFamily.IDENTITY:
        return IdentityTemporalCompressionAdapter()
    if resolved == TemporalCompressionAdapterFamily.FRAME_MEAN_POOL:
        return FrameMeanPoolTemporalCompressionAdapter()
    if resolved == TemporalCompressionAdapterFamily.TEMPORAL_LATENT_RESAMPLER_3D:
        if hidden_size is None:
            raise ValueError("Temporal latent resampler requires `hidden_size`.")
        return TemporalLatentResampler3D(
            hidden_size=hidden_size,
            input_dim=input_dim,
            compressed_tokens_per_frame=compressed_tokens_per_frame,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            max_frames=max_frames,
        )
    if resolved == TemporalCompressionAdapterFamily.VIDEO_FORMER_3D:
        if hidden_size is None:
            raise ValueError("Video-Former temporal compression requires `hidden_size`.")
        return VideoFormer3DTemporalCompressionAdapter(
            hidden_size=hidden_size,
            compressed_tokens_per_frame=compressed_tokens_per_frame,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            max_frames=max_frames,
        )
    raise ValueError(f"Unsupported temporal compression adapter family '{resolved}'.")
