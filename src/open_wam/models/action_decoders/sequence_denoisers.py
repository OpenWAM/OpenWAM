from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.configs import SequenceDenoiserFamily
from .vpp_replicas import DiffusionTransformerReplica


@dataclass
class PreparedSequenceMemory:
    """Encoder-side context memory reused across denoising steps."""

    memory: torch.Tensor


def _pool_goal_features(goal_features: torch.Tensor | None) -> torch.Tensor | None:
    if goal_features is None:
        return None
    if goal_features.ndim == 3:
        return goal_features.mean(dim=1)
    if goal_features.ndim == 2:
        return goal_features
    raise ValueError(
        "Sequence denoisers expect goal features shaped `[B, L, D]` or `[B, D]`, "
        f"got {tuple(goal_features.shape)}"
    )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        scale = math.log(10_000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -scale)
        embeddings = x[:, None].float() * frequencies[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class SequenceDenoiser(nn.Module, ABC):
    """Shared denoiser interface for sequence-native action decoders."""

    @abstractmethod
    def prepare_context(
        self,
        *,
        observation_tokens: torch.Tensor,
        goal_features: torch.Tensor | None,
        state_tokens: torch.Tensor | None,
    ) -> PreparedSequenceMemory:
        """Encode observation-conditioned context reused by multiple denoise steps."""

    @abstractmethod
    def denoise_actions(
        self,
        *,
        context: PreparedSequenceMemory,
        noised_actions: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Predict denoised actions from noisy actions and prepared context."""


class GenericTransformerSequenceDenoiser(SequenceDenoiser):
    """Current default denoiser based on stock PyTorch encoder/decoder blocks."""

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        goal_input_dim: int | None,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.goal_token_proj = nn.Linear(goal_input_dim or hidden_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.sigma_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Mish(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.action_emb = nn.Linear(action_dim, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.action_pred = nn.Linear(hidden_size, action_dim)

    def prepare_context(
        self,
        *,
        observation_tokens: torch.Tensor,
        goal_features: torch.Tensor | None,
        state_tokens: torch.Tensor | None,
    ) -> PreparedSequenceMemory:
        goal_token = _pool_goal_features(goal_features)
        if goal_token is not None:
            goal_token = self.goal_token_proj(goal_token)[:, None, :]
        else:
            goal_token = observation_tokens.new_zeros(observation_tokens.shape[0], 1, self.hidden_size)
        context_tokens = [goal_token, observation_tokens]
        if state_tokens is not None:
            context_tokens.append(state_tokens)
        memory = self.context_encoder(torch.cat(context_tokens, dim=1))
        return PreparedSequenceMemory(memory=self.context_norm(memory))

    def denoise_actions(
        self,
        *,
        context: PreparedSequenceMemory,
        noised_actions: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        sigma_hidden = self.sigma_emb(sigma)
        action_hidden = self.action_emb(noised_actions) + sigma_hidden[:, None, :].to(noised_actions.dtype)
        decoded = self.action_decoder(action_hidden, context.memory)
        return self.action_pred(decoded)


class _LayerNormNoBias(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class _ResidualMLP(nn.Module):
    def __init__(self, hidden_size: int, *, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Sequence denoiser attention requires hidden_size divisible by num_heads, "
                f"got hidden_size={hidden_size}, num_heads={num_heads}."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.causal = causal
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, *, context: torch.Tensor | None = None) -> torch.Tensor:
        key_value_source = x if context is None else context
        q = self._reshape_heads(self.query(x))
        k = self._reshape_heads(self.key(key_value_source))
        v = self._reshape_heads(self.value(key_value_source))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and context is None,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.hidden_size)
        return self.proj(attended)


class _ContextEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = _LayerNormNoBias(hidden_size)
        self.attn = _Attention(hidden_size, num_heads=num_heads, dropout=dropout, causal=False)
        self.mlp_norm = _LayerNormNoBias(hidden_size)
        self.mlp = _ResidualMLP(hidden_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class _AdaLNZero(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.modulation(condition).chunk(6, dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return shift + (x * scale)


class _FiLMDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = _LayerNormNoBias(hidden_size)
        self.self_attn = _Attention(hidden_size, num_heads=num_heads, dropout=dropout, causal=True)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = _Attention(hidden_size, num_heads=num_heads, dropout=dropout, causal=False)
        self.mlp_norm = _LayerNormNoBias(hidden_size)
        self.mlp = _ResidualMLP(hidden_size, dropout=dropout)
        self.adaln_zero = _AdaLNZero(hidden_size)

    def forward(self, x: torch.Tensor, *, sigma_context: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_zero(sigma_context)
        shift_msa = shift_msa[:, None, :]
        scale_msa = scale_msa[:, None, :]
        gate_msa = gate_msa[:, None, :]
        shift_mlp = shift_mlp[:, None, :]
        scale_mlp = scale_mlp[:, None, :]
        gate_mlp = gate_mlp[:, None, :]

        x_attn = _modulate(self.self_norm(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(x_attn)
        x = x + self.cross_attn(self.cross_norm(x), context=memory)
        x_mlp = _modulate(self.mlp_norm(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)
        return x


class FiLMDiffusionTransformerSequenceDenoiser(SequenceDenoiser):
    """Local `DiffusionTransformer` replica behind the generic denoiser interface."""

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.replica = DiffusionTransformerReplica(
            hidden_size=hidden_size,
            action_dim=action_dim,
            num_heads=num_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
        )

    def prepare_context(
        self,
        *,
        observation_tokens: torch.Tensor,
        goal_features: torch.Tensor | None,
        state_tokens: torch.Tensor | None,
    ) -> PreparedSequenceMemory:
        memory = self.replica.forward_enc_only(
            states={
                "state_images": observation_tokens,
                "state_obs": state_tokens,
            },
            goals=goal_features,
            uncond=False,
        )
        return PreparedSequenceMemory(memory=memory)

    def denoise_actions(
        self,
        *,
        context: PreparedSequenceMemory,
        noised_actions: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return self.replica.forward_dec_only(
            context=context.memory,
            actions=noised_actions,
            sigma=sigma,
        )


def build_sequence_denoiser(
    family: SequenceDenoiserFamily | str,
    *,
    hidden_size: int,
    action_dim: int,
    goal_input_dim: int | None = None,
    num_heads: int,
    encoder_layers: int,
    decoder_layers: int,
    dropout: float = 0.0,
) -> SequenceDenoiser:
    resolved = SequenceDenoiserFamily(family)
    if resolved == SequenceDenoiserFamily.GENERIC_TRANSFORMER:
        return GenericTransformerSequenceDenoiser(
            hidden_size=hidden_size,
            action_dim=action_dim,
            goal_input_dim=goal_input_dim,
            num_heads=num_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
        )
    if resolved == SequenceDenoiserFamily.FILM_DIFFUSION_TRANSFORMER:
        return FiLMDiffusionTransformerSequenceDenoiser(
            hidden_size=hidden_size,
            action_dim=action_dim,
            num_heads=num_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported sequence denoiser family '{resolved}'.")
