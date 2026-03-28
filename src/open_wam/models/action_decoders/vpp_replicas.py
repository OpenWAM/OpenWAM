from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import UninitializedParameter


class ReplicaLayerNorm(nn.Module):
    """LayerNorm with optional bias, matching the upstream VPP blocks."""

    def __init__(self, dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def _feed_forward_layer(dim: int, *, mult: int = 4, dropout: float = 0.0) -> nn.Module:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class ReplicaAttention(nn.Module):
    """Local port of the VPP transformer attention block."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        attn_dropout: float,
        resid_dropout: float,
        causal: bool,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Replica attention requires hidden_size divisible by num_heads, "
                f"got hidden_size={hidden_size}, num_heads={num_heads}."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.attn_dropout = attn_dropout
        self.resid_dropout = nn.Dropout(resid_dropout)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        custom_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = x if context is None else context
        q = self._reshape_heads(self.query(x))
        k = self._reshape_heads(self.key(source))
        v = self._reshape_heads(self.value(source))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=custom_attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.causal and context is None,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.hidden_size)
        return self.resid_dropout(self.proj(attended))


class ReplicaMLP(nn.Module):
    def __init__(self, hidden_size: int, *, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_size * 4, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class ReplicaBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        attn_dropout: float,
        resid_dropout: float,
        mlp_dropout: float,
        causal: bool,
        use_cross_attention: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.ln_1 = ReplicaLayerNorm(hidden_size, bias=bias)
        self.attn = ReplicaAttention(
            hidden_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            causal=causal,
            bias=bias,
        )
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.ln_3 = ReplicaLayerNorm(hidden_size, bias=bias)
            self.cross_attn = ReplicaAttention(
                hidden_size,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                causal=False,
                bias=bias,
            )
        self.ln_2 = ReplicaLayerNorm(hidden_size, bias=bias)
        self.mlp = ReplicaMLP(hidden_size, bias=bias, dropout=mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        custom_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_attn(self.ln_3(x), context=context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class ReplicaAdaLNZero(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True),
        )

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.modulation(condition).chunk(6, dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return shift + (x * scale)


class ReplicaConditionedBlock(ReplicaBlock):
    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        attn_dropout: float,
        resid_dropout: float,
        mlp_dropout: float,
        causal: bool,
        use_cross_attention: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            mlp_dropout=mlp_dropout,
            causal=causal,
            use_cross_attention=use_cross_attention,
            bias=bias,
        )
        self.ada_ln_zero = ReplicaAdaLNZero(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        condition: torch.Tensor,
        context: torch.Tensor | None = None,
        custom_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln_zero(condition)
        if shift_msa.ndim == 3 and shift_msa.shape[1] == 1:
            shift_msa = shift_msa[:, 0, :]
            scale_msa = scale_msa[:, 0, :]
            gate_msa = gate_msa[:, 0, :]
            shift_mlp = shift_mlp[:, 0, :]
            scale_mlp = scale_mlp[:, 0, :]
            gate_mlp = gate_mlp[:, 0, :]
        shift_msa = shift_msa[:, None, :]
        scale_msa = scale_msa[:, None, :]
        gate_msa = gate_msa[:, None, :]
        shift_mlp = shift_mlp[:, None, :]
        scale_mlp = scale_mlp[:, None, :]
        gate_mlp = gate_mlp[:, None, :]

        x_attn = _modulate(self.ln_1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_attn, custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_attn(self.ln_3(x), context=context, custom_attn_mask=custom_attn_mask)
        x_mlp = _modulate(self.ln_2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)
        return x


class ReplicaTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        num_layers: int,
        attn_dropout: float,
        resid_dropout: float,
        mlp_dropout: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ReplicaBlock(
                    hidden_size,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    mlp_dropout=mlp_dropout,
                    causal=False,
                    use_cross_attention=False,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = ReplicaLayerNorm(hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ReplicaTransformerFiLMDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        num_layers: int,
        attn_dropout: float,
        resid_dropout: float,
        mlp_dropout: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ReplicaConditionedBlock(
                    hidden_size,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    mlp_dropout=mlp_dropout,
                    causal=True,
                    use_cross_attention=True,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = ReplicaLayerNorm(hidden_size, bias=bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, condition=condition, context=context)
        return self.norm(x)


class ReplicaSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10_000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionTransformerReplica(nn.Module):
    """Local port of VPP's DiffusionTransformer for action denoising."""

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        dropout: float = 0.0,
        goal_conditioned: bool = True,
        goal_drop: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.goal_conditioned = goal_conditioned
        self.goal_drop = goal_drop

        self.tok_emb = nn.LazyLinear(hidden_size)
        self.goal_emb = nn.Sequential(
            nn.LazyLinear(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.lang_emb = nn.LazyLinear(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.proprio_drop = nn.Dropout(0.5)
        self.proprio_emb = nn.Sequential(
            nn.LazyLinear(hidden_size * 2),
            nn.Mish(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.encoder = ReplicaTransformerEncoder(
            hidden_size,
            num_heads=num_heads,
            num_layers=encoder_layers,
            attn_dropout=dropout,
            resid_dropout=dropout,
            mlp_dropout=dropout,
            bias=bias,
        )
        self.decoder = ReplicaTransformerFiLMDecoder(
            hidden_size,
            num_heads=num_heads,
            num_layers=decoder_layers,
            attn_dropout=dropout,
            resid_dropout=dropout,
            mlp_dropout=dropout,
            bias=bias,
        )
        self.sigma_emb = nn.Sequential(
            ReplicaSinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Mish(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.action_emb = nn.Linear(action_dim, hidden_size)
        self.action_pred = nn.Linear(hidden_size, action_dim)
        self.latent_encoder_emb: torch.Tensor | None = None
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if isinstance(module.weight, UninitializedParameter):
                return
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                if isinstance(module.bias, UninitializedParameter):
                    return
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _process_sigma_embeddings(self, sigma: torch.Tensor) -> torch.Tensor:
        sigmas = sigma.log() / 4
        embeddings = self.sigma_emb(sigmas)
        if embeddings.ndim == 2:
            embeddings = embeddings[:, None, :]
        return embeddings

    def _mask_goal(self, goal: torch.Tensor, *, force_mask: bool = False) -> torch.Tensor:
        if force_mask:
            return torch.zeros_like(goal)
        if self.training and self.goal_drop > 0.0:
            mask = torch.bernoulli(torch.ones_like(goal) * self.goal_drop)
            return goal * (1.0 - mask)
        return goal

    def _preprocess_goals(
        self,
        goals: torch.Tensor | None,
        *,
        state_length: int,
        uncond: bool = False,
    ) -> torch.Tensor | None:
        if goals is None:
            return None
        if goals.ndim == 2:
            goals = goals[:, None, :]
        if goals.shape[1] == state_length:
            goals = goals[:, :1, :]
        goals = self._mask_goal(goals, force_mask=uncond)
        return goals

    def _process_state_embeddings(
        self,
        states: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state_embed = self.tok_emb(states["state_images"])
        proprio = states.get("state_obs")
        proprio_embed = self.proprio_emb(proprio) if proprio is not None else None
        return state_embed, proprio_embed

    def forward_enc_only(
        self,
        *,
        states: dict[str, torch.Tensor],
        goals: torch.Tensor | None,
        uncond: bool = False,
    ) -> torch.Tensor:
        goals = self._preprocess_goals(goals, state_length=states["state_images"].shape[1], uncond=uncond)
        state_embed, proprio_embed = self._process_state_embeddings(states)
        components: list[torch.Tensor] = []
        if self.goal_conditioned:
            if goals is None:
                goal_embed = state_embed.new_zeros(state_embed.shape[0], 1, self.hidden_size)
            else:
                goal_embed = self.lang_emb(goals)
            components.append(goal_embed)
        components.append(state_embed)
        if proprio_embed is not None:
            components.append(self.proprio_drop(proprio_embed))
        context = self.encoder(torch.cat(components, dim=1))
        self.latent_encoder_emb = context
        return context

    def forward_dec_only(
        self,
        *,
        context: torch.Tensor,
        actions: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        sigma_context = self._process_sigma_embeddings(sigma)
        action_x = self.drop(self.action_emb(actions))
        decoded = self.decoder(action_x, sigma_context, context)
        return self.action_pred(decoded)


class ReplicaPerceiverAttentionLayer(nn.Module):
    """Local port of VPP's PerceiverAttentionLayer."""

    def __init__(self, dim: int, *, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        batch_size, n_features, _ = features.shape
        n_queries = latents.shape[1]
        x = self.norm_media(features)
        latents = self.norm_latents(latents)
        q = self.to_q(latents).reshape(batch_size, n_queries, self.heads, self.dim_head).permute(0, 2, 1, 3)
        kv_input = torch.cat((x, latents), dim=1)
        total_kv = kv_input.shape[1]
        k = self.to_k(kv_input).reshape(batch_size, total_kv, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(kv_input).reshape(batch_size, total_kv, self.heads, self.dim_head).permute(0, 2, 1, 3)
        sim = torch.einsum("bhqd,bhkd->bhqk", q * self.scale, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", alphas, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, n_queries, self.heads * self.dim_head)
        return self.to_out(out)


class ReplicaTemporalAttentionLayer(nn.Module):
    """Local port of VPP's temporal attention block used inside Video_Former_3D."""

    def __init__(self, dim: int, *, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = features.shape
        x = self.norm_media(features)
        q = self.to_q(x).reshape(batch_size, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = self.to_k(x).reshape(batch_size, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(batch_size, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3)
        sim = torch.einsum("bhqd,bhkd->bhqk", q * self.scale, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", alphas, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.heads * self.dim_head)
        return self.to_out(out)


class VideoFormer3DReplica(nn.Module):
    """Local port of VPP's `Video_Former_3D` with temporal mixing enabled."""

    def __init__(
        self,
        *,
        hidden_size: int,
        depth: int,
        compressed_tokens_per_frame: int,
        max_frames: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.compressed_tokens_per_frame = compressed_tokens_per_frame
        self.max_frames = max_frames
        self.goal_emb = nn.Sequential(
            nn.LazyLinear(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.latents = nn.Parameter(torch.randn(max_frames, compressed_tokens_per_frame, hidden_size))
        self.time_pos_emb = nn.Parameter(torch.randn(max_frames, 1, hidden_size))
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    (
                        ReplicaPerceiverAttentionLayer(hidden_size, dim_head=dim_head, heads=heads),
                        ReplicaTemporalAttentionLayer(hidden_size, dim_head=dim_head, heads=heads),
                        _feed_forward_layer(hidden_size, mult=ff_mult, dropout=dropout),
                    )
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x_f: torch.Tensor) -> torch.Tensor:
        if x_f.ndim != 4:
            raise ValueError(
                "VideoFormer3DReplica expects `[B, T, N, D]` features, "
                f"got {tuple(x_f.shape)}"
            )
        batch_size, frame_count, token_count, _ = x_f.shape
        if frame_count > self.max_frames:
            raise ValueError(
                f"VideoFormer3DReplica supports up to {self.max_frames} frames, got {frame_count}."
            )
        time_pos = self.time_pos_emb[:frame_count].unsqueeze(0).expand(batch_size, -1, -1, -1)
        x_f = self.goal_emb(x_f) + time_pos
        x_f = x_f.reshape(batch_size * frame_count, token_count, self.hidden_size)
        x = self.latents[:frame_count].unsqueeze(0).expand(batch_size, -1, -1, -1)
        x = x.reshape(batch_size * frame_count, self.compressed_tokens_per_frame, self.hidden_size)
        for perceiver_attn, temporal_attn, feed_forward in self.layers:
            x = x + perceiver_attn(x_f, x)
            x = x.reshape(batch_size, frame_count, self.compressed_tokens_per_frame, self.hidden_size)
            x = x.permute(0, 2, 1, 3).reshape(batch_size * self.compressed_tokens_per_frame, frame_count, self.hidden_size)
            x = x + temporal_attn(x)
            x = x.reshape(batch_size, self.compressed_tokens_per_frame, frame_count, self.hidden_size)
            x = x.permute(0, 2, 1, 3).reshape(batch_size * frame_count, self.compressed_tokens_per_frame, self.hidden_size)
            x = x + feed_forward(x)
        x = x.reshape(batch_size, frame_count, self.compressed_tokens_per_frame, self.hidden_size)
        x = x.reshape(batch_size, frame_count * self.compressed_tokens_per_frame, self.hidden_size)
        return self.norm(x)
