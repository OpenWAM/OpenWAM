from __future__ import annotations

import math
from dataclasses import replace

import torch
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch import nn

from open_wam.models.common import (
    PreparedAttentionProfile,
    build_chunked_temporal_exact_attention_profile,
    cache_backend_uses_slot_pool,
    clear_cache_backend_payload,
    init_cache_backend_payload,
    materialize_cache_backend_entries,
    merge_attention_cache_entries as _merge_attention_cache_entries,
    packed_slot_pool_query_sequence_ids as _packed_slot_pool_query_sequence_ids,
    prepend_cached_prefix_mask as _prepend_cached_prefix_mask,
    prepare_sdpa_mask as _prepare_sdpa_mask,
    resolve_cache_backend_spec,
    resolve_slot_pool_prefix_visibility as _resolve_slot_pool_prefix_visibility,
    retained_slot_pool_indices_for_current_write as _retained_slot_pool_indices_for_current_write,
    SlotPoolLayerState,
    unpatchify_video_tokens,
)
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.models.video_backbone.contracts import (
    AttentionCacheEntry,
    CacheBranchState,
    CacheState,
    CacheUpdateMetadata,
    replace_cache_branch_state,
    resolve_cache_branch_state,
)

from .contracts import (
    VisualCoreInput,
    VisualIntermediateReadout,
    VisualCoreOutput,
)
from .checkpoint_compat import RuntimeStreamCompatibilityParameters
from .context_encoders import (
    GeneralistModeContextEncoder,
    ProprioContextEncoder,
    ProprioHiddenContextEncoder,
)
from .runtime_programs import RuntimeStepInput, RuntimeStepOutput
from .sequence_adapters import (
    PreparedExactTrainSequence,
    prepare_exact_dual_stream_train_sequence,
    prepare_runtime_sequence,
)
from .shared_transformer_support import (
    SharedTransformerAttention,
    SharedTransformerBlock,
    SharedTransformerRotaryPositionalEmbedding,
    SharedTransformerTimeEmbedding,
    apply_rotary_emb as _apply_rotary_emb,
    feed_forward_with_materialized_params as _feed_forward_with_materialized_params,
    layer_norm_with_materialized_params as _layer_norm_with_materialized_params,
    linear_with_materialized_params as _linear_with_materialized_params,
    materialize_runtime_parameter as _materialize_runtime_parameter,
    rms_norm_with_materialized_weight as _rms_norm_with_materialized_weight,
    select_chunk_slices as _select_chunk_slices,
)


def _select_split_segments(tensor: torch.Tensor, lengths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    offset = 0
    segments: list[torch.Tensor] = []
    for length in lengths:
        segments.append(tensor.narrow(1, offset, length).clone())
        offset += length
    return tuple(segments)


class SharedVideoTransformerCore(nn.Module):
    """Shared Wan-style transformer core for all policy variants."""

    def __init__(
        self,
        config: SharedVideoTransformerConfig | None = None,
        *,
        action_dim: int | None = None,
        state_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SharedVideoTransformerConfig()
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"Expected hidden_size {self.config.hidden_size} to be divisible by num_heads {self.config.num_heads}."
            )
        self.action_dim = int(action_dim or 0)
        self.state_dim = int(state_dim or 0)
        self.inner_dim = self.config.hidden_size
        self.ffn_dim = self.config.ffn_dim or (self.config.hidden_size * self.config.mlp_ratio)
        self.patch_size = (
            self.config.patch_size_t,
            self.config.patch_size_h,
            self.config.patch_size_w,
        )
        self.rope = SharedTransformerRotaryPositionalEmbedding(self.config.hidden_size // self.config.num_heads)
        self.time_conditioner = SharedTransformerTimeEmbedding(self.config.hidden_size, self.config.freq_dim)
        self.action_time_conditioner = SharedTransformerTimeEmbedding(self.config.hidden_size, self.config.freq_dim)
        self.text_proj = PixArtAlphaTextProjection(self.config.text_dim, self.config.hidden_size, act_fn="gelu_tanh")
        self.action_text_proj = PixArtAlphaTextProjection(self.config.text_dim, self.config.hidden_size, act_fn="gelu_tanh")
        self.proprio_context_encoder: ProprioContextEncoder | None = None
        self.proprio_hidden_context_encoder: ProprioHiddenContextEncoder | None = None
        self.generalist_mode_context_encoder: GeneralistModeContextEncoder | None = None
        self.patch_embedding_mlp = nn.Linear(
            self.config.latent_channels * self.config.patch_size_t * self.config.patch_size_h * self.config.patch_size_w,
            self.config.hidden_size,
        )
        self.action_embedder = nn.Linear(max(self.action_dim, 1), self.config.hidden_size)
        self.runtime_stream_adapters = RuntimeStreamCompatibilityParameters(
            hidden_size=self.config.hidden_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )
        self.blocks = nn.ModuleList(
            [
                SharedTransformerBlock(
                    dim=self.config.hidden_size,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.config.num_heads,
                    cross_attn_norm=self.config.cross_attn_norm,
                    eps=self.config.latent_norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(self.config.hidden_size, self.config.latent_norm_eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, self.config.hidden_size) / self.config.hidden_size**0.5)
        self.proj_out = nn.Linear(
            self.config.hidden_size,
            self.config.latent_channels * self.config.patch_size_t * self.config.patch_size_h * self.config.patch_size_w,
        )
        self.action_proj_out = nn.Linear(self.config.hidden_size, max(self.action_dim, 1))
        self._exact_runtime_caches: dict[str, CacheState] = {}
        self._runtime_block_devices: tuple[torch.device, ...] = tuple()

    def configure_runtime_block_devices(
        self,
        devices: tuple[torch.device, ...],
        *,
        prep_device: torch.device | None = None,
        output_device: torch.device | None = None,
    ) -> None:
        if not devices:
            self._runtime_block_devices = tuple()
            return
        normalized = tuple(torch.device(device) for device in devices)
        self._runtime_block_devices = normalized
        input_device = torch.device(prep_device) if prep_device is not None else normalized[0]
        output_device = torch.device(output_device) if output_device is not None else normalized[-1]

        for module in (
            self.patch_embedding_mlp,
            self.action_embedder,
            self.time_conditioner,
            self.action_time_conditioner,
            self.text_proj,
            self.action_text_proj,
            self.proprio_context_encoder,
            self.proprio_hidden_context_encoder,
            self.generalist_mode_context_encoder,
            self.runtime_stream_adapters,
            self.rope,
        ):
            if module is None:
                continue
            module.to(device=input_device)

        for layer_index, block in enumerate(self.blocks):
            block.to(device=normalized[layer_index % len(normalized)])

        self.norm_out.to(device=output_device)
        self.proj_out.to(device=output_device)
        self.action_proj_out.to(device=output_device)
        self.scale_shift_table.data = self.scale_shift_table.data.to(device=output_device)

    def configure_proprio_context_encoder(self, *, enabled: bool, state_dim: int | None = None) -> None:
        if not enabled:
            self.proprio_context_encoder = None
            return
        resolved_state_dim = int(self.state_dim if state_dim is None else state_dim)
        if resolved_state_dim <= 0:
            raise ValueError("Proprio context mode requires a positive visual-tower state_dim.")
        if (
            self.proprio_context_encoder is not None
            and self.proprio_context_encoder.state_dim == resolved_state_dim
            and self.proprio_context_encoder.text_dim == self.config.text_dim
        ):
            return
        self.proprio_context_encoder = ProprioContextEncoder(
            state_dim=resolved_state_dim,
            text_dim=self.config.text_dim,
        )

    def configure_proprio_hidden_context_encoder(self, *, enabled: bool, state_dim: int | None = None) -> None:
        if not enabled:
            self.proprio_hidden_context_encoder = None
            return
        resolved_state_dim = int(self.state_dim if state_dim is None else state_dim)
        if resolved_state_dim <= 0:
            raise ValueError("Per-chunk proprio context mode requires a positive visual-tower state_dim.")
        if (
            self.proprio_hidden_context_encoder is not None
            and self.proprio_hidden_context_encoder.state_dim == resolved_state_dim
            and self.proprio_hidden_context_encoder.hidden_size == self.config.hidden_size
        ):
            return
        self.proprio_hidden_context_encoder = ProprioHiddenContextEncoder(
            state_dim=resolved_state_dim,
            hidden_size=self.config.hidden_size,
        )

    def configure_generalist_mode_context_encoder(self, *, enabled: bool) -> None:
        if not enabled:
            self.generalist_mode_context_encoder = None
            return
        if (
            self.generalist_mode_context_encoder is not None
            and self.generalist_mode_context_encoder.text_dim == self.config.text_dim
        ):
            return
        self.generalist_mode_context_encoder = GeneralistModeContextEncoder(text_dim=self.config.text_dim)

    def append_generalist_mode_context_token(
        self,
        text_emb: torch.Tensor,
        mode: object | None,
    ) -> torch.Tensor:
        if mode is None or self.generalist_mode_context_encoder is None:
            return text_emb
        if text_emb.ndim != 3:
            raise ValueError(
                "Generalist mode token appending expects text embeddings with shape [B, tokens, dim], "
                f"got {tuple(text_emb.shape)}."
            )
        if int(text_emb.shape[-1]) != int(self.config.text_dim):
            raise ValueError(
                "Text embedding dim mismatch for generalist mode appending, "
                f"got {text_emb.shape[-1]} and expected {self.config.text_dim}."
            )
        encoder = self.generalist_mode_context_encoder
        mode_tokens = encoder(mode, batch_size=int(text_emb.shape[0])).to(
            device=text_emb.device,
            dtype=text_emb.dtype,
        )
        return torch.cat([text_emb, mode_tokens[:, None, :]], dim=1)

    def append_proprio_context_tokens(
        self,
        text_emb: torch.Tensor,
        proprio_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Deprecated text-space proprio token path; use hidden additive context for new runs."""

        if proprio_state is None or self.proprio_context_encoder is None:
            return text_emb
        if text_emb.ndim != 3:
            raise ValueError(
                "Proprio context appending expects text embeddings with shape [B, tokens, dim], "
                f"got {tuple(text_emb.shape)}."
            )
        if int(text_emb.shape[-1]) != int(self.config.text_dim):
            raise ValueError(
                "Text embedding dim mismatch for proprio appending, "
                f"got {text_emb.shape[-1]} and expected {self.config.text_dim}."
            )
        if proprio_state.ndim == 2:
            proprio_state = proprio_state[:, None, :]
        if proprio_state.ndim != 3:
            raise ValueError(
                "Proprio context appending expects state with shape [B, state_dim] or [B, chunks, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        if int(proprio_state.shape[0]) != int(text_emb.shape[0]):
            raise ValueError(
                "Proprio/text batch mismatch, "
                f"got proprio batch {proprio_state.shape[0]} and text batch {text_emb.shape[0]}."
            )
        encoder = self.proprio_context_encoder
        batch_size, chunk_count, state_dim = proprio_state.shape
        proprio_state = proprio_state.to(device=encoder.proj.weight.device, dtype=encoder.proj.weight.dtype)
        proprio_tokens = encoder(proprio_state.reshape(batch_size * chunk_count, state_dim))
        proprio_tokens = proprio_tokens.reshape(batch_size, chunk_count, -1).to(
            device=text_emb.device,
            dtype=text_emb.dtype,
        )
        return torch.cat([text_emb, proprio_tokens], dim=1)

    def encode_proprio_hidden_context(
        self,
        proprio_state: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        encoder = self.proprio_hidden_context_encoder
        if encoder is None:
            raise ValueError("Per-chunk proprio context mode requires a configured hidden-context encoder.")
        if proprio_state.ndim == 2:
            proprio_state = proprio_state[:, None, :]
        if proprio_state.ndim != 3:
            raise ValueError(
                "Per-chunk proprio hidden context expects state with shape [B, F, state_dim] or [B, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        batch_size, frame_count, state_dim = proprio_state.shape
        proprio_state = proprio_state.to(device=encoder.proj.weight.device, dtype=encoder.proj.weight.dtype)
        hidden_context = encoder(proprio_state.reshape(batch_size * frame_count, state_dim))
        return hidden_context.reshape(batch_size, frame_count, -1).to(device=device, dtype=dtype)

    @staticmethod
    def _move_optional_tensor(tensor: torch.Tensor | None, *, device: torch.device, dtype: torch.dtype | None = None):
        if tensor is None:
            return None
        kwargs = {"device": device}
        if dtype is not None and tensor.is_floating_point():
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs)

    @classmethod
    def _cached_optional_tensor(
        cls,
        tensor: torch.Tensor | None,
        *,
        cache: dict[tuple[str, torch.device, torch.dtype | None], torch.Tensor],
        name: str,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        dtype_key = dtype if dtype is not None and tensor.is_floating_point() else None
        cache_key = (name, torch.device(device), dtype_key)
        cached = cache.get(cache_key)
        if cached is None:
            cached = cls._move_optional_tensor(tensor, device=torch.device(device), dtype=dtype)
            cache[cache_key] = cached
        return cached

    def _move_attention_profile(
        self,
        profile: PreparedAttentionProfile | None,
        *,
        device: torch.device,
    ) -> PreparedAttentionProfile | None:
        if profile is None:
            return None
        device = torch.device(device)
        has_block_masks = (
            profile.self_attention_block_mask is not None
            or profile.cross_attention_block_mask is not None
        )
        if has_block_masks:
            metadata = profile.metadata
            required_keys = (
                "latent_shape",
                "action_shape",
                "padded_length",
                "chunk_size",
                "window_size",
                "text_token_count",
            )
            if all(key in metadata for key in required_keys) and (
                "current_block_coupling" in metadata
                or "allow_joint_noisy_block_attention" in metadata
            ):
                current_block_coupling = metadata.get("current_block_coupling")
                if current_block_coupling is None:
                    current_block_coupling = (
                        "joint"
                        if bool(metadata["allow_joint_noisy_block_attention"])
                        else "video_then_action"
                    )
                return build_chunked_temporal_exact_attention_profile(
                    latent_shape=tuple(int(v) for v in metadata["latent_shape"]),
                    action_shape=tuple(int(v) for v in metadata["action_shape"]),
                    padded_length=int(metadata["padded_length"]),
                    chunk_size=int(metadata["chunk_size"]),
                    window_size=int(metadata["window_size"]),
                    patch_size=self.patch_size,
                    text_token_count=int(metadata["text_token_count"]),
                    base_text_token_count=(
                        None
                        if "base_text_token_count" not in metadata
                        else int(metadata["base_text_token_count"])
                    ),
                    proprio_context_token_count=int(metadata.get("proprio_context_token_count", 0)),
                    chunk_origin_frame=int(metadata.get("chunk_origin_frame", 0)),
                    prefix_condition_frames=int(metadata.get("prefix_condition_frames", 0)),
                    singleton_chunk_frame=(
                        None
                        if metadata.get("singleton_chunk_frame") is None
                        else int(metadata["singleton_chunk_frame"])
                    ),
                    action_context_mask=(
                        torch.tensor(
                            metadata["action_context_valid_tokens"],
                            device=device,
                            dtype=torch.bool,
                        )[None, :]
                        if metadata.get("action_context_valid_tokens") is not None
                        else None
                    ),
                    device=device,
                    build_dense_masks=(
                        profile.self_attention_mask is not None
                        or profile.cross_attention_mask is not None
                    ),
                    build_flex_masks=True,
                    current_block_coupling=str(current_block_coupling),
                    preserve_video_pretrain_history=bool(
                        metadata.get("preserve_video_pretrain_history", False)
                    ),
                    history_stream_visibility=metadata.get("history_stream_visibility"),
                    conditional_history_policy=metadata.get("conditional_history_policy"),
                )
            if profile.self_attention_mask is None and profile.cross_attention_mask is None:
                return profile
            return replace(
                profile,
                self_attention_mask=self._move_optional_tensor(profile.self_attention_mask, device=device),
                cross_attention_mask=self._move_optional_tensor(profile.cross_attention_mask, device=device),
                self_attention_block_mask=None,
                cross_attention_block_mask=None,
            )
        return replace(
            profile,
            self_attention_mask=self._move_optional_tensor(profile.self_attention_mask, device=device),
            cross_attention_mask=self._move_optional_tensor(profile.cross_attention_mask, device=device),
        )

    def _cached_attention_profile(
        self,
        profile: PreparedAttentionProfile | None,
        *,
        cache: dict[torch.device, PreparedAttentionProfile | None],
        device: torch.device,
    ) -> PreparedAttentionProfile | None:
        device = torch.device(device)
        if device not in cache:
            cache[device] = self._move_attention_profile(profile, device=device)
        return cache[device]

    @staticmethod
    def _move_slot_pool_layer_state(
        layer_state: SlotPoolLayerState | None,
        *,
        device: torch.device,
    ) -> SlotPoolLayerState | None:
        if layer_state is None:
            return None
        for name in ("key", "value", "slot_ids", "stream_ids", "slot_mask", "prediction_mask"):
            tensor = getattr(layer_state, name)
            if tensor is not None and tensor.device != device:
                setattr(layer_state, name, tensor.to(device=device))
        return layer_state

    def project_video_tokens_to_latents(
        self,
        *,
        hidden_states: torch.Tensor,
        token_grid,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "Expected shared-core video tokens with shape [B, seq, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        video_patch_prediction = self.proj_out(hidden_states)
        return unpatchify_video_tokens(
            video_patch_prediction,
            token_grid=token_grid,
            latent_channels=self.config.latent_channels,
        )

    def _require_exact_action_dim(self) -> None:
        if self.action_dim <= 0:
            raise ValueError(
                "SharedVideoTransformerCore exact-runtime path requires a positive action_dim. "
                "Construct the shared VisualTower with the experiment action_dim."
            )

    def clear_cache(self, cache_name: str) -> None:
        self._exact_runtime_caches.pop(cache_name, None)

    def clear_pred_cache(self, cache_name: str) -> None:
        cache_state = self._exact_runtime_caches.get(cache_name)
        if cache_state is None:
            return
        cleared_backend = clear_cache_backend_payload(cache_state.backend_payload, clear_predictions_only=True)
        self._exact_runtime_caches[cache_name] = CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=cache_state.cached_frames,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cleared_backend,
            payload=dict(cache_state.payload),
            self_attention_kv=materialize_cache_backend_entries(cleared_backend),
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=cache_state.update_metadata,
        )

    def clear_runtime_cache_state(self, cache_name: str) -> None:
        self.clear_cache(cache_name)

    def clear_runtime_prediction_cache(self, cache_name: str) -> None:
        self.clear_pred_cache(cache_name)

    def create_empty_cache(
        self,
        cache_name: str,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        backend_name: str = "slot_pool_exact",
        prefix_visibility_mode: str = "full_history",
    ) -> None:
        total_tokens = int((attn_window // 2) * latent_token_per_chunk + (attn_window // 2) * action_token_per_chunk)
        backend_spec = resolve_cache_backend_spec(backend_name)
        backend_payload = init_cache_backend_payload(
            backend_spec.name,
            num_layers=len(self.blocks),
            total_tokens=total_tokens,
            num_heads=self.config.num_heads,
            head_dim=self.config.hidden_size // self.config.num_heads,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            metadata={
                "cache_name": cache_name,
                "attn_window": attn_window,
                "latent_token_per_chunk": latent_token_per_chunk,
                "action_token_per_chunk": action_token_per_chunk,
                "prefix_visibility_mode": prefix_visibility_mode,
            },
        )
        self._exact_runtime_caches[cache_name] = CacheState(
            supported=True,
            current_start_frame=0,
            cached_frames=0,
            chunk_size=attn_window,
            capability="self_attn_only",
            backend_name=backend_spec.name,
            backend_payload=backend_payload,
            payload={
                "cache_name": cache_name,
                "attn_window": attn_window,
                "latent_token_per_chunk": latent_token_per_chunk,
                "action_token_per_chunk": action_token_per_chunk,
                "max_tokens": total_tokens,
            },
            self_attention_kv=materialize_cache_backend_entries(backend_payload),
            cross_attention_kv=tuple(),
            update_metadata=CacheUpdateMetadata(),
        )

    def initialize_runtime_cache_backend(
        self,
        cache_name: str,
        *,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        backend_name: str = "slot_pool_exact",
        prefix_visibility_mode: str = "full_history",
    ) -> None:
        self.create_empty_cache(
            cache_name,
            attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            backend_name=backend_name,
            prefix_visibility_mode=prefix_visibility_mode,
        )

    def _resolve_exact_cache_state(self, cache_name: str) -> CacheState | None:
        return self._exact_runtime_caches.get(cache_name)

    def _exact_text_hidden_states(self, text_emb: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return self.text_proj(text_emb.clone()).to(dtype=dtype)

    def prepare_exact_single_stream_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        action_mode: bool,
    ) -> dict[str, torch.Tensor]:
        """Prepare exact-runtime embeddings without executing transformer blocks."""

        noisy_latents = input_dict["noisy_latents"]
        hidden_states = self._input_embed(noisy_latents, input_type="action" if action_mode else "latent")
        text_hidden_states = self._exact_text_hidden_states(input_dict["text_emb"], dtype=hidden_states.dtype)
        rotary_emb = self.rope(input_dict["grid_id"])[:, :, None]
        temb, timestep_proj = self._time_embed(
            input_dict["timesteps"],
            int(noisy_latents.shape[-2]),
            int(noisy_latents.shape[-1]),
            dtype=hidden_states.dtype,
            action_mode=action_mode,
        )
        return {
            "hidden_states": hidden_states,
            "text_hidden_states": text_hidden_states,
            "rotary_emb": rotary_emb,
            "temb": temb,
            "timestep_proj": timestep_proj,
        }

    def _input_embed(self, latents: torch.Tensor, input_type: str = "latent") -> torch.Tensor:
        if input_type == "latent":
            hidden_states = rearrange(
                latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            return self.patch_embedding_mlp(hidden_states.clone())
        if input_type == "action":
            self._require_exact_action_dim()
            hidden_states = rearrange(latents, "b c f h w -> b (f h w) c")
            return self.action_embedder(hidden_states.clone())
        if input_type == "text":
            return self.text_proj(latents.clone())
        raise ValueError(f"Unsupported input_type={input_type!r}")

    def _time_embed(
        self,
        timesteps: torch.Tensor,
        height: int,
        width: int,
        *,
        dtype: torch.dtype,
        action_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_scale_h, patch_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (height // patch_scale_h) * (width // patch_scale_w),
            dim=1,
        )
        conditioner = self.action_time_conditioner if action_mode else self.time_conditioner
        temb, timestep_proj = conditioner(latent_time_steps, dtype=dtype)
        return temb.contiguous().clone(), timestep_proj.contiguous().clone()

    def forward_train(self, input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        prepared = prepare_exact_dual_stream_train_sequence(
            input_dict,
            config=self.config,
            patch_size=self.patch_size,
            model_dtype=self.patch_embedding_mlp.weight.dtype,
            input_embed=lambda tensor, input_type: self._input_embed(tensor, input_type=input_type),
            exact_text_hidden_states=lambda text_emb: self._exact_text_hidden_states(text_emb, dtype=self.patch_embedding_mlp.weight.dtype),
            time_embed=lambda timesteps, height, width, dtype, action_mode: self._time_embed(
                timesteps,
                height,
                width,
                dtype=dtype,
                action_mode=action_mode,
            ),
            rope=self.rope,
        )
        batch_size = prepared.batch_size
        hidden_states = prepared.hidden_states
        text_hidden_states = prepared.text_hidden_states
        rotary_emb = prepared.rotary_emb
        temb = prepared.temb
        timestep_proj = prepared.timestep_proj
        split_list = prepared.split_list
        exact_attention_profile = prepared.attention_profile

        for block in self.blocks:
            hidden_states, _, _ = block(
                hidden_states,
                encoder_hidden_states=text_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                attention_profile=exact_attention_profile,
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = _select_split_segments(
            hidden_states,
            tuple(int(length) for length in split_list),
        )
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=batch_size,
        )
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(
            action_hidden_states,
            "1 (b l) c -> b l c",
            b=batch_size,
        )
        return latent_hidden_states, action_hidden_states

    def _forward_exact_single_stream(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ) -> torch.Tensor:
        prepared = self.prepare_exact_single_stream_inputs(input_dict, action_mode=action_mode)
        hidden_states = prepared["hidden_states"]
        hidden_context = input_dict.get("hidden_context")
        if hidden_context is not None:
            if tuple(hidden_context.shape) != tuple(hidden_states.shape):
                raise ValueError(
                    "Exact single-stream hidden_context must match embedded hidden_states shape, "
                    f"got hidden_context={tuple(hidden_context.shape)}, hidden_states={tuple(hidden_states.shape)}."
                )
            hidden_states = hidden_states + hidden_context.to(device=hidden_states.device, dtype=hidden_states.dtype)
        text_hidden_states = prepared["text_hidden_states"]
        rotary_emb = prepared["rotary_emb"]
        temb = prepared["temb"]
        timestep_proj = prepared["timestep_proj"]
        cache_state = self._resolve_exact_cache_state(cache_name)
        cache_backend_name = cache_state.backend_name if cache_state is not None else None
        cache_backend_payload = cache_state.backend_payload if cache_state is not None else None
        detach_self_attention_cache = (
            bool(cache_state.payload.get("detach_self_attention_cache", True))
            if cache_state is not None
            else True
        )
        cache_current_token_count = 0
        if cache_state is not None and cache_state.update_metadata.update_kv_cache:
            # Exact single-stream cache writes are prefix-style: cache the visible
            # sequence being prefed unless the cache metadata narrows that span.
            tokens_per_frame = int(cache_state.payload.get("tokens_per_frame", 0))
            cached_frames = int(cache_state.cached_frames)
            if tokens_per_frame > 0 and cached_frames > 0:
                cache_current_token_count = tokens_per_frame * cached_frames
            else:
                cache_current_token_count = int(hidden_states.shape[1])
            cache_current_token_count = max(0, min(cache_current_token_count, int(hidden_states.shape[1])))
        next_self_attention_kv: list[AttentionCacheEntry] = []
        attention_mask = input_dict.get("attention_mask")
        cross_attention_mask = input_dict.get("cross_attention_mask")
        stream_id_value = 1 if action_mode else 0
        cache_backend_stream_ids = torch.full(
            (int(hidden_states.shape[1]),),
            stream_id_value,
            device=hidden_states.device,
            dtype=torch.long,
        )
        moved_tensor_cache: dict[tuple[str, torch.device, torch.dtype | None], torch.Tensor] = {}

        for layer_index, block in enumerate(self.blocks):
            block_device = (
                self._runtime_block_devices[layer_index % len(self._runtime_block_devices)]
                if self._runtime_block_devices
                else hidden_states.device
            )
            if hidden_states.device != block_device:
                hidden_states = hidden_states.to(device=block_device)
            block_text_hidden_states = self._cached_optional_tensor(
                text_hidden_states,
                cache=moved_tensor_cache,
                name="text_hidden_states",
                device=block_device,
                dtype=hidden_states.dtype,
            )
            block_timestep_proj = self._cached_optional_tensor(
                timestep_proj,
                cache=moved_tensor_cache,
                name="timestep_proj",
                device=block_device,
                dtype=hidden_states.dtype,
            )
            block_rotary_emb = self._cached_optional_tensor(
                rotary_emb,
                cache=moved_tensor_cache,
                name="rotary_emb",
                device=block_device,
            )
            block_attention_mask = self._cached_optional_tensor(
                attention_mask,
                cache=moved_tensor_cache,
                name="attention_mask",
                device=block_device,
            )
            block_cross_attention_mask = self._cached_optional_tensor(
                cross_attention_mask,
                cache=moved_tensor_cache,
                name="cross_attention_mask",
                device=block_device,
            )
            block_cache_backend_stream_ids = self._cached_optional_tensor(
                cache_backend_stream_ids,
                cache=moved_tensor_cache,
                name="cache_backend_stream_ids",
                device=block_device,
            )
            block_cache_backend_state = (
                cache_backend_payload.layer_states[layer_index]
                if cache_backend_uses_slot_pool(cache_backend_name)
                and cache_backend_payload is not None
                and layer_index < len(cache_backend_payload.layer_states)
                else None
            )
            block_cache_backend_state = self._move_slot_pool_layer_state(block_cache_backend_state, device=block_device)
            hidden_states, current_self_cache_entry, _ = block(
                hidden_states,
                encoder_hidden_states=block_text_hidden_states,
                temb=block_timestep_proj,
                rotary_emb=block_rotary_emb,
                attention_mask=block_attention_mask,
                cross_attention_mask=block_cross_attention_mask,
                self_attention_cache_backend_name=cache_backend_name,
                self_attention_cache_backend_state=block_cache_backend_state,
                cache_current_token_count=cache_current_token_count,
                detach_self_attention_cache=detach_self_attention_cache,
                self_attention_cache_update_mode=update_cache,
                self_attention_cache_stream_ids=block_cache_backend_stream_ids,
            )
            next_self_attention_kv.append(current_self_cache_entry or AttentionCacheEntry())

        output_device = self.scale_shift_table.device
        if hidden_states.device != output_device:
            hidden_states = hidden_states.to(device=output_device)
        temb = temb.to(device=output_device, dtype=hidden_states.dtype)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)

        if cache_state is not None:
            materialized_entries = (
                materialize_cache_backend_entries(cache_backend_payload)
                if cache_backend_uses_slot_pool(cache_backend_name)
                else tuple(next_self_attention_kv)
            )
            self._exact_runtime_caches[cache_name] = CacheState(
                supported=cache_state.supported,
                current_start_frame=cache_state.current_start_frame,
                cached_frames=cache_state.cached_frames,
                chunk_size=cache_state.chunk_size,
                capability=cache_state.capability,
                backend_name=cache_state.backend_name,
                backend_payload=cache_backend_payload,
                payload=dict(cache_state.payload),
                self_attention_kv=materialized_entries,
                cross_attention_kv=cache_state.cross_attention_kv,
                update_metadata=cache_state.update_metadata,
            )

        if action_mode:
            return self.action_proj_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return rearrange(hidden_states, "b l (n c) -> b (l n) c", n=math.prod(self.patch_size))

    def _forward_exact_dual_stream(
        self,
        prepared: PreparedExactTrainSequence,
        *,
        update_cache: int,
        cache_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = prepared.hidden_states
        text_hidden_states = prepared.text_hidden_states
        rotary_emb = prepared.rotary_emb
        temb = prepared.temb
        timestep_proj = prepared.timestep_proj
        split_list = prepared.split_list
        exact_attention_profile = prepared.attention_profile
        cache_backend_stream_ids = torch.cat(
            [
                torch.zeros(int(split_list[0]), device=hidden_states.device, dtype=torch.long),
                torch.zeros(int(split_list[1]), device=hidden_states.device, dtype=torch.long),
                torch.ones(int(split_list[2]), device=hidden_states.device, dtype=torch.long),
                torch.ones(int(split_list[3]), device=hidden_states.device, dtype=torch.long),
                torch.full((int(split_list[4]),), -1, device=hidden_states.device, dtype=torch.long),
            ],
            dim=0,
        )

        cache_state = self._resolve_exact_cache_state(cache_name)
        cache_backend_name = cache_state.backend_name if cache_state is not None else None
        cache_backend_payload = cache_state.backend_payload if cache_state is not None else None
        next_self_attention_kv: list[AttentionCacheEntry] = []
        moved_tensor_cache: dict[tuple[str, torch.device, torch.dtype | None], torch.Tensor] = {}
        attention_profile_cache: dict[torch.device, PreparedAttentionProfile | None] = {}

        for layer_index, block in enumerate(self.blocks):
            block_device = (
                self._runtime_block_devices[layer_index % len(self._runtime_block_devices)]
                if self._runtime_block_devices
                else hidden_states.device
            )
            if hidden_states.device != block_device:
                hidden_states = hidden_states.to(device=block_device)
            block_text_hidden_states = self._cached_optional_tensor(
                text_hidden_states,
                cache=moved_tensor_cache,
                name="text_hidden_states",
                device=block_device,
                dtype=hidden_states.dtype,
            )
            block_timestep_proj = self._cached_optional_tensor(
                timestep_proj,
                cache=moved_tensor_cache,
                name="timestep_proj",
                device=block_device,
                dtype=hidden_states.dtype,
            )
            block_rotary_emb = self._cached_optional_tensor(
                rotary_emb,
                cache=moved_tensor_cache,
                name="rotary_emb",
                device=block_device,
            )
            block_attention_profile = self._cached_attention_profile(
                exact_attention_profile,
                cache=attention_profile_cache,
                device=block_device,
            )
            block_cache_backend_stream_ids = self._cached_optional_tensor(
                cache_backend_stream_ids,
                cache=moved_tensor_cache,
                name="cache_backend_stream_ids",
                device=block_device,
            )
            block_cache_backend_state = (
                cache_backend_payload.layer_states[layer_index]
                if cache_backend_uses_slot_pool(cache_backend_name)
                and cache_backend_payload is not None
                and layer_index < len(cache_backend_payload.layer_states)
                else None
            )
            block_cache_backend_state = self._move_slot_pool_layer_state(block_cache_backend_state, device=block_device)
            hidden_states, current_self_cache_entry, _ = block(
                hidden_states,
                encoder_hidden_states=block_text_hidden_states,
                temb=block_timestep_proj,
                rotary_emb=block_rotary_emb,
                attention_profile=block_attention_profile,
                self_attention_cache_backend_name=cache_backend_name,
                self_attention_cache_backend_state=block_cache_backend_state,
                self_attention_cache_update_mode=update_cache,
                self_attention_cache_stream_ids=block_cache_backend_stream_ids,
            )
            next_self_attention_kv.append(current_self_cache_entry or AttentionCacheEntry())

        output_device = self.scale_shift_table.device
        if hidden_states.device != output_device:
            hidden_states = hidden_states.to(device=output_device)
        temb = temb.to(device=output_device, dtype=hidden_states.dtype)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = _select_split_segments(
            hidden_states,
            tuple(int(length) for length in split_list),
        )

        if cache_state is not None:
            materialized_entries = (
                materialize_cache_backend_entries(cache_backend_payload)
                if cache_backend_uses_slot_pool(cache_backend_name)
                else tuple(next_self_attention_kv)
            )
            self._exact_runtime_caches[cache_name] = CacheState(
                supported=cache_state.supported,
                current_start_frame=cache_state.current_start_frame,
                cached_frames=cache_state.cached_frames,
                chunk_size=cache_state.chunk_size,
                capability=cache_state.capability,
                backend_name=cache_state.backend_name,
                backend_payload=cache_backend_payload,
                payload=dict(cache_state.payload),
                self_attention_kv=materialized_entries,
                cross_attention_kv=cache_state.cross_attention_kv,
                update_metadata=cache_state.update_metadata,
            )

        video_prediction = self.proj_out(latent_hidden_states)
        video_prediction = rearrange(
            video_prediction,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=prepared.batch_size,
        )
        action_prediction = self.action_proj_out(action_hidden_states)
        action_prediction = rearrange(
            action_prediction,
            "1 (b l) c -> b l c",
            b=prepared.batch_size,
        )
        return video_prediction, action_prediction

    def execute_runtime_step(self, step_input: RuntimeStepInput) -> RuntimeStepOutput:
        prepared = prepare_runtime_sequence(
            step_input,
            exact_train_preparer=lambda payload: prepare_exact_dual_stream_train_sequence(
                payload,
                config=self.config,
                patch_size=self.patch_size,
                model_dtype=self.patch_embedding_mlp.weight.dtype,
                input_embed=lambda tensor, input_type: self._input_embed(tensor, input_type=input_type),
                exact_text_hidden_states=lambda text_emb: self._exact_text_hidden_states(
                    text_emb,
                    dtype=self.patch_embedding_mlp.weight.dtype,
                ),
                time_embed=lambda timesteps, height, width, dtype, action_mode: self._time_embed(
                    timesteps,
                    height,
                    width,
                    dtype=dtype,
                    action_mode=action_mode,
                ),
                rope=self.rope,
            ),
        )
        if prepared.mode == "core_input":
            if prepared.core_input is None:
                raise ValueError("Runtime sequence resolved to `core_input` without a core payload.")
            core_output = self.forward(prepared.core_input)
            core_output.aux.setdefault("runtime_program", step_input.program.name)
            core_output.aux.setdefault("sequence_family", step_input.program.sequence_family)
            return RuntimeStepOutput(
                tokens=core_output.tokens,
                core_output=core_output,
                projected_outputs={},
                cache_state=core_output.cache_state,
                aux={
                    **core_output.aux,
                    "runtime_program": step_input.program.name,
                    "sequence_family": step_input.program.sequence_family,
                    "stream_output_head_family": "none",
                },
            )
        if prepared.mode == "exact_train":
            if prepared.exact_train is None:
                raise ValueError("Exact-train runtime step requires prepared exact-train state.")
            hidden_states = prepared.exact_train.hidden_states
            text_hidden_states = prepared.exact_train.text_hidden_states
            rotary_emb = prepared.exact_train.rotary_emb
            temb = prepared.exact_train.temb
            timestep_proj = prepared.exact_train.timestep_proj
            split_list = prepared.exact_train.split_list
            exact_attention_profile = prepared.exact_train.attention_profile

            for block in self.blocks:
                hidden_states, _, _ = block(
                    hidden_states,
                    encoder_hidden_states=text_hidden_states,
                    temb=timestep_proj,
                    rotary_emb=rotary_emb,
                    attention_profile=exact_attention_profile,
                )

            temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
            shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)
            hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
            latent_hidden_states, _, action_hidden_states, _, _ = _select_split_segments(
                hidden_states,
                tuple(int(length) for length in split_list),
            )
            video_prediction = self.proj_out(latent_hidden_states)
            video_prediction = rearrange(
                video_prediction,
                "1 (b l) (n c) -> b (l n) c",
                n=math.prod(self.patch_size),
                b=prepared.exact_train.batch_size,
            )
            action_prediction = self.action_proj_out(action_hidden_states)
            action_prediction = rearrange(
                action_prediction,
                "1 (b l) c -> b l c",
                b=prepared.exact_train.batch_size,
            )
            return RuntimeStepOutput(
                projected_outputs={
                    "video_prediction": video_prediction,
                    "action_prediction": action_prediction,
                },
                aux={
                    "runtime_program": step_input.program.name,
                    "sequence_family": step_input.program.sequence_family,
                },
            )
        if prepared.mode == "exact_inference":
            if prepared.exact_inference is None:
                raise ValueError("Exact-inference runtime step requires prepared exact-inference state.")
            video_prediction, action_prediction = self._forward_exact_dual_stream(
                prepared.exact_inference,
                update_cache=prepared.update_cache,
                cache_name=prepared.cache_name,
            )
            return RuntimeStepOutput(
                projected_outputs={
                    "video_prediction": video_prediction,
                    "action_prediction": action_prediction,
                },
                cache_state=self._resolve_exact_cache_state(prepared.cache_name),
                aux={
                    "runtime_program": step_input.program.name,
                    "sequence_family": step_input.program.sequence_family,
                },
            )
        if prepared.mode == "exact_single_stream":
            if prepared.payload is None:
                raise ValueError("Exact single-stream runtime step requires `payload`.")
            tokens = self._forward_exact_single_stream(
                prepared.payload,
                update_cache=prepared.update_cache,
                cache_name=prepared.cache_name,
                action_mode=prepared.action_mode,
            )
            return RuntimeStepOutput(
                tokens=tokens,
                projected_outputs={"stream_prediction": tokens},
                cache_state=self._resolve_exact_cache_state(prepared.cache_name),
                aux={
                    "runtime_program": step_input.program.name,
                    "sequence_family": step_input.program.sequence_family,
                    "action_mode": prepared.action_mode,
                },
            )
        raise ValueError(f"Unsupported prepared runtime sequence mode {prepared.mode!r}.")

    def _resolve_stream_ids(
        self,
        stream_ids: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if stream_ids is None:
            return torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
        if stream_ids.ndim == 1:
            if stream_ids.shape[0] != seq_len:
                raise ValueError(f"Expected 1D stream_ids with length {seq_len}, got {tuple(stream_ids.shape)}")
            return stream_ids[None, :].expand(batch_size, -1).to(device=device, dtype=torch.long)
        if stream_ids.ndim == 2:
            if stream_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"Expected 2D stream_ids with shape {(batch_size, seq_len)}, got {tuple(stream_ids.shape)}"
                )
            return stream_ids.to(device=device, dtype=torch.long)
        raise ValueError(f"Expected stream_ids with ndim 1 or 2, got shape {tuple(stream_ids.shape)}")

    def _select_stream_tensor(
        self,
        video_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        stream_ids: torch.Tensor,
    ) -> torch.Tensor:
        if video_tensor.ndim == 3:
            mask = stream_ids[..., None].bool()
        elif video_tensor.ndim == 4:
            mask = stream_ids[..., None, None].bool()
        else:
            raise ValueError(f"Unsupported stream-conditioned tensor rank {video_tensor.ndim}")
        return torch.where(mask, action_tensor, video_tensor)

    def _resolve_encoder_hidden_states(
        self,
        core_input: VisualCoreInput,
        stream_ids: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        text_context = core_input.text_context
        if text_context is None and core_input.conditioning is not None:
            text_context = core_input.conditioning.text_context
        if text_context is None:
            return torch.zeros(batch_size, 1, self.config.hidden_size, device=device, dtype=dtype)
        text_context = text_context.to(device=device)
        if text_context.ndim == 2:
            text_context = text_context[:, None, :]
        if text_context.shape[-1] == self.config.hidden_size:
            video_hidden_states = text_context.to(dtype=dtype)
            action_hidden_states = video_hidden_states
        else:
            video_hidden_states = self.text_proj(text_context).to(dtype=dtype)
            action_hidden_states = self.action_text_proj(text_context).to(dtype=dtype)
        action_fraction = stream_ids.float().mean(dim=1, keepdim=True).unsqueeze(-1)
        return (1.0 - action_fraction) * video_hidden_states + action_fraction * action_hidden_states

    def forward(
        self,
        core_input: VisualCoreInput | dict[str, torch.Tensor],
        *,
        update_cache: int = 0,
        cache_name: str = "open_wam_exact",
        action_mode: bool = False,
        train_mode: bool = False,
    ) -> VisualCoreOutput | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(core_input, dict):
            if train_mode:
                return self.forward_train(core_input)
            return self._forward_exact_single_stream(
                core_input,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            )
        token_layout = core_input.token_layout
        hidden_states = core_input.tokens
        position_context = core_input.position_context
        grid_ids = core_input.grid_ids
        timestep_values = core_input.timestep_values
        attention_mask = core_input.attention_mask
        stream_ids_tensor = core_input.stream_ids
        batch_size, seq_len, _ = hidden_states.shape
        prep_device = (
            self.time_conditioner.time_embedder.linear_1.weight.device
            if self._runtime_block_devices
            else hidden_states.device
        )
        if hidden_states.device != prep_device:
            hidden_states = hidden_states.to(device=prep_device)
        if position_context is not None and position_context.device != prep_device:
            position_context = position_context.to(device=prep_device, dtype=hidden_states.dtype)
        if grid_ids is not None and grid_ids.device != prep_device:
            grid_ids = grid_ids.to(device=prep_device)
        if timestep_values is not None and timestep_values.device != prep_device:
            timestep_values = timestep_values.to(device=prep_device)
        if attention_mask is not None and attention_mask.device != prep_device:
            attention_mask = attention_mask.to(device=prep_device)
        if stream_ids_tensor is not None and stream_ids_tensor.device != prep_device:
            stream_ids_tensor = stream_ids_tensor.to(device=prep_device)
        device = prep_device
        dtype = hidden_states.dtype
        stream_ids = self._resolve_stream_ids(stream_ids_tensor, batch_size=batch_size, seq_len=seq_len, device=device)

        if position_context is not None and grid_ids is None:
            hidden_states = hidden_states + position_context

        if timestep_values is None:
            if core_input.timestep_context is not None:
                hidden_states = hidden_states + core_input.timestep_context
                video_temb = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device, dtype=dtype)
                video_timestep_proj = torch.zeros(
                    batch_size,
                    seq_len,
                    6,
                    self.config.hidden_size,
                    device=device,
                    dtype=dtype,
                )
                action_temb = video_temb
                action_timestep_proj = video_timestep_proj
            else:
                timestep_values = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
                video_temb, video_timestep_proj = self.time_conditioner(timestep_values, dtype=dtype)
                action_temb, action_timestep_proj = self.action_time_conditioner(timestep_values, dtype=dtype)
        else:
            timestep_values = timestep_values.to(device=device)
            video_temb, video_timestep_proj = self.time_conditioner(timestep_values, dtype=dtype)
            action_temb, action_timestep_proj = self.action_time_conditioner(timestep_values, dtype=dtype)

        temb = self._select_stream_tensor(video_temb, action_temb, stream_ids)
        timestep_proj = self._select_stream_tensor(video_timestep_proj, action_timestep_proj, stream_ids)

        rotary_grid_ids = grid_ids
        rotary_emb = self.rope(rotary_grid_ids.to(device=device))[:, :, None] if rotary_grid_ids is not None else None
        encoder_hidden_states = self._resolve_encoder_hidden_states(
            core_input,
            stream_ids=stream_ids,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        cache_update_metadata = core_input.cache_update_metadata or CacheUpdateMetadata()
        captured_readouts: list[VisualIntermediateReadout] = []
        requested_layers = (
            set(core_input.readout_request.capture_layer_indices)
            if core_input.readout_request is not None
            else set()
        )
        cache_branch = cache_update_metadata.cache_branch
        cache_metadata = core_input.sequence_metadata.metadata if core_input.sequence_metadata is not None else {}
        cacheable_video_tokens = int(cache_metadata.get("cacheable_video_tokens", 0))
        cache_reference_start = int(cache_metadata.get("cache_reference_start", 0))
        cache_reference_end = int(cache_metadata.get("cache_reference_end", cache_reference_start))
        tokens_per_frame = int(cache_metadata.get("tokens_per_frame", 0))
        max_cached_tokens = None
        if cache_update_metadata.max_cached_frames is not None and tokens_per_frame > 0:
            max_cached_tokens = cache_update_metadata.max_cached_frames * tokens_per_frame
        cached_prefix_visibility = None
        if (
            attention_mask is not None
            and cache_reference_end > cache_reference_start
            and attention_mask.shape[-1] >= cache_reference_end
        ):
            cached_prefix_visibility = attention_mask[..., cache_reference_start:cache_reference_end]

        next_self_attention_kv: list[AttentionCacheEntry] = []
        next_cross_attention_kv: list[AttentionCacheEntry] = []
        incoming_branch_state = resolve_cache_branch_state(core_input.cache_state, cache_branch)
        incoming_self_attention_kv = incoming_branch_state.self_attention_kv
        incoming_cross_attention_kv = incoming_branch_state.cross_attention_kv
        for layer_index, block in enumerate(self.blocks):
            block_device = (
                self._runtime_block_devices[layer_index % len(self._runtime_block_devices)]
                if self._runtime_block_devices
                else hidden_states.device
            )
            if hidden_states.device != block_device:
                hidden_states = hidden_states.to(device=block_device)
            block_timestep_proj = timestep_proj.to(device=block_device, dtype=hidden_states.dtype)
            block_encoder_hidden_states = encoder_hidden_states.to(device=block_device, dtype=hidden_states.dtype)
            block_rotary_emb = self._move_optional_tensor(rotary_emb, device=block_device)
            block_attention_mask = self._move_optional_tensor(attention_mask, device=block_device)
            block_cached_prefix_visibility = self._move_optional_tensor(
                cached_prefix_visibility,
                device=block_device,
                dtype=hidden_states.dtype,
            )
            hidden_states, current_self_cache_entry, current_cross_cache_entry = block(
                hidden_states,
                encoder_hidden_states=block_encoder_hidden_states,
                temb=block_timestep_proj,
                rotary_emb=block_rotary_emb,
                attention_mask=block_attention_mask,
                attention_profile=core_input.attention_profile,
                self_attention_cache_entry=(
                    incoming_self_attention_kv[layer_index]
                    if layer_index < len(incoming_self_attention_kv)
                    else None
                ),
                cross_attention_cache_entry=(
                    incoming_cross_attention_kv[layer_index]
                    if layer_index < len(incoming_cross_attention_kv)
                    else None
                ),
                cached_prefix_visibility=block_cached_prefix_visibility,
                cache_current_token_count=(
                    cacheable_video_tokens if cache_update_metadata.update_kv_cache and cacheable_video_tokens > 0 else 0
                ),
                cache_current_token_span=(
                    (cache_reference_start, cache_reference_end)
                    if cache_update_metadata.update_kv_cache and cache_reference_end > cache_reference_start
                    else None
                ),
            )
            existing_self_entry = incoming_self_attention_kv[layer_index] if layer_index < len(incoming_self_attention_kv) else None
            if cache_update_metadata.update_kv_cache and current_self_cache_entry is not None:
                next_self_attention_kv.append(
                    _merge_attention_cache_entries(
                        existing_self_entry,
                        current_self_cache_entry,
                        max_tokens=max_cached_tokens,
                    )
                )
            else:
                next_self_attention_kv.append(existing_self_entry or AttentionCacheEntry())
            existing_cross_entry = incoming_cross_attention_kv[layer_index] if layer_index < len(incoming_cross_attention_kv) else None
            if existing_cross_entry is not None and existing_cross_entry.key is not None and existing_cross_entry.value is not None:
                next_cross_attention_kv.append(existing_cross_entry)
            elif cache_update_metadata.update_cross_attention_cache and current_cross_cache_entry is not None:
                next_cross_attention_kv.append(current_cross_cache_entry)
            else:
                next_cross_attention_kv.append(AttentionCacheEntry())
            if layer_index in requested_layers:
                captured_readouts.append(
                    VisualIntermediateReadout(
                        layer_index=layer_index,
                        tokens=hidden_states,
                        token_layout=token_layout,
                        aux={"implementation": "shared_transformer"},
                    )
                )

        output_device = self.scale_shift_table.device
        if hidden_states.device != output_device:
            hidden_states = hidden_states.to(device=output_device)
        temb = temb.to(device=output_device, dtype=hidden_states.dtype)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)

        has_runtime_sequence = core_input.sequence_metadata is not None
        layer_cache_entries = (
            tuple(
                AttentionCacheEntry(
                    key=entry.key,
                    value=entry.value,
                    metadata={
                        **entry.metadata,
                        "layer_index": layer_index,
                        "sequence_length": int(entry.key.shape[2]) if entry.key is not None else seq_len,
                        "current_start_frame": cache_update_metadata.current_start_frame,
                        "implementation": "shared_transformer",
                    },
                )
                for layer_index, entry in enumerate(next_self_attention_kv)
            )
            if has_runtime_sequence
            else tuple()
        )
        cross_layer_cache_entries = (
            tuple(
                AttentionCacheEntry(
                    key=entry.key,
                    value=entry.value,
                    metadata={
                        **entry.metadata,
                        "layer_index": layer_index,
                        "current_start_frame": cache_update_metadata.current_start_frame,
                        "implementation": "shared_transformer",
                        "cache_kind": "cross_attention",
                    },
                )
                for layer_index, entry in enumerate(next_cross_attention_kv)
            )
            if has_runtime_sequence
            else tuple()
        )
        if core_input.cache_state is not None:
            branch_state = CacheBranchState(
                backend_name=incoming_branch_state.backend_name,
                backend_payload=incoming_branch_state.backend_payload,
                payload=dict(incoming_branch_state.payload),
                self_attention_kv=(
                    incoming_branch_state.self_attention_kv
                    if incoming_branch_state.self_attention_kv and not cache_update_metadata.update_kv_cache
                    else layer_cache_entries
                ),
                cross_attention_kv=(
                    incoming_branch_state.cross_attention_kv
                    if incoming_branch_state.cross_attention_kv and not cache_update_metadata.update_cross_attention_cache
                    else cross_layer_cache_entries
                ),
            )
            cache_state = replace_cache_branch_state(
                CacheState(
                    supported=core_input.cache_state.supported or has_runtime_sequence,
                    current_start_frame=cache_update_metadata.current_start_frame,
                    cached_frames=core_input.cache_state.cached_frames,
                    chunk_size=core_input.cache_state.chunk_size,
                    capability=(
                        core_input.cache_state.capability
                        if core_input.cache_state.capability != "none"
                        else ("self_attn_plus_cross_attn" if has_runtime_sequence else "none")
                    ),
                    backend_name=core_input.cache_state.backend_name,
                    backend_payload=core_input.cache_state.backend_payload,
                    payload=dict(core_input.cache_state.payload),
                    self_attention_kv=core_input.cache_state.self_attention_kv,
                    cross_attention_kv=core_input.cache_state.cross_attention_kv,
                    update_metadata=cache_update_metadata,
                    branch_states=dict(core_input.cache_state.branch_states),
                ),
                branch_name=cache_branch,
                branch_state=branch_state,
                mirror_to_top_level=cache_branch in {"default", "conditioned"},
            )
        else:
            cache_state = CacheState(
                supported=has_runtime_sequence,
                current_start_frame=cache_update_metadata.current_start_frame,
                cached_frames=0,
                chunk_size=seq_len,
                capability="self_attn_plus_cross_attn" if has_runtime_sequence else "none",
                backend_name="merged_prefix",
                backend_payload=None,
                payload={"stage": "shared_transformer_core", "implementation": "shared_transformer"},
                self_attention_kv=layer_cache_entries,
                cross_attention_kv=cross_layer_cache_entries,
                update_metadata=cache_update_metadata,
                branch_states={},
            )
        return VisualCoreOutput(
            tokens=hidden_states,
            token_layout=token_layout,
            cache_state=cache_state,
            intermediate_readouts=tuple(captured_readouts),
            aux={
                "implementation": "shared_transformer",
                "used_rotary": rotary_grid_ids is not None,
                "used_action_conditioner": bool((stream_ids != 0).any().item()),
                "has_sequence_metadata": core_input.sequence_metadata is not None,
                "structured_block_mode": "none",
                "structured_attention_mode": "none",
                "structured_attention_kernel": "none",
                "structured_cache_kernel": "none",
                "structured_attention_internal_mask": False,
                "structured_attention_full_cache_prefix": False,
                "structured_frequency_mode": "none",
                "structured_has_clean_prefix_frequencies": False,
                "structured_has_action_frequencies": False,
                "structured_has_state_frequencies": False,
                "structured_register_frame_shift": None,
                "structured_time_layout": "generic",
                "structured_position_layout": "generic",
                "structured_current_start_frame": None,
                "structured_observed_prefix_frames": None,
                "structured_action_register_length": None,
                "structured_state_register_length": None,
                "structured_clean_prefix_length": None,
                "cache_runtime_metadata": cache_update_metadata,
            },
        )


LingbotReplicaTimeEmbedding = SharedTransformerTimeEmbedding
LingbotReplicaRotaryPosEmbed = SharedTransformerRotaryPositionalEmbedding
LingbotReplicaAttention = SharedTransformerAttention
LingbotReplicaTransformerBlock = SharedTransformerBlock
LingbotReplicaVisualCore = SharedVideoTransformerCore
