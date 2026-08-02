from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import ParallelExactCacheWriteMode
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.common.cache_backend_contracts import (
    SlotPoolLayerState,
    cache_backend_uses_slot_pool,
)
from open_wam.models.visual_tower.exact_runtime import (
    initialize_exact_runtime_cache,
    resolve_runtime_module_dtype,
)

__all__ = [
    "ExactCacheContext",
    "ExactCacheInterfaceSpec",
    "build_clean_video_action_cache_stream_ids",
    "build_dual_stream_cache_stream_ids",
    "build_exact_cache_spec",
    "count_single_stream_action_tokens",
    "ensure_exact_cache_initialized",
    "ensure_exact_text_embeddings",
    "existing_exact_cache_attention_window",
    "restore_slot_pool_layer_metadata",
    "resolve_exact_cache_context",
    "set_slot_pool_layer_metadata",
    "validate_existing_exact_cache_attention_window",
]


@dataclass(frozen=True)
class ExactCacheInterfaceSpec:
    """Exact-runtime cache interface independent of rollout style."""

    write_mode: ParallelExactCacheWriteMode
    cache_batch_size_override: int | None = None
    token_batch_factor: int = 1
    prefix_visibility_mode: str = "full_history"


@dataclass(frozen=True)
class ExactCacheContext:
    """Resolved cache metadata shared across exact-runtime rollout paths."""

    cache_name: str
    cache_backend_name: str
    cache_initialized: bool
    batch_size: int
    latent_height: int
    latent_width: int
    use_cfg: bool
    device: torch.device
    model_dtype: torch.dtype


def build_dual_stream_cache_stream_ids(
    split_list: list[int] | tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Label packed video, action, and padding tokens for exact-cache writes."""

    return torch.cat(
        [
            torch.zeros(int(split_list[0]), device=device, dtype=torch.long),
            torch.zeros(int(split_list[1]), device=device, dtype=torch.long),
            torch.ones(int(split_list[2]), device=device, dtype=torch.long),
            torch.ones(int(split_list[3]), device=device, dtype=torch.long),
            torch.full((int(split_list[4]),), -1, device=device, dtype=torch.long),
        ],
        dim=0,
    )


def build_clean_video_action_cache_stream_ids(
    *,
    video_token_count: int,
    action_token_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Label a clean video/action cache commit without padding tokens."""

    return torch.cat(
        [
            torch.zeros(int(video_token_count), device=device, dtype=torch.long),
            torch.ones(int(action_token_count), device=device, dtype=torch.long),
        ],
        dim=0,
    )


def count_single_stream_action_tokens(actions: torch.Tensor) -> int:
    """Return the flattened token width of `[B, C, F, A, W]` action latents."""

    if actions.ndim != 5:
        raise ValueError(
            f"Expected action latents shaped [B, C, F, A, W], got {tuple(actions.shape)}."
        )
    return int(actions.shape[2]) * int(actions.shape[3]) * int(actions.shape[4])


SlotPoolLayerMetadataSnapshot = list[
    tuple[SlotPoolLayerState, dict[str, tuple[bool, Any]]]
]


def set_slot_pool_layer_metadata(
    transformer: torch.nn.Module,
    *,
    cache_name: str,
    updates: dict[str, Any],
) -> SlotPoolLayerMetadataSnapshot:
    """Apply reversible per-layer metadata to an initialized slot-pool cache."""

    if not updates or not hasattr(transformer, "_resolve_exact_cache_state"):
        return []
    cache_state = transformer._resolve_exact_cache_state(cache_name)
    if cache_state is None or not cache_backend_uses_slot_pool(cache_state.backend_name):
        return []
    cache_payload = cache_state.backend_payload
    layer_states = getattr(cache_payload, "layer_states", None)
    if layer_states is None:
        return []
    previous: SlotPoolLayerMetadataSnapshot = []
    for layer_state in layer_states:
        layer_previous: dict[str, tuple[bool, Any]] = {}
        for key, value in updates.items():
            layer_previous[key] = (
                key in layer_state.metadata,
                layer_state.metadata.get(key),
            )
            layer_state.metadata[key] = value
        previous.append((layer_state, layer_previous))
    return previous


def restore_slot_pool_layer_metadata(
    previous: SlotPoolLayerMetadataSnapshot,
) -> None:
    """Restore metadata captured by :func:`set_slot_pool_layer_metadata`."""

    for layer_state, layer_previous in previous:
        for key, (was_present, value) in layer_previous.items():
            if was_present:
                layer_state.metadata[key] = value
            else:
                layer_state.metadata.pop(key, None)


def ensure_exact_text_embeddings(
    text_emb: torch.Tensor | None,
    *,
    batch_size: int,
    backbone_config: SharedVideoTransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize exact-runtime text embeddings with the model contract."""

    if text_emb is None:
        return torch.zeros(
            batch_size,
            backbone_config.max_text_tokens,
            backbone_config.text_dim,
            device=device,
            dtype=dtype,
        )
    return text_emb.to(device=device, dtype=dtype)


def resolve_exact_cache_context(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    inference_config: InferenceConfig,
    infer_cache: dict[str, Any],
    batch_size: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
) -> tuple[ExactCacheContext, torch.Tensor, torch.Tensor | None]:
    """Resolve cache identity, CFG shape, dtype, and text inputs."""

    model_dtype = resolve_runtime_module_dtype(transformer)
    resolved_text_emb = ensure_exact_text_embeddings(
        text_emb,
        batch_size=batch_size,
        backbone_config=backbone_config,
        device=device,
        dtype=model_dtype,
    )
    use_cfg = bool(
        infer_cache.get(
            "use_cfg",
            inference_config.guidance_scale > 1.0
            or inference_config.action_guidance_scale > 1.0,
        )
    )
    if negative_text_emb is not None:
        resolved_negative_text_emb = ensure_exact_text_embeddings(
            negative_text_emb,
            batch_size=batch_size,
            backbone_config=backbone_config,
            device=device,
            dtype=model_dtype,
        )
    elif use_cfg:
        resolved_negative_text_emb = torch.zeros_like(resolved_text_emb)
    else:
        resolved_negative_text_emb = None
    return (
        ExactCacheContext(
            cache_name=str(infer_cache.get("cache_name", "open_wam_exact")),
            cache_backend_name=str(
                infer_cache.get("cache_backend_name", "slot_pool_exact")
            ),
            cache_initialized=bool(infer_cache.get("cache_initialized", False)),
            batch_size=batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
            use_cfg=use_cfg,
            device=device,
            model_dtype=model_dtype,
        ),
        resolved_text_emb,
        resolved_negative_text_emb,
    )


def build_exact_cache_spec(
    *,
    write_mode: ParallelExactCacheWriteMode | str,
    batch_size: int,
    use_cfg: bool,
    prefix_visibility_mode: str = "full_history",
) -> ExactCacheInterfaceSpec:
    """Resolve the cache write interface for one exact rollout."""

    write_mode = ParallelExactCacheWriteMode(write_mode)
    if write_mode == ParallelExactCacheWriteMode.JOINT_PACKED:
        # The shared runtime retains batch as the cache batch dimension.
        return ExactCacheInterfaceSpec(
            write_mode=write_mode,
            prefix_visibility_mode=prefix_visibility_mode,
        )
    return ExactCacheInterfaceSpec(
        write_mode=write_mode,
        prefix_visibility_mode=prefix_visibility_mode,
    )


def ensure_exact_cache_initialized(
    *,
    transformer: torch.nn.Module,
    policy_config: ParallelStreamPolicyConfig,
    inference_config: InferenceConfig,
    cache_context: ExactCacheContext,
    cache_spec: ExactCacheInterfaceSpec,
    attn_window: int | None = None,
    frame_chunk_size: int | None = None,
) -> ExactCacheContext:
    """Allocate or validate the exact cache required by a rollout."""

    if not inference_config.use_cache:
        return cache_context
    resolved_attn_window = int(
        policy_config.attn_window if attn_window is None else attn_window
    )
    if resolved_attn_window <= 0:
        raise ValueError(
            "Exact cache attention window must be positive, "
            f"got {resolved_attn_window}."
        )
    if cache_context.cache_initialized:
        validate_existing_exact_cache_attention_window(
            transformer,
            cache_name=cache_context.cache_name,
            requested_attn_window=resolved_attn_window,
        )
        return cache_context
    initialize_exact_runtime_cache(
        transformer,
        cache_name=cache_context.cache_name,
        attn_window=resolved_attn_window,
        batch_size=cache_context.batch_size,
        frame_chunk_size=max(
            1,
            int(
                inference_config.frame_chunk_size
                if frame_chunk_size is None
                else frame_chunk_size
            ),
        ),
        latent_height=cache_context.latent_height,
        latent_width=cache_context.latent_width,
        device=cache_context.device,
        action_per_frame=policy_config.action_per_frame,
        use_cfg=cache_context.use_cfg,
        cache_backend_name=cache_context.cache_backend_name,
        cache_batch_size_override=cache_spec.cache_batch_size_override,
        token_batch_factor=cache_spec.token_batch_factor,
        prefix_visibility_mode=cache_spec.prefix_visibility_mode,
    )
    return ExactCacheContext(
        cache_name=cache_context.cache_name,
        cache_backend_name=cache_context.cache_backend_name,
        cache_initialized=True,
        batch_size=cache_context.batch_size,
        latent_height=cache_context.latent_height,
        latent_width=cache_context.latent_width,
        use_cfg=cache_context.use_cfg,
        device=cache_context.device,
        model_dtype=cache_context.model_dtype,
    )


def validate_existing_exact_cache_attention_window(
    transformer: torch.nn.Module,
    *,
    cache_name: str,
    requested_attn_window: int,
) -> None:
    """Reject reuse of a cache initialized for a different window."""

    existing_attn_window = existing_exact_cache_attention_window(
        transformer,
        cache_name=cache_name,
    )
    if (
        existing_attn_window is not None
        and int(existing_attn_window) != int(requested_attn_window)
    ):
        raise ValueError(
            "Existing exact cache attention window does not match the requested rollout contract, "
            f"got existing={existing_attn_window}, requested={int(requested_attn_window)}. "
            "Reset the rollout session before switching joint-denoise conditioning modes."
        )


def existing_exact_cache_attention_window(
    transformer: torch.nn.Module,
    *,
    cache_name: str,
) -> int | None:
    """Read the active attention window from either exact cache backend."""

    if not hasattr(transformer, "_resolve_exact_cache_state"):
        return None
    cache_state = transformer._resolve_exact_cache_state(cache_name)
    if cache_state is None:
        return None
    payload_value = getattr(cache_state, "payload", {}).get("attn_window")
    if payload_value is not None:
        return int(payload_value)
    backend_payload = getattr(cache_state, "backend_payload", None)
    metadata = getattr(backend_payload, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("attn_window") is not None:
        return int(metadata["attn_window"])
    return None
