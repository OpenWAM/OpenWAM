from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import ParallelExactCacheWriteMode
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.models.visual_tower.exact_runtime import (
    initialize_exact_runtime_cache,
    resolve_runtime_module_dtype,
)

__all__ = [
    "ExactCacheContext",
    "ExactCacheInterfaceSpec",
    "build_exact_cache_spec",
    "ensure_exact_cache_initialized",
    "ensure_exact_text_embeddings",
    "existing_exact_cache_attention_window",
    "resolve_exact_cache_context",
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
