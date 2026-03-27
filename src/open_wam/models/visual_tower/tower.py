from __future__ import annotations

import torch
from torch import nn

from open_wam.data.raw_video import ViewPlacement
from open_wam.models.common import RolloutCursor
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.video_backbone.contracts import AttentionCacheEntry, CacheState, CacheUpdateMetadata

from .contracts import VisualCoreInput, VisualStageOutputs
from .core import LingbotVisualCore
from .decoder import VisualFeatureDecoder
from .frontend import LingbotVisualFrontend
from .grid_ids import build_video_grid_ids
from .reference_core_weights import ReferenceCoreLoadReport, load_reference_weights_into_replica_core
from .replica_core import LingbotReplicaVisualCore
from .reference_transformer import build_reference_transformer, preferred_reference_dtype

_MAX_CACHED_FRAMES_UNSET = object()


class VisualTower(nn.Module):
    """Stage-aware visual tower used by all policy variants."""

    def __init__(
        self,
        config: LingbotCompatibleVideoBackboneConfig | None = None,
        *,
        action_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LingbotCompatibleVideoBackboneConfig()
        self.action_dim = action_dim
        self.frontend = LingbotVisualFrontend(self.config)
        if self.config.implementation == "lingbot_replica":
            self.core = LingbotReplicaVisualCore(self.config)
        elif self.config.implementation == "dummy":
            self.core = LingbotVisualCore(self.config)
        else:
            raise ValueError(
                f"Unsupported backbone implementation '{self.config.implementation}'. "
                "Expected 'dummy' or 'lingbot_replica'."
            )
        self.decoder = VisualFeatureDecoder(self.config.hidden_size)
        self._lingbot_reference_transformers = nn.ModuleDict()
        self.reference_core_load_report: ReferenceCoreLoadReport | None = None
        if self.config.load_reference_core_weights:
            if self.config.implementation != "lingbot_replica":
                raise ValueError("`backbone.load_reference_core_weights` requires `backbone.implementation = lingbot_replica`.")
            if self.action_dim is None:
                raise ValueError("VisualTower requires `action_dim` to load LingBot reference weights into the shared core.")
            self.reference_core_load_report = load_reference_weights_into_replica_core(
                self.core,
                backbone_config=self.config,
                action_dim=self.action_dim,
            )

    def run_frontend(
        self,
        canonical_video,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        task_text: tuple[str | None, ...] | None = None,
        text_context=None,
        negative_text_context=None,
        preserve_stream_cache: bool = False,
    ):
        self._ensure_frontend_runtime_device(canonical_video.device)
        return self.frontend(
            canonical_video,
            placements=placements,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            preserve_stream_cache=preserve_stream_cache,
        )

    def run_frontend_from_latents(
        self,
        video_latents,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context=None,
        negative_text_context=None,
        canonical_video=None,
    ):
        self._ensure_frontend_runtime_device(video_latents.device)
        return self.frontend.from_video_latents(
            video_latents,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            canonical_video=canonical_video,
        )

    def reset_runtime_state(self) -> None:
        self.frontend.reset_runtime_state()

    def run_core(self, core_input: VisualCoreInput):
        core_output = self.core(core_input)
        core_output.aux.setdefault(
            "weight_source",
            "lingbot_reference_init" if self.reference_core_load_report is not None else "local_init",
        )
        if self.reference_core_load_report is not None:
            core_output.aux.setdefault("reference_core_loaded_keys", len(self.reference_core_load_report.loaded_keys))
        return core_output

    def cache_capability(self) -> str:
        if self.config.implementation == "lingbot_replica":
            return "self_attn_plus_cross_attn"
        return "none"

    def init_runtime_cache_state(
        self,
        *,
        cursor: RolloutCursor,
        stage: str,
        payload: dict[str, object] | None = None,
        cfg_mode: str = "none",
        update_kv_cache: bool = False,
        update_cross_attention_cache: bool = False,
        max_cached_frames: int | None | object = _MAX_CACHED_FRAMES_UNSET,
        sink_frames: int = 0,
        local_attn_window: int | None = None,
    ) -> CacheState:
        capability = self.cache_capability()
        resolved_max_cached_frames = (
            cursor.chunk_size if max_cached_frames is _MAX_CACHED_FRAMES_UNSET else max_cached_frames
        )
        resolved_payload = {"stage": stage, "block_index": cursor.block_index}
        if payload is not None:
            resolved_payload.update(payload)
        return CacheState(
            supported=capability != "none",
            current_start_frame=cursor.current_start_frame,
            cached_frames=0,
            chunk_size=cursor.chunk_size,
            capability=capability,
            payload=resolved_payload,
            update_metadata=CacheUpdateMetadata(
                current_start_frame=cursor.current_start_frame,
                update_kv_cache=update_kv_cache,
                update_cross_attention_cache=update_cross_attention_cache,
                cfg_mode=cfg_mode,
                max_cached_frames=resolved_max_cached_frames,
                sink_frames=sink_frames,
                local_attn_window=local_attn_window,
            ),
        )

    def resolve_runtime_cache_state(
        self,
        cache_state: CacheState | None,
        *,
        cursor: RolloutCursor,
        stage: str,
        payload: dict[str, object] | None = None,
        cfg_mode: str = "none",
        update_kv_cache: bool = False,
        update_cross_attention_cache: bool = False,
        max_cached_frames: int | None | object = _MAX_CACHED_FRAMES_UNSET,
        sink_frames: int = 0,
        local_attn_window: int | None = None,
    ) -> CacheState:
        """Resolve a runtime cache state for one rollout step.

        Stateless variants use this to obtain an explicit no-op cache object,
        while cache-aware variants can pass through an existing backbone-owned
        cache without reimplementing initialization guards.
        """

        if isinstance(cache_state, CacheState):
            return cache_state
        return self.init_runtime_cache_state(
            cursor=cursor,
            stage=stage,
            payload=payload,
            cfg_mode=cfg_mode,
            update_kv_cache=update_kv_cache,
            update_cross_attention_cache=update_cross_attention_cache,
            max_cached_frames=max_cached_frames,
            sink_frames=sink_frames,
            local_attn_window=local_attn_window,
        )

    def build_runtime_cache_update_metadata(
        self,
        cache_state: CacheState,
        *,
        current_start_frame: int,
        update_kv_cache: bool = False,
        update_cross_attention_cache: bool | None = None,
        cfg_mode: str | None = None,
    ) -> CacheUpdateMetadata:
        """Build one cache-update instruction from the shared runtime state."""

        previous_metadata = cache_state.update_metadata
        return CacheUpdateMetadata(
            current_start_frame=current_start_frame,
            update_kv_cache=update_kv_cache,
            update_cross_attention_cache=(
                previous_metadata.update_cross_attention_cache
                if update_cross_attention_cache is None
                else update_cross_attention_cache
            ),
            cfg_mode=previous_metadata.cfg_mode if cfg_mode is None else cfg_mode,
            max_cached_frames=previous_metadata.max_cached_frames,
            sink_frames=previous_metadata.sink_frames,
            local_attn_window=previous_metadata.local_attn_window,
        )

    def truncate_runtime_cache_state(
        self,
        cache_state: CacheState,
        *,
        tokens_per_frame: int | None = None,
    ) -> CacheState:
        """Apply the shared retention policy to a cache state.

        The first cache-aware rollout users mainly need a rolling-window policy.
        The helper also understands a simple sink-plus-local-window layout so
        future variants can reuse the same retention vocabulary.
        """

        if not cache_state.supported:
            return cache_state

        resolved_tokens_per_frame = tokens_per_frame
        if resolved_tokens_per_frame is None:
            payload_tokens_per_frame = cache_state.payload.get("tokens_per_frame")
            if isinstance(payload_tokens_per_frame, int) and payload_tokens_per_frame > 0:
                resolved_tokens_per_frame = payload_tokens_per_frame
        if resolved_tokens_per_frame is None or resolved_tokens_per_frame <= 0:
            return cache_state

        metadata = cache_state.update_metadata
        max_cached_frames = metadata.max_cached_frames
        sink_frames = max(0, metadata.sink_frames)
        local_attn_window = metadata.local_attn_window
        if max_cached_frames is None and local_attn_window is None:
            return cache_state

        sink_tokens = sink_frames * resolved_tokens_per_frame
        local_window_tokens = (
            None
            if local_attn_window is None
            else max(0, local_attn_window) * resolved_tokens_per_frame
        )
        max_cached_tokens = (
            None
            if max_cached_frames is None
            else max(0, max_cached_frames) * resolved_tokens_per_frame
        )

        truncated_self_attention = tuple(
            self._truncate_attention_cache_entry(
                entry,
                max_cached_tokens=max_cached_tokens,
                sink_tokens=sink_tokens,
                local_window_tokens=local_window_tokens,
            )
            for entry in cache_state.self_attention_kv
        )
        truncated_cross_attention = tuple(cache_state.cross_attention_kv)

        retained_frame_cap = cache_state.cached_frames
        if max_cached_frames is not None:
            retained_frame_cap = min(retained_frame_cap, max_cached_frames)
        if local_attn_window is not None:
            retained_frame_cap = min(retained_frame_cap, sink_frames + max(0, local_attn_window))

        return CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=retained_frame_cap,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            payload=dict(cache_state.payload),
            self_attention_kv=truncated_self_attention,
            cross_attention_kv=truncated_cross_attention,
            update_metadata=cache_state.update_metadata,
        )

    def advance_runtime_cache_state(
        self,
        cache_state: CacheState,
        *,
        next_cursor: RolloutCursor,
        payload_updates: dict[str, object] | None = None,
        tokens_per_frame: int | None = None,
        cached_frames_increment: int | None = None,
    ) -> CacheState:
        """Advance one runtime cache state to the next rollout cursor."""

        increment = next_cursor.chunk_size if cached_frames_increment is None else cached_frames_increment
        next_payload = dict(cache_state.payload)
        next_payload["block_index"] = next_cursor.block_index
        if tokens_per_frame is not None:
            next_payload["tokens_per_frame"] = tokens_per_frame
        if payload_updates is not None:
            next_payload.update(payload_updates)

        next_cache_state = CacheState(
            supported=cache_state.supported,
            current_start_frame=next_cursor.current_start_frame,
            cached_frames=cache_state.cached_frames + increment,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            payload=next_payload,
            self_attention_kv=cache_state.self_attention_kv,
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=CacheUpdateMetadata(
                current_start_frame=next_cursor.current_start_frame,
                update_kv_cache=cache_state.update_metadata.update_kv_cache,
                update_cross_attention_cache=cache_state.update_metadata.update_cross_attention_cache,
                cfg_mode=cache_state.update_metadata.cfg_mode,
                max_cached_frames=cache_state.update_metadata.max_cached_frames,
                sink_frames=cache_state.update_metadata.sink_frames,
                local_attn_window=cache_state.update_metadata.local_attn_window,
            ),
        )
        return self.truncate_runtime_cache_state(
            next_cache_state,
            tokens_per_frame=tokens_per_frame,
        )

    def clear_runtime_cache_state(
        self,
        cache_state: CacheState | None,
        *,
        cursor: RolloutCursor,
        stage: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> CacheState:
        """Clear cached tensors while preserving the shared cache policy."""

        resolved_cache = self.resolve_runtime_cache_state(
            cache_state,
            cursor=cursor,
            stage=stage or "runtime_reset",
            payload=payload,
        )
        next_payload = dict(resolved_cache.payload)
        if payload is not None:
            next_payload.update(payload)
        if stage is not None:
            next_payload["stage"] = stage
        return CacheState(
            supported=resolved_cache.supported,
            current_start_frame=cursor.current_start_frame,
            cached_frames=0,
            chunk_size=cursor.chunk_size,
            capability=resolved_cache.capability,
            payload=next_payload,
            self_attention_kv=tuple(),
            cross_attention_kv=tuple(),
            update_metadata=CacheUpdateMetadata(
                current_start_frame=cursor.current_start_frame,
                update_kv_cache=False,
                update_cross_attention_cache=False,
                cfg_mode=resolved_cache.update_metadata.cfg_mode,
                max_cached_frames=resolved_cache.update_metadata.max_cached_frames,
                sink_frames=resolved_cache.update_metadata.sink_frames,
                local_attn_window=resolved_cache.update_metadata.local_attn_window,
            ),
        )

    def run_default_core(self, frontend_output):
        batch_size, seq_len, _ = frontend_output.video_tokens.shape
        return self.run_core(
            VisualCoreInput(
                tokens=frontend_output.video_tokens,
                token_layout=frontend_output.token_grid,
                grid_ids=build_video_grid_ids(
                    frontend_output.token_grid,
                    device=frontend_output.video_tokens.device,
                ),
                timestep_values=frontend_output.video_tokens.new_zeros((batch_size, seq_len), dtype=frontend_output.video_tokens.dtype),
                stream_ids=frontend_output.video_tokens.new_zeros((batch_size, seq_len), dtype=frontend_output.video_tokens.dtype).long(),
                text_context=frontend_output.conditioning.text_context,
                conditioning=frontend_output.conditioning,
            )
        )

    def run_decode(self, frontend_output, core_output):
        return self.decoder(frontend_output=frontend_output, core_output=core_output)

    def get_lingbot_reference_transformer(self, *, action_dim: int) -> nn.Module:
        key = str(action_dim)
        if key not in self._lingbot_reference_transformers:
            self._lingbot_reference_transformers[key] = build_reference_transformer(self.config, action_dim=action_dim)
        return self._lingbot_reference_transformers[key]

    def ensure_lingbot_reference_transformer_device(self, *, action_dim: int, device) -> nn.Module:
        transformer = self.get_lingbot_reference_transformer(action_dim=action_dim)
        device = torch.device(device)
        target_dtype = preferred_reference_dtype(device)
        needs_move = False
        for parameter in transformer.parameters():
            if parameter.device != device:
                needs_move = True
                break
            if parameter.is_floating_point() and parameter.dtype != target_dtype:
                needs_move = True
                break
        if not needs_move:
            for buffer in transformer.buffers():
                if buffer.device != device:
                    needs_move = True
                    break
                if buffer.is_floating_point() and buffer.dtype != target_dtype:
                    needs_move = True
                    break
        if needs_move:
            transformer.to(device=device, dtype=target_dtype)
        return transformer

    def _ensure_frontend_runtime_device(self, device) -> None:
        device = torch.device(device)
        if any(parameter.device != device for parameter in self.frontend.parameters()):
            self.frontend.to(device=device)
            return
        if any(buffer.device != device for buffer in self.frontend.buffers()):
            self.frontend.to(device=device)

    def reset_lingbot_reference_runtime(self, *, action_dim: int, cache_name: str = "open_wam_exact") -> None:
        transformer = self.get_lingbot_reference_transformer(action_dim=action_dim)
        try:
            transformer.clear_pred_cache(cache_name)
        except KeyError:
            pass
        try:
            transformer.clear_cache(cache_name)
        except KeyError:
            pass

    def forward_default(
        self,
        canonical_video,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        task_text: tuple[str | None, ...] | None = None,
        include_decode: bool = False,
    ) -> VisualStageOutputs:
        frontend_output = self.run_frontend(canonical_video, placements=placements, task_text=task_text)
        core_output = self.run_default_core(frontend_output)
        decode_output = self.run_decode(frontend_output, core_output) if include_decode else None
        return VisualStageOutputs(frontend=frontend_output, core=core_output, decode=decode_output)

    def _truncate_attention_cache_entry(
        self,
        entry: AttentionCacheEntry,
        *,
        max_cached_tokens: int | None,
        sink_tokens: int,
        local_window_tokens: int | None,
    ) -> AttentionCacheEntry:
        if entry.key is None or entry.value is None:
            return entry
        sequence_length = entry.key.shape[2]
        if sequence_length == 0:
            return entry

        if max_cached_tokens is not None and sequence_length <= max_cached_tokens:
            return entry

        total_tokens = sequence_length
        target_local_tokens = local_window_tokens
        if max_cached_tokens is not None:
            if sink_tokens >= max_cached_tokens:
                keep_indices = torch.arange(min(max_cached_tokens, total_tokens), device=entry.key.device)
                return self._slice_attention_cache_entry(entry, keep_indices)
            tail_budget = max(0, max_cached_tokens - sink_tokens)
            if target_local_tokens is None:
                target_local_tokens = tail_budget
            else:
                target_local_tokens = min(target_local_tokens, tail_budget)

        if target_local_tokens is None:
            if max_cached_tokens is None:
                return entry
            keep_indices = torch.arange(total_tokens - max_cached_tokens, total_tokens, device=entry.key.device)
            return self._slice_attention_cache_entry(entry, keep_indices)

        sink_tokens = min(sink_tokens, total_tokens)
        remaining_tokens = max(0, total_tokens - sink_tokens)
        target_local_tokens = min(target_local_tokens, remaining_tokens)
        if sink_tokens + target_local_tokens >= total_tokens:
            return entry

        head_indices = (
            torch.arange(sink_tokens, device=entry.key.device)
            if sink_tokens > 0
            else torch.empty(0, dtype=torch.long, device=entry.key.device)
        )
        tail_indices = torch.arange(
            total_tokens - target_local_tokens,
            total_tokens,
            device=entry.key.device,
        )
        keep_indices = torch.cat((head_indices, tail_indices), dim=0)
        return self._slice_attention_cache_entry(entry, keep_indices)

    def _slice_attention_cache_entry(
        self,
        entry: AttentionCacheEntry,
        keep_indices: torch.Tensor,
    ) -> AttentionCacheEntry:
        key = entry.key.index_select(2, keep_indices)
        value = entry.value.index_select(2, keep_indices)
        next_metadata = dict(entry.metadata)
        next_metadata["sequence_length"] = int(key.shape[2])
        return AttentionCacheEntry(key=key, value=value, metadata=next_metadata)
