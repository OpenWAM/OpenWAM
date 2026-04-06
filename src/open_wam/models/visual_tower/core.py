from __future__ import annotations

import torch
from torch import nn

from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.video_backbone.contracts import AttentionCacheEntry, CacheState, CacheUpdateMetadata

from .contracts import VisualCoreInput, VisualCoreOutput, VisualIntermediateReadout
from .runtime_programs import RuntimeStepInput, RuntimeStepOutput
from .sequence_adapters import prepare_runtime_sequence


def _prepare_attention_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        if attention_mask.shape != (seq_len, seq_len):
            raise ValueError(
                "Expected 2D attention mask with shape [seq_len, seq_len], "
                f"got {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype == torch.bool:
            float_mask = torch.zeros_like(attention_mask, dtype=dtype, device=device)
            float_mask = float_mask.masked_fill(~attention_mask.to(device=device), float("-inf"))
            return float_mask
        return attention_mask.to(device=device, dtype=dtype)
    if attention_mask.ndim == 3:
        if attention_mask.shape != (batch_size, seq_len, seq_len):
            raise ValueError(
                "Expected 3D attention mask with shape [B, seq_len, seq_len], "
                f"got {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype == torch.bool:
            float_mask = torch.zeros_like(attention_mask, dtype=dtype, device=device)
            float_mask = float_mask.masked_fill(~attention_mask.to(device=device), float("-inf"))
        else:
            float_mask = attention_mask.to(device=device, dtype=dtype)
        return float_mask[:, None, :, :].expand(batch_size, num_heads, seq_len, seq_len).reshape(
            batch_size * num_heads,
            seq_len,
            seq_len,
        )
    raise ValueError(
        "Expected attention mask with shape [seq_len, seq_len] or [B, seq_len, seq_len], "
        f"got {tuple(attention_mask.shape)}"
    )


class SimpleTransformerBlock(nn.Module):
    """Small transformer block that accepts optional batch-specific masks."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        prepared_mask = _prepare_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            num_heads=self.num_heads,
            seq_len=seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        normed = self.norm1(hidden_states)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=prepared_mask, need_weights=False)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PackedSequenceVisualCore(nn.Module):
    """Shared visual core over a generic packed token sequence."""

    def __init__(self, config: SharedVideoTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or SharedVideoTransformerConfig()
        self.blocks = nn.ModuleList(
            [
                SimpleTransformerBlock(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_size)

    def forward(self, core_input: VisualCoreInput) -> VisualCoreOutput:
        hidden_states = core_input.tokens
        captured_readouts: list[VisualIntermediateReadout] = []
        requested_layers = (
            set(core_input.readout_request.capture_layer_indices)
            if core_input.readout_request is not None
            else set()
        )
        if core_input.position_context is not None:
            hidden_states = hidden_states + core_input.position_context
        if core_input.timestep_context is not None:
            hidden_states = hidden_states + core_input.timestep_context
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, attention_mask=core_input.attention_mask)
            if layer_index in requested_layers:
                captured_readouts.append(
                    VisualIntermediateReadout(
                        layer_index=layer_index,
                        tokens=hidden_states,
                        token_layout=core_input.token_layout,
                        aux={"implementation": "packed_sequence_core"},
                    )
                )
        hidden_states = self.final_norm(hidden_states)
        cache_update_metadata = core_input.cache_update_metadata or CacheUpdateMetadata()
        has_runtime_sequence = core_input.sequence_metadata is not None
        layer_cache_entries = (
            tuple(
                AttentionCacheEntry(
                    metadata={
                        "layer_index": layer_index,
                        "sequence_length": layer_seq_len,
                        "current_start_frame": cache_update_metadata.current_start_frame,
                    }
                )
                for layer_index, layer_seq_len in enumerate([hidden_states.shape[1]] * len(self.blocks))
            )
            if has_runtime_sequence
            else tuple()
        )
        if core_input.cache_state is not None:
            cache_state = CacheState(
                supported=core_input.cache_state.supported or has_runtime_sequence,
                current_start_frame=cache_update_metadata.current_start_frame,
                cached_frames=core_input.cache_state.cached_frames,
                chunk_size=core_input.cache_state.chunk_size,
                capability=(
                    core_input.cache_state.capability
                    if core_input.cache_state.capability != "none"
                    else ("layer_placeholder" if has_runtime_sequence else "none")
                ),
                backend_name=core_input.cache_state.backend_name,
                backend_payload=core_input.cache_state.backend_payload,
                payload=dict(core_input.cache_state.payload),
                self_attention_kv=(
                    core_input.cache_state.self_attention_kv
                    if core_input.cache_state.self_attention_kv
                    else layer_cache_entries
                ),
                cross_attention_kv=core_input.cache_state.cross_attention_kv,
                update_metadata=cache_update_metadata,
                branch_states=dict(core_input.cache_state.branch_states),
            )
        else:
            cache_state = CacheState(
                supported=has_runtime_sequence,
                current_start_frame=cache_update_metadata.current_start_frame,
                cached_frames=0,
                chunk_size=hidden_states.shape[1],
                capability="layer_placeholder" if has_runtime_sequence else "none",
                backend_name="merged_prefix",
                backend_payload=None,
                payload={"stage": "visual_core"},
                self_attention_kv=layer_cache_entries,
                update_metadata=cache_update_metadata,
                branch_states={},
            )
        return VisualCoreOutput(
            tokens=hidden_states,
            token_layout=core_input.token_layout,
            cache_state=cache_state,
            intermediate_readouts=tuple(captured_readouts),
            aux={
                "used_attention_mask": core_input.attention_mask is not None,
                "has_sequence_metadata": core_input.sequence_metadata is not None,
                "cache_runtime_metadata": cache_update_metadata,
            },
        )

    def execute_runtime_step(self, step_input: RuntimeStepInput) -> RuntimeStepOutput:
        prepared = prepare_runtime_sequence(step_input, hidden_size=self.config.hidden_size)
        if prepared.mode != "core_input" or prepared.core_input is None:
            raise ValueError(
                "PackedSequenceVisualCore only supports runtime programs that resolve to `core_input`."
            )
        core_output = self.forward(prepared.core_input)
        core_output.aux.setdefault("runtime_program", step_input.program.name)
        core_output.aux.setdefault("sequence_family", step_input.program.sequence_family)
        return RuntimeStepOutput(
            tokens=core_output.tokens,
            core_output=core_output,
            cache_state=core_output.cache_state,
            aux={
                **core_output.aux,
                "runtime_program": step_input.program.name,
                "sequence_family": step_input.program.sequence_family,
            },
        )


LingbotVisualCore = PackedSequenceVisualCore
