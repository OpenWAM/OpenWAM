"""Sample-isolated attention and layout composition, independent of model topology."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any

import torch

from .attention_backends import _resolve_compiled_create_block_mask, create_block_mask
from .packed_token_layout import PackedTokenLayout
from .attention_contracts import PreparedAttentionProfile


def _backend_mask(
    predicate,
    query_count: int,
    key_count: int,
    device: torch.device,
    *,
    build_sparse: bool | None = None,
):
    """Materialize only tiny CPU masks; CUDA retains sparse FlexAttention masks."""
    if query_count <= 0 or key_count <= 0:
        raise ValueError("Sequence attention requires nonempty query and key segments.")

    def bounded_predicate(b, h, q, k):
        # Flex kernels may evaluate rounded tile positions beyond real lengths.
        # Clamp before metadata gathers and explicitly mask those positions out.
        return (
            (q < query_count)
            & (k < key_count)
            & predicate(b, h, q.clamp(0, query_count - 1), k.clamp(0, key_count - 1))
        )

    sparse = device.type == "cuda" if build_sparse is None else build_sparse
    if not sparse:
        return bounded_predicate(
            None,
            None,
            torch.arange(query_count, device=device)[:, None],
            torch.arange(key_count, device=device)[None, :],
        ), None
    if create_block_mask is None:
        raise RuntimeError("Variable-length CUDA training requires FlexAttention.")
    compiled = _resolve_compiled_create_block_mask() if device.type == "cuda" else None
    return None, (compiled or create_block_mask)(
        bounded_predicate,
        1,
        1,
        query_count,
        key_count,
        device=str(device),
        _compile=compiled is not None,
    )


def combine_sequence_layouts(
    layouts: Sequence[PackedTokenLayout],
    stream_lengths: Sequence[Sequence[int]],
    stream_slots: Sequence[Sequence[int]],
) -> tuple[PackedTokenLayout, torch.Tensor]:
    """Arrange sample-local layouts stream-major, adding only transport padding.

    Each input contains one sequence with contiguous stream segments. Streams
    may be absent (zero length); sequence identities and validity remain explicit.
    """
    if not layouts or not stream_lengths or len(stream_lengths) != len(stream_slots):
        raise ValueError(
            "Sequence packing requires layouts and matching stream extents."
        )
    count = len(layouts)
    if any(len(row) != count for row in (*stream_lengths, *stream_slots)):
        raise ValueError("Each stream must describe every sequence.")
    names = [item.name for item in fields(PackedTokenLayout) if item.name != "metadata"]
    parts = {name: [] for name in names}
    local_indices = []
    offsets = [0] * count
    for lengths, slots in zip(stream_lengths, stream_slots, strict=True):
        for index, (layout, length, capacity) in enumerate(
            zip(layouts, lengths, slots, strict=True)
        ):
            start = offsets[index]
            if length < 0 or capacity < length or start + length > layout.token_count:
                raise ValueError(
                    "Stream extents do not fit their prepared token layout."
                )
            local_indices.append(
                torch.cat(
                    [
                        torch.arange(start, start + length, device=layout.device),
                        torch.full(
                            (capacity - length,),
                            -1,
                            device=layout.device,
                            dtype=torch.long,
                        ),
                    ]
                )
            )
            for name in names:
                value = getattr(layout, name)[start : start + length]
                if name == "seq_id":
                    value = torch.where(value >= 0, index, -1)
                fill = False if value.dtype == torch.bool else -1
                parts[name].append(
                    torch.cat([value, value.new_full((capacity - length,), fill)])
                )
            offsets[index] += length
    if any(
        offset != layout.token_count
        for offset, layout in zip(offsets, layouts, strict=True)
    ):
        raise ValueError(
            "Stream extents must cover the complete prepared token layout."
        )
    return PackedTokenLayout(
        **{name: torch.cat(values) for name, values in parts.items()}
    ), torch.cat(local_indices)


def build_sequence_batch_self_attention(
    profiles: Sequence[PreparedAttentionProfile],
    stream_lengths: Sequence[Sequence[int]],
    stream_slots: Sequence[Sequence[int]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, Any | None, PackedTokenLayout]:
    """Compose prepared sample-local rules without interpreting their semantics."""
    if not profiles or any(
        profile.token_layout is None
        or profile.token_layout.token_count == 0
        or profile.self_attention_visibility is None
        for profile in profiles
    ):
        raise ValueError(
            "Sequence batching requires a prepared token layout and visibility rule."
        )
    layouts = [profile.token_layout for profile in profiles]
    layout, local_indices = combine_sequence_layouts(
        layouts, stream_lengths, stream_slots
    )
    rules = tuple(profile.self_attention_visibility for profile in profiles)
    counts = tuple(item.token_count for item in layouts)

    def visibility(q, k):
        visible = torch.zeros_like(q + k, dtype=torch.bool)
        for index, (rule, count) in enumerate(zip(rules, counts, strict=True)):
            # Every predicate is evaluated by Flex, even for a different sample.
            # Bound local gathers before selecting the query's own rule.
            local = rule(
                local_indices[q].clamp(0, count - 1),
                local_indices[k].clamp(0, count - 1),
            )
            visible = visible | ((layout.seq_id[q] == index) & local)
        return visible

    dense, sparse = build_sequence_self_attention(layout, visibility, device=device)
    return dense, sparse, layout


def build_sequence_self_attention(
    layout: PackedTokenLayout,
    visibility: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, Any | None]:
    """Compose a consumer's local visibility with sequence/padding isolation."""

    def predicate(b, h, q, k):
        del b, h
        same_sample = (layout.seq_id[q] >= 0) & (layout.seq_id[q] == layout.seq_id[k])
        valid = layout.valid_as_query[q] & layout.valid_as_kv[k]
        return (same_sample & valid & visibility(q, k)) | (
            (layout.seq_id[q] < 0) & (q == k)
        )

    return _backend_mask(predicate, layout.token_count, layout.token_count, device)


def build_sequence_batch_cross_attention(
    query_lengths: Sequence[int],
    query_slots: Sequence[int],
    text_lengths: Sequence[int],
    local_masks: Sequence[torch.Tensor | None],
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, Any | None]:
    """Keep text/proprio conditioning in the source sample, including local masks."""
    if not query_lengths or not (
        len(query_lengths) == len(query_slots) == len(text_lengths) == len(local_masks)
    ):
        raise ValueError(
            "Cross-attention extents must describe the same nonempty sample batch."
        )
    if any(
        length < 0 or slots < length or text <= 0
        for length, slots, text in zip(
            query_lengths, query_slots, text_lengths, strict=True
        )
    ):
        raise ValueError(
            "Cross-attention requires valid query slots and nonempty per-sample context."
        )
    query_ids = torch.cat(
        [
            torch.full((slots,), i, device=device, dtype=torch.long)
            for i, slots in enumerate(query_slots)
        ]
    )
    text_ids = torch.cat(
        [
            torch.full((length,), i, device=device, dtype=torch.long)
            for i, length in enumerate(text_lengths)
        ]
    )
    # Local query-dependent masks are uncommon (legacy text-space proprio).
    # Keep their own small matrices, rather than building a dense global square.
    if any(mask is not None for mask in local_masks):
        max_text = max(text_lengths)
        rows = []
        for nq, sq, nt, mask in zip(
            query_lengths, query_slots, text_lengths, local_masks, strict=True
        ):
            local = torch.ones((sq, max_text), dtype=torch.bool, device=device)
            if mask is not None:
                local[:nq, :nt] = mask.reshape(nq, nt).to(
                    device=device, dtype=torch.bool
                )
            rows.append(local)
        permitted = torch.cat(rows)
        text_position = torch.cat(
            [torch.arange(n, device=device) for n in text_lengths]
        )

        def predicate(b, h, q, k):
            del b, h
            return (query_ids[q] == text_ids[k]) & permitted[q, text_position[k]]
    else:

        def predicate(b, h, q, k):
            del b, h
            return query_ids[q] == text_ids[k]

    return _backend_mask(predicate, sum(query_slots), sum(text_lengths), device)
