"""Sequence-isolated attention for padded and ragged dual-expert batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from typing import Any

import torch

from open_wam.configs import CurrentBlockCoupling, HistoryStreamVisibility
from open_wam.models.common.attention_backends import (
    _resolve_compiled_create_block_mask,
    create_block_mask,
)
from open_wam.models.common.attention_contracts import PreparedAttentionProfile
from open_wam.models.common.chunked_attention_visibility import (
    _build_chunked_self_attention_visibility,
)
from open_wam.models.common.packed_token_layout import (
    PackedTokenLayout,
    build_exact_video_action_token_layout,
)


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


def build_sequence_batch_self_attention(
    profiles: Sequence[PreparedAttentionProfile],
    video_lengths: Sequence[int],
    action_lengths: Sequence[int],
    video_slots: Sequence[int],
    action_slots: Sequence[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, Any | None, PackedTokenLayout]:
    """Pack `[sample V copies...] [sample A copies...]`, preserving each layout.

    Padded execution retains dummy slots but gives them only a diagonal edge.
    Dummy tokens can never become K/V for a real query, including at later layers.
    """
    if not profiles:
        raise ValueError("Sequence attention requires at least one sample.")
    shared_keys = (
        "chunk_size",
        "window_size",
        "current_block_coupling",
        "history_stream_visibility",
        "prefix_condition_frames",
        "conditional_history_policy",
    )
    meta = profiles[0].metadata
    if CurrentBlockCoupling(meta["current_block_coupling"]) not in {
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.JOINT,
    }:
        raise ValueError(
            "Sequence-batch attention currently supports VTA and Joint only."
        )
    for profile in profiles:
        if any(profile.metadata[key] != meta[key] for key in shared_keys):
            raise ValueError(
                "A sequence batch must share chunk/window and coupling semantics."
            )
        if profile.metadata["singleton_chunk_frame"] is not None:
            raise ValueError(
                "Variable-length batches do not yet support singleton chunk cutoffs."
            )
        if profile.metadata["conditional_history_policy"] != "none":
            raise ValueError(
                "Variable-length batches support ordinary VTA/Joint history only."
            )

    names = [
        field.name for field in fields(PackedTokenLayout) if field.name != "metadata"
    ]
    videos: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    actions: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for sample_id, (profile, nv, na, sv, sa) in enumerate(
        zip(
            profiles,
            video_lengths,
            action_lengths,
            video_slots,
            action_slots,
            strict=True,
        )
    ):
        item = profile.metadata
        _, _, frames, height, width = item["latent_shape"]
        _, _, action_frames, action_height, action_width = item["action_shape"]
        action_valid = item["action_context_valid_tokens"]
        action_mask = (
            None
            if action_valid is None
            else torch.tensor(
                action_valid,
                device=device,
                dtype=torch.bool,
            ).reshape(1, -1, 1)
        )
        layout = build_exact_video_action_token_layout(
            batch_size=1,
            latent_frames=frames,
            latent_height=height,
            latent_width=width,
            action_frames=action_frames,
            action_height=action_height,
            action_width=action_width,
            patch_size=(1, 1, 1),
            chunk_size=item["chunk_size"],
            chunk_origin_frame=item["chunk_origin_frame"],
            current_block_coupling=item["current_block_coupling"],
            device=device,
            action_context_mask=action_mask,
            prefix_condition_frames=item["prefix_condition_frames"],
        )
        if layout.token_count != nv + na or sv < nv or sa < na:
            raise ValueError(
                "Prepared sequence token lengths do not match their attention layout."
            )
        for name in names:
            value = getattr(layout, name)
            if name == "seq_id":
                value = torch.full_like(value, sample_id)
            for destination, start, length, slots in (
                (videos, 0, nv, sv),
                (actions, nv, na, sa),
            ):
                real = value[start : start + length]
                fill = False if real.dtype == torch.bool else -1
                destination[name].append(
                    torch.cat(
                        [
                            real,
                            real.new_full((slots - length,), fill),
                        ]
                    )
                )
    layout = PackedTokenLayout(
        **{name: torch.cat(videos[name] + actions[name]) for name in names}
    )

    def predicate(b, h, q, k):
        del b, h
        visible = _build_chunked_self_attention_visibility(
            q_seq=layout.seq_id[q],
            kv_seq=layout.seq_id[k],
            q_block_id=layout.block_id[q],
            kv_block_id=layout.block_id[k],
            q_chunk=layout.chunk_id[q],
            kv_chunk=layout.chunk_id[k],
            q_noise=layout.noise_id[q],
            kv_noise=layout.noise_id[k],
            q_stream=layout.stream_id[q],
            kv_stream=layout.stream_id[k],
            q_effective_frame=layout.frame_id[q],
            kv_effective_frame=layout.frame_id[k],
            q_valid=layout.valid_as_query[q],
            kv_valid=layout.valid_as_kv[k],
            window_size=meta["window_size"],
            chunk_size=meta["chunk_size"],
            chunk_origin_frame=0,
            prefix_condition_frames=meta["prefix_condition_frames"],
            singleton_chunk_frame=None,
            current_block_coupling=meta["current_block_coupling"],
            history_stream_visibility=HistoryStreamVisibility(
                meta["history_stream_visibility"]
            ),
            conditional_history_policy=meta["conditional_history_policy"],
        )
        return visible | ((layout.seq_id[q] < 0) & (q == k))

    dense, sparse = _backend_mask(
        predicate, layout.token_count, layout.token_count, device
    )
    return dense, sparse, layout


def build_sequence_batch_cross_attention(
    query_lengths: Sequence[int],
    query_slots: Sequence[int],
    text_lengths: Sequence[int],
    local_masks: Sequence[torch.Tensor | None],
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, Any | None]:
    """Keep text/proprio conditioning in the source sample, including local masks."""
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
