from __future__ import annotations

from collections.abc import Sequence
from operator import index
from typing import Any

import torch

from open_wam.contracts.video import CanonicalViewLayout, ViewPlacement


def assemble_latent_views(
    latents_by_slot: Sequence[torch.Tensor],
    *,
    slots: Sequence[str],
    canvas_view_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Assemble 1-4 same-resolution latent views into a deterministic canvas."""

    latents = tuple(latents_by_slot)
    slot_names = tuple(str(slot) for slot in slots)
    if len(latents) != len(slot_names):
        raise ValueError(
            f"Expected one latent tensor per slot, got {len(latents)} tensors "
            f"and {len(slot_names)} slots."
        )
    if not 1 <= len(latents) <= 4:
        raise ValueError(
            f"Mixed-video latent view assembly supports 1 to 4 views, got {len(latents)}."
        )
    first = latents[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("Latent views must be torch.Tensor instances.")
    if first.ndim != 4:
        raise ValueError(
            f"Expected latent views shaped [C,T,H,W], got {tuple(first.shape)}."
        )
    channels, frames, height, width = (int(value) for value in first.shape)
    for slot, latent in zip(slot_names, latents, strict=True):
        if not isinstance(latent, torch.Tensor):
            raise TypeError(f"Latent view {slot!r} must be a torch.Tensor.")
        if latent.ndim != 4:
            raise ValueError(
                f"Expected latent view {slot!r} shaped [C,T,H,W], "
                f"got {tuple(latent.shape)}."
            )
        if tuple(int(value) for value in latent.shape) != (
            channels,
            frames,
            height,
            width,
        ):
            raise ValueError(
                "Mixed-video latent view assembly requires same-resolution views, "
                f"got first={(channels, frames, height, width)} and "
                f"{slot!r}={tuple(latent.shape)}."
            )
        if latent.dtype != first.dtype:
            raise ValueError(
                "Mixed-video latent view assembly requires one dtype, "
                f"got first={first.dtype} and {slot!r}={latent.dtype}."
            )
        if latent.device != first.device:
            raise ValueError(
                "Mixed-video latent view assembly requires one device, "
                f"got first={first.device} and {slot!r}={latent.device}."
            )

    if canvas_view_count is None:
        resolved_canvas_views = len(latents)
    else:
        if isinstance(canvas_view_count, bool):
            raise ValueError(
                "Mixed-video latent assembly canvas_view_count must be an integer."
            )
        try:
            resolved_canvas_views = index(canvas_view_count)
        except TypeError as exc:
            raise ValueError(
                "Mixed-video latent assembly canvas_view_count must be an integer."
            ) from exc
    if not 1 <= resolved_canvas_views <= 4:
        raise ValueError(
            "Mixed-video latent assembly canvas supports 1 to 4 views, "
            f"got {resolved_canvas_views}."
        )
    if resolved_canvas_views < len(latents):
        raise ValueError(
            f"Assembly canvas for {resolved_canvas_views} views cannot hold "
            f"{len(latents)} selected views."
        )
    canvas_height, canvas_width = _latent_assembly_canvas_shape(
        resolved_canvas_views,
        view_height=height,
        view_width=width,
    )
    canvas = first.new_zeros(channels, frames, canvas_height, canvas_width)
    placements = _latent_assembly_placements(
        selected_view_count=len(latents),
        canvas_view_count=resolved_canvas_views,
        view_height=height,
        view_width=width,
    )
    view_placements: list[ViewPlacement] = []
    for slot, latent, (top, left) in zip(
        slot_names,
        latents,
        placements,
        strict=True,
    ):
        canvas[:, :, top : top + height, left : left + width] = latent
        view_placements.append(
            ViewPlacement(
                source_name=slot,
                canonical_name=slot,
                top=int(top),
                left=int(left),
                height=int(height),
                width=int(width),
            )
        )
    layout = CanonicalViewLayout(
        canvas_height=int(canvas_height),
        canvas_width=int(canvas_width),
        placements=tuple(view_placements),
    )
    return canvas.contiguous(), layout.to_metadata()


def _latent_assembly_canvas_shape(
    view_count: int,
    *,
    view_height: int,
    view_width: int,
) -> tuple[int, int]:
    if view_count == 1:
        return int(view_height), int(view_width)
    if view_count == 2:
        return int(view_height), int(view_width) * 2
    if view_count in {3, 4}:
        return int(view_height) * 2, int(view_width) * 2
    raise ValueError(
        f"Mixed-video latent assembly supports 1 to 4 views, got {view_count}."
    )


def _latent_assembly_placements(
    *,
    selected_view_count: int,
    canvas_view_count: int,
    view_height: int,
    view_width: int,
) -> tuple[tuple[int, int], ...]:
    canvas_height, canvas_width = _latent_assembly_canvas_shape(
        canvas_view_count,
        view_height=view_height,
        view_width=view_width,
    )
    if selected_view_count == 1:
        return (
            (
                max(0, (canvas_height - view_height) // 2),
                max(0, (canvas_width - view_width) // 2),
            ),
        )
    if selected_view_count == 2:
        return ((0, 0), (0, view_width))
    if selected_view_count == 3:
        return (
            (0, 0),
            (0, view_width),
            (view_height, max(0, (canvas_width - view_width) // 2)),
        )
    if selected_view_count == 4:
        return (
            (0, 0),
            (0, view_width),
            (view_height, 0),
            (view_height, view_width),
        )
    raise ValueError(
        "Mixed-video latent assembly supports 1 to 4 views, "
        f"got {selected_view_count}."
    )


assemble_mixed_video_latent_views = assemble_latent_views


__all__ = [
    "assemble_latent_views",
    "assemble_mixed_video_latent_views",
]
