from __future__ import annotations

import torch

from open_wam.models.video_backbone.contracts import TokenGridMetadata


def build_mesh_id(
    f: int,
    h: int,
    w: int,
    t: float,
    *,
    f_w: float = 1.0,
    f_shift: float = 0.0,
    action: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build LingBot-style mesh ids with shape `[4, f * h * w]`."""

    f_idx = torch.arange(f, device=device, dtype=torch.float32) + float(f_shift)
    f_idx = f_idx * float(f_w)
    h_idx = torch.arange(h, device=device, dtype=torch.float32)
    w_idx = torch.arange(w, device=device, dtype=torch.float32)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.arange(1, h + 1, device=device, dtype=torch.float32) / float(h + 1)).view(1, h, 1)
        ff = ff + ff_offset
        hh = torch.full_like(hh, -1.0)
        ww = torch.full_like(ww, -1.0)
    grid_id = torch.cat(
        [
            ff.unsqueeze(0),
            hh.unsqueeze(0),
            ww.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)
    t_row = torch.full_like(grid_id[:1], float(t))
    return torch.cat([grid_id, t_row], dim=0)


def build_video_grid_ids(
    token_grid: TokenGridMetadata,
    *,
    device: torch.device,
    frame_shift: float = 0.0,
) -> torch.Tensor:
    """Build LingBot-style video grid ids for one video token sequence."""

    patch_t, _, _ = token_grid.patch_size
    post_patch_frames = token_grid.num_frames // patch_t
    return build_mesh_id(
        f=post_patch_frames,
        h=token_grid.patches_per_frame_h,
        w=token_grid.patches_per_frame_w,
        t=0.0,
        f_shift=frame_shift,
        device=device,
    )


def build_action_grid_ids(
    *,
    num_frames: int,
    action_per_frame: int,
    device: torch.device,
    frame_shift: float = 0.0,
) -> torch.Tensor:
    """Build LingBot-style action grid ids for a packed action-token stream."""

    return build_mesh_id(
        f=num_frames,
        h=action_per_frame,
        w=1,
        t=0.0,
        f_shift=frame_shift,
        action=True,
        device=device,
    )


def build_sequence_grid_ids(length: int, *, device: torch.device, offset: float = 0.0) -> torch.Tensor:
    """Build simple sequential grid ids for non-video packed tokens."""

    return build_mesh_id(f=length, h=1, w=1, t=0.0, f_shift=offset, device=device)


def build_block_register_grid_ids(
    *,
    num_blocks: int,
    tokens_per_block: int,
    device: torch.device,
    frame_shift: float = 0.0,
    stream_marker: float = -1.0,
) -> torch.Tensor:
    """Build rollout-aware grid ids for structured register-token streams.

    This keeps register tokens aligned to their corresponding future-video
    blocks instead of treating the entire register suffix as one flat 1D
    sequence. `stream_marker` lets different register streams occupy distinct
    spatial marker lanes while still sharing the same temporal block index.
    """

    if num_blocks <= 0 or tokens_per_block <= 0:
        return torch.zeros(4, 0, device=device, dtype=torch.float32)
    frame_ids = (
        torch.arange(num_blocks, device=device, dtype=torch.float32) + float(frame_shift)
    ).repeat_interleave(tokens_per_block)
    marker = torch.full_like(frame_ids, float(stream_marker))
    zeros = torch.zeros_like(frame_ids)
    return torch.stack([frame_ids, marker, marker, zeros], dim=0)
