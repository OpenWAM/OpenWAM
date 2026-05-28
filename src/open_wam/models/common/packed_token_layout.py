from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch

from open_wam.configs import CurrentBlockCoupling


class PackedTokenKind(IntEnum):
    """Token kind ids used by packed video/action attention layouts."""

    VIDEO_NOISY = 0
    VIDEO_CLEAN = 1
    ACTION_NOISY = 2
    ACTION_CLEAN = 3
    TEXT = 4
    PROPRIO = 5
    PADDING = -1


class PackedTokenStream(IntEnum):
    """Stream ids used by packed video/action attention layouts."""

    VIDEO = 0
    ACTION = 1
    TEXT = 2
    PROPRIO = 3
    PADDING = -1


@dataclass(frozen=True)
class PackedTokenLayout:
    """Source-of-truth metadata for packed transformer tokens.

    The important distinction is that a token can be valid as a query while
    being invalid as K/V context. Strict one-frame startup uses this for dummy
    action-prefix tokens: their rows must remain finite, but no valid token
    should attend to them as context.
    """

    token_kind: torch.Tensor
    seq_id: torch.Tensor
    frame_id: torch.Tensor
    chunk_id: torch.Tensor
    block_id: torch.Tensor
    stream_id: torch.Tensor
    noise_id: torch.Tensor
    valid_as_query: torch.Tensor
    valid_as_kv: torch.Tensor
    valid_for_loss: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fields = (
            self.token_kind,
            self.seq_id,
            self.frame_id,
            self.chunk_id,
            self.block_id,
            self.stream_id,
            self.noise_id,
            self.valid_as_query,
            self.valid_as_kv,
            self.valid_for_loss,
        )
        lengths = {int(field.numel()) for field in fields}
        if len(lengths) != 1:
            raise ValueError(f"PackedTokenLayout fields must have equal lengths, got {sorted(lengths)}.")
        for field_name, value in (
            ("token_kind", self.token_kind),
            ("seq_id", self.seq_id),
            ("frame_id", self.frame_id),
            ("chunk_id", self.chunk_id),
            ("block_id", self.block_id),
            ("stream_id", self.stream_id),
            ("noise_id", self.noise_id),
            ("valid_as_query", self.valid_as_query),
            ("valid_as_kv", self.valid_as_kv),
            ("valid_for_loss", self.valid_for_loss),
        ):
            if value.ndim != 1:
                raise ValueError(f"PackedTokenLayout.{field_name} must be 1-D, got {tuple(value.shape)}.")

    @property
    def token_count(self) -> int:
        return int(self.seq_id.numel())

    @property
    def device(self) -> torch.device:
        return self.seq_id.device

    def with_padding(self, padded_length: int) -> PackedTokenLayout:
        padded_length = int(padded_length)
        if padded_length < 0:
            raise ValueError(f"Expected padded_length >= 0, got {padded_length}.")
        if padded_length == 0:
            return self
        return PackedTokenLayout(
            token_kind=torch.nn.functional.pad(
                self.token_kind,
                (0, padded_length),
                value=int(PackedTokenKind.PADDING),
            ),
            seq_id=torch.nn.functional.pad(self.seq_id, (0, padded_length), value=-1),
            frame_id=torch.nn.functional.pad(self.frame_id, (0, padded_length), value=-1),
            chunk_id=torch.nn.functional.pad(self.chunk_id, (0, padded_length), value=-1),
            block_id=torch.nn.functional.pad(self.block_id, (0, padded_length), value=-1),
            stream_id=torch.nn.functional.pad(
                self.stream_id,
                (0, padded_length),
                value=int(PackedTokenStream.PADDING),
            ),
            noise_id=torch.nn.functional.pad(self.noise_id, (0, padded_length), value=-1),
            valid_as_query=torch.nn.functional.pad(self.valid_as_query, (0, padded_length), value=False),
            valid_as_kv=torch.nn.functional.pad(self.valid_as_kv, (0, padded_length), value=False),
            valid_for_loss=torch.nn.functional.pad(self.valid_for_loss, (0, padded_length), value=False),
            metadata={**self.metadata, "padded_length": padded_length},
        )


def flatten_action_token_mask(
    action_context_mask: torch.Tensor,
    *,
    batch_size: int,
    action_frames: int,
    action_height: int,
    action_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Return per-action-token K/V visibility in packed token order."""

    token_count = int(action_frames) * int(action_height) * int(action_width)
    mask = action_context_mask.to(device=device)
    if mask.ndim == 5:
        # [B, C, F, H, W] action-latent mask. Collapse action channels because
        # exact packed attention has one token per [F, H, W] slot.
        if tuple(int(dim) for dim in mask.shape[2:]) != (
            int(action_frames),
            int(action_height),
            int(action_width),
        ):
            raise ValueError(
                "action_context_mask shape does not match action token geometry: "
                f"mask={tuple(mask.shape)}, expected trailing=({action_frames}, {action_height}, {action_width})."
            )
        token_valid = mask.float().amax(dim=1).reshape(int(mask.shape[0]), token_count) > 0
    elif mask.ndim == 4:
        if tuple(int(dim) for dim in mask.shape[1:]) != (
            int(action_frames),
            int(action_height),
            int(action_width),
        ):
            raise ValueError(
                "action_context_mask shape does not match action token geometry: "
                f"mask={tuple(mask.shape)}, expected [B, {action_frames}, {action_height}, {action_width}]."
            )
        token_valid = mask.reshape(int(mask.shape[0]), token_count).bool()
    elif mask.ndim == 3 and int(mask.shape[1]) == token_count:
        # [B, T_action, C] sequence mask. Collapse feature/channel dim.
        token_valid = mask.float().amax(dim=-1) > 0
    elif mask.ndim == 2 and int(mask.shape[1]) == token_count:
        token_valid = mask.bool()
    else:
        raise ValueError(
            "Unsupported action_context_mask shape. Expected [B,C,F,H,W], [B,F,H,W], "
            f"[B,T,C], or [B,T] for token_count={token_count}; got {tuple(mask.shape)}."
        )

    if int(token_valid.shape[0]) != int(batch_size):
        if int(batch_size) == 1:
            # Packed shared profiles are batch-agnostic. A token is visible
            # only if every sample in the runtime batch says it is real.
            token_valid = token_valid.all(dim=0, keepdim=True)
        else:
            raise ValueError(
                "action_context_mask batch size does not match attention profile batch size: "
                f"mask_batch={int(token_valid.shape[0])}, profile_batch={int(batch_size)}."
            )
    return token_valid.reshape(-1).to(device=device, dtype=torch.bool)


def build_exact_video_action_token_layout(
    *,
    batch_size: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    action_frames: int,
    action_height: int,
    action_width: int,
    patch_size: tuple[int, int, int],
    chunk_size: int,
    chunk_origin_frame: int,
    current_block_coupling: CurrentBlockCoupling | str,
    device: torch.device,
    action_context_mask: torch.Tensor | None = None,
    prefix_condition_frames: int = 0,
) -> PackedTokenLayout:
    """Build `[V_noisy, V_clean, A_noisy, A_clean]` packed-token metadata."""

    batch_size = int(batch_size)
    latent_frames = int(latent_frames)
    latent_height = int(latent_height)
    latent_width = int(latent_width)
    action_frames = int(action_frames)
    action_height = int(action_height)
    action_width = int(action_width)
    patch_t, patch_h, patch_w = (int(v) for v in patch_size)
    chunk_size = int(chunk_size)
    chunk_origin_frame = int(chunk_origin_frame)
    coupling = CurrentBlockCoupling(current_block_coupling)
    if batch_size <= 0:
        raise ValueError(f"Expected batch_size > 0, got {batch_size}.")
    if patch_t <= 0 or patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"Expected positive patch_size, got {patch_size}.")
    if latent_frames % patch_t != 0 or latent_height % patch_h != 0 or latent_width % patch_w != 0:
        raise ValueError(
            "Latent shape must be divisible by patch_size, "
            f"got latent=({latent_frames}, {latent_height}, {latent_width}), patch={patch_size}."
        )
    if chunk_size <= 0:
        raise ValueError(f"Expected chunk_size > 0, got {chunk_size}.")
    prefix_condition_frames = max(0, int(prefix_condition_frames))
    if prefix_condition_frames > 0 and prefix_condition_frames >= latent_frames:
        raise ValueError(
            "`prefix_condition_frames` must be smaller than latent_frames, "
            f"got prefix_condition_frames={prefix_condition_frames}, latent_frames={latent_frames}."
        )

    latent_seq_id = (
        torch.arange(batch_size, device=device)[:, None, None, None]
        .expand(-1, latent_frames // patch_t, latent_height // patch_h, latent_width // patch_w)
        .flatten()
    )
    action_seq_id = (
        torch.arange(batch_size, device=device)[:, None, None, None]
        .expand(-1, action_frames, action_height, action_width)
        .flatten()
    )
    latent_token_valid = torch.ones_like(latent_seq_id, dtype=torch.bool)
    if action_context_mask is not None:
        action_token_valid = flatten_action_token_mask(
            action_context_mask,
            batch_size=batch_size,
            action_frames=action_frames,
            action_height=action_height,
            action_width=action_width,
            device=device,
        )
    else:
        action_token_valid = torch.ones_like(action_seq_id, dtype=torch.bool)

    latent_frame_id = (
        torch.arange(latent_frames // patch_t, device=device)[None, :, None, None]
        .expand(batch_size, -1, latent_height // patch_h, latent_width // patch_w)[None]
        .flatten()
    )
    action_frame_id = (
        torch.arange(action_frames, device=device)[None, :, None, None]
        .expand(batch_size, -1, action_height, action_width)[None]
        .flatten()
    )
    if prefix_condition_frames > 0:
        latent_target_frame_id = (latent_frame_id - prefix_condition_frames).clamp_min(0)
        target_latent_chunk_id = torch.div(
            latent_target_frame_id - chunk_origin_frame,
            chunk_size,
            rounding_mode="floor",
        )
        latent_chunk_id = torch.where(
            latent_frame_id < prefix_condition_frames,
            torch.zeros_like(target_latent_chunk_id),
            target_latent_chunk_id + 1,
        )
        action_chunk_id = (
            torch.div(action_frame_id - chunk_origin_frame, chunk_size, rounding_mode="floor") + 1
        )
    else:
        latent_chunk_id = torch.div(latent_frame_id - chunk_origin_frame, chunk_size, rounding_mode="floor")
        action_chunk_id = torch.div(action_frame_id - chunk_origin_frame, chunk_size, rounding_mode="floor")
    if prefix_condition_frames > 0:
        latent_block_id = torch.where(
            latent_frame_id < prefix_condition_frames,
            torch.zeros_like(latent_frame_id),
            latent_chunk_id * 2,
        )
        action_block_id = action_chunk_id * 2 + 1
    elif coupling in {CurrentBlockCoupling.ACTION_THEN_VIDEO, CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO}:
        latent_block_id = latent_chunk_id * 2 + 1
        action_block_id = action_chunk_id * 2
    else:
        latent_block_id = latent_chunk_id * 2
        action_block_id = action_chunk_id * 2 + 1

    seq_id = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)
    frame_id = torch.cat([latent_frame_id] * 2 + [action_frame_id] * 2)
    chunk_id = torch.cat([latent_chunk_id] * 2 + [action_chunk_id] * 2)
    block_id = torch.cat([latent_block_id] * 2 + [action_block_id] * 2)
    stream_id = torch.cat(
        [
            torch.full_like(latent_frame_id, int(PackedTokenStream.VIDEO)),
            torch.full_like(latent_frame_id, int(PackedTokenStream.VIDEO)),
            torch.full_like(action_frame_id, int(PackedTokenStream.ACTION)),
            torch.full_like(action_frame_id, int(PackedTokenStream.ACTION)),
        ]
    )
    noise_id = torch.cat(
        [
            torch.zeros_like(latent_frame_id),
            torch.ones_like(latent_frame_id),
            torch.zeros_like(action_frame_id),
            torch.ones_like(action_frame_id),
        ]
    )
    token_kind = torch.cat(
        [
            torch.full_like(latent_frame_id, int(PackedTokenKind.VIDEO_NOISY)),
            torch.full_like(latent_frame_id, int(PackedTokenKind.VIDEO_CLEAN)),
            torch.full_like(action_frame_id, int(PackedTokenKind.ACTION_NOISY)),
            torch.full_like(action_frame_id, int(PackedTokenKind.ACTION_CLEAN)),
        ]
    )
    # Dummy strict-startup action-prefix tokens must still be legal query rows
    # but never K/V context. Structural loss eligibility is narrower: only the
    # noisy target copies can be supervised, and objective-specific loss masks
    # can narrow this further upstream.
    valid_as_query = torch.cat([latent_token_valid] * 2 + [torch.ones_like(action_token_valid)] * 2)
    valid_as_kv = torch.cat([latent_token_valid] * 2 + [action_token_valid] * 2)
    valid_for_loss = torch.cat(
        [
            latent_token_valid,
            torch.zeros_like(latent_token_valid),
            action_token_valid,
            torch.zeros_like(action_token_valid),
        ]
    )
    return PackedTokenLayout(
        token_kind=token_kind,
        seq_id=seq_id,
        frame_id=frame_id,
        chunk_id=chunk_id,
        block_id=block_id,
        stream_id=stream_id,
        noise_id=noise_id,
        valid_as_query=valid_as_query,
        valid_as_kv=valid_as_kv,
        valid_for_loss=valid_for_loss,
        metadata={
            "batch_size": batch_size,
            "latent_frames": latent_frames,
            "action_frames": action_frames,
            "chunk_size": chunk_size,
            "chunk_origin_frame": chunk_origin_frame,
            "prefix_condition_frames": prefix_condition_frames,
            "current_block_coupling": coupling.value,
        },
    )
