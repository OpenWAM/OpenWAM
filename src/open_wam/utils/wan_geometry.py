from __future__ import annotations


WAN_TEMPORAL_CHUNK_SIZE = 4


def wan_safe_temporal_frame_count(num_frames: int, *, cache_initialized: bool) -> int:
    """Return raw frames consumed by Diffusers Wan VAE temporal chunking.

    AutoencoderKLWan encodes a fresh clip as frame 0 plus complete 4-frame
    groups after it. In streaming mode, every emitted latent comes from one
    complete 4-frame group. Incomplete tail frames are not encoded into a
    latent by the reference implementation.
    """

    if num_frames <= 0:
        raise ValueError(f"Wan VAE encoding requires at least one frame, got num_frames={num_frames}.")
    if cache_initialized:
        return WAN_TEMPORAL_CHUNK_SIZE * (num_frames // WAN_TEMPORAL_CHUNK_SIZE)
    return 1 + WAN_TEMPORAL_CHUNK_SIZE * ((num_frames - 1) // WAN_TEMPORAL_CHUNK_SIZE)


def wan_raw_frame_count_to_latent_count(num_frames: int) -> int:
    """Map a fresh Wan VAE raw-frame span to Diffusers' latent-frame count."""

    if num_frames <= 0:
        raise ValueError(f"Wan VAE encoding requires at least one frame, got num_frames={num_frames}.")
    return 1 + (num_frames - 1) // WAN_TEMPORAL_CHUNK_SIZE


def wan_fully_observed_latent_count(raw_observed_frames: int) -> int:
    """Count Wan latent frames whose full raw support lies inside the observed prefix."""

    if raw_observed_frames <= 0:
        raise ValueError(
            f"Wan observed-prefix mapping requires at least one raw frame, got {raw_observed_frames}."
        )
    return 1 + max(0, (raw_observed_frames - 1) // WAN_TEMPORAL_CHUNK_SIZE)
