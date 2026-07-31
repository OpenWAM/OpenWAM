"""Compatibility exports for shared WAN temporal geometry contracts.

New code should import these helpers from :mod:`open_wam.contracts`.
"""

from open_wam.contracts.video import (
    WAN_TEMPORAL_CHUNK_SIZE,
    wan_fully_observed_latent_count,
    wan_raw_frame_count_to_latent_count,
    wan_safe_temporal_frame_count,
)

__all__ = [
    "WAN_TEMPORAL_CHUNK_SIZE",
    "wan_fully_observed_latent_count",
    "wan_raw_frame_count_to_latent_count",
    "wan_safe_temporal_frame_count",
]
