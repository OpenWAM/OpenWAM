from __future__ import annotations

from .base import LinearActionDecoder


class MLPActionDecoder(LinearActionDecoder):
    """Default decoder for post-latent and parallel-stream variants."""
