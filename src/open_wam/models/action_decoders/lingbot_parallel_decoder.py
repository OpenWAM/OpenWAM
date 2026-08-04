"""Deprecated import facade for the parallel-stream action decoder."""

from .parallel_stream_decoder import (
    LingbotParallelActionDecoder,
    ParallelStreamActionDecoder,
)

_COMPATIBILITY_EXPORTS = (
    LingbotParallelActionDecoder,
    ParallelStreamActionDecoder,
)

__all__ = [
    "LingbotParallelActionDecoder",
    "ParallelStreamActionDecoder",
]
