from __future__ import annotations


def init_parallel_cache(frame_chunk_size: int) -> dict[str, int]:
    return {
        "current_frame_start": 0,
        "chunk_index": 0,
        "frame_chunk_size": frame_chunk_size,
    }


def advance_parallel_cache(cache: dict[str, int]) -> dict[str, int]:
    updated = dict(cache)
    updated["chunk_index"] = updated.get("chunk_index", 0) + 1
    updated["current_frame_start"] = updated.get("current_frame_start", 0) + updated.get("frame_chunk_size", 1)
    return updated
