from __future__ import annotations


def init_register_cache(num_frame_per_block: int) -> dict[str, int]:
    return {
        "current_start_frame": 0,
        "block_index": 0,
        "num_frame_per_block": num_frame_per_block,
    }


def advance_register_cache(cache: dict[str, int]) -> dict[str, int]:
    updated = dict(cache)
    updated["block_index"] = updated.get("block_index", 0) + 1
    updated["current_start_frame"] = updated.get("current_start_frame", 0) + updated.get("num_frame_per_block", 1)
    return updated
