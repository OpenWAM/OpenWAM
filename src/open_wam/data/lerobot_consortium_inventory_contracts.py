"""Dependency-light records and scalar coercion for consortium inventory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LeRobotConsortiumRepoTarget:
    """One input Hugging Face dataset repo selected for consortium indexing."""

    repo_id: str
    source_group: str


@dataclass(frozen=True)
class LeRobotConsortiumInventoryRow:
    """Spreadsheet-style inventory row for one HF LeRobot-format dataset."""

    source_group: str
    repo_id: str
    private: bool | None
    domain_type: str
    total_size_mb: float | None
    data_size_mb: float | None
    video_size_mb: float | None
    total_episodes: int | None
    total_frames: int | None
    total_tasks: int | None
    total_hours: float | None
    avg_seconds_per_episode: float | None
    fps: float | None
    observation_fps: float | None
    action_fps: float | None
    robot_type: str | None
    embodiment_type: str
    embodiment_confidence: str
    embodiment_reason: str
    action_dim: int | None
    action_shape: str | None
    state_dim: int | None
    state_shape: str | None
    visual_stream_count: int
    visual_stream_keys: str
    visual_dimensions: str
    visual_dtypes: str
    text_annotation_extent: str
    task_text_present: bool
    task_text_count: int | None
    task_text_examples: str
    temporal_dense_present: bool
    temporal_sparse_present: bool
    language_feature_keys: str
    readme_url: str
    dataset_url: str
    generation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    return None


__all__ = [
    "LeRobotConsortiumInventoryRow",
    "LeRobotConsortiumRepoTarget",
]
