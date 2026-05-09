from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class FdmAblationMode(StrEnum):
    """Forward-dynamics ablation mode for maintained M1.2 joint denoising."""

    FORCED_ACTION_JOINT_FDM = "forced_action_joint_fdm"
    VANILLA_JOINT_ROLLOUT = "vanilla_joint_rollout"
    CLEAN_ACTION_FEEDBACK = "clean_action_feedback"


class FdmStartPolicy(StrEnum):
    """Where to place t0 inside each selected trajectory window."""

    EARLY_MIDDLE = "early_middle"
    LATEST_FIT = "latest_fit"


@dataclass(frozen=True)
class FdmRunConfig:
    """Run-level choices that should be recorded in each manifest."""

    config_path: Path
    checkpoint_path: Path
    output_dir: Path
    horizon_frames: int
    trajectories_per_task: int
    seed: int
    video_fps: float
    modes: tuple[FdmAblationMode, ...] = (
        FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        FdmAblationMode.VANILLA_JOINT_ROLLOUT,
        FdmAblationMode.CLEAN_ACTION_FEEDBACK,
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FdmWindowSelection:
    """One deterministic held-out future-video window."""

    sample_index: int
    dataset_index: int
    task_key: str
    task_rank: int
    episode_index: int
    t0_frame: int
    horizon_frames: int
    generated_frames: int
    context_start_frame: int
    total_video_frames: int
    repo_root: str

    @property
    def target_start_frame(self) -> int:
        return self.t0_frame

    @property
    def target_end_frame(self) -> int:
        return self.t0_frame + self.horizon_frames

    @property
    def generation_end_frame(self) -> int:
        return self.t0_frame + self.generated_frames
