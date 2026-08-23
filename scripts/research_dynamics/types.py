from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from open_wam.configs import DynamicsObjective


class FdmAblationMode(StrEnum):
    """Forward-dynamics ablation mode for parallel-stream joint denoising."""

    FORCED_ACTION_JOINT_FDM = "forced_action_joint_fdm"
    VIDEO_CONDITIONED_ACTION = "video_conditioned_action"
    VANILLA_JOINT_ROLLOUT = "vanilla_joint_rollout"
    CLEAN_ACTION_FEEDBACK = "clean_action_feedback"


def dynamics_objective_for_ablation_mode(
    mode: FdmAblationMode,
) -> DynamicsObjective:
    """Translate a research intervention into the core model objective."""

    if mode in {
        FdmAblationMode.VANILLA_JOINT_ROLLOUT,
        FdmAblationMode.CLEAN_ACTION_FEEDBACK,
    }:
        return DynamicsObjective.JOINT
    if mode == FdmAblationMode.FORCED_ACTION_JOINT_FDM:
        return DynamicsObjective.ACTION_CONDITIONED_VIDEO
    if mode == FdmAblationMode.VIDEO_CONDITIONED_ACTION:
        return DynamicsObjective.VIDEO_CONDITIONED_ACTION
    raise AssertionError(f"Unhandled research dynamics mode {mode!r}.")


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
        FdmAblationMode.VIDEO_CONDITIONED_ACTION,
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
    target_start_offset_frames: int = 0
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_start_frame(self) -> int:
        return self.t0_frame + self.target_start_offset_frames

    @property
    def target_end_frame(self) -> int:
        return self.target_start_frame + self.horizon_frames

    @property
    def generation_end_frame(self) -> int:
        return self.target_start_frame + self.generated_frames
