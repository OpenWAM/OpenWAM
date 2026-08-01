"""Typed contracts for LIBERO rollout artifact production."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from open_wam.configs.enums import RolloutArtifactProfile


@dataclass(frozen=True)
class LiberoRolloutArtifactIdentity:
    """Stable coordinates used to derive one episode's artifact paths."""

    benchmark: str
    task_id: int
    prompt: str
    episode_idx: int
    success: bool
    suffix: str


@dataclass(frozen=True)
class LiberoRolloutArtifactOptions:
    """Output and video choices independent from simulator execution."""

    output_root: Path
    video_fps: float
    save_rollout_video: bool = False
    skip_comparison_video: bool = False


@dataclass(frozen=True)
class LiberoRolloutArtifactPayload:
    """Episode traces consumed only by artifact rendering and persistence."""

    real_observations: Sequence[Mapping[str, np.ndarray]]
    predicted_latent_chunks: Sequence[torch.Tensor]
    action_trace: Sequence[np.ndarray]
    chunk_events: Sequence[Mapping[str, object]]
    component_report: Mapping[str, object]


@dataclass(frozen=True)
class LiberoRolloutArtifactOutput:
    """Persisted artifact paths plus the path-enriched legacy summary."""

    summary: dict[str, object]
    summary_path: Path
    action_trace_path: Path
    chunk_events_path: Path
    component_report_path: Path
    comparison_video_path: Path | None
    rollout_video_path: Path | None


@dataclass(frozen=True)
class RolloutArtifactPolicy:
    """Artifact profile decisions shared by collection and persistence."""

    profile: RolloutArtifactProfile
    write_fallback_timeline_video: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", RolloutArtifactProfile(self.profile))

    @classmethod
    def from_value(
        cls,
        profile: RolloutArtifactProfile | str,
        *,
        write_fallback_timeline_video: bool = False,
    ) -> RolloutArtifactPolicy:
        return cls(
            profile=RolloutArtifactProfile(profile),
            write_fallback_timeline_video=bool(write_fallback_timeline_video),
        )

    @property
    def writes_rollout_video(self) -> bool:
        return self.profile in {
            RolloutArtifactProfile.STANDARD,
            RolloutArtifactProfile.DEBUG,
        }

    @property
    def writes_debug_artifacts(self) -> bool:
        return self.profile in {
            RolloutArtifactProfile.STANDARD,
            RolloutArtifactProfile.DEBUG,
        }

    @property
    def writes_fallback_timeline_video(self) -> bool:
        return (
            self.profile is RolloutArtifactProfile.DEBUG
            or self.write_fallback_timeline_video
        )

    @property
    def collects_video_records(self) -> bool:
        return self.writes_rollout_video or self.writes_fallback_timeline_video


@dataclass(frozen=True)
class LiberoRealtimeArtifactIdentity:
    """Stable coordinates used for one realtime episode's artifact stem."""

    benchmark: str
    task_id: int
    prompt: str
    episode_idx: int
    suffix: str


@dataclass(frozen=True)
class LiberoRealtimeArtifactOptions:
    """Realtime output choices independent from simulator execution."""

    output_root: Path
    video_fps: float
    action_per_frame: int
    policy: RolloutArtifactPolicy


@dataclass(frozen=True)
class LiberoRealtimeArtifactPayload:
    """Realtime traces consumed only by artifact rendering and persistence."""

    action_records: Sequence[Mapping[str, Any]]
    action_video_records: Sequence[Mapping[str, Any]]
    replan_records: Sequence[Mapping[str, Any]]
    extension_records: Sequence[Mapping[str, Any]]
    component_report: Mapping[str, Any]
    startup_debug_report: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LiberoRealtimeArtifactOutput:
    """Persisted realtime paths plus the path-enriched summary."""

    summary: dict[str, Any]
    summary_path: Path
    video_path: Path | None
    fallback_timeline_video_path: Path | None
    action_trace_path: Path | None
    replan_trace_path: Path | None
    extension_trace_path: Path | None
    load_report_path: Path | None
    startup_debug_path: Path | None


@dataclass(frozen=True)
class LiberoExactStartupDebugOptions:
    """Stable runtime metadata for one exact-rollout startup report."""

    prompt: str
    seed: int
    runtime_device: torch.device
    frontend_device: torch.device
    decode_device: torch.device
    reference_assets_device_policy: str
    runtime_mode: str
    video_num_inference_steps: int
    action_num_inference_steps: int
    guidance_scale: float
    action_guidance_scale: float
    frame_chunk_size: int
    action_per_frame: int
    exact_startup_bootstrap_padding: bool = False
    startup_warmup_debug: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LiberoExactStartupDebugPayload:
    """Observed and model-produced values summarized by a startup report."""

    first_observation: Mapping[str, np.ndarray]
    video_latents: torch.Tensor | None
    text_context: torch.Tensor | None
    negative_text_context: torch.Tensor | None
    session_text_context: torch.Tensor | None
    session_negative_text_context: torch.Tensor | None
    rng_before_startup_infer: Mapping[str, Any]
    rng_after_startup_infer: Mapping[str, Any]
    first_chunk_debug: Mapping[str, Any]
    chunk_action_pred: torch.Tensor | None
    raw_chunk_action_pred: torch.Tensor | None
    predicted_latents: torch.Tensor | None
