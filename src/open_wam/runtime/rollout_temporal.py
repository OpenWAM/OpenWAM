"""Derived temporal laws for rollout assembly, never a user-authored config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_wam.models.policy_variants import PolicyTemporalSpan
    from open_wam.pipelines import VariantPipeline


@dataclass(frozen=True)
class ResolvedRolloutTemporalContract:
    startup_frames: int
    controls_per_frame: int
    raw_frames_per_frame: int
    frame_chunk_size: int
    attention_window_size: int

    def __post_init__(self) -> None:
        if (
            min(
                self.startup_frames,
                self.controls_per_frame,
                self.raw_frames_per_frame,
                self.frame_chunk_size,
                self.attention_window_size,
            )
            <= 0
        ):
            raise ValueError("Rollout temporal dimensions must be positive.")

    @classmethod
    def from_pipeline(
        cls, pipeline: VariantPipeline
    ) -> ResolvedRolloutTemporalContract:
        contract = pipeline.policy_variant.rollout_contract
        geometry = pipeline.default_temporal_geometry
        return cls(
            startup_frames=contract.startup_observation_frames,
            controls_per_frame=contract.action_tokens_per_frame,
            raw_frames_per_frame=pipeline.visual_tower.frontend.temporal_stride,
            frame_chunk_size=geometry.frame_chunk_size,
            attention_window_size=geometry.attention_window_size,
        )

    @property
    def prediction_controls(self) -> int:
        return self.frame_chunk_size * self.controls_per_frame

    def control_for_frame(self, frame: int) -> int:
        if frame < self.startup_frames:
            raise ValueError(
                "Conditioning frames have no preceding generated controls."
            )
        return (frame - self.startup_frames) * self.controls_per_frame

    def frame_for_control(self, index: int) -> int:
        if index < 0:
            raise ValueError("Control indices must be nonnegative.")
        return self.startup_frames + index // self.controls_per_frame

    def control_offset(self, index: int) -> int:
        self.frame_for_control(index)
        return index % self.controls_per_frame

    def complete_control_end(self, index: int) -> int:
        return index - self.control_offset(index)

    def observed_span(self, start: int, end: int) -> PolicyTemporalSpan:
        from open_wam.models.policy_variants import PolicyTemporalSpan

        if self.control_offset(start) or self.control_offset(end) or end <= start:
            raise ValueError(
                "Observed history must contain complete model-frame groups."
            )
        return PolicyTemporalSpan(
            self.frame_for_control(start), (end - start) // self.controls_per_frame
        )

    def raw_window_frames(self, latent_frames: int) -> int:
        if latent_frames <= 0:
            raise ValueError("Observation windows must contain a positive frame count.")
        return 1 + (latent_frames - 1) * self.raw_frames_per_frame

    def validate_observations(
        self, span: PolicyTemporalSpan, *, observations: int, actions: int, latents: int
    ) -> None:
        count = span.frame_count * self.controls_per_frame
        if (
            observations != count + 1
            or actions != count
            or latents != span.frame_count + 1
        ):
            raise ValueError(
                "An observed interval must encode its anchor plus each complete "
                "executed model frame, with every actual executed control."
            )
