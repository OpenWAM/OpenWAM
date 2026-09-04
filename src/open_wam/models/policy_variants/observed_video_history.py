"""Policy-independent recurrent history for video-producing rollouts."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from open_wam.models.common.temporal_windows import (
    resolve_one_frame_conditioned_history_window,
)

from .contracts import (
    PolicyObservedHistory,
    PolicyTemporalGeometry,
    PolicyTemporalSpan,
)


def _validate_video_latents(video_latents: torch.Tensor, *, label: str) -> None:
    if video_latents.ndim != 5 or int(video_latents.shape[2]) <= 0:
        raise ValueError(
            f"{label} must be non-empty [B, C, T, H, W] latents, "
            f"got {tuple(video_latents.shape)}."
        )


@dataclass(frozen=True, slots=True)
class ObservedVideoHistoryState:
    """Real video history plus one outstanding speculative generation span.

    Generated frames are deliberately represented only by ``pending_span``.
    They enter persistent history only after the rollout commits corresponding
    real observations, preventing imagined video from leaking across replans.
    """

    video_latents: torch.Tensor
    observed_span: PolicyTemporalSpan
    model_frame_start: int
    temporal_geometry: PolicyTemporalGeometry
    chunk_origin_frame: int = 0
    prefix_frame_count: int = 1
    pending_span: PolicyTemporalSpan | None = None

    def __post_init__(self) -> None:
        _validate_video_latents(self.video_latents, label="Observed video history")
        if int(self.observed_span.frame_count) != int(self.video_latents.shape[2]):
            raise ValueError(
                "Observed-video span length must match its latent tensor, "
                f"span={self.observed_span.frame_count}, "
                f"latents={self.video_latents.shape[2]}."
            )
        if int(self.prefix_frame_count) != 1:
            raise ValueError(
                "Observed-video recurrent history requires exactly one external "
                f"condition frame, got {self.prefix_frame_count}."
            )
        if not 0 <= int(self.chunk_origin_frame) < int(
            self.temporal_geometry.frame_chunk_size
        ):
            raise ValueError(
                "Observed-video chunk origin must be canonical for its retained "
                f"window, got origin={self.chunk_origin_frame}, "
                f"chunk={self.temporal_geometry.frame_chunk_size}."
            )
        if self.pending_span is not None and int(
            self.pending_span.start_frame
        ) != int(self.observed_span.end_frame):
            raise ValueError(
                "Speculative video must start immediately after observed history; "
                f"observed_end={self.observed_span.end_frame}, "
                f"speculative_start={self.pending_span.start_frame}."
            )

    @classmethod
    def initialize(
        cls,
        video_latents: torch.Tensor,
        *,
        start_frame: int,
        model_frame_start: int | None = None,
        temporal_geometry: PolicyTemporalGeometry,
        chunk_origin_frame: int = 0,
        prefix_frame_count: int = 1,
    ) -> ObservedVideoHistoryState:
        _validate_video_latents(video_latents, label="Initial observed video")
        if int(prefix_frame_count) != 1:
            raise ValueError(
                "Observed-video recurrent history requires exactly one external "
                f"condition frame, got {prefix_frame_count}."
            )
        resolved_chunk_size = int(temporal_geometry.frame_chunk_size)
        resolved_model_frame_start = (
            int(start_frame)
            if model_frame_start is None
            else int(model_frame_start)
        )
        resolved_latents = video_latents.detach()
        resolved_start_frame = int(start_frame)
        resolved_chunk_origin = int(chunk_origin_frame) % resolved_chunk_size
        window = resolve_one_frame_conditioned_history_window(
            history_frames=int(resolved_latents.shape[2]),
            window_size=int(temporal_geometry.attention_window_size),
            frame_chunk_size=resolved_chunk_size,
            chunk_origin_frame=resolved_chunk_origin,
        )
        if window.dropped_frames > 0:
            resolved_latents = resolved_latents[
                :, :, window.dropped_frames :
            ].contiguous()
            resolved_start_frame += int(window.dropped_frames)
            resolved_model_frame_start += int(window.dropped_frames)
        resolved_chunk_origin = int(window.chunk_origin_frame)
        return cls(
            video_latents=resolved_latents,
            observed_span=PolicyTemporalSpan(
                start_frame=resolved_start_frame,
                frame_count=int(resolved_latents.shape[2]),
            ),
            model_frame_start=resolved_model_frame_start,
            temporal_geometry=temporal_geometry,
            chunk_origin_frame=resolved_chunk_origin,
            prefix_frame_count=int(prefix_frame_count),
        )

    def begin_generation(
        self,
        *,
        frame_count: int,
    ) -> tuple[ObservedVideoHistoryState, PolicyTemporalSpan]:
        if self.pending_span is not None:
            raise RuntimeError(
                "Speculative video history must be reconciled with real observations "
                "before another generation request."
            )
        generated_span = PolicyTemporalSpan(
            start_frame=int(self.observed_span.end_frame),
            frame_count=int(frame_count),
        )
        next_state = replace(
            self,
            pending_span=generated_span,
        )
        return next_state, generated_span

    def commit_observations(
        self,
        history: PolicyObservedHistory,
    ) -> ObservedVideoHistoryState:
        if self.pending_span is None:
            raise RuntimeError(
                "Observed-video reconciliation requires an outstanding speculative span."
            )
        commit = history.execution_commit
        if commit is None:
            raise ValueError(
                "Observed-video reconciliation requires a typed execution commit."
            )
        if commit.speculative_span != self.pending_span:
            raise ValueError(
                "Observed-video execution does not match the pending generation; "
                f"pending={self.pending_span}, committed={commit.speculative_span}."
            )
        executed_span = commit.executed_span
        if int(executed_span.start_frame) != int(self.observed_span.end_frame):
            raise ValueError(
                "Observed-video commits must continue the real history without gaps; "
                f"observed_end={self.observed_span.end_frame}, "
                f"executed_start={executed_span.start_frame}."
            )
        _validate_video_latents(history.video_latents, label="Committed observed video")
        if int(history.video_latents.shape[2]) != int(executed_span.frame_count):
            raise ValueError(
                "Committed observed-video latent count must equal executed model "
                f"frames, got latents={history.video_latents.shape[2]}, "
                f"executed={executed_span.frame_count}."
            )
        if (
            tuple(history.video_latents.shape[:2])
            != tuple(self.video_latents.shape[:2])
            or tuple(history.video_latents.shape[-2:])
            != tuple(self.video_latents.shape[-2:])
        ):
            raise ValueError(
                "Committed and retained video history must share batch, channel, "
                "and spatial dimensions."
            )
        committed_latents = history.video_latents.detach().to(
            device=self.video_latents.device,
            dtype=self.video_latents.dtype,
        )
        resolved_window_size = int(self.temporal_geometry.attention_window_size)
        model_frame_chunk_size = int(self.temporal_geometry.frame_chunk_size)
        combined_latents = torch.cat(
            [self.video_latents, committed_latents], dim=2
        )
        retained_latents = combined_latents
        retained_start_frame = int(self.observed_span.start_frame)
        retained_model_frame_start = int(self.model_frame_start)
        retained_chunk_origin = int(self.chunk_origin_frame)
        window = resolve_one_frame_conditioned_history_window(
            history_frames=int(combined_latents.shape[2]),
            window_size=resolved_window_size,
            frame_chunk_size=model_frame_chunk_size,
            chunk_origin_frame=retained_chunk_origin,
        )
        if window.dropped_frames > 0:
            retained_latents = combined_latents[
                :, :, window.dropped_frames :
            ].contiguous()
            retained_start_frame += int(window.dropped_frames)
            retained_model_frame_start += int(window.dropped_frames)
        retained_chunk_origin = int(window.chunk_origin_frame)
        return replace(
            self,
            video_latents=retained_latents,
            observed_span=PolicyTemporalSpan(
                start_frame=retained_start_frame,
                frame_count=int(retained_latents.shape[2]),
            ),
            model_frame_start=retained_model_frame_start,
            chunk_origin_frame=retained_chunk_origin,
            pending_span=None,
        )


__all__ = ["ObservedVideoHistoryState"]
