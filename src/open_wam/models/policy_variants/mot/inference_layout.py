from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from open_wam.configs import MoTGeneralistTrainingMode

from .contracts import MoTRuntimeState


@dataclass(frozen=True)
class MoTConditionalRolloutInputs:
    """Validated tensor overrides for joint, FDM, and IDM packed rollouts."""

    forced_action_latents: torch.Tensor | None
    commit_action_latents: torch.Tensor | None
    video_condition_latents: torch.Tensor | None


@dataclass(frozen=True)
class MoTPackedHistoryWindow:
    """Frame-aligned recurrent history selected for one packed inference step."""

    video_latents: torch.Tensor | None
    action_latents: torch.Tensor | None
    action_tokens: int
    frames: int
    max_frames: int
    hidden_proprio_sequence: torch.Tensor | None

    def prepend_video(self, current_video: torch.Tensor) -> torch.Tensor:
        if self.video_latents is None:
            return current_video
        return torch.cat([self.video_latents, current_video], dim=2)


@dataclass(frozen=True)
class MoTPackedHistory:
    """Normalized persistent history before per-step window selection."""

    video_latents: torch.Tensor | None
    action_latents: torch.Tensor | None
    hidden_proprio_states: torch.Tensor | None

    @classmethod
    def from_runtime_state(
        cls,
        runtime_state: MoTRuntimeState,
        *,
        layout: MoTPackedInferenceLayout,
        current_video_latents: torch.Tensor,
    ) -> MoTPackedHistory:
        video_latents = runtime_state.past_clean_latents
        if video_latents is not None:
            video_latents = video_latents.to(device=layout.device, dtype=layout.dtype)
            if (
                video_latents.shape[0] != layout.batch_size
                or video_latents.shape[1] != current_video_latents.shape[1]
            ):
                raise ValueError(
                    "M5 packed video history shape does not match current video latents, "
                    f"got past={tuple(video_latents.shape)}, current={tuple(current_video_latents.shape)}."
                )
            if video_latents.shape[-2:] != current_video_latents.shape[-2:]:
                raise ValueError(
                    "M5 packed video history spatial shape does not match current video latents, "
                    f"got past={tuple(video_latents.shape)}, current={tuple(current_video_latents.shape)}."
                )

        action_latents = runtime_state.past_clean_actions
        if action_latents is not None:
            action_latents = action_latents.to(device=layout.device, dtype=layout.dtype)
            if (
                action_latents.shape[0] != layout.batch_size
                or action_latents.shape[-1] != layout.action_dim
            ):
                raise ValueError(
                    "M5 packed action history shape does not match current action shape, "
                    f"got past_actions={tuple(action_latents.shape)}, "
                    f"batch_size={layout.batch_size}, action_dim={layout.action_dim}."
                )
            if action_latents.shape[1] % layout.action_tokens_per_frame != 0:
                raise ValueError(
                    "M5 packed action history length must be divisible by action_tokens_per_frame, "
                    f"got past_action_tokens={action_latents.shape[1]}, "
                    f"action_tokens_per_frame={layout.action_tokens_per_frame}."
                )

        return cls(
            video_latents=video_latents,
            action_latents=action_latents,
            hidden_proprio_states=runtime_state.past_hidden_proprio_states,
        )

    def select_window(
        self,
        *,
        layout: MoTPackedInferenceLayout,
        history_window_frames: int,
        hidden_proprio_state: torch.Tensor | None,
        require_hidden_proprio_history: bool,
    ) -> MoTPackedHistoryWindow:
        history_video_frames = 0 if self.video_latents is None else int(self.video_latents.shape[2])
        source_action_tokens = 0 if self.action_latents is None else int(self.action_latents.shape[1])
        history_action_frames = source_action_tokens // layout.action_tokens_per_frame
        history_frames = min(history_video_frames, history_action_frames)
        max_history_frames = max(0, int(history_window_frames) - layout.frame_chunk_size)
        if max_history_frames > 0:
            history_frames = min(history_frames, max_history_frames)
        else:
            history_frames = 0

        if history_frames > 0:
            video_latents = self.video_latents[:, :, -history_frames:].contiguous()
            action_tokens = history_frames * layout.action_tokens_per_frame
            action_latents = self.action_latents[:, -action_tokens:].contiguous()
        else:
            video_latents = None
            action_latents = None
            action_tokens = 0

        history_hidden_proprio = self.hidden_proprio_states
        if history_hidden_proprio is not None and history_frames > 0:
            history_hidden_proprio = history_hidden_proprio.to(
                device=layout.device,
                dtype=layout.dtype,
            )
            if int(history_hidden_proprio.shape[0]) != layout.batch_size:
                raise ValueError(
                    "M5 packed hidden proprio history batch size does not match current batch, "
                    f"got history={tuple(history_hidden_proprio.shape)}, "
                    f"batch_size={layout.batch_size}."
                )
            history_hidden_proprio = history_hidden_proprio[:, -history_frames:].contiguous()
        else:
            history_hidden_proprio = None
        if (
            history_frames > 0
            and require_hidden_proprio_history
            and history_hidden_proprio is None
        ):
            raise ValueError("M5 per-chunk additive proprio inference is missing hidden proprio history.")

        current_hidden_proprio = None
        if hidden_proprio_state is not None:
            current_hidden_proprio = hidden_proprio_state.to(
                device=layout.device,
                dtype=layout.dtype,
            )[:, None, :].expand(
                -1,
                layout.current_video_sequence_frames,
                -1,
            )
        if history_hidden_proprio is None:
            hidden_proprio_sequence = current_hidden_proprio
        elif current_hidden_proprio is None:
            hidden_proprio_sequence = history_hidden_proprio
        else:
            hidden_proprio_sequence = torch.cat(
                [history_hidden_proprio, current_hidden_proprio],
                dim=1,
            )

        return MoTPackedHistoryWindow(
            video_latents=video_latents,
            action_latents=action_latents,
            action_tokens=action_tokens,
            frames=history_frames,
            max_frames=max_history_frames,
            hidden_proprio_sequence=hidden_proprio_sequence,
        )


@dataclass(frozen=True)
class MoTPackedInferenceLayout:
    """Current-chunk geometry and tensor coercion for packed MoT inference."""

    batch_size: int
    action_dim: int
    configured_action_horizon: int
    frame_chunk_size: int
    action_tokens_per_frame: int
    current_video_prefix_frames: int
    current_video_sequence_frames: int
    current_action_prefix_tokens: int
    current_action_sequence_tokens: int
    video_channels: int
    video_height: int
    video_width: int
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        positive_fields = {
            "batch_size": self.batch_size,
            "action_dim": self.action_dim,
            "configured_action_horizon": self.configured_action_horizon,
            "frame_chunk_size": self.frame_chunk_size,
            "action_tokens_per_frame": self.action_tokens_per_frame,
            "current_video_sequence_frames": self.current_video_sequence_frames,
            "current_action_sequence_tokens": self.current_action_sequence_tokens,
            "video_channels": self.video_channels,
            "video_height": self.video_height,
            "video_width": self.video_width,
        }
        invalid = {
            name: int(value)
            for name, value in positive_fields.items()
            if int(value) <= 0
        }
        if invalid:
            raise ValueError(
                "MoT packed inference layout requires positive dimensions, "
                f"got {invalid}."
            )
        if self.current_video_prefix_frames < 0 or self.current_action_prefix_tokens < 0:
            raise ValueError(
                "MoT packed inference layout prefixes must be non-negative, "
                f"got video={self.current_video_prefix_frames}, "
                f"action={self.current_action_prefix_tokens}."
            )
        expected_video_frames = self.current_video_prefix_frames + self.frame_chunk_size
        if self.current_video_sequence_frames != expected_video_frames:
            raise ValueError(
                "MoT packed inference video sequence must contain its prefix plus one chunk, "
                f"got sequence={self.current_video_sequence_frames}, "
                f"prefix={self.current_video_prefix_frames}, chunk={self.frame_chunk_size}."
            )
        if self.current_action_sequence_tokens <= self.current_action_prefix_tokens:
            raise ValueError(
                "MoT packed inference action sequence must contain generated tokens after its prefix, "
                f"got sequence={self.current_action_sequence_tokens}, "
                f"prefix={self.current_action_prefix_tokens}."
            )

    def resolve_conditional_rollout_inputs(
        self,
        extra: Mapping[str, Any],
        *,
        generalist_rollout_mode: MoTGeneralistTrainingMode,
        current_clean_video: torch.Tensor,
    ) -> MoTConditionalRolloutInputs:
        forced_action_latents = self._coerce_action_tensor(
            extra,
            "mot_forced_action_latents",
            required=(
                generalist_rollout_mode
                == MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO
            ),
            generalist_rollout_mode=generalist_rollout_mode,
        )
        commit_action_latents = self._coerce_action_tensor(
            extra,
            "mot_commit_action_latents",
            required=False,
            generalist_rollout_mode=generalist_rollout_mode,
        )
        video_condition_latents = self._coerce_video_tensor(
            extra,
            "mot_video_condition_latents",
            required=(
                generalist_rollout_mode
                == MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION
            ),
            generalist_rollout_mode=generalist_rollout_mode,
            current_clean_video=current_clean_video,
        )
        return MoTConditionalRolloutInputs(
            forced_action_latents=forced_action_latents,
            commit_action_latents=commit_action_latents,
            video_condition_latents=video_condition_latents,
        )

    def compose_current_action_sequence(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if self.current_action_prefix_tokens <= 0:
            return action_tokens
        invalid_prefix = action_tokens.new_zeros(
            action_tokens.shape[0],
            self.current_action_prefix_tokens,
            action_tokens.shape[-1],
        )
        return torch.cat([invalid_prefix, action_tokens], dim=1)

    def _context_tensor(
        self,
        extra: Mapping[str, Any],
        key: str,
    ) -> torch.Tensor | None:
        value = extra.get(key)
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"M5 GJD rollout context {key!r} must be a torch.Tensor, "
                f"got {type(value)!r}."
            )
        return value.to(device=self.device, dtype=self.dtype)

    def _coerce_action_tensor(
        self,
        extra: Mapping[str, Any],
        key: str,
        *,
        required: bool,
        generalist_rollout_mode: MoTGeneralistTrainingMode,
    ) -> torch.Tensor | None:
        value = self._context_tensor(extra, key)
        if value is None:
            if required:
                raise ValueError(
                    f"M5 GJD rollout mode {generalist_rollout_mode.value!r} "
                    f"requires {key!r}."
                )
            return None
        if value.ndim != 3:
            raise ValueError(
                f"M5 GJD rollout context {key!r} must have shape [B, H, D], "
                f"got {tuple(value.shape)}."
            )
        if int(value.shape[0]) != self.batch_size or int(value.shape[-1]) != self.action_dim:
            raise ValueError(
                f"M5 GJD rollout context {key!r} shape does not match current action shape, "
                f"got {tuple(value.shape)}, expected batch={self.batch_size}, "
                f"action_dim={self.action_dim}."
            )
        if int(value.shape[1]) == self.configured_action_horizon:
            return value.contiguous()
        if int(value.shape[1]) == self.current_action_sequence_tokens:
            return value[:, self.current_action_prefix_tokens :].contiguous()
        raise ValueError(
            f"M5 GJD rollout context {key!r} must contain either "
            f"action_horizon={self.configured_action_horizon} or "
            f"current_action_sequence_tokens={self.current_action_sequence_tokens} tokens, "
            f"got {value.shape[1]}."
        )

    def _coerce_video_tensor(
        self,
        extra: Mapping[str, Any],
        key: str,
        *,
        required: bool,
        generalist_rollout_mode: MoTGeneralistTrainingMode,
        current_clean_video: torch.Tensor,
    ) -> torch.Tensor | None:
        value = self._context_tensor(extra, key)
        if value is None:
            if required:
                raise ValueError(
                    f"M5 GJD rollout mode {generalist_rollout_mode.value!r} "
                    f"requires {key!r}."
                )
            return None
        if value.ndim != 5:
            raise ValueError(
                f"M5 GJD rollout context {key!r} must have shape [B, C, T, H, W], "
                f"got {tuple(value.shape)}."
            )
        expected_prefix = (
            self.batch_size,
            self.video_channels,
            self.video_height,
            self.video_width,
        )
        got_prefix = (
            int(value.shape[0]),
            int(value.shape[1]),
            int(value.shape[-2]),
            int(value.shape[-1]),
        )
        if got_prefix != expected_prefix:
            raise ValueError(
                f"M5 GJD rollout context {key!r} shape does not match current video shape, "
                f"got {tuple(value.shape)}, expected batch/channels/spatial={expected_prefix}."
            )
        if int(value.shape[2]) == self.frame_chunk_size:
            if self.current_video_prefix_frames <= 0:
                return value.contiguous()
            return torch.cat(
                [
                    current_clean_video[:, :, : self.current_video_prefix_frames],
                    value,
                ],
                dim=2,
            ).contiguous()
        if int(value.shape[2]) == self.current_video_sequence_frames:
            return value.contiguous()
        raise ValueError(
            f"M5 GJD rollout context {key!r} must contain either "
            f"frame_chunk_size={self.frame_chunk_size} or "
            f"current_video_sequence_frames={self.current_video_sequence_frames} frames, "
            f"got {value.shape[2]}."
        )
