"""Model-space rollout state, independent of transformer and feature storage."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VideoActionRolloutState:
    """Conditioning and aligned video/action history owned by one session.

    Feature caches are derived from this state and never own temporal position.
    Missing action groups, including t0, remain explicit masked placeholders.
    """

    text_context: torch.Tensor | None = None
    proprio_state: torch.Tensor | None = None
    hidden_proprio_state: torch.Tensor | None = None
    past_hidden_proprio_states: torch.Tensor | None = None
    past_clean_latents: torch.Tensor | None = None
    past_clean_actions: torch.Tensor | None = None
    past_clean_action_mask: torch.Tensor | None = None
    pending_predicted_video_frames: int = 0
    # Latest generation's advance; observed seeds and completed commits use zero.
    chunk_advance_frames: int = 0
    generalist_mode_text_token_count: int = 0

    def require_complete_video_history(self) -> None:
        """Do not reinterpret old video as a later, unobserved frame."""
        if self.pending_predicted_video_frames < self.chunk_advance_frames:
            raise ValueError(
                "Commit replacement video history for the preceding action-only "
                "interval before generating another chunk."
            )
