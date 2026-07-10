from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ActionChunk, DynamicsPrediction, PlanningContext


@dataclass(frozen=True)
class ActionTintDynamics:
    """Cheap fake dynamics model that tints the last frame by action mean."""

    view_key: str = "agentview_image"
    video_frames: int = 4

    def predict(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> DynamicsPrediction:
        del seed
        if self.view_key not in context.views:
            raise KeyError(f"ActionTintDynamics missing view {self.view_key!r}.")
        frame = np.asarray(context.views[self.view_key], dtype=np.float32)
        action_bias = float(np.tanh(np.mean(action_chunk.actions[:, : min(3, action_chunk.actions.shape[1])]))) * 32.0
        video = np.repeat(frame[None], int(self.video_frames), axis=0)
        video = np.clip(video + action_bias, 0, 255).astype(np.uint8)
        next_views = dict(context.views)
        next_views[self.view_key] = video[-1]
        return DynamicsPrediction(
            predicted_video=video,
            next_context=context.with_prediction(predicted_video=video, views=next_views),
            metadata={"dynamics": "action_tint", "action_bias": action_bias},
        )
