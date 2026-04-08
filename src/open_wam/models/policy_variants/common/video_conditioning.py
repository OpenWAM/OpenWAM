from __future__ import annotations

from open_wam.configs import VideoConditionInputSpace
from open_wam.models.visual_tower import VisualStageOutputs

from ..contracts import VideoConditionWindowContext
from .layouts import tokens_to_frame_major


def build_local_video_condition_window(
    *,
    visual_outputs: VisualStageOutputs,
    input_space: str,
    local_window_frames: int,
    current_frame_index: int,
    action_chunk_anchor_mode: str,
    source_stage: str,
    observed_frame_count: int = 1,
) -> VideoConditionWindowContext:
    """Build a typed local frame-token window for decoder-side video conditioning."""

    input_space = VideoConditionInputSpace(str(input_space))
    if input_space == VideoConditionInputSpace.VIDEO_LATENT:
        frame_tokens = tokens_to_frame_major(
            visual_outputs.frontend.video_tokens,
            visual_outputs.frontend.token_grid,
        )
        source_family = "frontend_video_tokens"
        source_metadata = {
            "encoded_from": visual_outputs.frontend.input_source,
        }
    elif input_space == VideoConditionInputSpace.RGB_VIDEO:
        if visual_outputs.frontend.input_source != "canonical_rgb":
            raise ValueError(
                "Method-4 `rgb_video` conditioning requires an RGB-backed frontend pass. "
                "This run entered the frontend from precomputed latents instead. "
                "Use `video_latent` conditioning for latent-first runs, or execute the method-4 path "
                "from raw RGB views so the shared frontend/VAE encodes the condition window."
            )
        frame_tokens = tokens_to_frame_major(
            visual_outputs.frontend.video_tokens,
            visual_outputs.frontend.token_grid,
        )
        source_family = "encoded_rgb_frontend_video_tokens"
        source_metadata = {
            "encoded_from": "canonical_rgb",
            "rgb_encoder": "shared_frontend_vae",
        }
    else:  # pragma: no cover - enum validation should prevent this
        raise ValueError(f"Unsupported method-4 video condition input space {input_space!r}.")
    if int(current_frame_index) != 0:
        raise ValueError(
            "Method-4 local video-condition windows currently support only "
            "`current_frame_index = 0` for rollout-window decoding. Non-zero sliding-window "
            "alignment is not implemented yet."
        )
    if local_window_frames > frame_tokens.shape[1]:
        raise ValueError(
            "Local video condition window requires enough frontend frames, "
            f"got local_window_frames={local_window_frames}, available_frames={frame_tokens.shape[1]}."
        )
    local_tokens = frame_tokens[:, :local_window_frames]
    return VideoConditionWindowContext(
        local_window_tokens=local_tokens,
        token_grid=visual_outputs.frontend.token_grid,
        source_stage=source_stage,
        input_space=str(input_space),
        local_window_frames=int(local_window_frames),
        current_frame_index=int(current_frame_index),
        current_action_index=0,
        action_chunk_anchor_mode=str(action_chunk_anchor_mode),
        observed_frame_count=int(observed_frame_count),
        previous_context_frames=0,
        metadata={
            "source_family": source_family,
            "deferred_previous_context": True,
            **source_metadata,
        },
    )
