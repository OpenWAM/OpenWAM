from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.models.common import RegisterSequenceLayout


@dataclass(frozen=True)
class StreamOutputHeadSpec:
    """Declarative description of one backbone-owned stream output head."""

    stream_name: str
    head_name: str
    projection_mode: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_action_span(layout: RegisterSequenceLayout) -> tuple[int, int]:
    if not layout.action_block_spans:
        return (0, 0)
    return (layout.action_block_spans[0][0], layout.action_block_spans[-1][1])


def project_runtime_stream_outputs(
    *,
    family: str,
    hidden_states: torch.Tensor,
    token_layout: object | None,
    video_projector,
    action_projector,
) -> dict[str, torch.Tensor]:
    """Project shared-backbone hidden states into named stream outputs.

    This keeps method-2 flow heads backbone-owned instead of variant-owned.
    The register-attached variant still decides *which* runtime family to use,
    but the actual hidden-state-to-flow projection now lives alongside the
    shared transformer weights.
    """

    if family == "none":
        return {}
    if family != "structured_joint_flow":
        raise ValueError(
            f"Unsupported runtime stream output-head family {family!r}. "
            "Expected 'structured_joint_flow' or 'none'."
        )
    if not isinstance(token_layout, RegisterSequenceLayout):
        raise ValueError(
            "Structured joint-flow output heads require a RegisterSequenceLayout token layout."
        )

    # The shared runtime executor returns the full packed hidden-state sequence.
    # Output-head families are responsible for knowing which token spans should
    # be projected back into each modality-specific prediction space.
    noisy_video_start, noisy_video_end = token_layout.noisy_video_span
    action_start, action_end = _resolve_action_span(token_layout)
    outputs = {
        "video_patch_flow": video_projector(hidden_states[:, noisy_video_start:noisy_video_end, :]),
        "action_flow": (
            action_projector(hidden_states[:, action_start:action_end, :])
            if action_end > action_start
            else hidden_states.new_zeros(
                hidden_states.shape[0],
                0,
                action_projector.out_features,
            )
        ),
    }
    return outputs
