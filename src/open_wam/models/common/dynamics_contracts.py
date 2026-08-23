"""Typed architecture-neutral contracts for dynamics inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs.enums import (
    DynamicsObjective,
    HistoryStreamVisibility,
)

from .attention_contracts import (
    CONDITIONAL_HISTORY_POLICY_NONE,
    normalize_conditional_history_policy,
)


@dataclass(frozen=True, slots=True)
class DynamicsRolloutGeometry:
    """Architecture-neutral temporal and history contract for one rollout call."""

    frame_chunk_size: int
    attention_window_size: int
    history_stream_visibility: HistoryStreamVisibility
    conditional_history_policy: str | None

    def __post_init__(self) -> None:
        if int(self.frame_chunk_size) <= 0:
            raise ValueError(
                "Dynamics rollout `frame_chunk_size` must be positive, "
                f"got {self.frame_chunk_size!r}."
            )
        if int(self.attention_window_size) <= 0:
            raise ValueError(
                "Dynamics rollout `attention_window_size` must be positive, "
                f"got {self.attention_window_size!r}."
            )
        object.__setattr__(
            self,
            "history_stream_visibility",
            HistoryStreamVisibility(self.history_stream_visibility),
        )
        normalized_history_policy = normalize_conditional_history_policy(
            self.conditional_history_policy
        )
        object.__setattr__(
            self,
            "conditional_history_policy",
            None
            if normalized_history_policy == CONDITIONAL_HISTORY_POLICY_NONE
            else normalized_history_policy,
        )


@dataclass(frozen=True, slots=True)
class DynamicsRolloutRequest:
    """Canonical inputs for one joint, FDM, or IDM rollout.

    Action tensors use the policy's model space and ``[B, T, D]`` layout.
    Video tensors use latent space and ``[B, C, T, H, W]`` layout. The optional
    ``history_action`` is the clean action sequence appended to recurrent
    history after this diagnostic chunk; FDM defaults it to ``clean_action``.
    Policy backends own conversion from these layouts to native token packs.
    """

    objective: DynamicsObjective | str | None = None
    clean_action: torch.Tensor | None = None
    clean_video: torch.Tensor | None = None
    history_action: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.objective is not None and not isinstance(
            self.objective,
            DynamicsObjective,
        ):
            try:
                objective = DynamicsObjective(self.objective)
            except ValueError as exc:
                supported = ", ".join(item.value for item in DynamicsObjective)
                raise ValueError(
                    f"Unsupported dynamics objective {self.objective!r}. "
                    f"Supported objectives: {supported}."
                ) from exc
            object.__setattr__(self, "objective", objective)
        _validate_rollout_tensor(
            self.clean_action,
            name="clean_action",
            expected_ndim=3,
        )
        _validate_rollout_tensor(
            self.clean_video,
            name="clean_video",
            expected_ndim=5,
        )
        _validate_rollout_tensor(
            self.history_action,
            name="history_action",
            expected_ndim=3,
        )


def _validate_rollout_tensor(
    value: torch.Tensor | None,
    *,
    name: str,
    expected_ndim: int,
) -> None:
    if value is None:
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Dynamics rollout `{name}` must be a torch.Tensor, "
            f"got {type(value).__name__}."
        )
    if value.ndim != expected_ndim:
        raise ValueError(
            f"Dynamics rollout `{name}` must have {expected_ndim} dimensions, "
            f"got shape={tuple(value.shape)}."
        )


__all__ = ["DynamicsRolloutGeometry", "DynamicsRolloutRequest"]
