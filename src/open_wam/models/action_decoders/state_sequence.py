from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.configs import StateSequenceAdapterFamily
from open_wam.models.policy_variants.contracts import DecoderSequenceContext


def _align_sequence_length(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
    if sequence.shape[1] == target_length:
        return sequence
    return F.interpolate(
        sequence.transpose(1, 2),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


class StateSequenceAdapter(nn.Module, ABC):
    """Shared adapter for state/proprio sequence conditioning."""

    @abstractmethod
    def forward(
        self,
        sequence_context: DecoderSequenceContext,
        *,
        target_length: int,
        target_hidden_size: int,
    ) -> torch.Tensor | None:
        """Return a `[B, T, D]` state-conditioning sequence or `None`."""


class IdentityStateSequenceAdapter(StateSequenceAdapter):
    """Pass through already-projected state sequences."""

    def forward(
        self,
        sequence_context: DecoderSequenceContext,
        *,
        target_length: int,
        target_hidden_size: int,
    ) -> torch.Tensor | None:
        del target_hidden_size
        state_sequence = sequence_context.state_sequence
        if state_sequence is None:
            return None
        if state_sequence.ndim == 2:
            state_sequence = state_sequence[:, None, :]
        if state_sequence.ndim != 3:
            raise ValueError(
                "Identity state adapter expects `[B, T, D]` or `[B, D]`, "
                f"got {tuple(state_sequence.shape)}"
            )
        return _align_sequence_length(state_sequence, target_length)


class LinearStateSequenceAdapter(StateSequenceAdapter):
    """Project raw state/proprio sequences into decoder hidden space."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.state_proj = nn.Linear(input_dim, hidden_size)

    def forward(
        self,
        sequence_context: DecoderSequenceContext,
        *,
        target_length: int,
        target_hidden_size: int,
    ) -> torch.Tensor | None:
        del target_hidden_size
        state_sequence = sequence_context.state_sequence
        if state_sequence is None:
            return None
        if state_sequence.ndim == 2:
            state_sequence = state_sequence[:, None, :]
        if state_sequence.ndim != 3:
            raise ValueError(
                "Linear state adapter expects `[B, T, D]` or `[B, D]`, "
                f"got {tuple(state_sequence.shape)}"
            )
        projected = self.state_proj(state_sequence)
        return _align_sequence_length(projected, target_length)


def build_state_sequence_adapter(
    family: StateSequenceAdapterFamily | str,
    *,
    input_dim: int,
    hidden_size: int,
) -> StateSequenceAdapter:
    resolved = StateSequenceAdapterFamily(family)
    if resolved == StateSequenceAdapterFamily.IDENTITY:
        return IdentityStateSequenceAdapter()
    if resolved == StateSequenceAdapterFamily.LINEAR:
        return LinearStateSequenceAdapter(input_dim=input_dim, hidden_size=hidden_size)
    raise ValueError(f"Unsupported state sequence adapter family '{resolved}'.")
