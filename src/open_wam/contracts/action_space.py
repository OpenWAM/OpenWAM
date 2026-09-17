"""Bidirectional action representation contract shared by data and models."""

from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class ActionSpaceAdapter(Protocol):
    """A single bidirectional transform shared by plans and executed history."""

    @property
    def source_dim(self) -> int: ...
    def to_model(self, actions: torch.Tensor) -> torch.Tensor: ...
    def to_source(self, actions: torch.Tensor) -> torch.Tensor: ...
