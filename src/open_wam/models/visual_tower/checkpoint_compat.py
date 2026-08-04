from __future__ import annotations

from torch import nn


class RuntimeStreamCompatibilityParameters(nn.Module):
    """Retain historical runtime-stream parameter keys for current checkpoints.

    Selected historical parallel-stream/dual-expert exports contain the
    ``runtime_stream_adapters.{action,state}_register_adapter`` and
    ``runtime_stream_adapters.role_embedding`` keys. The retired traditional
    Method 2 runtime no longer executes these modules, but their names,
    registration order, and tensor shapes remain part of the strict checkpoint
    and optimizer-state contract.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int = 0,
        state_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.action_register_adapter = (
            nn.Sequential(
                nn.Linear(self.action_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            if self.action_dim > 0
            else None
        )
        self.state_register_adapter = (
            nn.Sequential(
                nn.Linear(self.state_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            if self.state_dim > 0
            else None
        )
        self.role_embedding = nn.Embedding(2, hidden_size)
