"""Learned visual-context projections attached by the shared transformer core."""

from __future__ import annotations

import torch
from torch import nn


class ProprioContextEncoder(nn.Module):
    """Deprecated adapter that projects proprio state into text-context space."""

    def __init__(self, state_dim: int, text_dim: int) -> None:
        super().__init__()
        state_dim = int(state_dim)
        text_dim = int(text_dim)
        if state_dim <= 0:
            raise ValueError(f"Expected positive proprio state_dim, got {state_dim}.")
        if text_dim <= 0:
            raise ValueError(f"Expected positive text_dim, got {text_dim}.")
        self.state_dim = state_dim
        self.text_dim = text_dim
        self.proj = nn.Linear(state_dim, text_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, proprio_state: torch.Tensor) -> torch.Tensor:
        if proprio_state.ndim != 2:
            raise ValueError(
                "Proprio context encoder expects anchor state with shape [B, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        if int(proprio_state.shape[-1]) != self.state_dim:
            raise ValueError(
                "Proprio state dim mismatch for context encoder, "
                f"got {proprio_state.shape[-1]} and expected {self.state_dim}."
            )
        return self.proj(proprio_state)


class ProprioHiddenContextEncoder(nn.Module):
    """Project proprio state into additive transformer hidden context."""

    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        state_dim = int(state_dim)
        hidden_size = int(hidden_size)
        if state_dim <= 0:
            raise ValueError(f"Expected positive proprio state_dim, got {state_dim}.")
        if hidden_size <= 0:
            raise ValueError(f"Expected positive hidden_size, got {hidden_size}.")
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.proj = nn.Linear(state_dim, hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, proprio_state: torch.Tensor) -> torch.Tensor:
        if proprio_state.ndim != 2:
            raise ValueError(
                "Proprio hidden context encoder expects state with shape [B, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        if int(proprio_state.shape[-1]) != self.state_dim:
            raise ValueError(
                "Proprio hidden state dim mismatch, "
                f"got {proprio_state.shape[-1]} and expected {self.state_dim}."
            )
        return self.proj(proprio_state)


class GeneralistModeContextEncoder(nn.Module):
    """Learned text-space control token for GJD conditioning mode."""

    MODE_TO_INDEX = {
        "joint": 0,
        "action_conditioned_video": 1,
        "video_conditioned_action": 2,
    }

    def __init__(self, text_dim: int) -> None:
        super().__init__()
        text_dim = int(text_dim)
        if text_dim <= 0:
            raise ValueError(f"Expected positive text_dim, got {text_dim}.")
        self.text_dim = text_dim
        self.embedding = nn.Embedding(len(self.MODE_TO_INDEX), text_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    @classmethod
    def _index_for_mode(cls, mode: object) -> int:
        key = str(getattr(mode, "value", mode))
        try:
            return cls.MODE_TO_INDEX[key]
        except KeyError as exc:
            supported = ", ".join(sorted(cls.MODE_TO_INDEX))
            raise ValueError(f"Unsupported generalist mode {key!r}. Supported modes: {supported}.") from exc

    def _indices_for_modes(self, modes: object, *, batch_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(modes, torch.Tensor):
            indices = modes.to(device=device, dtype=torch.long).reshape(-1)
            if int(indices.numel()) > 0:
                min_index = int(indices.min().item())
                max_index = int(indices.max().item())
                if min_index < 0 or max_index >= len(self.MODE_TO_INDEX):
                    raise ValueError(
                        "Generalist mode tensor indices must be in "
                        f"[0, {len(self.MODE_TO_INDEX) - 1}], got min={min_index}, max={max_index}."
                    )
        elif isinstance(modes, str):
            index = self._index_for_mode(modes)
            indices = torch.full((batch_size,), index, device=device, dtype=torch.long)
        elif isinstance(modes, (list, tuple)):
            resolved = [self._index_for_mode(mode) for mode in modes]
            indices = torch.tensor(resolved, device=device, dtype=torch.long)
        else:
            index = self._index_for_mode(modes)
            indices = torch.full((batch_size,), index, device=device, dtype=torch.long)
        if int(indices.numel()) == 1 and batch_size != 1:
            indices = indices.expand(batch_size)
        if int(indices.numel()) != int(batch_size):
            raise ValueError(
                "Generalist mode token count must match text batch size, "
                f"got modes={int(indices.numel())} and batch={batch_size}."
            )
        return indices

    def forward(self, modes: object, *, batch_size: int) -> torch.Tensor:
        indices = self._indices_for_modes(
            modes,
            batch_size=int(batch_size),
            device=self.embedding.weight.device,
        )
        return self.embedding(indices)


__all__ = [
    "GeneralistModeContextEncoder",
    "ProprioContextEncoder",
    "ProprioHiddenContextEncoder",
]
