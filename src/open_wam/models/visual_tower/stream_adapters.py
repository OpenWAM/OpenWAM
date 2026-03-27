from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


def _build_token_timestep_context(values: torch.Tensor, hidden_size: int) -> torch.Tensor:
    values = values.float()
    if hidden_size <= 0:
        return torch.zeros(*values.shape, 0, device=values.device, dtype=values.dtype)
    half_dim = max(1, hidden_size // 2)
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=values.device, dtype=values.dtype)
    exponent = exponent / max(half_dim - 1, 1)
    freqs = torch.exp(exponent)
    angles = values[..., None] * freqs
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < hidden_size:
        pad = torch.zeros(
            *embedding.shape[:-1],
            hidden_size - embedding.shape[-1],
            device=embedding.device,
            dtype=embedding.dtype,
        )
        embedding = torch.cat([embedding, pad], dim=-1)
    return embedding[..., :hidden_size]


@dataclass(frozen=True)
class StreamInputAdapterSpec:
    """Declarative description of one backbone-owned stream tokenizer."""

    stream_name: str
    adapter_name: str
    role_name: str = "none"
    use_timestep_context: bool = True
    use_role_embedding: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class PreparedStreamInput:
    """Backbone-ready tokens produced by one shared stream adapter."""

    stream_name: str
    tokens: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class SharedRuntimeStreamAdapters(nn.Module):
    """Backbone-owned tokenizers for non-visual runtime streams."""

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

    def prepare_stream_inputs(
        self,
        *,
        family: str,
        action_inputs: torch.Tensor | None,
        state_inputs: torch.Tensor | None,
        action_timesteps: torch.Tensor | None,
        state_timesteps: torch.Tensor | None,
        action_adapter_name: str = "mlp",
        state_adapter_name: str = "mlp",
        use_state_adapter: bool = True,
    ) -> dict[str, PreparedStreamInput]:
        if family != "structured_register_streams":
            raise ValueError(
                f"Unsupported stream input adapter family {family!r}. "
                "Expected 'structured_register_streams'."
            )
        if action_adapter_name != "mlp":
            raise ValueError(f"Unsupported action stream adapter {action_adapter_name!r}. Expected 'mlp'.")
        if state_adapter_name != "mlp":
            raise ValueError(f"Unsupported state stream adapter {state_adapter_name!r}. Expected 'mlp'.")
        if action_inputs is None or action_timesteps is None:
            raise ValueError("Structured register streams require `action_inputs` and `action_timesteps`.")
        if self.action_register_adapter is None:
            raise ValueError("Shared runtime stream adapters were constructed without an action stream adapter.")

        action_tokens = self.action_register_adapter(action_inputs)
        action_tokens = action_tokens + _build_token_timestep_context(action_timesteps, self.hidden_size)
        action_tokens = action_tokens + self.role_embedding.weight[0][None, None, :]

        if use_state_adapter and state_inputs is not None and state_timesteps is not None:
            if self.state_register_adapter is None:
                raise ValueError("Shared runtime stream adapters were constructed without a state stream adapter.")
            state_tokens = self.state_register_adapter(state_inputs)
            state_tokens = state_tokens + _build_token_timestep_context(state_timesteps, self.hidden_size)
            state_tokens = state_tokens + self.role_embedding.weight[1][None, None, :]
        else:
            state_tokens = action_tokens.new_zeros((action_tokens.shape[0], 0, self.hidden_size))

        return {
            "action_register": PreparedStreamInput(
                stream_name="action_register",
                tokens=action_tokens,
                metadata={
                    "adapter_family": family,
                    "adapter_name": action_adapter_name,
                    "role_name": "action_register",
                },
            ),
            "state_register": PreparedStreamInput(
                stream_name="state_register",
                tokens=state_tokens,
                metadata={
                    "adapter_family": family,
                    "adapter_name": state_adapter_name,
                    "role_name": "state_register",
                    "enabled": bool(use_state_adapter and state_inputs is not None and state_timesteps is not None),
                },
            ),
        }
