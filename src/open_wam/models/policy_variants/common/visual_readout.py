from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from open_wam.configs import VisualReadoutConfig, VisualReadoutFusionMode, VisualReadoutSourceFamily
from open_wam.models.visual_tower import VisualCoreOutput, VisualReadoutRequest


@dataclass
class ResolvedVisualReadout:
    """Resolved visual-token readout returned to a policy variant."""

    tokens: torch.Tensor
    token_layout: Any | None
    source_stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SharedVisualReadout(nn.Module):
    """Reusable visual-readout selector and fusion helper for policy variants."""

    def __init__(self, config: VisualReadoutConfig | None, *, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.concat_project = None
        self.layer_weights = None
        if config is None:
            return
        if config.source_family == VisualReadoutSourceFamily.CORE_MULTI_LAYER_TOKENS:
            layer_count = len(config.layer_indices)
            if config.fusion_mode == VisualReadoutFusionMode.CONCAT_PROJECT:
                self.concat_project = nn.Linear(hidden_size * layer_count, hidden_size)
            elif config.fusion_mode == VisualReadoutFusionMode.LEARNED_WEIGHTED_SUM:
                self.layer_weights = nn.Parameter(torch.zeros(layer_count))

    def requested_capture(self) -> VisualReadoutRequest | None:
        if self.config is None:
            return None
        if self.config.source_family == VisualReadoutSourceFamily.CORE_LAYER_TOKENS:
            return VisualReadoutRequest(capture_layer_indices=(int(self.config.layer_index),))
        if self.config.source_family == VisualReadoutSourceFamily.CORE_MULTI_LAYER_TOKENS:
            return VisualReadoutRequest(capture_layer_indices=tuple(int(value) for value in self.config.layer_indices))
        return None

    def _lookup_intermediate_readout(self, core_output: VisualCoreOutput, *, layer_index: int) -> torch.Tensor:
        for readout in core_output.intermediate_readouts:
            if int(readout.layer_index) == int(layer_index):
                return readout.tokens
        available = tuple(int(readout.layer_index) for readout in core_output.intermediate_readouts)
        raise ValueError(
            f"Requested visual core layer {layer_index}, but only captured intermediate readouts {available}."
        )

    def _fuse_layers(self, layer_tokens: list[torch.Tensor]) -> torch.Tensor:
        if self.config is None:
            raise RuntimeError("Cannot fuse layers without a visual readout config.")
        if self.config.fusion_mode == VisualReadoutFusionMode.MEAN:
            return torch.stack(layer_tokens, dim=0).mean(dim=0)
        if self.config.fusion_mode == VisualReadoutFusionMode.LEARNED_WEIGHTED_SUM:
            if self.layer_weights is None:
                raise RuntimeError("Learned weighted-sum fusion requested without initialized weights.")
            weights = torch.softmax(self.layer_weights, dim=0)
            stacked = torch.stack(layer_tokens, dim=0)
            return (stacked * weights[:, None, None, None]).sum(dim=0)
        if self.config.fusion_mode == VisualReadoutFusionMode.CONCAT_PROJECT:
            if self.concat_project is None:
                raise RuntimeError("Concat-project fusion requested without an initialized projection.")
            return self.concat_project(torch.cat(layer_tokens, dim=-1))
        raise ValueError(f"Unsupported visual readout fusion mode {self.config.fusion_mode!r}.")

    def resolve_from_core(self, core_output: VisualCoreOutput) -> ResolvedVisualReadout:
        if self.config is None or self.config.source_family == VisualReadoutSourceFamily.FINAL_CORE_TOKENS:
            return ResolvedVisualReadout(
                tokens=core_output.tokens,
                token_layout=core_output.token_layout,
                source_stage="core",
                metadata={"source_family": VisualReadoutSourceFamily.FINAL_CORE_TOKENS},
            )
        if self.config.source_family == VisualReadoutSourceFamily.CORE_LAYER_TOKENS:
            if self.config.layer_index is None:
                raise ValueError("Visual readout requires `layer_index` for `core_layer_tokens`.")
            layer_tokens = self._lookup_intermediate_readout(core_output, layer_index=int(self.config.layer_index))
            return ResolvedVisualReadout(
                tokens=layer_tokens,
                token_layout=core_output.token_layout,
                source_stage=f"core_layer_{int(self.config.layer_index)}",
                metadata={
                    "source_family": self.config.source_family,
                    "layer_index": int(self.config.layer_index),
                },
            )
        if self.config.source_family == VisualReadoutSourceFamily.CORE_MULTI_LAYER_TOKENS:
            layer_indices = tuple(int(value) for value in self.config.layer_indices)
            fused = self._fuse_layers(
                [self._lookup_intermediate_readout(core_output, layer_index=value) for value in layer_indices]
            )
            return ResolvedVisualReadout(
                tokens=fused,
                token_layout=core_output.token_layout,
                source_stage="core_multi_layer",
                metadata={
                    "source_family": self.config.source_family,
                    "layer_indices": layer_indices,
                    "fusion_mode": str(self.config.fusion_mode),
                },
            )
        raise ValueError(
            "SharedVisualReadout currently supports only core-based visual readout families, "
            f"got {self.config.source_family!r}."
        )
