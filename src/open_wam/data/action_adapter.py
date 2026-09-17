"""Configured action conversion at the data/environment boundary."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from open_wam.configs import ActionMappingConfig, ActionNormalizationConfig
from .action_mapping import (
    apply_action_mapping,
    inverse_action_mapping,
    resolve_action_source_dim,
)
from .action_normalization import normalize_action_targets, denormalize_action_targets


@dataclass(frozen=True)
class ConfiguredActionAdapter:
    mapping: ActionMappingConfig
    normalization: ActionNormalizationConfig
    model_dim: int

    @property
    def source_dim(self) -> int:
        return resolve_action_source_dim(self.mapping, self.model_dim)

    def to_model(self, actions: torch.Tensor) -> torch.Tensor:
        shape = actions.shape
        normalized = normalize_action_targets(
            actions.float(), normalization=self.normalization
        )
        mapped = apply_action_mapping(
            normalized.reshape(-1, shape[-1]),
            torch.ones_like(normalized).reshape(-1, shape[-1]),
            self.mapping,
            target_dim=self.model_dim,
        ).actions
        return mapped.reshape(*shape[:-1], self.model_dim)

    def to_source(self, actions: torch.Tensor) -> torch.Tensor:
        return denormalize_action_targets(
            inverse_action_mapping(actions, self.mapping),
            normalization=self.normalization,
        )
