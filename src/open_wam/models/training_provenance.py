"""Inference eligibility derived from a checkpoint's resolved training routes.

This describes configured supervision, not measured checkpoint quality. The
checkpoint loader remains responsible for selecting its resolved configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from open_wam.configs.enums import DynamicsObjective

if TYPE_CHECKING:
    from open_wam.configs.data_contracts import DynamicsRouteConfig


@dataclass(frozen=True)
class PolicyTrainingProvenance:
    objectives: frozenset[DynamicsObjective]

    @classmethod
    def from_routes(
        cls, routes: Iterable[DynamicsRouteConfig]
    ) -> PolicyTrainingProvenance:
        return cls(frozenset(route.mode for route in routes if route.weight > 0))

    def require(self, objective: DynamicsObjective | None) -> None:
        if objective is not None and objective not in self.objectives:
            raise ValueError(
                f"Inference requires a positive `{objective.value}` training route "
                "in the checkpoint's resolved configuration."
            )
