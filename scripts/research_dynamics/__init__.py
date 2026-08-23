"""Checkout-only research diagnostics for conditional GJD dynamics.

This namespace is intentionally excluded from the open_wam wheel. Use the
stable top-level scripts documented in this directory instead of importing it
as a library API.
"""

from open_wam.data.counterfactual_actions import (
    ACTION_BRANCH_SPECS,
    BRANCH_PRESETS,
    ActionBranchSpec,
    apply_action_branch,
)

from .types import (
    FdmAblationMode,
    FdmRunConfig,
    FdmWindowSelection,
    dynamics_objective_for_ablation_mode,
)

__all__ = [
    "ACTION_BRANCH_SPECS",
    "BRANCH_PRESETS",
    "ActionBranchSpec",
    "FdmAblationMode",
    "FdmRunConfig",
    "FdmWindowSelection",
    "apply_action_branch",
    "dynamics_objective_for_ablation_mode",
]
