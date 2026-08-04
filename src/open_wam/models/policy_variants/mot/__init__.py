"""Deprecated compatibility package for :mod:`..dual_expert`.

The implementation has one owner. Historical submodule imports are registered
as aliases to that owner so classes and module globals retain identity.
"""

from __future__ import annotations

import importlib
import sys
import warnings

from ..dual_expert import DualExpertPolicyVariant


warnings.warn(
    "`open_wam.models.policy_variants.mot` is deprecated; use `dual_expert`.",
    FutureWarning,
    stacklevel=2,
)

_SUBMODULES = (
    "attention",
    "attention_cached",
    "attention_packed",
    "attention_unpacked",
    "cache_execution",
    "cache_state",
    "conditioning",
    "contracts",
    "coupling_semantics",
    "dual_stream_execution",
    "generalist_modes",
    "inference_backend",
    "inference_layout",
    "joint_denoise_inference",
    "modules",
    "observed_history",
    "packed_block",
    "packed_inference",
    "packed_training",
    "rollout_geometry",
    "runtime",
    "runtime_routes",
    "runtime_routing",
    "sequence_layout",
    "split_cache_inference",
    "unpacked_training",
    "variant",
)

for _name in _SUBMODULES:
    _module = importlib.import_module(f"..dual_expert.{_name}", __package__)
    for _attribute in tuple(vars(_module)):
        if _attribute.startswith("DualExpert"):
            setattr(
                _module,
                f"MoT{_attribute.removeprefix('DualExpert')}",
                getattr(_module, _attribute),
            )
    sys.modules[f"{__name__}.{_name}"] = _module

MoTPolicyVariant = DualExpertPolicyVariant

__all__ = ["MoTPolicyVariant"]
