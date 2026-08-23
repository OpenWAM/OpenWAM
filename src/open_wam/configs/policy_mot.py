"""Deprecated compatibility facade for :mod:`policy_dual_expert`."""

from .policy_dual_expert import DualExpertPolicyConfig

MoTPolicyConfig = DualExpertPolicyConfig

__all__ = ["MoTPolicyConfig"]
