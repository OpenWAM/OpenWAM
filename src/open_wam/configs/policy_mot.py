"""Deprecated compatibility facade for :mod:`policy_dual_expert`."""

from .policy_dual_expert import (
    DualExpertPolicyConfig,
    _coerce_generalist_denoising_mode_probs,
    _coerce_mot_generalist_training_mode_probs,
)

MoTPolicyConfig = DualExpertPolicyConfig
_COMPATIBILITY_EXPORTS = (
    _coerce_generalist_denoising_mode_probs,
    _coerce_mot_generalist_training_mode_probs,
)

__all__ = ["MoTPolicyConfig"]
