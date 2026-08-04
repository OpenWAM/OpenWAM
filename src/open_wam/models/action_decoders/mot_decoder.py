"""Deprecated compatibility facade for the dual-expert decoder."""

from .dual_expert_decoder import DualExpertActionDecoder

MoTActionDecoder = DualExpertActionDecoder

__all__ = ["MoTActionDecoder"]
