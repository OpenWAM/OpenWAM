"""Compatibility imports for the pre-0.2 backbone config location.

New code should import these contracts from :mod:`open_wam.configs`.
"""

from open_wam.configs.backbone import (
    LingbotCompatibleVideoBackboneConfig,
    SharedVideoTransformerConfig,
    normalize_backbone_implementation,
    resolve_stage_attention_mode,
)

__all__ = [
    "LingbotCompatibleVideoBackboneConfig",
    "SharedVideoTransformerConfig",
    "normalize_backbone_implementation",
    "resolve_stage_attention_mode",
]
