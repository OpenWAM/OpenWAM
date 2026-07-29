"""Compatibility surface for the removed traditional Method-2 runtime."""

from .deprecation import RegisterAttachedObsoleteError
from .variant import RegisterAttachedPolicyVariant

__all__ = ["RegisterAttachedObsoleteError", "RegisterAttachedPolicyVariant"]
