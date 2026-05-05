"""Obsolete traditional Method 2 register-attached policy variant.

The implementation is retained for historical reference only. Calls into this
variant raise an explicit warning and error.
"""

from .deprecation import RegisterAttachedObsoleteError
from .variant import RegisterAttachedPolicyVariant

__all__ = ["RegisterAttachedObsoleteError", "RegisterAttachedPolicyVariant"]
