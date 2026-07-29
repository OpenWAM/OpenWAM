from __future__ import annotations

from .deprecation import raise_register_attached_obsolete


class RegisterAttachedPolicyVariant:
    """Compatibility stub for the removed traditional Method-2 runtime."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise_register_attached_obsolete(stacklevel=2)
