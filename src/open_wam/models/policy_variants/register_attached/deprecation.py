from __future__ import annotations

import warnings


REGISTER_ATTACHED_OBSOLETE_MESSAGE = (
    "Traditional Method 2 `register_attached` is obsolete and intentionally disabled. "
    "This packed joint video/action/register runtime is kept in the tree only as a "
    "historical reference. Use the maintained Method 2 parallel-stream "
    "`lingbot_exact_action_conditioned` / joint-denoise path instead."
)


class RegisterAttachedObsoleteError(RuntimeError):
    """Raised when obsolete traditional Method 2 register-attached code is invoked."""


def warn_register_attached_obsolete(*, stacklevel: int = 2) -> None:
    warnings.warn(
        REGISTER_ATTACHED_OBSOLETE_MESSAGE,
        RuntimeWarning,
        stacklevel=stacklevel,
    )


def raise_register_attached_obsolete(*, stacklevel: int = 2) -> None:
    warn_register_attached_obsolete(stacklevel=stacklevel)
    raise RegisterAttachedObsoleteError(REGISTER_ATTACHED_OBSOLETE_MESSAGE)
