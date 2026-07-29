"""Compatibility stub for the removed traditional Method-2 smoke."""

from __future__ import annotations

_MESSAGE = (
    "Traditional Method 2 register_attached is obsolete and intentionally disabled. "
    "Use parallel_stream with lingbot_exact_action_conditioned / joint-denoise instead."
)


def main() -> None:
    raise SystemExit(_MESSAGE)


if __name__ == "__main__":
    main()
