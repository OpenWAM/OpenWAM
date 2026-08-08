"""Vendored LingBot reference modules."""

from __future__ import annotations

from open_wam._shims.loader import ensure_flash_attn_shims


ensure_flash_attn_shims()

__all__ = ["WanTransformer3DModel"]


def __getattr__(name: str):
    if name == "WanTransformer3DModel":
        from .model import WanTransformer3DModel

        globals()[name] = WanTransformer3DModel
        return WanTransformer3DModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
