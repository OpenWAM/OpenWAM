from __future__ import annotations

from typing import Any

from open_wam.configs import ActionDecoderName
from open_wam.registry import BuilderRegistry


POLICY_VARIANT_BUILDERS = BuilderRegistry[type[Any], Any]("policy variant builder")
ACTION_DECODER_BUILDERS = BuilderRegistry[ActionDecoderName, Any]("action decoder builder")


__all__ = ["ACTION_DECODER_BUILDERS", "POLICY_VARIANT_BUILDERS"]
