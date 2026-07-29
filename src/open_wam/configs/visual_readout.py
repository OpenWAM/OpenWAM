from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .coercion import coerce_enum
from .enums import (
    VisualReadoutFusionMode,
    VisualReadoutSourceFamily,
    coerce_fields,
)


@dataclass(frozen=True)
class VisualReadoutConfig:
    """Shared visual-readout selection used by post-visual policy variants."""

    source_family: VisualReadoutSourceFamily
    layer_index: int | None = None
    layer_indices: tuple[int, ...] = field(default_factory=tuple)
    fusion_mode: VisualReadoutFusionMode = VisualReadoutFusionMode.NONE
    diffusion_extract_timestep: int = 20
    diffusion_extract_step_time: int = 1

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "source_family": VisualReadoutSourceFamily,
                "fusion_mode": VisualReadoutFusionMode,
            },
        )
        if self.source_family == VisualReadoutSourceFamily.CORE_LAYER_TOKENS:
            if self.layer_index is None:
                raise ValueError("`core_layer_tokens` requires `layer_index`.")
            if self.layer_indices:
                raise ValueError("`core_layer_tokens` should not set `layer_indices`.")
            if self.fusion_mode != VisualReadoutFusionMode.NONE:
                raise ValueError("`core_layer_tokens` requires `fusion_mode = none`.")
        elif self.source_family == VisualReadoutSourceFamily.CORE_MULTI_LAYER_TOKENS:
            if len(self.layer_indices) < 2:
                raise ValueError("`core_multi_layer_tokens` requires at least two `layer_indices`.")
            if self.layer_index is not None:
                raise ValueError("`core_multi_layer_tokens` should not set `layer_index`.")
            if self.fusion_mode == VisualReadoutFusionMode.NONE:
                raise ValueError("`core_multi_layer_tokens` requires a non-`none` fusion mode.")
        else:
            if self.layer_indices and self.source_family != VisualReadoutSourceFamily.DIFFUSION_FEATURE_TOKENS:
                raise ValueError(
                    "`layer_indices` is only valid for `core_multi_layer_tokens` or diffusion-feature readouts."
                )
            if self.layer_index is not None and self.source_family != VisualReadoutSourceFamily.DIFFUSION_FEATURE_TOKENS:
                raise ValueError(
                    "`layer_index` is only valid for `core_layer_tokens` or diffusion-feature readouts."
                )
        if self.source_family != VisualReadoutSourceFamily.DIFFUSION_FEATURE_TOKENS:
            if self.diffusion_extract_timestep != 20 or self.diffusion_extract_step_time != 1:
                raise ValueError(
                    "Diffusion extraction controls are only valid for `diffusion_feature_tokens`."
                )
        if self.diffusion_extract_step_time <= 0:
            raise ValueError("`diffusion_extract_step_time` must be positive.")


def parse_visual_readout_config(
    raw_value: Mapping[str, Any] | None,
) -> VisualReadoutConfig | None:
    """Parse an optional visual feature-selection contract."""

    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("Expected `visual_readout` to be a mapping.")
    layer_index = raw_value.get("layer_index")
    layer_indices_raw = raw_value.get("layer_indices")
    if layer_indices_raw is None:
        layer_indices: tuple[int, ...] = ()
    elif isinstance(layer_indices_raw, (list, tuple)):
        layer_indices = tuple(int(value) for value in layer_indices_raw)
    else:
        raise ValueError(
            "Expected `visual_readout.layer_indices` to be a list or tuple."
        )
    return VisualReadoutConfig(
        source_family=coerce_enum(
            VisualReadoutSourceFamily,
            raw_value["source_family"],
        ),
        layer_index=(None if layer_index is None else int(layer_index)),
        layer_indices=layer_indices,
        fusion_mode=coerce_enum(
            VisualReadoutFusionMode,
            raw_value.get("fusion_mode", VisualReadoutFusionMode.NONE),
        ),
        diffusion_extract_timestep=int(
            raw_value.get("diffusion_extract_timestep", 20)
        ),
        diffusion_extract_step_time=int(
            raw_value.get("diffusion_extract_step_time", 1)
        ),
    )
