from __future__ import annotations

from typing import TypeVar

import torch

from open_wam.configs.enums import StrEnum

ModeEnumT = TypeVar("ModeEnumT", bound=StrEnum)


def sample_conditioning_mode(
    probs: dict[ModeEnumT, float],
    *,
    enum_cls: type[ModeEnumT],
    device: torch.device,
    error_label: str,
) -> ModeEnumT:
    """Sample one enum-backed conditioning mode from normalized probabilities."""

    modes = tuple(enum_cls)
    weights = torch.tensor([float(probs.get(mode, 0.0)) for mode in modes], device=device, dtype=torch.float32)
    if float(weights.sum().item()) <= 0.0:
        raise ValueError(f"{error_label} probabilities must have positive total weight.")
    index = int(torch.multinomial(weights, num_samples=1).item())
    return modes[index]


def should_drop_text_for_conditioning_mode(
    mode: ModeEnumT,
    *,
    joint_mode: ModeEnumT,
    drop_text_conditioning: bool | None,
) -> bool:
    """Resolve text-drop semantics for joint-vs-conditional denoising modes."""

    if drop_text_conditioning is not None:
        return bool(drop_text_conditioning)
    return mode != joint_mode
