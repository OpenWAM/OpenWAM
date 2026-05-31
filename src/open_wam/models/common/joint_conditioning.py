from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import torch

from open_wam.configs.enums import StrEnum

ModeEnumT = TypeVar("ModeEnumT", bound=StrEnum)


@dataclass(frozen=True)
class JointConditioningModeSemantics:
    """Shared GJD mode contract used by M1 and M5.

    This object captures method-agnostic semantics only. Each policy variant
    still owns its artifact layout and applies these decisions to its local
    tensors.
    """

    mode_value: str
    clean_action_noisy_slot: bool
    clean_video_noisy_slot: bool
    action_loss_active: bool
    video_loss_active: bool
    drop_text_conditioning: bool
    force_clean_video_condition: bool
    conditional_history_chunks: int

    @property
    def is_joint(self) -> bool:
        return self.action_loss_active and self.video_loss_active

    @property
    def is_conditional(self) -> bool:
        return not self.is_joint

    def attention_window_size(self, *, fallback_window_size: int) -> int:
        if self.conditional_history_chunks > 0:
            return one_history_chunk_block_window()
        return max(1, int(fallback_window_size))


def sample_conditioning_mode(
    probs: dict[ModeEnumT, float],
    *,
    enum_cls: type[ModeEnumT],
    device: torch.device,
    error_label: str,
) -> ModeEnumT:
    """Sample one enum-backed conditioning mode from normalized probabilities.

    FSDP-wrapped GJD forwards must choose the same conditioning branch on every
    rank. If ranks diverge, they can enqueue different FSDP all-gathers and hit
    NCCL watchdog timeouts. Rank 0 owns the stochastic draw and broadcasts the
    selected enum index.
    """

    modes = tuple(enum_cls)
    weights = torch.tensor([float(probs.get(mode, 0.0)) for mode in modes], device=device, dtype=torch.float32)
    if float(weights.sum().item()) <= 0.0:
        raise ValueError(f"{error_label} probabilities must have positive total weight.")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            index_tensor = torch.multinomial(weights, num_samples=1).to(device=device, dtype=torch.long)
        else:
            index_tensor = torch.zeros(1, device=device, dtype=torch.long)
        torch.distributed.broadcast(index_tensor, src=0)
    else:
        index_tensor = torch.multinomial(weights, num_samples=1).to(device=device, dtype=torch.long)
    index = int(index_tensor.item())
    return modes[index]


def mode_value(mode: StrEnum | str) -> str:
    return mode.value if isinstance(mode, StrEnum) else str(mode)


def resolve_generalist_joint_conditioning_semantics(
    mode: ModeEnumT | str,
    *,
    joint_mode: ModeEnumT,
    action_conditioned_video_mode: ModeEnumT,
    video_conditioned_action_mode: ModeEnumT,
    drop_text_conditioning: bool | None = None,
) -> JointConditioningModeSemantics:
    """Resolve the shared GJD semantics for one sampled mode.

    M5 is the canonical behavior:
    - joint: denoise video and action; keep configured noisy video condition.
    - action-conditioned-video/FDM: clean action is exposed in the noisy action
      slot, action loss is masked, video loss remains active, and task text is
      dropped by default.
    - video-conditioned-action/IDM: clean video is exposed in the noisy video
      slot, video loss is masked, action loss remains active, and task text is
      dropped by default.

    Conditional modes also force clean video condition slots and use one local
    history chunk.
    """

    resolved_mode = mode_value(mode)
    joint_value = joint_mode.value
    action_conditioned_video_value = action_conditioned_video_mode.value
    video_conditioned_action_value = video_conditioned_action_mode.value
    if resolved_mode == joint_value:
        return JointConditioningModeSemantics(
            mode_value=joint_value,
            clean_action_noisy_slot=False,
            clean_video_noisy_slot=False,
            action_loss_active=True,
            video_loss_active=True,
            drop_text_conditioning=bool(drop_text_conditioning) if drop_text_conditioning is not None else False,
            force_clean_video_condition=False,
            conditional_history_chunks=0,
        )
    if resolved_mode == action_conditioned_video_value:
        return JointConditioningModeSemantics(
            mode_value=action_conditioned_video_value,
            clean_action_noisy_slot=True,
            clean_video_noisy_slot=False,
            action_loss_active=False,
            video_loss_active=True,
            drop_text_conditioning=True,
            force_clean_video_condition=True,
            conditional_history_chunks=1,
        )
    if resolved_mode == video_conditioned_action_value:
        return JointConditioningModeSemantics(
            mode_value=video_conditioned_action_value,
            clean_action_noisy_slot=False,
            clean_video_noisy_slot=True,
            action_loss_active=True,
            video_loss_active=False,
            drop_text_conditioning=True,
            force_clean_video_condition=True,
            conditional_history_chunks=1,
        )
    supported = ", ".join(
        sorted(
            {
                joint_value,
                action_conditioned_video_value,
                video_conditioned_action_value,
            }
        )
    )
    raise ValueError(f"Unsupported generalist joint-conditioning mode {resolved_mode!r}. Supported modes: {supported}.")


def is_conditional_joint_conditioning_mode(
    mode: ModeEnumT | str,
    *,
    joint_mode: ModeEnumT,
    action_conditioned_video_mode: ModeEnumT,
    video_conditioned_action_mode: ModeEnumT,
) -> bool:
    semantics = resolve_generalist_joint_conditioning_semantics(
        mode,
        joint_mode=joint_mode,
        action_conditioned_video_mode=action_conditioned_video_mode,
        video_conditioned_action_mode=video_conditioned_action_mode,
    )
    return semantics.is_conditional


def generalist_joint_conditioning_window_size(
    mode: ModeEnumT | str,
    *,
    joint_mode: ModeEnumT,
    action_conditioned_video_mode: ModeEnumT,
    video_conditioned_action_mode: ModeEnumT,
    fallback_window_size: int,
) -> int:
    semantics = resolve_generalist_joint_conditioning_semantics(
        mode,
        joint_mode=joint_mode,
        action_conditioned_video_mode=action_conditioned_video_mode,
        video_conditioned_action_mode=video_conditioned_action_mode,
    )
    return semantics.attention_window_size(fallback_window_size=fallback_window_size)


def should_drop_text_for_conditioning_mode(
    mode: ModeEnumT,
    *,
    joint_mode: ModeEnumT,
    drop_text_conditioning: bool | None,
) -> bool:
    """Resolve text-drop semantics for joint-vs-conditional denoising modes."""

    if mode != joint_mode:
        return True
    if drop_text_conditioning is not None:
        return bool(drop_text_conditioning)
    return False


def one_history_chunk_block_window() -> int:
    """Return the block-local window that covers one full previous V/A chunk.

    Packed M1/M5 joint layouts assign video and action chunks to adjacent block
    ids. The farthest immediate-history edge is current action -> previous
    video, which is three block ids away.
    """

    return 3
