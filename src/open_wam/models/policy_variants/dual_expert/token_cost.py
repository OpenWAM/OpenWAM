"""Exact real-token admission counts for supported packed training batches."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Protocol

import torch

from open_wam.configs import (
    BatchingMode,
    ExperimentConfig,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.contracts.sample_metadata import (
    DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_METADATA_KEY,
    DYNAMICS_CONDITIONAL_HISTORY_POLICY_METADATA_KEY,
    DYNAMICS_CONDITIONAL_LAYOUT_METADATA_KEY,
    DYNAMICS_ROUTING_BUCKET_METADATA_KEY,
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY,
    DYNAMICS_ROUTING_MODE_METADATA_KEY,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
)


class LatentTokenAdmissionBatch(Protocol):
    """Read-only batch inputs consumed by token admission, independent of loaders.

    Producers supply tensors and their original per-sample extents structurally;
    no inheritance, tensor copying, or data-layer implementation is required.
    The counter still validates values and shapes at runtime before admission.
    """

    @property
    def video_latents(self) -> torch.Tensor: ...

    @property
    def actions(self) -> torch.Tensor: ...

    @property
    def action_mask(self) -> torch.Tensor | None: ...

    @property
    def state(self) -> torch.Tensor | None: ...

    @property
    def state_mask(self) -> torch.Tensor | None: ...

    @property
    def text_context(self) -> torch.Tensor | None: ...

    @property
    def negative_text_context(self) -> torch.Tensor | None: ...

    @property
    def canonical_video(self) -> torch.Tensor | None: ...

    @property
    def condition_latents(self) -> torch.Tensor | None: ...

    @property
    def proprio_context_state(self) -> torch.Tensor | None: ...

    @property
    def proprio_context_state_mask(self) -> torch.Tensor | None: ...

    @property
    def proprio_context_frames(self) -> torch.Tensor | None: ...

    @property
    def proprio_context_frames_mask(self) -> torch.Tensor | None: ...

    @property
    def batching_mode(self) -> BatchingMode: ...

    @property
    def sequence_lengths(self) -> tuple[int, ...] | list[int]: ...

    @property
    def tensor_lengths(self) -> Mapping[str, tuple[int | None, ...] | list[int | None]]: ...

    @property
    def metadata(self) -> tuple[Mapping[str, object], ...] | list[Mapping[str, object]]: ...


_TENSOR_TIME_AXES = {
    "video_latents": 2,
    "actions": 1,
    "action_mask": 1,
    "state": 1,
    "state_mask": 1,
    "text_context": 1,
    "negative_text_context": 1,
    "condition_latents": 2,
    "proprio_context_state": 1,
    "proprio_context_state_mask": 1,
    "proprio_context_frames": 1,
    "proprio_context_frames_mask": 1,
}
_DYNAMICS_METADATA_KEYS = (
    DYNAMICS_ROUTING_MODE_METADATA_KEY,
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
    DYNAMICS_ROUTING_BUCKET_METADATA_KEY,
    DYNAMICS_CONDITIONAL_LAYOUT_METADATA_KEY,
    DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_METADATA_KEY,
    DYNAMICS_CONDITIONAL_HISTORY_POLICY_METADATA_KEY,
)


def _bounded_lengths(
    value, *, name: str, size: int, capacity: int, optional: bool = False
) -> tuple[int | None, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f"Token admission requires one original {name} length per sample.")
    if any(
        not (optional and length is None)
        and (
            isinstance(length, bool)
            or not isinstance(length, Integral)
            or not (0 if optional else 1) <= length <= capacity
        )
        for length in value
    ):
        raise ValueError(
            f"Original {name} lengths must be {'nonnegative integers or None' if optional else 'positive integers'} bounded by "
            f"the physical CPU tensor capacity ({capacity})."
        )
    return tuple(None if length is None else int(length) for length in value)


def dual_expert_token_costs(
    *, config: ExperimentConfig, batch: LatentTokenAdmissionBatch
) -> tuple[int, ...]:
    """Count real self-attention/FFN tokens before transferring a batch to CUDA.

    For each original sample, the count is ``2 * (T + prefix) * Hpatch *
    Wpatch + 2 * action_rows``. The two copies are clean and noisy streams.
    Legacy-prefix training adds one VIDEO frame; it does not add action rows.
    Default Joint/full and per-chunk additive proprio do not add token slots.

    Dataset-materialized padding and masked action rows still execute and are
    counted. Only padding introduced by collation is excluded. Text/proprio
    cross-attention K/V lengths are not included in this self-token count; it is
    an exact token admission unit, not an exact GPU-memory prediction.
    """
    if (
        config.data.batching.mode is not BatchingMode.PACKED
        or batch.batching_mode is not BatchingMode.PACKED
    ):
        raise ValueError("Token admission currently requires packed batching mode.")
    policy = config.policy_variant
    if not isinstance(policy, DualExpertPolicyConfig) or policy.program not in {
        VideoActionProgram.VIDEO_THEN_ACTION,
        VideoActionProgram.JOINT,
    }:
        raise ValueError("Token admission currently supports DualExpert VTA/Joint only.")
    if config.data.dynamics_routing.routes or policy.generalist_mode_text_token:
        raise ValueError("Token admission does not support dynamics routing or mode tokens.")
    if policy.sequence_contract not in {
        VideoActionSequenceContract.DEFAULT,
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    }:
        raise ValueError("Token admission does not support this sequence/prefix contract.")
    prefix = int(
        policy.sequence_contract
        is VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    patch = (
        config.backbone.patch_size_t,
        config.backbone.patch_size_h,
        config.backbone.patch_size_w,
    )
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in patch):
        raise ValueError("Token admission requires positive integer patch sizes.")
    if patch[0] != 1:
        raise ValueError("Token admission requires frame-aligned temporal patch size 1.")
    if batch.canonical_video is not None:
        raise ValueError("Token admission does not support online canonical RGB inputs.")
    video, actions = batch.video_latents, batch.actions
    if (
        not isinstance(video, torch.Tensor)
        or video.ndim != 5
        or not isinstance(actions, torch.Tensor)
        or actions.ndim != 3
        or min(video.shape) <= 0
        or min(actions.shape) <= 0
    ):
        raise ValueError("Token admission requires video [B,C,T,H,W] and actions [B,A,D].")
    size = int(video.shape[0])
    height, width = map(int, video.shape[-2:])
    if height % patch[1] or width % patch[2]:
        raise ValueError("Token admission requires spatial shapes divisible by patch sizes.")
    if not isinstance(batch.tensor_lengths, Mapping):
        raise ValueError("Token admission requires original tensor_lengths metadata.")
    lengths = _bounded_lengths(
        batch.sequence_lengths, name="sequence", size=size, capacity=int(video.shape[2])
    )
    if min(lengths) < 2:
        raise ValueError("Token admission requires at least two materialized video frames.")
    actual_lengths = {}
    for name, axis in _TENSOR_TIME_AXES.items():
        tensor = getattr(batch, name)
        if tensor is None:
            continue
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.ndim <= axis
            or tensor.shape[0] != size
        ):
            raise ValueError(f"Token admission requires a correctly batched CPU {name} tensor.")
        actual_lengths[name] = _bounded_lengths(
            batch.tensor_lengths.get(name),
            name=name,
            size=size,
            capacity=int(tensor.shape[axis]),
            optional=name not in {"video_latents", "actions", "action_mask"},
        )
    if actual_lengths["video_latents"] != lengths:
        raise ValueError("sequence_lengths and video_latents tensor_lengths disagree.")
    action_lengths = actual_lengths["actions"]
    if any(a % t or a < t for t, a in zip(lengths, action_lengths, strict=True)):
        raise ValueError("Token admission requires a positive integral action/frame ratio.")
    if prefix:
        condition = batch.condition_latents
        if (
            not policy.use_condition_latents
            or condition is None
            or condition.ndim != 5
            or condition.shape[1] != video.shape[1]
            or condition.shape[-2:] != video.shape[-2:]
            or any(length is None or length <= 0 for length in actual_lengths["condition_latents"])
        ):
            raise ValueError("Legacy-prefix token admission requires matching condition latents.")
    if not isinstance(batch.metadata, (tuple, list)) or len(batch.metadata) != size:
        raise ValueError("Token admission requires metadata for every original sample.")
    for metadata, frames in zip(batch.metadata, lengths, strict=True):
        if not isinstance(metadata, Mapping):
            raise ValueError("Token admission requires each sample metadata to be a mapping.")
        if any(metadata.get(key) is not None for key in _DYNAMICS_METADATA_KEYS):
            raise ValueError("Token admission does not support routed/dynamics sample metadata.")
        geometry = tuple(
            metadata.get(key)
            for key in ("sampled_chunk_size", "sampled_window_size")
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value <= 0
            for value in geometry
        ):
            raise ValueError(
                "Token admission requires positive integer sampled_chunk_size and "
                "sampled_window_size stamped on every logical-batch sample; "
                "per-microbatch fallback geometry would change sample semantics."
            )
        if geometry[0] > frames:
            raise ValueError("Token admission requires sampled_chunk_size <= sample frames.")
    patches_per_frame = (height // int(patch[1])) * (width // int(patch[2]))
    return tuple(
        2 * (frames + prefix) * patches_per_frame + 2 * action_rows
        for frames, action_rows in zip(lengths, action_lengths, strict=True)
    )


__all__ = ["dual_expert_token_costs"]
