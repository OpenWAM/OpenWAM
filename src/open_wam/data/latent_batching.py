"""Explicit collation and bounded length bucketing for latent sequences."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from itertools import islice

import torch
from torch.utils.data import Sampler

from open_wam.configs import BatchingConfig, BatchingMode

from .latent_contracts import (
    LATENT_SAMPLE_TENSOR_AXES,
    LatentWAMBatch,
    LatentWAMSample,
    _metadata_with_action_stats,
    collate_latent_wam_samples,
)


class LengthBucketSampler(Sampler[int]):
    """Reorder finite pools without changing the underlying sampled multiset.

    Pool boundaries are multiples of the fixed microbatch size. Any dropped
    tail belongs to the original sampler stream, before length sorting. Epoch
    forwarding and deterministic ordering preserve loader-cursor resume.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        *,
        length_for_index: Callable[[int], int],
        batch_size: int,
        pool_size: int,
        drop_last: bool,
    ) -> None:
        if batch_size <= 0 or pool_size <= 0:
            raise ValueError("Bucket batch_size and pool_size must be positive.")
        self.sampler = sampler
        self.length_for_index = length_for_index
        self.batch_size = int(batch_size)
        self.pool_size = max(batch_size, (pool_size // batch_size) * batch_size)
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        size = len(self.sampler)  # type: ignore[arg-type]
        return size - size % self.batch_size if self.drop_last else size

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.sampler, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))

    def __iter__(self) -> Iterator[int]:
        source = iter(islice(iter(self.sampler), len(self)))
        while pool := list(islice(source, self.pool_size)):
            # Python's stable sort keeps equal-length replacement draws intact.
            yield from sorted(pool, key=self.length_for_index)


def _pad_axis(value: torch.Tensor, axis: int, size: int) -> torch.Tensor:
    if int(value.shape[axis]) == size:
        return value
    shape = list(value.shape)
    shape[axis] = size
    padded = value.new_zeros(shape)
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(0, value.shape[axis])
    padded[tuple(slices)] = value
    return padded


@dataclass(frozen=True)
class LatentBatchCollator:
    """Keep original per-sample extents while padding the transport batch.

    Padded and packed modes share this transport contract. The policy runtime
    owns masking or removing transformer padding, not the dataset adapter.
    """

    config: BatchingConfig

    def __call__(self, samples: list[LatentWAMSample]) -> LatentWAMBatch:
        if self.config.mode is BatchingMode.STRICT:
            return collate_latent_wam_samples(samples)
        if not samples:
            raise ValueError("Cannot collate an empty latent batch.")
        if any(
            sample.video_latents.ndim != 4 or sample.actions.ndim != 2
            for sample in samples
        ):
            raise ValueError("Expected sample video [C,T,H,W] and actions [A,D].")
        lengths = tuple(int(sample.video_latents.shape[1]) for sample in samples)
        if min(lengths) < 1:
            raise ValueError(
                "Latent batches require at least one video frame per sample."
            )
        for sample, length in zip(samples, lengths, strict=True):
            if sample.negative_text_context is not None and (
                sample.text_context is None
                or sample.text_context.shape != sample.negative_text_context.shape
            ):
                raise ValueError(
                    "Positive/negative text contexts must have matching shapes within each sample."
                )
            if (
                sample.action_mask is not None
                and sample.action_mask.shape != sample.actions.shape
            ):
                raise ValueError(
                    "Action masks must match each sample's action tensor shape."
                )
        multiple = self.config.pad_to_multiple_of
        padded_frames = math.ceil(max(lengths) / multiple) * multiple
        originals = [
            replace(sample, metadata=dict(sample.metadata)) for sample in samples
        ]
        metadata = [_metadata_with_action_stats(sample) for sample in originals]
        # Synthesizing an action mask makes all newly introduced padding inert.
        samples = [
            replace(
                sample,
                action_mask=torch.ones_like(sample.actions)
                if sample.action_mask is None
                else sample.action_mask,
            )
            for sample in originals
        ]
        payload: dict[str, torch.Tensor | None] = {}
        tensor_lengths: dict[str, tuple[int | None, ...]] = {}
        for name, axis in LATENT_SAMPLE_TENSOR_AXES.items():
            values = [getattr(sample, name) for sample in samples]
            if all(value is None for value in values):
                payload[name] = None
                continue
            tensors = [value for value in values if value is not None]
            extents = tuple(
                None if value is None else int(value.shape[axis]) for value in values
            )
            shapes = {
                tuple(value.shape[:axis]) + tuple(value.shape[axis + 1 :])
                for value in tensors
            }
            if len(shapes) != 1:
                raise ValueError(
                    f"Non-temporal dimensions must match for latent field {name!r}."
                )
            tensor_lengths[name] = extents
            if name in ("actions", "action_mask"):
                ratios = {
                    extent // length
                    for extent, length in zip(extents, lengths, strict=True)
                }
                aligned = all(
                    extent % length == 0
                    for extent, length in zip(extents, lengths, strict=True)
                )
                target = (
                    padded_frames * next(iter(ratios))
                    if aligned and len(ratios) == 1
                    else max(extents)
                )
            elif extents == lengths:
                target = padded_frames
            else:
                target = max(extent for extent in extents if extent is not None)
            empty_shape = list(tensors[0].shape)
            empty_shape[axis] = target
            payload[name] = torch.stack(
                [
                    tensors[0].new_zeros(empty_shape)
                    if value is None
                    else _pad_axis(value, axis, target)
                    for value in values
                ]
            )
        return LatentWAMBatch(
            **payload,
            task_text=tuple(sample.task_text for sample in samples),
            metadata=tuple(metadata),
            batching_mode=self.config.mode,
            sequence_lengths=lengths,
            tensor_lengths=tensor_lengths,
        )
