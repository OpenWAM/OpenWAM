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
        if any(sample.canonical_video is not None for sample in samples):
            raise ValueError(
                "Variable-length latent batching does not support online canonical RGB tensors."
            )
        if any(
            sample.video_latents.ndim != 4 or sample.actions.ndim != 2
            for sample in samples
        ):
            raise ValueError("Expected sample video [C,T,H,W] and actions [A,D].")
        lengths = tuple(int(sample.video_latents.shape[1]) for sample in samples)
        if min(lengths) < 2:
            raise ValueError(
                "Variable-length policy batches require at least two latent frames per sample."
            )
        ratios = []
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
            if sample.actions.shape[0] <= 0 or sample.actions.shape[0] % length:
                raise ValueError(
                    "Latent batching requires a positive integral number of action rows per video frame."
                )
            ratios.append(int(sample.actions.shape[0]) // length)
        if len(set(ratios)) != 1:
            raise ValueError(
                "All samples in a latent batch must share action rows per video frame."
            )
        multiple = self.config.pad_to_multiple_of
        padded_frames = math.ceil(max(lengths) / multiple) * multiple
        originals = [
            replace(sample, metadata=dict(sample.metadata)) for sample in samples
        ]
        metadata = self._shared_geometry_metadata(originals, lengths)
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
        axes = {
            "video_latents": 1,
            "actions": 0,
            "action_mask": 0,
            "state": 0,
            "state_mask": 0,
            "text_context": 0,
            "negative_text_context": 0,
            "condition_latents": 1,
            "proprio_context_state": 0,
            "proprio_context_state_mask": 0,
            "proprio_context_frames": 0,
            "proprio_context_frames_mask": 0,
        }
        payload: dict[str, torch.Tensor | None] = {}
        tensor_lengths: dict[str, tuple[int, ...]] = {}
        for name, axis in axes.items():
            values = [getattr(sample, name) for sample in samples]
            if all(value is None for value in values):
                payload[name] = None
                continue
            if any(value is None for value in values):
                raise ValueError(
                    f"Inconsistent optional field {name!r} across latent batch."
                )
            tensors = [value for value in values if value is not None]
            extents = tuple(int(value.shape[axis]) for value in tensors)
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
                target = padded_frames * ratios[0]
            elif extents == lengths:
                target = padded_frames
            else:
                target = max(extents)
            payload[name] = torch.stack(
                [_pad_axis(value, axis, target) for value in tensors]
            )
        return LatentWAMBatch(
            **payload,
            task_text=tuple(sample.task_text for sample in samples),
            metadata=tuple(metadata),
            batching_mode=self.config.mode,
            sequence_lengths=lengths,
            tensor_lengths=tensor_lengths,
        )

    def _shared_geometry_metadata(
        self, samples: list[LatentWAMSample], lengths: tuple[int, ...]
    ) -> list[dict[str, object]]:
        keys = ("sampled_chunk_size", "sampled_window_size", "history_frames")
        has_geometry = ["sampled_chunk_size" in sample.metadata for sample in samples]
        if any(has_geometry) and not all(has_geometry):
            raise ValueError(
                "A latent batch cannot mix explicit and absent sampled geometry."
            )
        shared_chunk = None
        shared_window = None
        if all(has_geometry):
            if any("sampled_window_size" not in sample.metadata for sample in samples):
                raise ValueError("Sampled chunk geometry requires sampled_window_size.")
            shared_chunk = min(
                int(samples[0].metadata["sampled_chunk_size"]), min(lengths)
            )
            shared_window = int(samples[0].metadata["sampled_window_size"])
            if shared_chunk <= 0 or shared_window <= 0:
                raise ValueError("Sampled chunk/window sizes must be positive.")
        output = []
        for sample, length in zip(samples, lengths, strict=True):
            row = _metadata_with_action_stats(sample)
            row["batching_sequence_length"] = length
            row["valid_video_frames"] = int(
                row.get("segment_valid_latent_frames", length)
            )
            if shared_chunk is not None and shared_window is not None:
                row["batching_original_geometry"] = {
                    key: row[key] for key in keys if key in row
                }
                row["sampled_chunk_size"] = shared_chunk
                row["sampled_window_size"] = shared_window
                # Explicit loss boundaries retain their own history semantics.
                if "loss_frame_start" not in row:
                    row["history_frames"] = max(
                        1,
                        min(
                            math.ceil(shared_window / 2) * shared_chunk,
                            length - shared_chunk,
                        ),
                    )
            output.append(row)
        return output
