"""Sample-level Generalist Joint Denoising source and mode routing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import torch
from torch.utils.data import Dataset, Sampler

from open_wam.configs import (
    DataConfig,
    GeneralistDenoisingMode,
    GeneralistDynamicsMixtureConfig,
    WindowSamplingMode,
)
from open_wam.contracts import (
    GENERALIST_TRAINING_BUCKET_METADATA_KEY,
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
)

from .conditional_dynamics_layout import project_real_conditional_sample_to_target_only

# Compatibility re-exports preserve the historical mixture-module import path.
from .counterfactual_dynamics_dataset import EncodedCounterfactualDynamicsLatentDataset
from .counterfactual_dynamics_materialization import (
    COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY,
    COUNTERFACTUAL_CONTRACT_T0_PLUS_FUTURE,
    COUNTERFACTUAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE,
    COUNTERFACTUAL_STATE_KEY,
)
from .counterfactual_source_order import _balanced_counterfactual_source_indices
from .distributed_sampling import PaddedEpochOffsetDistributedSampler
from .latent_contracts import LatentWAMSample

_COUNTERFACTUAL_COMPATIBILITY_EXPORTS = (
    COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY,
    COUNTERFACTUAL_CONTRACT_T0_PLUS_FUTURE,
    COUNTERFACTUAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE,
    COUNTERFACTUAL_STATE_KEY,
    _balanced_counterfactual_source_indices,
)


REAL_DEMO_SOURCE = "real_demo"
COUNTERFACTUAL_DYNAMICS_SOURCE = "counterfactual_dynamics"
JOINT_MODE = "joint"
ACTION_CONDITIONED_VIDEO_MODE = "action_conditioned_video"
VIDEO_CONDITIONED_ACTION_MODE = "video_conditioned_action"


@dataclass(frozen=True)
class GeneralistMixtureBucket:
    name: str
    source: str
    mode: str
    weight: float
    drop_text: bool


class GeneralistDynamicsMixtureDataset(Dataset[LatentWAMSample]):
    """Sample-level mixture for the opt-in generalist dynamics paradigm."""

    def __init__(
        self,
        *,
        real_dataset: Dataset[LatentWAMSample],
        counterfactual_dataset: Dataset[LatentWAMSample] | None,
        mixture_config: GeneralistDynamicsMixtureConfig,
        split: str,
        fixed_mode: GeneralistDenoisingMode | str | None = None,
    ) -> None:
        self.real_dataset = real_dataset
        self.counterfactual_dataset = counterfactual_dataset
        self.mixture_config = mixture_config
        self.split = str(split)
        self.fixed_mode = (
            None
            if fixed_mode is None
            else GeneralistDenoisingMode(fixed_mode).value
        )
        self.buckets = _build_mixture_buckets(
            mixture_config,
            fixed_mode=self.fixed_mode,
        )
        active_sources = {bucket.source for bucket in self.buckets}
        if REAL_DEMO_SOURCE in active_sources and len(real_dataset) <= 0:
            raise ValueError(
                "Generalist dynamics mixture assigns positive real-demo weight but the dataset is empty."
            )
        if COUNTERFACTUAL_DYNAMICS_SOURCE in active_sources:
            if counterfactual_dataset is None:
                raise ValueError(
                    "Generalist dynamics mixture assigns positive counterfactual weight but no encoded "
                    "counterfactual dataset was provided."
                )
            if len(counterfactual_dataset) <= 0:
                raise ValueError(
                    "Generalist dynamics mixture assigns positive counterfactual weight but the dataset is empty."
                )
        self._distributed_draw_group_size = 1
        self._distributed_epoch_size = 0
        active_lengths = []
        if REAL_DEMO_SOURCE in active_sources:
            active_lengths.append(len(real_dataset))
        if COUNTERFACTUAL_DYNAMICS_SOURCE in active_sources:
            assert counterfactual_dataset is not None
            active_lengths.append(len(counterfactual_dataset))
        base_length = max(active_lengths)
        self._length = max(1, int(round(base_length * float(mixture_config.length_multiplier))))

    def __len__(self) -> int:
        return self._length

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int]:
        return GeneralistDynamicsMixtureTrainSampler(self, world_size=world_size, rank=rank)

    def set_distributed_draw_group_size(self, world_size: int) -> None:
        self.set_distributed_draw_geometry(world_size=world_size)

    def set_distributed_draw_geometry(self, *, world_size: int, epoch_size: int | None = None) -> None:
        group_size = max(1, int(world_size))
        self._distributed_draw_group_size = group_size
        if epoch_size is None:
            epoch_size = int(math.ceil(len(self) / float(group_size))) * group_size
        self._distributed_epoch_size = max(len(self), int(epoch_size))

    def build_source_view(
        self,
        *,
        source: str,
        mode: str,
        bucket_name: str,
        drop_text: bool,
        spread_indices: bool = False,
    ) -> Dataset[LatentWAMSample]:
        bucket = GeneralistMixtureBucket(
            name=str(bucket_name),
            source=str(source),
            mode=str(mode),
            weight=1.0,
            drop_text=bool(drop_text),
        )
        return GeneralistDynamicsSourceViewDataset(
            self,
            bucket=bucket,
            spread_indices=spread_indices,
        )

    def has_source(self, source: str) -> bool:
        """Return whether a concrete source dataset is available for validation."""

        if source == REAL_DEMO_SOURCE:
            return len(self.real_dataset) > 0
        if source == COUNTERFACTUAL_DYNAMICS_SOURCE:
            return (
                self.counterfactual_dataset is not None
                and len(self.counterfactual_dataset) > 0
            )
        return False

    def source_dataset(self, source: str) -> Dataset[LatentWAMSample]:
        """Resolve one concrete source or fail with a source-specific error."""

        if source == REAL_DEMO_SOURCE and self.has_source(source):
            return self.real_dataset
        if source == COUNTERFACTUAL_DYNAMICS_SOURCE and self.has_source(source):
            assert self.counterfactual_dataset is not None
            return self.counterfactual_dataset
        raise ValueError(
            f"Generalist dynamics source {source!r} is not available for split {self.split!r}."
        )

    def __getitem__(self, index: int) -> LatentWAMSample:
        index = int(index)
        group_size = max(1, int(getattr(self, "_distributed_draw_group_size", 1)))
        epoch_size = int(getattr(self, "_distributed_epoch_size", 0) or len(self))
        epoch = index // max(1, epoch_size)
        epoch_index = index % max(1, epoch_size)
        if group_size == 1:
            rng = random.Random(int(self.mixture_config.seed) + index * 1_000_003)
            bucket = _sample_bucket(self.buckets, rng)
            source_rng = rng
        else:
            # FSDP requires every rank to enter the same sharded module path in
            # the same order. Coordinate the source/mode bucket per distributed
            # step, then vary the source-row draw by rank for data diversity.
            draw_group = index // group_size
            rank_offset = epoch_index % group_size
            bucket_rng = random.Random(int(self.mixture_config.seed) + draw_group * 1_000_003)
            bucket = _sample_bucket(self.buckets, bucket_rng)
            source_rng = random.Random(
                int(self.mixture_config.seed)
                + draw_group * 1_000_003
                + (rank_offset + 1) * 9176
            )
        if bucket.source == REAL_DEMO_SOURCE:
            sample_index = _draw_source_index(self.real_dataset, rng=source_rng, epoch=epoch)
            sample = self.real_dataset[sample_index]
        elif bucket.source == COUNTERFACTUAL_DYNAMICS_SOURCE:
            if self.counterfactual_dataset is None:  # pragma: no cover - constructor validates active buckets.
                raise RuntimeError("Counterfactual bucket selected without a counterfactual dataset.")
            sample_index = _draw_source_index(self.counterfactual_dataset, rng=source_rng, epoch=epoch)
            sample = self.counterfactual_dataset[sample_index]
        else:
            raise ValueError(f"Unsupported generalist source bucket {bucket.source!r}.")
        if _uses_real_target_only_conditional_layout(bucket):
            sample = project_real_conditional_sample_to_target_only(sample)
        return _with_generalist_metadata(
            sample,
            bucket=bucket,
            split=self.split,
            source_index=sample_index,
        )


class GeneralistDynamicsSourceViewDataset(Dataset[LatentWAMSample]):
    """Deterministic source projection that preserves mixture sample transforms."""

    def __init__(
        self,
        mixture_dataset: GeneralistDynamicsMixtureDataset,
        *,
        bucket: GeneralistMixtureBucket,
        spread_indices: bool = False,
    ) -> None:
        self.mixture_dataset = mixture_dataset
        self.bucket = bucket
        self.spread_indices = bool(spread_indices)
        self.source_dataset = mixture_dataset.source_dataset(bucket.source)
        self._spread_source_indices = _balanced_source_indices_for_dataset(self.source_dataset) if self.spread_indices else None
        self._uses_balanced_source_indices = (
            self._spread_source_indices is not None and len(self._spread_source_indices) > 0
        )
        self._source_spread_stride = _source_view_spread_stride(len(self.source_dataset))

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> LatentWAMSample:
        source_index = int(index)
        if self._uses_balanced_source_indices:
            source_index = int(self._spread_source_indices[source_index % len(self._spread_source_indices)])
        elif self.spread_indices and len(self.source_dataset) > 1:
            source_index = (source_index * self._source_spread_stride) % len(self.source_dataset)
        sample = self.source_dataset[source_index]
        if _uses_real_target_only_conditional_layout(self.bucket):
            sample = project_real_conditional_sample_to_target_only(sample)
        sample = _with_generalist_metadata(
            sample,
            bucket=self.bucket,
            split=self.mixture_dataset.split,
            source_index=source_index,
        )
        if self.spread_indices:
            metadata = dict(sample.metadata)
            metadata["generalist_source_view_index"] = int(index)
            metadata["generalist_source_view_order"] = "balanced" if self._uses_balanced_source_indices else "stride"
            metadata["generalist_source_view_stride"] = (
                1 if self._uses_balanced_source_indices else int(self._source_spread_stride)
            )
            sample = replace(sample, metadata=metadata)
        return sample


class GeneralistDynamicsMixtureTrainSampler(PaddedEpochOffsetDistributedSampler):
    """Epoch-offset sampler for mixed real/counterfactual dynamics draws."""

    def __init__(self, dataset: GeneralistDynamicsMixtureDataset, *, world_size: int = 1, rank: int = 0) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message="Generalist dynamics mixture sampling requires a non-empty dataset.",
        )
        self.dataset.set_distributed_draw_geometry(
            world_size=self.world_size,
            epoch_size=self._total_size,
        )


def build_generalist_dynamics_mixture_datasets(
    *,
    data_config: DataConfig,
    train_dataset: Dataset[LatentWAMSample],
    val_dataset: Dataset[LatentWAMSample],
    fixed_mode: GeneralistDenoisingMode | str | None = None,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    mixture_config = data_config.generalist_dynamics_mixture
    buckets = _build_mixture_buckets(mixture_config, fixed_mode=fixed_mode)
    uses_counterfactual = any(
        bucket.source == COUNTERFACTUAL_DYNAMICS_SOURCE
        for bucket in buckets
    )
    train_counterfactual: Dataset[LatentWAMSample] | None = None
    val_counterfactual: Dataset[LatentWAMSample] | None = None
    if uses_counterfactual:
        if mixture_config.train_latent_root is None:
            raise ValueError(
                "`generalist_training_paradigm = dynamics_routed` requires "
                "`data.generalist_dynamics_mixture.train_latent_root` when a counterfactual "
                "bucket has positive weight."
            )
        train_counterfactual = EncodedCounterfactualDynamicsLatentDataset(
            data_config,
            mixture_config.train_latent_root,
            split="train",
        )
        val_root = mixture_config.val_latent_root
        if val_root is None:
            if not mixture_config.allow_train_latent_root_for_val:
                raise ValueError(
                    "`generalist_training_paradigm = dynamics_routed` requires "
                    "`data.generalist_dynamics_mixture.val_latent_root` for validation when a "
                    "counterfactual bucket has positive weight. Set "
                    "`allow_train_latent_root_for_val: true` only for local debug runs."
                )
            val_root = mixture_config.train_latent_root
        val_counterfactual = EncodedCounterfactualDynamicsLatentDataset(
            data_config,
            val_root,
            split="val",
        )
    return (
        GeneralistDynamicsMixtureDataset(
            real_dataset=train_dataset,
            counterfactual_dataset=train_counterfactual,
            mixture_config=mixture_config,
            split="train",
            fixed_mode=fixed_mode,
        ),
        GeneralistDynamicsMixtureDataset(
            real_dataset=val_dataset,
            counterfactual_dataset=val_counterfactual,
            mixture_config=mixture_config,
            split="val",
            fixed_mode=fixed_mode,
        ),
    )


def _build_mixture_buckets(
    config: GeneralistDynamicsMixtureConfig,
    *,
    fixed_mode: GeneralistDenoisingMode | str | None = None,
) -> tuple[GeneralistMixtureBucket, ...]:
    buckets = (
        GeneralistMixtureBucket(
            name="real_joint",
            source=REAL_DEMO_SOURCE,
            mode=JOINT_MODE,
            weight=float(config.real_joint_weight),
            drop_text=False,
        ),
        GeneralistMixtureBucket(
            name="real_action_conditioned_video",
            source=REAL_DEMO_SOURCE,
            mode=ACTION_CONDITIONED_VIDEO_MODE,
            weight=float(config.real_action_conditioned_video_weight),
            drop_text=True,
        ),
        GeneralistMixtureBucket(
            name="real_video_conditioned_action",
            source=REAL_DEMO_SOURCE,
            mode=VIDEO_CONDITIONED_ACTION_MODE,
            weight=float(config.real_video_conditioned_action_weight),
            drop_text=True,
        ),
        GeneralistMixtureBucket(
            name="counterfactual_action_conditioned_video",
            source=COUNTERFACTUAL_DYNAMICS_SOURCE,
            mode=ACTION_CONDITIONED_VIDEO_MODE,
            weight=float(config.counterfactual_action_conditioned_video_weight),
            drop_text=True,
        ),
        GeneralistMixtureBucket(
            name="counterfactual_video_conditioned_action",
            source=COUNTERFACTUAL_DYNAMICS_SOURCE,
            mode=VIDEO_CONDITIONED_ACTION_MODE,
            weight=float(config.counterfactual_video_conditioned_action_weight),
            drop_text=True,
        ),
    )
    resolved_fixed_mode = (
        None
        if fixed_mode is None
        else GeneralistDenoisingMode(fixed_mode).value
    )
    active = tuple(
        bucket
        for bucket in buckets
        if bucket.weight > 0.0
        and (resolved_fixed_mode is None or bucket.mode == resolved_fixed_mode)
    )
    if not active:
        suffix = (
            ""
            if resolved_fixed_mode is None
            else f" for fixed mode {resolved_fixed_mode!r}"
        )
        raise ValueError(
            "`data.generalist_dynamics_mixture` must contain at least one positive source weight"
            f"{suffix}."
        )
    return active


def _sample_bucket(buckets: tuple[GeneralistMixtureBucket, ...], rng: random.Random) -> GeneralistMixtureBucket:
    total = sum(bucket.weight for bucket in buckets)
    draw = rng.random() * total
    cursor = 0.0
    for bucket in buckets:
        cursor += bucket.weight
        if draw <= cursor:
            return bucket
    return buckets[-1]


def _draw_source_index(dataset: Dataset[LatentWAMSample], *, rng: random.Random, epoch: int) -> int:
    local_index = int(rng.randrange(len(dataset)))
    if _dataset_uses_epoch_offset_draw_keys(dataset):
        return int(epoch) * len(dataset) + local_index
    return local_index


def _dataset_uses_epoch_offset_draw_keys(dataset: Dataset[LatentWAMSample]) -> bool:
    explicit = getattr(dataset, "uses_epoch_offset_draw_keys", None)
    if explicit is not None:
        return bool(explicit)
    sample_construction = getattr(getattr(dataset, "data_config", None), "sample_construction", None)
    return (
        getattr(sample_construction, "mode", None) == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT
        and callable(getattr(dataset, "_draw_hierarchical_sample", None))
    )


def _source_view_spread_stride(length: int) -> int:
    if length <= 1:
        return 1
    stride = max(1, int(length) // 10 + 1)
    while math.gcd(stride, int(length)) != 1:
        stride += 1
        if stride >= int(length):
            return 1
    return stride


def _balanced_source_indices_for_dataset(dataset: Dataset[LatentWAMSample]) -> tuple[int, ...] | None:
    build_indices = getattr(dataset, "build_balanced_source_indices", None)
    if not callable(build_indices):
        return None
    indices = tuple(int(index) for index in build_indices())
    if not indices:
        return None
    return indices


def _with_generalist_metadata(
    sample: LatentWAMSample,
    *,
    bucket: GeneralistMixtureBucket,
    split: str,
    source_index: int,
) -> LatentWAMSample:
    metadata = dict(sample.metadata)
    metadata.update(
        {
            GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY: bucket.mode,
            GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY: bool(bucket.drop_text),
            GENERALIST_TRAINING_SOURCE_METADATA_KEY: bucket.source,
            GENERALIST_TRAINING_BUCKET_METADATA_KEY: bucket.name,
            "generalist_training_split": split,
            "generalist_source_index": int(source_index),
        }
    )
    text_context = sample.text_context
    task_text = sample.task_text
    if bucket.drop_text:
        task_text = None
        if sample.negative_text_context is not None:
            text_context = sample.negative_text_context.clone()
        elif text_context is not None:
            text_context = torch.zeros_like(text_context)
    return replace(
        sample,
        task_text=task_text,
        text_context=text_context,
        metadata=metadata,
    )


def _uses_real_target_only_conditional_layout(bucket: GeneralistMixtureBucket) -> bool:
    return bucket.source == REAL_DEMO_SOURCE and bucket.mode in {
        ACTION_CONDITIONED_VIDEO_MODE,
        VIDEO_CONDITIONED_ACTION_MODE,
    }
