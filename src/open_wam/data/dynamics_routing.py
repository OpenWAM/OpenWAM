"""Typed sample-level source and objective routing for dynamics training."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import Dataset, Sampler

from open_wam.configs import (
    DataConfig,
    DynamicsObjective,
    DynamicsRoutingConfig,
    DynamicsSource,
)
from open_wam.contracts import (
    DYNAMICS_ROUTING_BUCKET_METADATA_KEY,
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY,
    DYNAMICS_ROUTING_MODE_METADATA_KEY,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
)

from .distributed_sampling import PaddedEpochOffsetDistributedSampler
from .encoded_dynamics_dataset import (
    EncodedDynamicsLatentDataset,
    EncodedDynamicsResources,
)
from .latent_contracts import LatentWAMSample


@dataclass(frozen=True)
class DynamicsRouteBucket:
    name: str
    source: DynamicsSource
    mode: DynamicsObjective
    weight: float

    @property
    def drop_text(self) -> bool:
        """Whether this objective removes task-text conditioning."""

        return self.mode.is_conditional


@dataclass(frozen=True)
class DynamicsRouteKey:
    """Identity of one concrete source/objective dataset view."""

    source: DynamicsSource
    mode: DynamicsObjective

    @classmethod
    def from_values(
        cls,
        source: DynamicsSource | str,
        mode: DynamicsObjective | str,
    ) -> DynamicsRouteKey:
        return cls(source=DynamicsSource(source), mode=DynamicsObjective(mode))


@dataclass(frozen=True)
class DynamicsDatasetPlan:
    """Exact dataset inputs required by the active dynamics routes."""

    buckets: tuple[DynamicsRouteBucket, ...]

    @property
    def requires_planning(self) -> bool:
        return any(bucket.mode == DynamicsObjective.JOINT for bucket in self.buckets)

    @property
    def requires_real_conditional(self) -> bool:
        return any(
            bucket.mode.is_conditional and bucket.source == DynamicsSource.REAL_DEMO
            for bucket in self.buckets
        )

    @property
    def requires_counterfactual(self) -> bool:
        return any(
            bucket.mode.is_conditional
            and bucket.source == DynamicsSource.COUNTERFACTUAL_DYNAMICS
            for bucket in self.buckets
        )

    @property
    def requires_encoded_dynamics(self) -> bool:
        return self.requires_real_conditional or self.requires_counterfactual

    @property
    def encoded_sources(self) -> tuple[DynamicsSource, ...]:
        return tuple(
            source
            for source, required in (
                (DynamicsSource.REAL_DEMO, self.requires_real_conditional),
                (DynamicsSource.COUNTERFACTUAL_DYNAMICS, self.requires_counterfactual),
            )
            if required
        )


@runtime_checkable
class DynamicsSourceViewProvider(Protocol):
    """Dataset extension contract for source-specific validation views."""

    def has_route(
        self,
        *,
        source: DynamicsSource | str,
        mode: DynamicsObjective | str,
    ) -> bool: ...

    def build_source_view(
        self,
        *,
        source: DynamicsSource | str,
        mode: DynamicsObjective | str,
        bucket_name: str,
        spread_indices: bool = False,
    ) -> Dataset[LatentWAMSample]: ...


class DynamicsRoutingDataset(Dataset[LatentWAMSample]):
    """Route draws across typed source datasets and dynamics objectives."""

    def __init__(
        self,
        *,
        route_datasets: Mapping[DynamicsRouteKey, Dataset[LatentWAMSample]],
        routing_config: DynamicsRoutingConfig,
        split: str,
        fixed_mode: DynamicsObjective | str | None = None,
    ) -> None:
        normalized_routes: dict[DynamicsRouteKey, Dataset[LatentWAMSample]] = {}
        for raw_key, dataset in route_datasets.items():
            if not isinstance(raw_key, DynamicsRouteKey):
                raise TypeError(
                    "Dynamics route datasets must use DynamicsRouteKey keys, "
                    f"got {type(raw_key).__name__}."
                )
            key = DynamicsRouteKey.from_values(raw_key.source, raw_key.mode)
            if key in normalized_routes:
                raise ValueError(
                    f"Dynamics route {key.source.value!r}/{key.mode.value!r} "
                    "was provided more than once."
                )
            normalized_routes[key] = dataset
        self._route_datasets = normalized_routes
        self.routing_config = routing_config
        self.split = str(split)
        self.fixed_mode = (
            None
            if fixed_mode is None
            else DynamicsObjective(fixed_mode)
        )
        self.buckets = _build_route_buckets(
            routing_config,
            fixed_mode=self.fixed_mode,
        )
        active_keys = {
            DynamicsRouteKey(source=bucket.source, mode=bucket.mode)
            for bucket in self.buckets
        }
        for key in active_keys:
            dataset = self._route_datasets.get(key)
            if dataset is None:
                raise ValueError(
                    "Dynamics routing assigns positive weight to route "
                    f"{key.source.value!r}/{key.mode.value!r}, but no dataset "
                    "was provided for it."
                )
            if len(dataset) <= 0:
                raise ValueError(
                    "Dynamics routing assigns positive weight to route "
                    f"{key.source.value!r}/{key.mode.value!r}, but that dataset is empty."
                )
        self._distributed_draw_group_size = 1
        self._distributed_epoch_size = 0
        active_lengths = [len(self._route_datasets[key]) for key in active_keys]
        base_length = max(active_lengths)
        self._length = max(1, int(round(base_length * float(routing_config.length_multiplier))))

    def __len__(self) -> int:
        return self._length

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> Sampler[int]:
        return DynamicsRoutingDistributedSampler(
            self,
            world_size=world_size,
            rank=rank,
        )

    def build_validation_sampler(
        self,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> Sampler[int]:
        """Coordinate validation routes across ranks, including padded tails."""

        return DynamicsRoutingDistributedSampler(
            self,
            world_size=world_size,
            rank=rank,
        )

    def set_distributed_draw_geometry(
        self,
        *,
        world_size: int,
        epoch_size: int | None = None,
    ) -> None:
        group_size = max(1, int(world_size))
        self._distributed_draw_group_size = group_size
        if epoch_size is None:
            epoch_size = int(math.ceil(len(self) / float(group_size))) * group_size
        self._distributed_epoch_size = max(len(self), int(epoch_size))

    def build_source_view(
        self,
        *,
        source: DynamicsSource | str,
        mode: DynamicsObjective | str,
        bucket_name: str,
        spread_indices: bool = False,
    ) -> Dataset[LatentWAMSample]:
        bucket = DynamicsRouteBucket(
            name=str(bucket_name),
            source=DynamicsSource(source),
            mode=DynamicsObjective(mode),
            weight=1.0,
        )
        return DynamicsSourceViewDataset(
            self,
            bucket=bucket,
            spread_indices=spread_indices,
        )

    def has_route(
        self,
        *,
        source: DynamicsSource | str,
        mode: DynamicsObjective | str,
    ) -> bool:
        """Return whether one source/objective dataset view is available."""

        try:
            key = DynamicsRouteKey.from_values(source, mode)
        except ValueError:
            return False
        dataset = self._route_datasets.get(key)
        return dataset is not None and len(dataset) > 0

    def route_dataset(
        self,
        *,
        source: DynamicsSource | str,
        mode: DynamicsObjective | str,
    ) -> Dataset[LatentWAMSample]:
        """Resolve one concrete route or fail with a route-specific error."""

        try:
            key = DynamicsRouteKey.from_values(source, mode)
        except ValueError as exc:
            raise ValueError(f"Unknown dynamics route {source!r}/{mode!r}.") from exc
        dataset = self._route_datasets.get(key)
        if dataset is not None and len(dataset) > 0:
            return dataset
        raise ValueError(
            f"Dynamics route {key.source.value!r}/{key.mode.value!r} is not "
            f"available for split {self.split!r}."
        )

    def __getitem__(self, index: int) -> LatentWAMSample:
        index = int(index)
        group_size = max(1, int(getattr(self, "_distributed_draw_group_size", 1)))
        epoch_size = int(getattr(self, "_distributed_epoch_size", 0) or len(self))
        epoch_index = index % max(1, epoch_size)
        if group_size == 1:
            rng = random.Random(int(self.routing_config.seed) + index * 1_000_003)
            bucket = _sample_bucket(self.buckets, rng)
            source_rng = rng
        else:
            # FSDP requires every rank to enter the same sharded module path in
            # the same order. Coordinate the source/mode bucket per distributed
            # step, then vary the source-row draw by rank for data diversity.
            draw_group = index // group_size
            rank_offset = epoch_index % group_size
            bucket_rng = random.Random(int(self.routing_config.seed) + draw_group * 1_000_003)
            bucket = _sample_bucket(self.buckets, bucket_rng)
            source_rng = random.Random(
                int(self.routing_config.seed)
                + draw_group * 1_000_003
                + (rank_offset + 1) * 9176
            )
        source_dataset = self.route_dataset(
            source=bucket.source,
            mode=bucket.mode,
        )
        sample_index = _draw_source_index(source_dataset, rng=source_rng)
        sample = source_dataset[sample_index]
        return _with_routing_metadata(
            sample,
            bucket=bucket,
            split=self.split,
            source_index=sample_index,
        )


class DynamicsSourceViewDataset(Dataset[LatentWAMSample]):
    """Deterministic source projection that preserves routing transforms."""

    def __init__(
        self,
        routing_dataset: DynamicsRoutingDataset,
        *,
        bucket: DynamicsRouteBucket,
        spread_indices: bool = False,
    ) -> None:
        self.routing_dataset = routing_dataset
        self.bucket = bucket
        self.spread_indices = bool(spread_indices)
        self.source_dataset = routing_dataset.route_dataset(
            source=bucket.source,
            mode=bucket.mode,
        )
        self._spread_source_indices = (
            _balanced_source_indices_for_dataset(self.source_dataset)
            if self.spread_indices
            else None
        )
        self._uses_balanced_source_indices = (
            self._spread_source_indices is not None and len(self._spread_source_indices) > 0
        )
        self._source_spread_stride = _source_view_spread_stride(len(self.source_dataset))

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> LatentWAMSample:
        source_index = int(index)
        if self._uses_balanced_source_indices:
            source_index = int(
                self._spread_source_indices[
                    source_index % len(self._spread_source_indices)
                ]
            )
        elif self.spread_indices and len(self.source_dataset) > 1:
            source_index = (source_index * self._source_spread_stride) % len(self.source_dataset)
        sample = self.source_dataset[source_index]
        sample = _with_routing_metadata(
            sample,
            bucket=self.bucket,
            split=self.routing_dataset.split,
            source_index=source_index,
        )
        if self.spread_indices:
            metadata = dict(sample.metadata)
            metadata["generalist_source_view_index"] = int(index)
            metadata["generalist_source_view_order"] = (
                "balanced" if self._uses_balanced_source_indices else "stride"
            )
            metadata["generalist_source_view_stride"] = (
                1 if self._uses_balanced_source_indices else int(self._source_spread_stride)
            )
            sample = replace(sample, metadata=metadata)
        return sample


class DynamicsRoutingDistributedSampler(PaddedEpochOffsetDistributedSampler):
    """Epoch-offset sampler for coordinated source/objective route draws."""

    def __init__(
        self,
        dataset: DynamicsRoutingDataset,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message="Dynamics routing requires a non-empty dataset.",
        )
        self.dataset.set_distributed_draw_geometry(
            world_size=self.world_size,
            epoch_size=self._total_size,
        )


def build_dynamics_routing_datasets(
    *,
    data_config: DataConfig,
    train_dataset: Dataset[LatentWAMSample] | None = None,
    val_dataset: Dataset[LatentWAMSample] | None = None,
    fixed_mode: DynamicsObjective | str | None = None,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    routing_config = data_config.dynamics_routing
    dataset_plan = resolve_dynamics_dataset_plan(
        data_config,
        fixed_mode=fixed_mode,
    )
    buckets = dataset_plan.buckets
    train_real_dynamics: Dataset[LatentWAMSample] | None = None
    val_real_dynamics: Dataset[LatentWAMSample] | None = None
    train_counterfactual: Dataset[LatentWAMSample] | None = None
    val_counterfactual: Dataset[LatentWAMSample] | None = None
    if dataset_plan.requires_encoded_dynamics:
        if routing_config.train_latent_root is None:
            raise ValueError(
                "A positive conditional dynamics route requires a rollout-local "
                "target-only encoded root at "
                "`data.dynamics_routing.train_latent_root`."
            )
        val_root = routing_config.val_latent_root
        if val_root is None:
            if not routing_config.allow_train_latent_root_for_val:
                raise ValueError(
                    "A positive conditional dynamics route requires "
                    "`data.dynamics_routing.val_latent_root` for validation. Set "
                    "`allow_train_latent_root_for_val: true` only for local debug runs."
                )
            val_root = routing_config.train_latent_root
        train_resources = EncodedDynamicsResources.load(
            data_config,
            routing_config.train_latent_root,
        )
        val_resources = train_resources.with_artifact_root(val_root)
        if dataset_plan.requires_real_conditional:
            train_real_dynamics = EncodedDynamicsLatentDataset(
                data_config,
                train_resources,
                split="train",
                source=DynamicsSource.REAL_DEMO,
            )
            val_real_dynamics = EncodedDynamicsLatentDataset(
                data_config,
                val_resources,
                split="val",
                source=DynamicsSource.REAL_DEMO,
            )
        if dataset_plan.requires_counterfactual:
            train_counterfactual = EncodedDynamicsLatentDataset(
                data_config,
                train_resources,
                split="train",
                source=DynamicsSource.COUNTERFACTUAL_DYNAMICS,
            )
            val_counterfactual = EncodedDynamicsLatentDataset(
                data_config,
                val_resources,
                split="val",
                source=DynamicsSource.COUNTERFACTUAL_DYNAMICS,
            )
    return (
        DynamicsRoutingDataset(
            route_datasets=_build_route_dataset_map(
                buckets=buckets,
                planning_dataset=train_dataset,
                real_conditional_dataset=train_real_dynamics,
                counterfactual_dataset=train_counterfactual,
            ),
            routing_config=routing_config,
            split="train",
            fixed_mode=fixed_mode,
        ),
        DynamicsRoutingDataset(
            route_datasets=_build_route_dataset_map(
                buckets=buckets,
                planning_dataset=val_dataset,
                real_conditional_dataset=val_real_dynamics,
                counterfactual_dataset=val_counterfactual,
            ),
            routing_config=routing_config,
            split="val",
            fixed_mode=fixed_mode,
        ),
    )


def _build_route_dataset_map(
    *,
    buckets: tuple[DynamicsRouteBucket, ...],
    planning_dataset: Dataset[LatentWAMSample] | None,
    real_conditional_dataset: Dataset[LatentWAMSample] | None,
    counterfactual_dataset: Dataset[LatentWAMSample] | None,
) -> dict[DynamicsRouteKey, Dataset[LatentWAMSample]]:
    route_datasets: dict[DynamicsRouteKey, Dataset[LatentWAMSample]] = {}
    for bucket in buckets:
        key = DynamicsRouteKey(source=bucket.source, mode=bucket.mode)
        if bucket.mode == DynamicsObjective.JOINT:
            if bucket.source != DynamicsSource.REAL_DEMO:
                raise ValueError(
                    "Joint planning routes require source=real_demo; encoded "
                    "counterfactual windows are conditional-dynamics data."
                )
            dataset = planning_dataset
        elif bucket.source == DynamicsSource.REAL_DEMO:
            dataset = real_conditional_dataset
        else:
            dataset = counterfactual_dataset
        if dataset is None:
            raise ValueError(
                f"No dataset was built for dynamics route "
                f"{bucket.source.value!r}/{bucket.mode.value!r}."
            )
        route_datasets[key] = dataset
    return route_datasets


def resolve_dynamics_dataset_plan(
    data_config: DataConfig,
    *,
    fixed_mode: DynamicsObjective | str | None = None,
) -> DynamicsDatasetPlan:
    """Resolve only the dataset sources that positive routes can consume."""

    return DynamicsDatasetPlan(
        buckets=_build_route_buckets(
            data_config.dynamics_routing,
            fixed_mode=fixed_mode,
        )
    )


def _build_route_buckets(
    config: DynamicsRoutingConfig,
    *,
    fixed_mode: DynamicsObjective | str | None = None,
) -> tuple[DynamicsRouteBucket, ...]:
    resolved_fixed_mode = (
        None if fixed_mode is None else DynamicsObjective(fixed_mode)
    )
    conflicting_routes = tuple(
        route
        for route in config.active_routes
        if resolved_fixed_mode is not None and route.mode != resolved_fixed_mode
    )
    if conflicting_routes:
        route_labels = ", ".join(route.bucket_name for route in conflicting_routes)
        raise ValueError(
            f"Fixed dynamics mode {resolved_fixed_mode.value!r} accepts only matching "
            f"routes; remove conflicting routes: {route_labels}."
        )

    buckets = tuple(
        DynamicsRouteBucket(
            name=route.bucket_name,
            source=route.source,
            mode=route.mode,
            weight=float(route.weight),
        )
        for route in config.active_routes
    )
    if not buckets:
        suffix = (
            ""
            if resolved_fixed_mode is None
            else f" for fixed mode {resolved_fixed_mode.value!r}"
        )
        raise ValueError(
            "`data.dynamics_routing` must contain at least one positive source weight"
            f"{suffix}."
        )
    return buckets


def _sample_bucket(
    buckets: tuple[DynamicsRouteBucket, ...],
    rng: random.Random,
) -> DynamicsRouteBucket:
    total = sum(bucket.weight for bucket in buckets)
    draw = rng.random() * total
    cursor = 0.0
    for bucket in buckets:
        cursor += bucket.weight
        if draw <= cursor:
            return bucket
    return buckets[-1]


def _draw_source_index(
    dataset: Dataset[LatentWAMSample],
    *,
    rng: random.Random,
) -> int:
    return int(rng.randrange(len(dataset)))


def _source_view_spread_stride(length: int) -> int:
    if length <= 1:
        return 1
    stride = max(1, int(length) // 10 + 1)
    while math.gcd(stride, int(length)) != 1:
        stride += 1
        if stride >= int(length):
            return 1
    return stride


def _balanced_source_indices_for_dataset(
    dataset: Dataset[LatentWAMSample],
) -> tuple[int, ...] | None:
    build_indices = getattr(dataset, "build_balanced_source_indices", None)
    if not callable(build_indices):
        return None
    indices = tuple(int(index) for index in build_indices())
    if not indices:
        return None
    return indices


def _with_routing_metadata(
    sample: LatentWAMSample,
    *,
    bucket: DynamicsRouteBucket,
    split: str,
    source_index: int,
) -> LatentWAMSample:
    metadata = dict(sample.metadata)
    metadata.update(
        {
            DYNAMICS_ROUTING_MODE_METADATA_KEY: bucket.mode.value,
            DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY: bool(bucket.drop_text),
            DYNAMICS_ROUTING_SOURCE_METADATA_KEY: bucket.source.value,
            DYNAMICS_ROUTING_BUCKET_METADATA_KEY: bucket.name,
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
