from __future__ import annotations

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from open_wam.configs import (
    BatchAdapterName,
    BatchingMode,
    ExperimentConfig,
)
from open_wam.configs.enums import SampleOrderMode, SampleWeightMode
from open_wam.configs.policy_video_action import resolve_fixed_conditioning_mode
from open_wam.data import (
    build_dynamics_routing_datasets,
    build_train_val_datasets,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    collate_wam_samples,
    preflight_encoded_dynamics_artifact,
    resolve_dataset_loader_spec,
    resolve_dynamics_dataset_plan,
)
from open_wam.data.artifacts import DatasetArtifactStatus
from open_wam.data.latent_batching import LatentBatchCollator, LengthBucketSampler
from open_wam.data.registries import preflight_dataset_artifacts


def build_runtime_dataloaders(config: ExperimentConfig, strategy) -> tuple[DataLoader, DataLoader]:
    if _uses_dynamics_routing(config):
        _validate_dynamics_source_sampling(config)
    if config.trainer.batch_adapter == BatchAdapterName.LATENTS:
        if _uses_dynamics_routing(config):
            if config.data.batching.mode is BatchingMode.STRICT and (
                config.data.train_batch_size != 1 or config.data.val_batch_size != 1
            ):
                raise ValueError(
                    "Active `data.dynamics_routing.routes` with strict batching require "
                    "`data.train_batch_size = data.val_batch_size = 1` because routed sources may have "
                    "different temporal lengths and dynamics-routed runtimes use one objective per segment."
                )
            fixed_mode = resolve_fixed_conditioning_mode(config.policy_variant)
            dataset_plan = resolve_dynamics_dataset_plan(
                config.data,
                fixed_mode=fixed_mode,
            )
            train_dataset = None
            val_dataset = None
            if dataset_plan.requires_planning:
                train_dataset, val_dataset = build_train_val_latent_datasets(
                    config.data
                )
            train_dataset, val_dataset = build_dynamics_routing_datasets(
                data_config=config.data,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                fixed_mode=fixed_mode,
            )
        else:
            train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
        train_loader_spec = resolve_dataset_loader_spec(
            train_dataset,
            split="train",
            world_size=strategy.world_size,
            rank=strategy.rank,
        )
        val_loader_spec = resolve_dataset_loader_spec(
            val_dataset,
            split="val",
            world_size=strategy.world_size,
            rank=strategy.rank,
        )
        train_sampler = train_loader_spec.sampler
        if train_sampler is None and strategy.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                shuffle=True,
                num_replicas=strategy.world_size,
                rank=strategy.rank,
            )
        val_sampler = val_loader_spec.sampler
        if val_sampler is None and strategy.distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                shuffle=False,
                num_replicas=strategy.world_size,
                rank=strategy.rank,
            )
        if config.data.batching.mode is not BatchingMode.STRICT:
            return (
                _build_variable_length_loader(
                    config, train_dataset, train_sampler,
                    batch_size=config.data.train_batch_size,
                    shuffle=train_loader_spec.shuffle,
                    train=True,
                ),
                _build_variable_length_loader(
                    config, val_dataset, val_sampler,
                    batch_size=config.data.val_batch_size,
                    shuffle=val_loader_spec.shuffle,
                    train=False,
                ),
            )
        return (
            DataLoader(
                train_dataset,
                batch_size=config.data.train_batch_size,
                shuffle=train_sampler is None and train_loader_spec.shuffle,
                num_workers=config.data.num_workers,
                sampler=train_sampler,
                collate_fn=collate_latent_wam_samples,
            ),
            DataLoader(
                val_dataset,
                batch_size=config.data.val_batch_size,
                shuffle=val_sampler is None and val_loader_spec.shuffle,
                num_workers=config.data.num_workers,
                sampler=val_sampler,
                collate_fn=collate_latent_wam_samples,
            ),
        )
    train_dataset, val_dataset = build_train_val_datasets(config.data)
    train_loader_spec = resolve_dataset_loader_spec(
        train_dataset,
        split="train",
        world_size=strategy.world_size,
        rank=strategy.rank,
    )
    val_loader_spec = resolve_dataset_loader_spec(
        val_dataset,
        split="val",
        world_size=strategy.world_size,
        rank=strategy.rank,
    )
    train_sampler = train_loader_spec.sampler
    if train_sampler is None and strategy.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=strategy.world_size, rank=strategy.rank)
    val_sampler = val_loader_spec.sampler
    if val_sampler is None and strategy.distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=strategy.world_size, rank=strategy.rank)
    return (
        DataLoader(
            train_dataset,
            batch_size=config.data.train_batch_size,
            shuffle=train_sampler is None and train_loader_spec.shuffle,
            num_workers=config.data.num_workers,
            sampler=train_sampler,
            collate_fn=collate_wam_samples,
        ),
        DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            shuffle=val_loader_spec.shuffle,
            num_workers=config.data.num_workers,
            sampler=val_sampler,
            collate_fn=collate_wam_samples,
        ),
    )




def _build_variable_length_loader(
    config: ExperimentConfig, dataset, sampler, *, batch_size: int, shuffle: bool, train: bool,
) -> DataLoader:
    batching = config.data.batching
    drop_last = bool(train and batching.drop_last_train)
    if batch_size <= 0:
        raise ValueError("Latent batching requires a positive rank-local batch size.")
    if sampler is None:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    if drop_last and len(sampler) < batch_size:
        raise ValueError("Rank-local sample count is smaller than one full training batch.")
    if train and not drop_last and len(sampler) % batch_size:
        raise ValueError(
            "Variable-length training requires fixed rank-local sample counts; "
            "enable data.batching.drop_last_train for an incomplete final batch."
        )
    if batching.mode.groups_by_length:
        if not getattr(sampler, "supports_reordering", True):
            raise ValueError(
                "Bucket batching cannot reorder this source sampler; use padded or packed."
            )
        length_hint = getattr(dataset, "batching_length_hint", None)
        if not callable(length_hint):
            raise ValueError("Bucket batching requires a dataset batching_length_hint(index).")
        sampler = LengthBucketSampler(
            sampler,
            length_for_index=length_hint,
            batch_size=batch_size,
            pool_size=batching.bucket_pool_size,
            drop_last=drop_last,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=LatentBatchCollator(batching),
        drop_last=drop_last,
    )


def preflight_runtime_dataset_artifacts(
    config: ExperimentConfig,
) -> tuple[DatasetArtifactStatus, ...]:
    """Preflight exactly the dataset sources the configured routes can consume."""

    if not _uses_dynamics_routing(config):
        return preflight_dataset_artifacts(config.data)

    fixed_mode = resolve_fixed_conditioning_mode(config.policy_variant)
    dataset_plan = resolve_dynamics_dataset_plan(
        config.data,
        fixed_mode=fixed_mode,
    )
    statuses: tuple[DatasetArtifactStatus, ...] = ()
    if dataset_plan.requires_planning:
        statuses += preflight_dataset_artifacts(config.data)
    if not dataset_plan.requires_encoded_dynamics:
        return statuses

    routing = config.data.dynamics_routing
    val_root = routing.val_latent_root
    if val_root is None and routing.allow_train_latent_root_for_val:
        val_root = routing.train_latent_root
    roots = (
        (routing.train_latent_root, "data.dynamics_routing.train_latent_root"),
        (val_root, "data.dynamics_routing.val_latent_root"),
    )
    seen_roots: set[str] = set()
    for root, config_path in roots:
        key = "" if root is None else str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        statuses += preflight_encoded_dynamics_artifact(
            root,
            sources=dataset_plan.encoded_sources,
            config_path=config_path,
        )
    return statuses


def _uses_dynamics_routing(config: ExperimentConfig) -> bool:
    return bool(config.data.dynamics_routing.active_routes)


def _validate_dynamics_source_sampling(config: ExperimentConfig) -> None:
    if config.trainer.batch_adapter != BatchAdapterName.LATENTS:
        raise ValueError(
            "Active `data.dynamics_routing.routes` require "
            "`trainer.batch_adapter=latents` because the dynamics source router wraps latent datasets."
        )
    sample_construction = config.data.sample_construction
    if sample_construction.sample_order_mode != SampleOrderMode.REPLACEMENT:
        raise ValueError(
            "`data.sample_construction.sample_order_mode` must be `replacement` "
            "with active `data.dynamics_routing.routes` because route weights "
            "define replacement probabilities."
        )
    if sample_construction.sample_weight_mode != SampleWeightMode.UNIFORM:
        raise ValueError(
            "`data.sample_construction.sample_weight_mode` must be `uniform` with "
            "active `data.dynamics_routing.routes` because the dynamics router "
            "owns source sampling and only preserves parity for uniform replacement draws."
        )
