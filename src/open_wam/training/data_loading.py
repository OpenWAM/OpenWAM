from __future__ import annotations

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from open_wam.configs import BatchAdapterName, ExperimentConfig
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
from open_wam.data.registries import preflight_dataset_artifacts


def build_runtime_dataloaders(config: ExperimentConfig, strategy) -> tuple[DataLoader, DataLoader]:
    if _uses_dynamics_routing(config):
        _validate_dynamics_source_sampling(config)
    if config.trainer.batch_adapter == BatchAdapterName.LATENTS:
        if _uses_dynamics_routing(config):
            if config.data.train_batch_size != 1 or config.data.val_batch_size != 1:
                raise ValueError(
                    "Active `data.dynamics_routing.routes` currently require "
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
