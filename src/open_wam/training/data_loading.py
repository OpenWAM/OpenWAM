from __future__ import annotations

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from open_wam.configs import BatchAdapterName, ExperimentConfig
from open_wam.configs.enums import GeneralistTrainingParadigm, SampleWeightMode
from open_wam.data import (
    build_generalist_dynamics_mixture_datasets,
    build_train_val_datasets,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    collate_wam_samples,
    resolve_dataset_loader_spec,
)


def build_runtime_dataloaders(config: ExperimentConfig, strategy) -> tuple[DataLoader, DataLoader]:
    if _uses_mixed_dynamics_paradigm(config):
        _validate_mixed_dynamics_source_sampling(config)
    if config.trainer.batch_adapter == BatchAdapterName.LATENTS:
        train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
        if _uses_mixed_dynamics_paradigm(config):
            if config.data.train_batch_size != 1 or config.data.val_batch_size != 1:
                raise ValueError(
                    "`generalist_training_paradigm = mixed_dynamics` currently requires "
                    "`data.train_batch_size = data.val_batch_size = 1` because mixed samples may have "
                    "different temporal lengths and GJD runtimes use one forced mode per segment."
                )
            train_dataset, val_dataset = build_generalist_dynamics_mixture_datasets(
                data_config=config.data,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
        train_loader_spec = resolve_dataset_loader_spec(
            train_dataset,
            split="train",
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
        val_sampler = (
            DistributedSampler(val_dataset, shuffle=False, num_replicas=strategy.world_size, rank=strategy.rank)
            if strategy.distributed
            else None
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
                shuffle=False,
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


def _uses_mixed_dynamics_paradigm(config: ExperimentConfig) -> bool:
    paradigm = getattr(config.policy_variant, "generalist_training_paradigm", None)
    return paradigm == GeneralistTrainingParadigm.MIXED_DYNAMICS


def _validate_mixed_dynamics_source_sampling(config: ExperimentConfig) -> None:
    if config.trainer.batch_adapter != BatchAdapterName.LATENTS:
        raise ValueError(
            "`policy_variant.generalist_training_paradigm=mixed_dynamics` requires "
            "`trainer.batch_adapter=latents` because the mixed-dynamics source mixture wraps latent datasets."
        )
    sample_construction = config.data.sample_construction
    if sample_construction.sample_weight_mode != SampleWeightMode.UNIFORM:
        raise ValueError(
            "`data.sample_construction.sample_weight_mode` must be `uniform` with "
            "`policy_variant.generalist_training_paradigm=mixed_dynamics` because the mixed-dynamics "
            "wrapper owns source sampling and only preserves parity for uniform replacement draws."
        )
