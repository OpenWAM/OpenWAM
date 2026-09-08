"""Bounded, checkpoint-free comparison using the unmodified training microstep.

Launch once per mode with four processes. Reports are benchmark artifacts only;
the input configuration and checkpoint are read-only. Compilation and filesystem
cache effects are reported, not silently interpreted as steady-state speedups.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def stable_seed(*values):
    digest = hashlib.sha256(":".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def plan_positions(indices, lengths, *, warmup_samples, mode, pool_size):
    """Keep warmup and measured multisets separate, including repeated draws."""
    positions = list(range(len(indices)))
    if mode != "bucket":
        return positions
    result = []
    for partition in (positions[:warmup_samples], positions[warmup_samples:]):
        for start in range(0, len(partition), pool_size):
            pool = partition[start : start + pool_size]
            result.extend(sorted(pool, key=lambda position: lengths[position]))
    return result


class ReproducibleDataset:
    """Make each finite draw independent of workers, collation, and reordering."""

    def __init__(self, dataset, indices, *, seed, rank, geometry, chunk, window):
        self.dataset = dataset
        self.indices = indices
        self.seed = seed
        self.rank = rank
        self.geometry = geometry
        self.chunk = chunk
        self.window = window

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        import numpy as np
        import torch

        index = self.indices[position]
        seed = stable_seed(self.seed, self.rank, position, index)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            # CPU RNG only: worker processes must never initialize CUDA.
            with torch.random.fork_rng(devices=[]):
                random.seed(seed)
                np.random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                sample = self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        metadata = dict(sample.metadata)
        metadata["benchmark_draw_position"] = int(position)
        metadata["benchmark_source_index"] = int(index)
        metadata["benchmark_materialization_seed"] = seed
        metadata["benchmark_original_geometry"] = {
            key: metadata.get(key)
            for key in ("sampled_chunk_size", "sampled_window_size", "history_frames")
        }
        if self.geometry == "fixed":
            length = int(sample.video_latents.shape[1])
            if length < self.chunk:
                raise ValueError("Fixed geometry requires every sampled sequence >= chunk size.")
            if "loss_frame_start" in metadata:
                raise ValueError("Fixed geometry benchmark requires the full-segment loss contract.")
            metadata["sampled_chunk_size"] = self.chunk
            metadata["sampled_window_size"] = self.window
            metadata["history_frames"] = max(
                1, min(math.ceil(self.window / 2) * self.chunk, length - self.chunk)
            )
        return dataclasses.replace(sample, metadata=metadata)


class MemoryLogSink:
    def __init__(self):
        self.metrics = {}

    def log_metrics(self, *, step, phase, metrics):
        self.metrics[int(step)] = dict(metrics)

    def log_event(self, *, name, payload):
        pass

    def close(self):
        pass


def local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def benchmark_batch_layout(mode, max_tokens=None):
    """Logical loader size and accumulation; token-limited GPU sizes are dynamic."""
    if max_tokens is not None:
        if mode != "packed" or isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("--max-tokens requires packed mode and a positive integer.")
        return 6, 1
    return (1, 6) if mode == "strict" else (2, 3)


def describe_batch(batch, cfg, *, token_costs=None):
    lengths = tuple(batch.sequence_lengths) or (int(batch.video_latents.shape[2]),)
    height, width = map(int, batch.video_latents.shape[-2:])
    patch = (cfg.backbone.patch_size_t, cfg.backbone.patch_size_h, cfg.backbone.patch_size_w)
    if patch[0] != 1 or height % patch[1] or width % patch[2]:
        raise ValueError("Throughput counters require the configured frame-aligned patch grid.")
    tokens_per_frame = (height // patch[1]) * (width // patch[2])
    samples = []
    for length, metadata in zip(lengths, batch.metadata, strict=True):
        valid = int(metadata.get("segment_valid_latent_frames", length))
        samples.append({
            "draw_position": metadata["benchmark_draw_position"],
            "source_index": metadata["benchmark_source_index"],
            "episode_index": metadata.get("episode_index"),
            "latent_frames": int(length),
            "valid_latent_frames": valid,
            "valid_video_patch_tokens": valid * tokens_per_frame,
            "sampled_chunk_size": metadata.get("sampled_chunk_size"),
            "sampled_window_size": metadata.get("sampled_window_size"),
            "history_frames": metadata.get("history_frames"),
            "sample_start_frame": metadata.get("sample_start_frame"),
            "sample_end_frame": metadata.get("sample_end_frame"),
        })
    report = {
        "samples": samples,
        "transport_latent_capacity": int(batch.video_latents.shape[0] * batch.video_latents.shape[2]),
        "materialized_latent_frames": sum(lengths),
        "valid_latent_frames": sum(sample["valid_latent_frames"] for sample in samples),
        "valid_video_patch_tokens": sum(sample["valid_video_patch_tokens"] for sample in samples),
    }
    if token_costs is not None:
        costs = tuple(int(value) for value in token_costs)
        if len(costs) != len(samples) or any(value <= 0 for value in costs):
            raise ValueError("Expected one positive transformer token cost per original sample.")
        report["sample_transformer_tokens"] = costs
        report["transformer_tokens"] = sum(costs)
    return report


def describe_physical_batch_from_cost_cache(batch, cfg, costs_by_draw=None):
    """Use CPU-admitted scalar costs with a physical batch already on device."""
    costs = None if costs_by_draw is None else tuple(
        costs_by_draw[int(metadata["benchmark_draw_position"])] for metadata in batch.metadata
    )
    return describe_batch(batch, cfg, token_costs=costs)


def install_probes(runtime, *, describe_physical_batch=None):
    """Tiny tensor-only probes; their symmetric overhead stays inside timing."""
    import torch

    probes = []
    for key in ("action_expert", "video_expert"):
        chosen = [
            (name, parameter) for name, parameter in runtime.model.named_parameters()
            if parameter.requires_grad and key in name and local_tensor(parameter).numel()
        ]
        probes.extend(chosen[:2])
    if not probes:
        probes = [(name, parameter) for name, parameter in runtime.model.named_parameters()
                  if parameter.requires_grad and local_tensor(parameter).numel()][:4]
    if not probes:
        raise RuntimeError("No trainable local parameter shards available for update proof.")
    pending = {"losses": [], "updates": [], "physical_batches": []}
    original_forward = runtime.step_executor.forward_train
    original_step = runtime.optimizer.step

    def forward(batch):
        result = original_forward(batch)
        pending["losses"].append(result.loss.detach())
        if describe_physical_batch is not None:
            # The callback records only Python metadata, never live GPU tensors.
            pending["physical_batches"].append(describe_physical_batch(batch))
        return result

    def step(*args, **kwargs):
        snapshots = []
        gradient_records = []
        for name, parameter in probes:
            before = local_tensor(parameter).detach().reshape(-1)[:4096].clone()
            snapshots.append((name, parameter, before))
            if parameter.grad is not None:
                gradient = local_tensor(parameter.grad).detach().reshape(-1)[:4096]
                if gradient.numel():
                    gradient_records.append((name, torch.isfinite(gradient).all(), gradient.abs().amax()))
        result = original_step(*args, **kwargs)
        deltas = [(name, (local_tensor(parameter).detach().reshape(-1)[:before.numel()] - before).abs().amax())
                  for name, parameter, before in snapshots]
        pending["updates"].append((gradient_records, deltas))
        return result

    runtime.step_executor.forward_train = forward
    runtime.optimizer.step = step
    return pending, [name for name, _ in probes]


def read_probes(pending):
    losses = [float(value.item()) for value in pending["losses"]]
    updates = []
    for gradients, deltas in pending["updates"]:
        updates.append({
            "gradient_probes": [{"parameter": name, "finite": bool(finite.item()), "abs_max": float(maximum.item())}
                                for name, finite, maximum in gradients],
            "parameter_delta_probes": [{"parameter": name, "abs_max": float(delta.item())}
                                       for name, delta in deltas],
        })
    pending["losses"].clear()
    pending["updates"].clear()
    physical_batches = list(pending["physical_batches"])
    pending["physical_batches"].clear()
    return {"microbatch_losses": losses, "updates": updates, "physical_batches": physical_batches}


def rank_update_admission(rank_row):
    """Require positive proof on this rank/update, not on a peer or earlier step."""
    proof = rank_row["proof"]
    losses, updates = proof["microbatch_losses"], proof["updates"]
    physical_batches = rank_row.get("physical_batches", rank_row["batches"])
    physical_samples = [sample for batch in physical_batches for sample in batch["samples"]]
    logical_samples = [sample for batch in rank_row["batches"] for sample in batch["samples"]]
    budget = rank_row.get("configured_max_tokens")
    gradients = [probe for update in updates for probe in update["gradient_probes"]]
    deltas = [probe for update in updates for probe in update["parameter_delta_probes"]]
    checks = {
        "complete_microbatch_losses": bool(losses) and len(losses) == len(physical_batches),
        "original_samples_preserved": len(physical_samples) == len(logical_samples),
        "original_sample_order_preserved": [sample.get("draw_position") for sample in physical_samples]
        == [sample.get("draw_position") for sample in logical_samples],
        "physical_batches_within_token_budget": budget is None or bool(physical_batches) and all(
            isinstance(batch.get("transformer_tokens"), int)
            and 0 < batch["transformer_tokens"] <= budget for batch in physical_batches
        ),
        "exactly_one_optimizer_update": len(updates) == 1,
        "finite_losses": bool(losses) and all(math.isfinite(loss) for loss in losses),
        "finite_gradient_probes": bool(gradients) and all(
            probe["finite"] and math.isfinite(probe["abs_max"]) for probe in gradients
        ),
        "nonzero_gradient_probe": any(probe["abs_max"] > 0 for probe in gradients),
        "finite_parameter_update_probes": bool(deltas) and all(
            math.isfinite(probe["abs_max"]) for probe in deltas
        ),
        "nonzero_parameter_update_probe": any(probe["abs_max"] > 0 for probe in deltas),
    }
    return {"checks": checks, "passed": all(checks.values())}


def summarize(rows, *, world_size):
    measured = [row for row in rows if not row["warmup"]]
    durations = [max(rank["wall_seconds"] for rank in row["ranks"]) for row in measured]
    total_seconds = sum(durations)
    sample_count = sum(len(batch["samples"]) for row in measured for rank in row["ranks"] for batch in rank["batches"])
    valid_frames = sum(batch["valid_latent_frames"] for row in measured for rank in row["ranks"] for batch in rank["batches"])
    valid_tokens = sum(batch["valid_video_patch_tokens"] for row in measured for rank in row["ranks"] for batch in rank["batches"])
    capacity = sum(batch["transport_latent_capacity"] for row in measured for rank in row["ranks"]
                   for batch in rank.get("physical_batches", rank["batches"]))
    materialized = sum(batch["materialized_latent_frames"] for row in measured for rank in row["ranks"] for batch in rank["batches"])
    all_rank_rows = [rank for row in measured for rank in row["ranks"]]
    losses = [loss for rank in all_rank_rows for loss in rank["proof"]["microbatch_losses"]]
    gradient_probes = [probe for rank in all_rank_rows for update in rank["proof"]["updates"] for probe in update["gradient_probes"]]
    delta_probes = [probe for rank in all_rank_rows for update in rank["proof"]["updates"] for probe in update["parameter_delta_probes"]]
    rank_update_checks = []
    complete_rank_sets = True
    aligned_physical_microsteps = True
    for row in measured:
        rank_ids = [rank["rank"] for rank in row["ranks"]]
        complete_rank_sets = complete_rank_sets and sorted(rank_ids) == list(range(world_size))
        aligned_physical_microsteps = aligned_physical_microsteps and len({
            len(rank.get("physical_batches", rank["batches"])) for rank in row["ranks"]
        }) == 1
        for rank in row["ranks"]:
            rank_update_checks.append({
                "optimizer_step": row["optimizer_step"], "rank": rank["rank"],
                **rank_update_admission(rank),
            })
    all_rank_updates_passed = bool(rank_update_checks) and complete_rank_sets and aligned_physical_microsteps and all(
        check["passed"] for check in rank_update_checks
    )
    physical_batches = [batch for rank in all_rank_rows for batch in rank.get("physical_batches", rank["batches"])]
    physical_token_counts = [batch["transformer_tokens"] for batch in physical_batches if "transformer_tokens" in batch]
    return {
        "world_size": world_size,
        "measured_updates": len(measured),
        "synchronized_update_seconds": durations,
        "mean_update_seconds": statistics.mean(durations),
        "median_update_seconds": statistics.median(durations),
        "min_update_seconds": min(durations),
        "max_update_seconds": max(durations),
        "measured_seconds": total_seconds,
        "samples": sample_count,
        "samples_per_second": sample_count / total_seconds,
        "valid_latent_frames": valid_frames,
        "valid_latent_frames_per_second": valid_frames / total_seconds,
        "valid_video_patch_tokens": valid_tokens,
        "valid_video_patch_tokens_per_second": valid_tokens / total_seconds,
        "transport_padding_fraction": 1.0 - materialized / capacity,
        "logical_batch_sizes": [len(batch["samples"]) for rank in all_rank_rows for batch in rank["batches"]],
        "physical_microbatch_sizes": [len(batch["samples"]) for batch in physical_batches],
        "physical_microbatches_per_rank_update": [len(rank.get("physical_batches", rank["batches"])) for rank in all_rank_rows],
        "physical_max_tokens": max(physical_token_counts) if physical_token_counts else None,
        "configured_max_tokens": sorted({rank["configured_max_tokens"] for rank in all_rank_rows if rank.get("configured_max_tokens") is not None}),
        "max_cuda_peak_allocated_bytes": max(rank["cuda_peak_allocated_bytes"] for rank in all_rank_rows),
        "max_cuda_peak_reserved_bytes": max(rank["cuda_peak_reserved_bytes"] for rank in all_rank_rows),
        "mean_rank_max_data_wait_seconds": statistics.mean(max(rank["data_wait_seconds"] for rank in row["ranks"]) for row in measured),
        "finite_loss": bool(losses) and all(math.isfinite(loss) for loss in losses),
        "finite_gradient_probes": bool(gradient_probes) and all(probe["finite"] for probe in gradient_probes),
        "finite_parameter_update_probes": bool(delta_probes) and all(math.isfinite(probe["abs_max"]) for probe in delta_probes),
        "nonzero_gradient_probe": any(probe["abs_max"] > 0 for probe in gradient_probes),
        "nonzero_parameter_update_probe": any(probe["abs_max"] > 0 for probe in delta_probes),
        "complete_measured_rank_sets": complete_rank_sets,
        "aligned_physical_microstep_counts": aligned_physical_microsteps,
        "rank_update_admission": rank_update_checks,
        "all_rank_updates_passed": all_rank_updates_passed,
        "passed": all_rank_updates_passed,
        "notes": [
            "Duration is the maximum synchronized per-rank wall time per complete optimizer update.",
            "Data wait is time inside next(loader); worker prefetch may overlap GPU work.",
            "Tiny gradient/update probes are included in timings; JSON I/O and aggregation are excluded.",
            "Video patch tokens count valid data once, not duplicated clean/noisy streams or condition prefixes.",
            "Packing still uses padded transport tensors; transport padding is not transformer padding.",
            "Transport padding statistics describe actual GPU microbatches; logical loader batches may be split before device transfer.",
            "Token-budget mode uses logical loader size 6 and accumulation 1, not a fixed physical GPU batch of 6.",
            "Unseen shapes may compile during measured updates; this is not automatically a steady-state estimate.",
            "Gradients are checked by runtime global norm when configured plus bounded shard probes, not a second complete gradient scan.",
            "Admission requires finite loss and finite/nonzero gradient and parameter-update evidence on every measured rank/update; an active peer cannot mask a stalled rank.",
            "Diffusion draws need not match bitwise between microbatch arrangements; this is a speed benchmark, not numerical parity proof.",
        ],
    }


def run(args):
    source_root = Path(args.source_root).resolve()
    if not (source_root / "src/open_wam/__init__.py").is_file():
        raise FileNotFoundError("The requested source root does not contain the expected package.")
    sys.path.insert(0, str(source_root / "src"))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils.data import DataLoader
    from open_wam.configs import BatchingMode, load_experiment_config, resolve_experiment_config, validate_experiment_config_runtime_contract
    from open_wam.configs.enums import serialize_enum_values
    from open_wam.data.latent_batching import LatentBatchCollator, LengthBucketSampler
    from open_wam.models.policy_variants.dual_expert.token_cost import dual_expert_token_costs
    from open_wam.runtime.checkpoints import normalize_checkpoint_state_dict
    from open_wam.training.checkpoints import CheckpointManager
    import open_wam.training.runtime as runtime_module

    runtime_path = Path(runtime_module.__file__).resolve()
    if not runtime_path.is_relative_to(source_root):
        raise RuntimeError("The imported runtime does not belong to the requested source snapshot.")

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world_size != 4 or not torch.cuda.is_available():
        raise ValueError("This comparison requires exactly four CUDA workers.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"rank{rank}.json").exists():
        raise FileExistsError("Use a fresh output directory for each benchmark invocation.")
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["WANDB_MODE"] = "disabled"
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = load_experiment_config(config_path)
    if cfg.data.sample_construction.segment_max_frames != args.expected_sequence_cap:
        raise ValueError("Refusing to change or silently shorten the baseline sequence cap.")
    mode = BatchingMode(args.mode)
    logical_batch_size, accumulation = benchmark_batch_layout(args.mode, args.max_tokens)
    original_training = cfg.training
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, train_batch_size=logical_batch_size,
                                 batching=dataclasses.replace(cfg.data.batching, mode=mode,
                                                             bucket_pool_size=args.bucket_pool_size,
                                                             max_tokens=args.max_tokens)),
        training=dataclasses.replace(cfg.training, gradient_accumulation_steps=accumulation),
        trainer=dataclasses.replace(
            cfg.trainer, devices=4, enable_checkpointing=False, save_interval=None,
            export_runtime_backbone=False, initialize_weights_from=str(checkpoint), resume_from=None,
            default_root_dir=str(output / "runtime"), checkpoint_dir=str(output / "checkpoints_disabled"),
            run_name=f"latent_batch_benchmark_{args.mode}", limit_val_batches=0,
            validation_interval=None, enable_jsonl_logging=False, enable_wandb=False,
            wandb_project=None, wandb_entity=None, wandb_mode="disabled", log_every_n_steps=1,
        ),
    )
    cfg = validate_experiment_config_runtime_contract(resolve_experiment_config(cfg))
    if dataclasses.replace(cfg.training, gradient_accumulation_steps=original_training.gradient_accumulation_steps) != original_training:
        raise ValueError("Only gradient accumulation may differ in the optimizer/training configuration.")
    program = cfg.policy_variant.program.value
    expected_history = "full" if program == "joint" else "video_only"
    if program not in ("joint", "video_then_action") or cfg.policy_variant.history_stream_visibility.value != expected_history:
        raise ValueError("Unexpected task program or historical-stream visibility.")

    class CheckedCheckpointManager(CheckpointManager):
        def initialize_weights(self, *, path, model, map_location="cpu"):
            with FakeTensorMode():
                payload = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
            state = normalize_checkpoint_state_dict(payload)
            expected = model.state_dict()
            missing, extra = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
            mismatch = {name: [list(expected[name].shape), list(state[name].shape)]
                        for name in set(expected) & set(state) if expected[name].shape != state[name].shape}
            if missing or extra or mismatch:
                raise ValueError({"missing": missing, "unexpected": extra, "shape_mismatch": mismatch})
            write_json(output / f"checkpoint_rank{rank}.json", {"checkpoint": str(path), "expected_keys": len(expected), "checkpoint_keys": len(state), "strict_schema_match": True})
            del expected, state, payload
            return super().initialize_weights(path=path, model=model, map_location=map_location)

        def save(self, *unused_args, **unused_kwargs):
            raise RuntimeError("Checkpoint writes are forbidden in the benchmark harness.")

    runtime_module.CheckpointManager = CheckedCheckpointManager
    initialization_started = time.perf_counter()
    print(json.dumps({"stage": "initializing", "mode": args.mode, "rank": rank, "program": program}), flush=True)
    runtime = runtime_module.TrainingRuntime.from_config(cfg)
    runtime.log_sink.close()
    runtime.log_sink = MemoryLogSink()
    sampler = runtime.train_loader.sampler
    if isinstance(sampler, LengthBucketSampler):
        sampler = sampler.sampler
    epoch_setter = getattr(sampler, "set_epoch", None)
    if callable(epoch_setter):
        epoch_setter(0)
    sample_count = (args.warmup_updates + args.measure_updates) * 6
    indices = list(itertools.islice(iter(sampler), sample_count))
    if len(indices) != sample_count:
        raise ValueError("Underlying rank-local sampler is too short for the requested bounded comparison.")
    dataset = runtime.train_loader.dataset
    length_hint = getattr(dataset, "batching_length_hint", None)
    if not callable(length_hint):
        raise ValueError("Expected local uniform-segment dataset with metadata length hints.")
    lengths = [int(length_hint(index)) for index in indices]
    positions = plan_positions(indices, lengths, warmup_samples=args.warmup_updates * 6,
                               mode=args.mode, pool_size=args.bucket_pool_size)
    reproducible = ReproducibleDataset(dataset, indices, seed=args.seed, rank=rank,
                                      geometry=args.geometry_policy, chunk=args.chunk_size, window=args.window_size)
    runtime.train_loader = DataLoader(
        reproducible, batch_size=logical_batch_size, sampler=positions,
        num_workers=cfg.data.num_workers, collate_fn=LatentBatchCollator(cfg.data.batching),
        drop_last=True, generator=torch.Generator().manual_seed(seed),
    )
    plan = {"source_indices": indices, "length_hints": lengths, "execution_draw_positions": positions,
            "warmup_samples_per_rank": args.warmup_updates * 6,
            "measured_multiset_sha256": hashlib.sha256(json.dumps(sorted(indices[args.warmup_updates * 6:])).encode()).hexdigest()}
    manifest = {
        "arguments": vars(args), "rank": rank, "world_size": world_size,
        "effective_global_batch_size": logical_batch_size * accumulation * world_size,
        "logical_loader_batch_size": logical_batch_size,
        "logical_gradient_accumulation_steps": accumulation,
        "configured_max_tokens": args.max_tokens,
        "physical_gpu_batch_size": "dynamic; recorded per forward" if args.max_tokens is not None else logical_batch_size,
        "config_path": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runtime_source": str(runtime_path), "runtime_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns},
        "program": program, "history_stream_visibility": cfg.policy_variant.history_stream_visibility.value,
        "sequence_contract": cfg.policy_variant.sequence_contract.value,
        "initialization_seconds": time.perf_counter() - initialization_started,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(runtime.strategy.device), "plan": plan,
        "effective_config": serialize_enum_values(dataclasses.asdict(cfg)),
        "geometry_note": "Controlled fixed C/W benchmark; original sampled geometry is retained in metadata." if args.geometry_policy == "fixed" else "Production geometry: each sample retains its own chunk/window and history metadata across batching modes.",
        "scheduler_note": "Original num_steps and optimizer/scheduler hyperparameters retained; fresh optimizer initialized from identical model weights.",
    }
    write_json(output / f"manifest_rank{rank}.json", manifest)
    token_cost_by_draw = {}

    def describe_physical_batch(batch):
        # Admission validates CPU batches. Reuse those scalar costs for GPU
        # forward reports instead of copying tensors back or bypassing guards.
        return describe_physical_batch_from_cost_cache(
            batch, cfg, token_cost_by_draw if mode is BatchingMode.PACKED else None
        )

    pending, probe_names = install_probes(runtime, describe_physical_batch=describe_physical_batch)
    runtime.strategy.zero_grad(runtime.optimizer)
    iterator = iter(runtime.train_loader)
    local_rows, aggregate_rows = [], []
    for update_index in range(args.warmup_updates + args.measure_updates):
        torch.cuda.synchronize(runtime.strategy.device)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(runtime.strategy.device)
        start = time.perf_counter()
        data_wait, batches, token_plans = 0.0, [], []
        for _ in range(accumulation):
            data_start = time.perf_counter()
            batch = next(iterator)
            data_wait += time.perf_counter() - data_start
            costs = dual_expert_token_costs(config=cfg, batch=batch) if mode is BatchingMode.PACKED else None
            if costs is not None:
                token_cost_by_draw.update(
                    (int(metadata["benchmark_draw_position"]), int(cost))
                    for metadata, cost in zip(batch.metadata, costs, strict=True)
                )
            batches.append(describe_batch(batch, cfg, token_costs=costs))
            runtime._train_micro_step(batch)
            if args.max_tokens is not None:
                token_plans.append(getattr(runtime, "last_token_batch_plan", None))
        torch.cuda.synchronize(runtime.strategy.device)
        elapsed = time.perf_counter() - start
        proof = read_probes(pending)
        physical_batches = proof.pop("physical_batches")
        row = {
            "rank": rank, "optimizer_step": int(runtime.train_state.optimizer_step),
            "warmup": update_index < args.warmup_updates, "wall_seconds": elapsed,
            "data_wait_seconds": data_wait,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.strategy.device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(runtime.strategy.device),
            "batches": batches, "physical_batches": physical_batches, "proof": proof,
            "configured_max_tokens": args.max_tokens, "token_batch_plans": token_plans,
            "runtime_metrics": runtime.log_sink.metrics.get(update_index + 1, {}),
        }
        if row["optimizer_step"] != update_index + 1:
            raise RuntimeError("Harness did not execute exactly one complete optimizer update.")
        local_rows.append(row)
        write_json(output / f"rank{rank}.json", {"manifest": manifest, "probe_parameters": probe_names, "updates": local_rows, "completed": False})
        gathered = [None] * world_size
        dist.all_gather_object(gathered, row)
        if rank == 0:
            aggregate_rows.append({"optimizer_step": update_index + 1, "warmup": row["warmup"], "ranks": gathered})
            write_json(output / "progress.json", {"mode": args.mode, "updates": aggregate_rows})
            print(json.dumps({"stage": "optimizer_update", "mode": args.mode,
                              "optimizer_step": update_index + 1, "warmup": row["warmup"],
                              "rank_max_seconds": max(item["wall_seconds"] for item in gathered),
                              "rank_min_seconds": min(item["wall_seconds"] for item in gathered),
                              "logical_batches_per_rank": len(batches),
                              "physical_microbatches_per_rank": [len(item["physical_batches"]) for item in gathered],
                              "physical_gpu_batch_sizes_by_rank": [[len(batch["samples"]) for batch in item["physical_batches"]] for item in gathered],
                              "max_peak_allocated_gib": max(item["cuda_peak_allocated_bytes"] for item in gathered) / 2**30}), flush=True)
    write_json(output / f"rank{rank}.json", {"manifest": manifest, "probe_parameters": probe_names, "updates": local_rows, "completed": True})
    if rank == 0:
        summary = summarize(aggregate_rows, world_size=world_size)
        summary["mode"] = args.mode
        summary["candidate"] = "packed_token_budget" if args.max_tokens is not None else args.mode
        write_json(output / "summary.json", summary)
        print(json.dumps({"stage": "finished", **summary}), flush=True)
        if not summary["passed"]:
            raise RuntimeError("Finite-loss/gradient/parameter-update proof failed.")
    dist.barrier()
    dist.destroy_process_group()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root")
    result.add_argument("--config")
    result.add_argument("--checkpoint")
    result.add_argument("--output-dir")
    result.add_argument("--mode", choices=("strict", "bucket", "padded", "packed"))
    result.add_argument("--max-tokens", type=int, help="Packed-mode physical token ceiling; uses logical loader batch 6 and accumulation 1.")
    result.add_argument("--warmup-updates", type=int, default=2)
    result.add_argument("--measure-updates", type=int, default=5)
    result.add_argument("--seed", type=int, default=20260907)
    result.add_argument("--bucket-pool-size", type=int, default=128)
    result.add_argument("--expected-sequence-cap", type=int, default=1000)
    result.add_argument("--geometry-policy", choices=("fixed", "sampled"), default="fixed")
    result.add_argument("--chunk-size", type=int, default=4)
    result.add_argument("--window-size", type=int, default=64)
    result.add_argument("--self-test", action="store_true")
    return result


def self_test():
    indices = [3, 3, 9, 1, 8, 2, 2, 5, 6, 0, 7, 4]
    lengths = [100, 100, 90, 300, 80, 200, 200, 50, 60, 20, 70, 40]
    order = plan_positions(indices, lengths, warmup_samples=6, mode="bucket", pool_size=128)
    assert sorted(order[:6]) == list(range(6))
    assert sorted(order[6:]) == list(range(6, 12))
    assert [lengths[i] for i in order[:6]] == sorted(lengths[:6])
    assert order.index(0) < order.index(1)
    for mode in ("strict", "padded", "packed"):
        assert plan_positions(indices, lengths, warmup_samples=6, mode=mode, pool_size=128) == list(range(12))
    assert stable_seed(1, 2, 3) == stable_seed(1, 2, 3) != stable_seed(1, 2, 4)
    print("Benchmark plan/self-test passed; GPU execution is not covered by this test.")


if __name__ == "__main__":
    args = parser().parse_args()
    if args.self_test:
        self_test()
        raise SystemExit(0)
    if not all((args.source_root, args.config, args.checkpoint, args.output_dir, args.mode)):
        parser().error("source-root, config, checkpoint, output-dir, and mode are required")
    if args.warmup_updates < 1 or args.measure_updates < 1 or args.bucket_pool_size < 2 or args.bucket_pool_size % 2 or args.chunk_size < 1 or args.window_size < 1:
        parser().error("Require positive warmup/measure/geometry and an even bucket-pool-size >= 2")
    try:
        benchmark_batch_layout(args.mode, args.max_tokens)
    except ValueError as error:
        parser().error(str(error))
    try:
        run(args)
    except BaseException as error:
        write_json(Path(args.output_dir) / f"failure_rank{os.environ.get('RANK', 'unknown')}.json", {
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "arguments": vars(args),
        })
        raise
