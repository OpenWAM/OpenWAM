from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from open_wam.configs import load_experiment_config
from open_wam.models.common.rollout_history import resolve_execute_action_steps
from open_wam.models.policy_variants import PolicyInferContext
from open_wam.models.policy_variants.dual_expert.inference_backend import (
    ensure_dual_expert_inference_backend,
)
from open_wam.models.policy_variants.dual_expert.rollout_geometry import (
    resolve_dual_expert_rollout_cache_window_frames,
)
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training import TrainingRuntime
from open_wam.training.runtime import _normalize_optimizer_state_dtypes
from open_wam.training.state import TrainState
from open_wam.utils import apply_config_overrides

from .dual_expert_refactor_artifacts import (
    EXACT_COMPARISON_TOLERANCE,
    ComparisonTolerance,
    collect_tensor_fingerprints,
    compare_characterization_reports,
    load_latent_batch_fixture,
    load_rollout_input_fixture,
    scalar_tensor_values,
    tensor_fingerprint,
    tensor_tree_schema,
    write_json_atomic,
)
from .dual_expert_refactor_contract import (
    CHARACTERIZATION_ASSET_IDS,
    CACHE_ROLLOVER_ASSET_IDS,
    DEFAULT_INFERENCE_CONTRACT,
    FULL_STATE_RESUME_ASSET_ID,
    GJD_INFERENCE_CONTRACT,
    METHOD_BY_ASSET_ID,
    DualExpertMethodSpec,
    DualExpertTrainingProfile,
    GJDTrainingMode,
    apply_gjd_ablation,
    apply_ground_truth_training_profile,
    inference_scenarios_for,
    load_characterization_assets,
    resolve_characterization_asset_id,
    training_scenarios_for,
)
from .dual_expert_refactor_end_to_end import stage_model_only_checkpoint
from .dual_expert_refactor_fixtures import CONFIG_ROOT, configure_characterization_paths
from .dual_expert_refactor_provenance import (
    apply_checkpoint_provenance_policy,
    assert_checkpoint_provenance,
    build_checkpoint_source_contract_config,
    checkpoint_provenance_report,
)

WORKER_SEED = 20260727
INFERENCE_CHARACTERIZATION_CHUNKS = 3
CACHE_ROLLOVER_CHARACTERIZATION_CHUNKS = 17
RESUME_REPORT_SCHEMA_VERSION = 2
TRAINED_DUAL_EXPERT_KEY_PREFIXES = (
    "policy_variant.packed_block_stack.",
    "policy_variant.action_expert.",
    "visual_tower.core.",
)
STATE_DIGEST_CHUNK_BYTES = 16 * 1024 * 1024
RESUME_GRADIENT_TOLERANCE = ComparisonTolerance(
    absolute=5e-4,
    relative=5e-3,
)
RESUME_DISTRIBUTED_AGGREGATE_TOLERANCE = ComparisonTolerance(
    absolute=0.25,
    relative=0.0,
)
RESUME_DISTRIBUTED_AGGREGATE_DELTA_TOLERANCE = ComparisonTolerance(
    absolute=2 * RESUME_DISTRIBUTED_AGGREGATE_TOLERANCE.absolute,
    relative=0.0,
)


def build_training_characterization_config(
    *,
    method: DualExpertMethodSpec,
    assets,
    output_root: Path,
    world_size: int,
):
    config = load_experiment_config(CONFIG_ROOT / f"{method.config_name}.yaml")
    config = configure_characterization_paths(
        config,
        assets,
        include_counterfactual=method.is_gjd,
    )
    if method.is_gjd:
        config = apply_gjd_ablation(config, method)
    else:
        # Strict source checkpoints are phase-two checkpoints. The
        # random-segment fixture still carries its sampled chunk/window
        # geometry, which the policy consumes ahead of these config fallbacks.
        config = apply_ground_truth_training_profile(
            config,
            DualExpertTrainingProfile.FULL_SEGMENT_W64,
        )
    return apply_config_overrides(
        config,
        {
            "data.num_workers": 0,
            "trainer.devices": world_size,
            "trainer.strategy": "fsdp",
            "trainer.accelerator": "gpu",
            "trainer.precision": "bf16-mixed",
            "trainer.checkpoint_mode": "model_only",
            "trainer.enable_checkpointing": False,
            "trainer.save_interval": None,
            "trainer.export_runtime_backbone": False,
            "trainer.enable_jsonl_logging": False,
            "trainer.enable_wandb": False,
            "trainer.wandb_mode": "disabled",
            "trainer.validation_interval": None,
            "trainer.default_root_dir": str(output_root),
            "trainer.run_name": f"characterize_{method.asset_id}",
            "trainer.resume_from": str(assets.checkpoint_for(method.asset_id)),
        },
    )


def run_training_characterization(
    *,
    method: DualExpertMethodSpec,
    assets,
    fixture_root: Path,
    output_path: Path,
    seed: int,
    allow_provenance_mismatch: bool,
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    output_root = output_path.parent / "runtime"
    _seed_everything(seed)
    source_contract_config = build_checkpoint_source_contract_config(method)
    provenance_report = apply_checkpoint_provenance_policy(
        checkpoint_provenance_report(
            method=method,
            expected_config=source_contract_config,
            resolved_config_path=assets.checkpoint_config_for(method.asset_id),
        ),
        accepted_origin_mismatch_fields=(
            assets.checkpoints[method.asset_id].accepted_origin_mismatch_fields
        ),
    )
    if not allow_provenance_mismatch:
        assert_checkpoint_provenance(
            provenance_report,
            asset_id=method.asset_id,
        )
    config = build_training_characterization_config(
        method=method,
        assets=assets,
        output_root=output_root,
        world_size=world_size,
    )
    runtime = TrainingRuntime.from_config(config)
    checkpoint_report = _distributed_training_checkpoint_compatibility(
        runtime=runtime,
        checkpoint_path=assets.checkpoint_for(method.asset_id),
    )
    _assert_checkpoint_compatibility(
        checkpoint_report,
        asset_id=method.asset_id,
    )
    reports: dict[str, Any] = {}
    optimizer_step_report: dict[str, Any] | None = None
    try:
        scenarios = training_scenarios_for(method)
        for scenario_offset, scenario in enumerate(scenarios):
            scenario_seed = seed + scenario_offset * 101
            if runtime.strategy.is_main_process:
                print(
                    "[dual_expert_characterization] "
                    f"phase=training asset={method.asset_id} "
                    f"scenario={scenario.scenario_id} status=start",
                    flush=True,
                )
            _seed_everything(scenario_seed)
            runtime.strategy.zero_grad(runtime.optimizer)
            torch.cuda.reset_peak_memory_stats(runtime.strategy.device)
            batch = load_latent_batch_fixture(
                fixture_root / f"{scenario.fixture_id}.json"
            )
            device_batch = runtime.step_executor.batch_adapter.move_to_device(
                batch,
                runtime.strategy.device,
            )
            runtime.model.train()
            runtime.strategy.set_gradient_sync(runtime.model, enabled=True)
            with runtime.strategy.autocast_context():
                result = runtime.step_executor.forward_train(device_batch)
            runtime.strategy.backward(result.loss)
            report = _training_scenario_report(
                runtime=runtime,
                result=result,
                batch=batch,
                method=method,
                scenario_id=scenario.scenario_id,
                fixture_id=scenario.fixture_id,
                seed=scenario_seed,
            )
            _assert_training_scenario(report, method=method, scenario=scenario)
            reports[scenario.scenario_id] = report
            if scenario_offset == len(scenarios) - 1:
                optimizer_step_report = _execute_characterization_optimizer_step(
                    runtime=runtime,
                    scenario_id=scenario.scenario_id,
                )
                _assert_optimizer_step(
                    optimizer_step_report,
                    method=method,
                )
            if runtime.strategy.is_main_process:
                print(
                    "[dual_expert_characterization] "
                    f"phase=training asset={method.asset_id} "
                    f"scenario={scenario.scenario_id} status=passed "
                    f"loss={float(report['metrics']['loss']):.8f}",
                    flush=True,
                )
            runtime.strategy.zero_grad(runtime.optimizer)
            del result
            del device_batch
            del batch
            gc.collect()
            torch.cuda.empty_cache()

        if runtime.strategy.is_main_process:
            write_json_atomic(
                output_path,
                {
                    "schema_version": 2,
                    "phase": "training",
                    "asset_id": method.asset_id,
                    "config_name": method.config_name,
                    "checkpoint": str(assets.checkpoint_for(method.asset_id)),
                    "checkpoint_size_bytes": assets.checkpoint_for(method.asset_id)
                    .stat()
                    .st_size,
                    "checkpoint_load": checkpoint_report,
                    "checkpoint_provenance": provenance_report,
                    "world_size": world_size,
                    "precision": str(config.trainer.precision.value),
                    "strategy": str(config.trainer.strategy.value),
                    "runtime_profile": (
                        "gjd_standard"
                        if method.is_gjd
                        else DualExpertTrainingProfile.FULL_SEGMENT_W64.value
                    ),
                    "scenarios": reports,
                    "optimizer_step": optimizer_step_report,
                },
            )
    finally:
        runtime.log_sink.close()
        runtime.strategy.close()


def run_full_state_resume_characterization(
    *,
    method: DualExpertMethodSpec,
    assets,
    fixture_root: Path,
    output_path: Path,
    seed: int,
    allow_provenance_mismatch: bool,
) -> None:
    """Compare uninterrupted and full-state-resumed continuation on real M5 GJD."""

    if method.asset_id != FULL_STATE_RESUME_ASSET_ID:
        raise ValueError(
            "Full-state resume characterization requires the widest shared-state "
            f"sentinel {FULL_STATE_RESUME_ASSET_ID!r}, got {method.asset_id!r}."
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    source_contract_config = build_checkpoint_source_contract_config(method)
    provenance_report = apply_checkpoint_provenance_policy(
        checkpoint_provenance_report(
            method=method,
            expected_config=source_contract_config,
            resolved_config_path=assets.checkpoint_config_for(method.asset_id),
        ),
        accepted_origin_mismatch_fields=(
            assets.checkpoints[method.asset_id].accepted_origin_mismatch_fields
        ),
    )
    if not allow_provenance_mismatch:
        assert_checkpoint_provenance(
            provenance_report,
            asset_id=method.asset_id,
        )

    runtime_root = output_path.parent / "resume_runtime"
    checkpoint_root = runtime_root / "checkpoints"
    source_model_checkpoint = stage_model_only_checkpoint(
        assets.checkpoint_for(method.asset_id),
        runtime_root / "source_checkpoint",
    )
    base_config = build_training_characterization_config(
        method=method,
        assets=assets,
        output_root=runtime_root / "uninterrupted",
        world_size=world_size,
    )
    base_config = apply_config_overrides(
        base_config,
        {
            "training.gradient_accumulation_steps": 1,
            "trainer.checkpoint_mode": "full_training_state",
            "trainer.checkpoint_dir": str(checkpoint_root),
            "trainer.default_root_dir": str(runtime_root / "uninterrupted"),
            "trainer.run_name": "characterize_full_state_resume",
            # This characterization starts from frozen model weights, then proves
            # that the checkpoint it writes preserves the complete runtime state.
            "trainer.resume_from": str(source_model_checkpoint),
        },
    )
    fixture_path = fixture_root / "gjd_real_joint.json"
    owner_strategy = None
    runtime = None
    resumed_runtime = None
    try:
        _seed_everything(seed)
        runtime = TrainingRuntime.from_config(base_config)
        owner_strategy = runtime.strategy
        _log_resume_progress(runtime, "runtime_initialized")
        checkpoint_report = _distributed_training_checkpoint_compatibility(
            runtime=runtime,
            checkpoint_path=assets.checkpoint_for(method.asset_id),
        )
        _assert_checkpoint_compatibility(
            checkpoint_report,
            asset_id=method.asset_id,
        )
        runtime.train_state = TrainState(run_name="characterize_full_state_resume")

        first_update = _execute_resume_update(
            runtime=runtime,
            fixture_path=fixture_path,
            method=method,
            seed=seed,
            scenario_id="before_checkpoint",
        )
        _log_resume_progress(runtime, "first_update_complete")
        saved_state = _distributed_runtime_state_digest(runtime)
        _log_resume_progress(runtime, "pre_save_state_hashed")
        checkpoint_dir = runtime.checkpoint_manager.save(
            step=runtime.train_state.optimizer_step,
            model=runtime.strategy.unwrap_model(runtime.model),
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            train_state=runtime.train_state,
            strategy_state=runtime.strategy.state_dict(),
        )
        runtime.train_state.last_checkpoint_path = str(checkpoint_dir)
        checkpoint_artifacts = _checkpoint_artifact_report(checkpoint_dir)
        _log_resume_progress(runtime, "full_state_checkpoint_saved")
        post_save_state = _distributed_runtime_state_digest(runtime)
        save_mutation_differences = _runtime_state_digest_differences(
            saved_state,
            post_save_state,
        )
        if save_mutation_differences:
            raise AssertionError(
                "Full-state save mutated the live runtime state:\n- "
                + "\n- ".join(save_mutation_differences)
            )
        _log_resume_progress(runtime, "post_save_state_verified")

        uninterrupted_update = _execute_resume_update(
            runtime=runtime,
            fixture_path=fixture_path,
            method=method,
            seed=seed + 1,
            scenario_id="after_checkpoint",
        )
        _log_resume_progress(runtime, "uninterrupted_update_complete")
        uninterrupted_state = _distributed_runtime_state_digest(runtime)
        _log_resume_progress(runtime, "uninterrupted_state_hashed")

        runtime.log_sink.close()
        runtime.strategy.barrier()
        del runtime
        runtime = None
        gc.collect()
        torch.cuda.empty_cache()

        resumed_config = apply_config_overrides(
            base_config,
            {
                "trainer.resume_from": str(checkpoint_dir),
                "trainer.checkpoint_dir": str(runtime_root / "resumed_checkpoints"),
                "trainer.default_root_dir": str(runtime_root / "resumed"),
            },
        )
        _seed_everything(seed)
        resumed_runtime = TrainingRuntime.from_config(resumed_config)
        _log_resume_progress(resumed_runtime, "resumed_runtime_loaded")
        resumed_loaded_state = _distributed_runtime_state_digest(resumed_runtime)
        load_differences = _runtime_state_digest_differences(
            saved_state,
            resumed_loaded_state,
        )
        if load_differences:
            raise AssertionError(
                "Full-state checkpoint did not restore exactly:\n- "
                + "\n- ".join(load_differences)
            )
        _log_resume_progress(resumed_runtime, "restored_state_verified")

        resumed_update = _execute_resume_update(
            runtime=resumed_runtime,
            fixture_path=fixture_path,
            method=method,
            seed=seed + 1,
            scenario_id="after_checkpoint",
        )
        _log_resume_progress(resumed_runtime, "resumed_update_complete")
        resumed_final_state = _distributed_runtime_state_digest(resumed_runtime)
        continuation_differences = _resume_update_differences(
            uninterrupted_update,
            resumed_update,
        )
        final_state_differences = _runtime_state_contract_differences(
            uninterrupted_state,
            resumed_final_state,
        )
        if continuation_differences or final_state_differences:
            sections = []
            if continuation_differences:
                sections.append(
                    "Resumed update diverged from uninterrupted continuation:\n- "
                    + "\n- ".join(continuation_differences)
                )
            if final_state_differences:
                sections.append(
                    "Resumed final runtime state diverged from uninterrupted "
                    "state:\n- " + "\n- ".join(final_state_differences)
                )
            raise AssertionError("\n\n".join(sections))
        _log_resume_progress(resumed_runtime, "final_state_verified")

        if resumed_runtime.strategy.is_main_process:
            write_json_atomic(
                output_path,
                {
                    "schema_version": RESUME_REPORT_SCHEMA_VERSION,
                    "phase": "resume",
                    "asset_id": method.asset_id,
                    "config_name": method.config_name,
                    "checkpoint_provenance": provenance_report,
                    "world_size": world_size,
                    "precision": str(base_config.trainer.precision.value),
                    "strategy": str(base_config.trainer.strategy.value),
                    "checkpoint_mode": str(base_config.trainer.checkpoint_mode.value),
                    "gradient_accumulation_steps": int(
                        base_config.training.gradient_accumulation_steps
                    ),
                    "checkpoint_load": checkpoint_report,
                    "checkpoint_artifacts": checkpoint_artifacts,
                    "first_update": _resume_update_report_contract(
                        first_update,
                        preserve_output_values=True,
                    ),
                    "uninterrupted_update": _resume_update_report_contract(
                        uninterrupted_update,
                        preserve_output_values=False,
                    ),
                    "resumed_update": _resume_update_report_contract(
                        resumed_update,
                        preserve_output_values=False,
                    ),
                    "restored_state_contract": _state_digest_contract(saved_state),
                    "final_state_contract": _state_digest_contract(uninterrupted_state),
                    "restored_state_exact": True,
                    "save_is_nonmutating": True,
                    "continuation_matches_within_distributed_tolerance": True,
                    "final_state_contract_exact": True,
                },
            )
    finally:
        if resumed_runtime is not None:
            resumed_runtime.log_sink.close()
            resumed_runtime.strategy.barrier()
            resumed_runtime.strategy.close()
        if runtime is not None:
            runtime.log_sink.close()
            runtime.strategy.barrier()
        if owner_strategy is not None:
            owner_strategy.close()


def _log_resume_progress(runtime: TrainingRuntime, status: str) -> None:
    if runtime.strategy.is_main_process:
        print(
            "[dual_expert_characterization] "
            f"phase=resume asset={FULL_STATE_RESUME_ASSET_ID} status={status}",
            flush=True,
        )


def run_inference_characterization(
    *,
    method: DualExpertMethodSpec,
    assets,
    fixture_root: Path,
    output_path: Path,
    seed: int,
    allow_provenance_mismatch: bool,
    chunk_count: int = INFERENCE_CHARACTERIZATION_CHUNKS,
    require_cache_rollover: bool = False,
) -> None:
    _seed_everything(seed)
    source_contract_config = build_checkpoint_source_contract_config(method)
    provenance_report = apply_checkpoint_provenance_policy(
        checkpoint_provenance_report(
            method=method,
            expected_config=source_contract_config,
            resolved_config_path=assets.checkpoint_config_for(method.asset_id),
        ),
        accepted_origin_mismatch_fields=(
            assets.checkpoints[method.asset_id].accepted_origin_mismatch_fields
        ),
    )
    if not allow_provenance_mismatch:
        assert_checkpoint_provenance(
            provenance_report,
            asset_id=method.asset_id,
        )
    config = _build_inference_characterization_config(method=method, assets=assets)
    pipeline = build_variant_pipeline_from_config(config)
    checkpoint_report = _load_pipeline_checkpoint(
        pipeline,
        assets.checkpoint_for(method.asset_id),
    )
    device = torch.device("cuda", 0)
    pipeline.to(device=device)
    if hasattr(pipeline.policy_variant, "_maybe_initialize_action_expert"):
        pipeline.policy_variant._maybe_initialize_action_expert(pipeline.visual_tower)
    backend_report = ensure_dual_expert_inference_backend(pipeline, config)
    pipeline.eval()
    fixture = load_rollout_input_fixture(fixture_root / "mot_streaming_startup.json")
    reports: dict[str, Any] = {}
    try:
        scenarios = inference_scenarios_for(method)
        if require_cache_rollover:
            if method.asset_id not in CACHE_ROLLOVER_ASSET_IDS:
                raise ValueError(
                    "Cache-rollover characterization is limited to shared-runtime "
                    f"sentinels {CACHE_ROLLOVER_ASSET_IDS}, got {method.asset_id!r}."
                )
            scenarios = scenarios[:1]
        for scenario_offset, scenario in enumerate(scenarios):
            scenario_seed = seed + scenario_offset * 101
            print(
                "[dual_expert_characterization] "
                f"phase=inference asset={method.asset_id} "
                f"scenario={scenario.scenario_id} status=start",
                flush=True,
            )
            _seed_everything(scenario_seed)
            torch.cuda.reset_peak_memory_stats(device)
            reports[scenario.scenario_id] = _run_inference_scenario(
                pipeline=pipeline,
                fixture=fixture,
                method=method,
                mode=scenario.mode,
                seed=scenario_seed,
                device=device,
                chunk_count=chunk_count,
                require_cache_rollover=require_cache_rollover,
            )
            print(
                "[dual_expert_characterization] "
                f"phase=inference asset={method.asset_id} "
                f"scenario={scenario.scenario_id} status=passed",
                flush=True,
            )
        contract = (
            GJD_INFERENCE_CONTRACT if method.is_gjd else DEFAULT_INFERENCE_CONTRACT
        )
        payload = {
            "schema_version": 1,
            "phase": ("cache_rollover" if require_cache_rollover else "inference"),
            "asset_id": method.asset_id,
            "config_name": method.config_name,
            "checkpoint": str(assets.checkpoint_for(method.asset_id)),
            "checkpoint_size_bytes": assets.checkpoint_for(method.asset_id)
            .stat()
            .st_size,
            "checkpoint_load": checkpoint_report,
            "checkpoint_provenance": provenance_report,
            "backend": backend_report,
            "rollout_contract": {
                "frontend_encode_mode": contract.frontend_encode_mode,
                "inference_window_size": contract.inference_window_size,
                "startup_model_obs_frames": contract.startup_model_obs_frames,
                "startup_env_init_steps": contract.startup_env_init_steps,
                "model_frame_chunk_size": contract.model_frame_chunk_size,
                "action_per_frame": contract.action_per_frame,
                "action_horizon": contract.action_horizon,
                "execute_action_steps": contract.execute_action_steps,
                "max_timestep": contract.max_timestep,
                "max_chunks": contract.max_chunks,
            },
            "scenarios": reports,
        }
        if require_cache_rollover:
            payload["cache_rollover_contract"] = {
                "characterization_chunks": chunk_count,
                "cache_window_frames": resolve_dual_expert_rollout_cache_window_frames(
                    window_size=contract.inference_window_size,
                    frame_chunk_size=contract.model_frame_chunk_size,
                ),
            }
        write_json_atomic(
            output_path,
            payload,
        )
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


def _build_inference_characterization_config(*, method: DualExpertMethodSpec, assets):
    config = load_experiment_config(CONFIG_ROOT / f"{method.config_name}.yaml")
    config = configure_characterization_paths(
        config,
        assets,
        include_counterfactual=method.is_gjd,
    )
    if method.is_gjd:
        config = apply_gjd_ablation(config, method)
    else:
        config = apply_ground_truth_training_profile(
            config,
            DualExpertTrainingProfile.FULL_SEGMENT_W64,
        )
    return apply_config_overrides(
        config,
        {
            "trainer.devices": 1,
            "trainer.strategy": "single_device",
            "trainer.accelerator": "gpu",
            "trainer.precision": "bf16-mixed",
            "trainer.enable_wandb": False,
            "trainer.wandb_mode": "disabled",
            "trainer.enable_jsonl_logging": False,
        },
    )


def _load_pipeline_checkpoint(
    pipeline: torch.nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint, raw_state, state_dict = _load_checkpoint_state_dict(checkpoint_path)
    tensor_key_count = len(state_dict)
    runtime_keys = set(pipeline.state_dict())
    loadable_state = {
        key: value for key, value in state_dict.items() if key in runtime_keys
    }
    missing, unexpected = pipeline.load_state_dict(loadable_state, strict=False)
    policy_variant = getattr(pipeline, "policy_variant", None)
    has_action_weights = any(
        key.startswith("policy_variant.action_expert.") or ".action_block." in key
        for key in state_dict
    )
    missing_action_weights = any(
        key.startswith("policy_variant.action_expert.") or ".action_block." in key
        for key in missing
    )
    if (
        policy_variant is not None
        and hasattr(policy_variant, "_action_expert_initialized")
        and has_action_weights
        and not missing_action_weights
    ):
        policy_variant._action_expert_initialized = True
    report = {
        "tensor_keys": tensor_key_count,
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_key_preview": list(missing[:20]),
        "unexpected_key_preview": list(unexpected[:20]),
        "trained_missing_keys": [key for key in missing if _is_trained_dual_expert_key(key)],
        "trained_unexpected_keys": [
            key for key in unexpected if _is_trained_dual_expert_key(key)
        ],
    }
    _assert_checkpoint_compatibility(
        report,
        asset_id=checkpoint_path.parent.name,
    )
    del checkpoint
    del raw_state
    del state_dict
    gc.collect()
    return report


def _distributed_training_checkpoint_compatibility(
    *,
    runtime: TrainingRuntime,
    checkpoint_path: Path,
) -> dict[str, Any]:
    report: dict[str, Any] | None = None
    if runtime.strategy.is_main_process:
        model = runtime.strategy.unwrap_model(runtime.model)
        expected_keys = {
            normalized
            for name, _ in model.named_parameters(remove_duplicate=False)
            if _is_trained_dual_expert_key(normalized := _normalize_checkpoint_key(name))
        }
        checkpoint, raw_state, state_dict = _load_checkpoint_state_dict(checkpoint_path)
        checkpoint_keys = set(state_dict)
        missing = sorted(expected_keys - checkpoint_keys)
        unexpected = sorted(
            key for key in checkpoint_keys - expected_keys if _is_trained_dual_expert_key(key)
        )
        report = {
            "tensor_keys": len(checkpoint_keys),
            "missing_key_count": len(missing),
            "unexpected_key_count": len(unexpected),
            "missing_key_preview": missing[:20],
            "unexpected_key_preview": unexpected[:20],
            "trained_missing_keys": missing,
            "trained_unexpected_keys": unexpected,
        }
        del checkpoint
        del raw_state
        del state_dict
        gc.collect()
    if dist.is_initialized():
        payload = [report]
        dist.broadcast_object_list(payload, src=0)
        report = payload[0]
    if not isinstance(report, dict):
        raise TypeError("Failed to distribute checkpoint compatibility report.")
    return report


def _load_checkpoint_state_dict(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint mapping at {checkpoint_path}.")
    raw_state = checkpoint.get(
        "state_dict",
        checkpoint.get("model_state_dict", checkpoint),
    )
    if not isinstance(raw_state, dict):
        raise TypeError(
            "Checkpoint must be a raw state dict or contain state_dict/model_state_dict."
        )
    state_dict = {
        _normalize_checkpoint_key(str(key)): value
        for key, value in raw_state.items()
        if isinstance(value, torch.Tensor)
    }
    return checkpoint, raw_state, state_dict


def _normalize_checkpoint_key(key: str) -> str:
    return key.removeprefix("pipeline.")


def _is_trained_dual_expert_key(key: str) -> bool:
    return key.startswith(TRAINED_DUAL_EXPERT_KEY_PREFIXES)


def _assert_checkpoint_compatibility(
    report: dict[str, Any],
    *,
    asset_id: str,
) -> None:
    missing = list(report.get("trained_missing_keys") or ())
    unexpected = list(report.get("trained_unexpected_keys") or ())
    if not missing and not unexpected:
        return
    details: list[str] = []
    if missing:
        details.append(
            f"missing {len(missing)} trained DualExpert keys: {', '.join(missing[:8])}"
        )
    if unexpected:
        details.append(
            "contains "
            f"{len(unexpected)} unexpected trained DualExpert keys: "
            f"{', '.join(unexpected[:8])}"
        )
    raise AssertionError(f"Checkpoint {asset_id!r} " + "; ".join(details))


def _run_inference_scenario(
    *,
    pipeline,
    fixture,
    method: DualExpertMethodSpec,
    mode: GJDTrainingMode | None,
    seed: int,
    device: torch.device,
    chunk_count: int,
    require_cache_rollover: bool,
) -> dict[str, Any]:
    views = {name: value.to(device=device) for name, value in fixture.views.items()}
    state = fixture.state.to(device=device)
    text_context = fixture.text_context.to(device=device)
    negative_text_context = fixture.negative_text_context.to(device=device)
    action_conditioning = fixture.action_conditioning.to(device=device)
    video_conditioning = fixture.video_conditioning.to(device=device)
    pipeline.visual_tower.reset_runtime_state()
    with torch.inference_mode():
        canonical_batch = pipeline.canonicalize(views)
        canonical_video = canonical_batch.video.to(device=device)
        frontend_output = pipeline.visual_tower.run_frontend(
            canonical_video,
            placements=canonical_batch.placements,
            task_text=fixture.task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            preserve_stream_cache=False,
        )
        runtime_dtype = pipeline.visual_tower.core.patch_embedding_mlp.weight.dtype
        visual_outputs = pipeline.prepare_visual_outputs_from_latents(
            frontend_output.video_latents.to(
                device=device,
                dtype=runtime_dtype,
            ),
            task_text=fixture.task_text,
            text_context=frontend_output.conditioning.text_context.to(
                device=device,
                dtype=runtime_dtype,
            ),
            negative_text_context=frontend_output.conditioning.negative_text_context.to(
                device=device,
                dtype=runtime_dtype,
            ),
            canonical_video=canonical_video,
        )
        resolved_text_context = frontend_output.conditioning.text_context.to(
            device=device,
            dtype=runtime_dtype,
        )
        resolved_negative_text_context = (
            frontend_output.conditioning.negative_text_context.to(
                device=device,
                dtype=runtime_dtype,
            )
        )
        infer_state = None
        next_input_latents = frontend_output.video_latents.to(
            device=device,
            dtype=runtime_dtype,
        )
        chunks: list[dict[str, Any]] = []
        progression: list[dict[str, int | None]] = []
        for chunk_index in range(chunk_count):
            chunk_seed = seed + chunk_index
            _seed_everything(chunk_seed)
            if chunk_index > 0:
                visual_outputs = pipeline.prepare_visual_outputs_from_latents(
                    next_input_latents,
                    task_text=fixture.task_text,
                    text_context=resolved_text_context,
                    negative_text_context=resolved_negative_text_context,
                )
            extra = _inference_extra(
                fixture=fixture,
                mode=mode,
                device=device,
                action_conditioning=action_conditioning,
                video_conditioning=video_conditioning,
            )
            infer_output = pipeline.forward_infer_step_from_visual_outputs(
                visual_outputs,
                context=PolicyInferContext(state=state, extra=extra),
                infer_state=infer_state,
            )
            chunk_report, next_input_latents = _inference_chunk_report(
                infer_output=infer_output,
                input_latents=visual_outputs.frontend.video_latents,
                action_conditioning=action_conditioning,
                video_conditioning=video_conditioning,
                method=method,
                mode=mode,
                chunk_index=chunk_index,
                seed=chunk_seed,
            )
            chunks.append(chunk_report)
            infer_state = infer_output.policy_output.next_state
            progression.append(_inference_state_progression(infer_state))
            del infer_output
            completed_chunks = chunk_index + 1
            if require_cache_rollover and (
                completed_chunks % 4 == 0 or completed_chunks == chunk_count
            ):
                print(
                    "[dual_expert_characterization] "
                    f"phase=cache_rollover asset={method.asset_id} "
                    f"mode={mode or 'default'} chunks={completed_chunks}/{chunk_count}",
                    flush=True,
                )

    _assert_inference_progression(
        progression,
        method=method,
        mode=mode,
        expected_chunk_count=chunk_count,
    )
    if require_cache_rollover:
        _assert_cache_rollover(
            progression,
            method=method,
            mode=mode,
        )
    return {
        "seed": seed,
        "mode": None if mode is None else mode.value,
        "chunk_count": chunk_count,
        "input": {
            "view_shapes": {
                name: list(value.shape) for name, value in sorted(views.items())
            },
            "state": tensor_fingerprint(state),
            "action_conditioning": (
                tensor_fingerprint(action_conditioning)
                if mode == GJDTrainingMode.FDM
                else None
            ),
            "video_conditioning": (
                tensor_fingerprint(video_conditioning)
                if mode == GJDTrainingMode.IDM
                else None
            ),
        },
        "frontend_video_latents": tensor_fingerprint(frontend_output.video_latents),
        "chunks": chunks,
        "state_progression": progression,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _inference_extra(
    *,
    fixture,
    mode: GJDTrainingMode | None,
    device: torch.device,
    action_conditioning: torch.Tensor,
    video_conditioning: torch.Tensor,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "task_text": fixture.task_text,
        "action_device": str(device),
        "dual_expert_inference_window_size": 30,
    }
    if mode is not None:
        extra.update(
            {
                "action_conditioning_mode": mode.value,
                "dual_expert_generalist_rollout_mode": mode.value,
            }
        )
    if mode == GJDTrainingMode.FDM:
        extra["dual_expert_forced_action_latents"] = action_conditioning
        extra["dual_expert_commit_action_latents"] = action_conditioning
    elif mode == GJDTrainingMode.IDM:
        extra["dual_expert_video_condition_latents"] = video_conditioning
        extra["dual_expert_commit_action_latents"] = action_conditioning
    return extra


def _inference_chunk_report(
    *,
    infer_output,
    input_latents: torch.Tensor,
    action_conditioning: torch.Tensor,
    video_conditioning: torch.Tensor,
    method: DualExpertMethodSpec,
    mode: GJDTrainingMode | None,
    chunk_index: int,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    action_pred = infer_output.decoder_output.action_pred
    predicted_latents = infer_output.decoder_output.aux.get("predicted_latents")
    if not isinstance(predicted_latents, torch.Tensor):
        predicted_latents = infer_output.policy_output.aux.get("predicted_latents")
    if not isinstance(predicted_latents, torch.Tensor):
        raise AssertionError(  # noqa: TRY004 - this is an output invariant
            f"{method.asset_id}/{mode or 'default'} returned no predicted latents."
        )
    for name, value in (
        ("action_pred", action_pred),
        ("predicted_latents", predicted_latents),
        ("input_latents", input_latents),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise AssertionError(
                f"{method.asset_id}/{mode or 'default'} chunk {chunk_index} "
                f"produced non-finite {name}."
            )
    if tuple(action_pred.shape) != tuple(action_conditioning.shape):
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} chunk {chunk_index} returned "
            f"action shape {tuple(action_pred.shape)}, expected "
            f"{tuple(action_conditioning.shape)}."
        )
    if tuple(predicted_latents.shape) != tuple(video_conditioning.shape):
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} chunk {chunk_index} returned "
            f"video-latent shape {tuple(predicted_latents.shape)}, expected "
            f"{tuple(video_conditioning.shape)}."
        )
    next_state = infer_output.policy_output.next_state
    selected_state_fingerprints = collect_tensor_fingerprints(
        next_state.variant_state,
        root="next_state.variant_state",
        max_tensors=8,
    )
    selected_state_fingerprints.update(
        collect_tensor_fingerprints(
            next_state.cache,
            root="next_state.cache",
            max_tensors=4,
        )
    )
    contract = GJD_INFERENCE_CONTRACT if method.is_gjd else DEFAULT_INFERENCE_CONTRACT
    execute_action_steps = resolve_execute_action_steps(
        None,
        action_horizon=int(action_pred.shape[1]),
        action_per_frame=contract.action_per_frame,
    )
    if execute_action_steps != contract.execute_action_steps:
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} chunk {chunk_index} resolved "
            f"{execute_action_steps} executed actions, expected "
            f"{contract.execute_action_steps}."
        )
    committed_action_pred = action_pred[:, :execute_action_steps]
    report = {
        "chunk_index": chunk_index,
        "seed": seed,
        "input_latents": tensor_fingerprint(input_latents),
        "outputs": {
            "action_pred": tensor_fingerprint(action_pred),
            "committed_action_pred": tensor_fingerprint(committed_action_pred),
            "predicted_latents": tensor_fingerprint(predicted_latents),
            "policy_features": tensor_fingerprint(
                infer_output.policy_output.policy_features
            ),
        },
        "policy_metrics": scalar_tensor_values(infer_output.policy_output.aux),
        "action_execution": {
            "predicted_action_steps": int(action_pred.shape[1]),
            "execute_action_steps": execute_action_steps,
            "execute_frame_chunk_size": (
                execute_action_steps // contract.action_per_frame
            ),
        },
        "state_fingerprints": selected_state_fingerprints,
        "state_schema": tensor_tree_schema(
            next_state,
            root="next_state",
            max_entries=192,
        ),
    }
    return report, predicted_latents.detach()


def _inference_state_progression(state) -> dict[str, int | None]:
    runtime_state = state.variant_state
    cursor = state.cursor
    past_latents = getattr(runtime_state, "past_clean_latents", None)
    past_actions = getattr(runtime_state, "past_clean_actions", None)
    return {
        "step_index": int(state.step_index),
        "cursor_current_start_frame": int(cursor.current_start_frame),
        "cursor_block_index": int(cursor.block_index),
        "past_clean_latent_frames": (
            None if past_latents is None else int(past_latents.shape[2])
        ),
        "past_clean_action_steps": (
            None if past_actions is None else int(past_actions.shape[1])
        ),
        "pending_predicted_video_frames": int(
            getattr(runtime_state, "pending_predicted_video_frames", 0)
        ),
        "next_condition_frame_start": int(
            getattr(runtime_state, "next_condition_frame_start", 0)
        ),
    }


def _assert_inference_progression(
    progression: list[dict[str, int | None]],
    *,
    method: DualExpertMethodSpec,
    mode: GJDTrainingMode | None,
    expected_chunk_count: int = INFERENCE_CHARACTERIZATION_CHUNKS,
) -> None:
    if len(progression) != expected_chunk_count:
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} produced "
            f"{len(progression)} recurrent states."
        )
    expected_steps = list(range(1, expected_chunk_count + 1))
    actual_steps = [int(item["step_index"]) for item in progression]
    if actual_steps != expected_steps:
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} state steps "
            f"{actual_steps}, expected {expected_steps}."
        )
    cursor_starts = [int(item["cursor_current_start_frame"]) for item in progression]
    if any(
        next_start <= current_start
        for current_start, next_start in pairwise(cursor_starts)
    ):
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} did not advance its "
            f"rollout cursor monotonically: {cursor_starts}."
        )


def _assert_cache_rollover(
    progression: list[dict[str, int | None]],
    *,
    method: DualExpertMethodSpec,
    mode: GJDTrainingMode | None,
) -> None:
    contract = GJD_INFERENCE_CONTRACT if method.is_gjd else DEFAULT_INFERENCE_CONTRACT
    cache_window_frames = resolve_dual_expert_rollout_cache_window_frames(
        window_size=contract.inference_window_size,
        frame_chunk_size=contract.model_frame_chunk_size,
    )
    latent_frames = [item["past_clean_latent_frames"] for item in progression]
    action_steps = [item["past_clean_action_steps"] for item in progression]
    if any(value is None for value in latent_frames + action_steps):
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} did not expose packed cache sizes."
        )
    resolved_latent_frames = [
        int(value) for value in latent_frames if value is not None
    ]
    resolved_action_steps = [int(value) for value in action_steps if value is not None]
    expected_action_steps = cache_window_frames * contract.action_per_frame
    if max(resolved_latent_frames) != cache_window_frames:
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} retained "
            f"{max(resolved_latent_frames)} latent frames, expected cache cap "
            f"{cache_window_frames}."
        )
    if max(resolved_action_steps) != expected_action_steps:
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} retained "
            f"{max(resolved_action_steps)} action steps, expected cache cap "
            f"{expected_action_steps}."
        )
    if len(resolved_latent_frames) < 2 or resolved_latent_frames[-2:] != [
        cache_window_frames,
        cache_window_frames,
    ]:
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} did not demonstrate latent "
            f"cache rollover at {cache_window_frames} frames: "
            f"{resolved_latent_frames}."
        )
    if resolved_action_steps[-2:] != [
        expected_action_steps,
        expected_action_steps,
    ]:
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} did not demonstrate action "
            f"cache rollover at {expected_action_steps} steps: "
            f"{resolved_action_steps}."
        )
    if int(progression[-1]["cursor_current_start_frame"]) <= cache_window_frames:
        raise AssertionError(
            f"{method.asset_id}/{mode or 'default'} cursor did not advance beyond "
            f"the retained cache window: {progression[-1]}."
        )


def _training_scenario_report(
    *,
    runtime: TrainingRuntime,
    result,
    batch,
    method: DualExpertMethodSpec,
    scenario_id: str,
    fixture_id: str,
    seed: int,
) -> dict[str, Any]:
    decoder = result.output.decoder_output
    policy = result.output.policy_output
    metadata = dict(batch.metadata[0]) if batch.metadata else {}
    action_mask = batch.action_mask
    leading_action_steps = DEFAULT_INFERENCE_CONTRACT.action_per_frame
    tensor_outputs = {
        "decoder.action_pred": tensor_fingerprint(decoder.action_pred),
        "policy.policy_features": tensor_fingerprint(policy.policy_features),
    }
    tensor_outputs.update(
        collect_tensor_fingerprints(
            decoder.aux,
            root="decoder.aux",
            max_tensors=24,
        )
    )
    tensor_outputs.update(
        collect_tensor_fingerprints(
            policy.aux,
            root="policy.aux",
            max_tensors=24,
        )
    )
    tensor_outputs.update(_schema_v1_decoder_artifact_fingerprints(policy))
    return {
        "scenario_id": scenario_id,
        "fixture_id": fixture_id,
        "seed": seed,
        "coupling": method.coupling,
        "mode": metadata.get("generalist_training_mode_override"),
        "source": metadata.get("generalist_training_source", "real_demo"),
        "input": {
            "video_latents": tensor_fingerprint(batch.video_latents),
            "actions": tensor_fingerprint(batch.actions),
            "action_mask": (
                tensor_fingerprint(action_mask) if action_mask is not None else None
            ),
            "leading_action_mask_sum": (
                float(action_mask[:, :leading_action_steps].sum().item())
                if action_mask is not None
                else None
            ),
            "future_action_mask_sum": (
                float(action_mask[:, leading_action_steps:].sum().item())
                if action_mask is not None
                else None
            ),
            "condition_latents_present": batch.condition_latents is not None,
            "state_present": batch.state is not None,
            "sampled_chunk_size": metadata.get("sampled_chunk_size"),
            "sampled_window_size": metadata.get("sampled_window_size"),
            "loss_frame_start": metadata.get("loss_frame_start"),
            "loss_frame_end": metadata.get("loss_frame_end"),
            "history_frames": metadata.get("history_frames"),
            "singleton_chunk_frame": metadata.get("singleton_chunk_frame"),
            "chunk_origin_frame": metadata.get("chunk_origin_frame"),
        },
        "metrics": scalar_tensor_values(result.metrics),
        "policy_metrics": scalar_tensor_values(policy.metrics),
        "outputs": tensor_outputs,
        "gradients": _distributed_gradient_summary(runtime.model),
        "cuda_peak_memory_bytes": int(
            torch.cuda.max_memory_allocated(runtime.strategy.device)
        ),
    }


def _schema_v1_decoder_artifact_fingerprints(policy) -> dict[str, dict[str, Any]]:
    """Project typed decoder payloads onto the immutable report-v1 paths."""

    envelope = policy.decoder_artifacts
    if envelope is None:
        return {}
    payload = envelope.payload
    action = getattr(payload, "action", None)
    if action is None:
        return {}
    fingerprints = collect_tensor_fingerprints(
        action,
        root="policy.aux.mot_train_artifacts.action",
        max_tensors=16,
    )
    video = getattr(payload, "video", None)
    if video is not None:
        fingerprints.update(
            collect_tensor_fingerprints(
                video,
                root="policy.aux.mot_train_artifacts.video",
                max_tensors=16,
            )
        )
    return fingerprints


def _distributed_gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    device = _distributed_reduction_device(model)
    fields = (
        "parameter_tensors",
        "parameter_elements",
        "finite_elements",
        "nonzero_elements",
        "absolute_sum",
        "squared_sum",
    )
    summaries: dict[str, dict[str, torch.Tensor]] = defaultdict(
        lambda: {
            field: torch.zeros((), device=device, dtype=torch.float64)
            for field in fields
        }
    )
    maxima: dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros((), device=device, dtype=torch.float32)
    )
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        local = _local_tensor(grad.detach()).float()
        group = _gradient_group(name)
        finite = torch.isfinite(local)
        finite_values = local.masked_fill(~finite, 0.0)
        summaries[group]["parameter_tensors"] += 1
        summaries[group]["parameter_elements"] += local.numel()
        summaries[group]["finite_elements"] += finite.sum().to(device)
        summaries[group]["nonzero_elements"] += (finite_values != 0).sum().to(device)
        summaries[group]["absolute_sum"] += (
            finite_values.abs().sum().double().to(device)
        )
        summaries[group]["squared_sum"] += (
            finite_values.square().sum().double().to(device)
        )
        if finite_values.numel():
            maxima[group] = torch.maximum(
                maxima[group],
                finite_values.abs().max().to(device),
            )

    groups = sorted(set(summaries) | set(maxima))
    resolved: dict[str, Any] = {}
    for group in groups:
        values = summaries[group]
        for value in values.values():
            if dist.is_initialized():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
        maximum = maxima[group]
        if dist.is_initialized():
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        resolved[group] = {
            "parameter_tensors": int(values["parameter_tensors"].item()),
            "parameter_elements": int(values["parameter_elements"].item()),
            "finite_elements": int(values["finite_elements"].item()),
            "nonzero_elements": int(values["nonzero_elements"].item()),
            "absolute_sum": float(values["absolute_sum"].item()),
            "l2_norm": float(torch.sqrt(values["squared_sum"]).item()),
            "max_abs": float(maximum.item()),
        }
    return resolved


def _execute_characterization_optimizer_step(
    *,
    runtime: TrainingRuntime,
    scenario_id: str,
) -> dict[str, Any]:
    """Apply the production optimizer sequence to the frozen-batch gradient.

    A single backward of the unscaled frozen loss is mathematically equivalent
    to accumulating that same frozen microbatch N times with each loss divided
    by the configured accumulation count. The full CLI smoke separately
    exercises TrainingRuntime's microstep counter and update branch.
    """

    configured_accumulation = max(
        1,
        int(runtime.config.training.gradient_accumulation_steps),
    )
    before = _distributed_parameter_summary(runtime.model)
    probes = _select_local_parameter_probes(runtime.model)
    lr_before = tuple(float(value) for value in runtime.scheduler.get_last_lr())
    scheduler_epoch_before = int(runtime.scheduler.last_epoch)

    runtime.strategy.unscale_(runtime.optimizer)
    max_grad_norm = runtime.config.training.max_grad_norm
    grad_norm = (
        runtime.strategy.clip_grad_norm_(runtime.model.parameters(), max_grad_norm)
        if max_grad_norm is not None
        else None
    )
    if grad_norm is not None and not bool(torch.isfinite(grad_norm).item()):
        raise RuntimeError(
            "Characterization optimizer step produced a non-finite gradient norm."
        )
    _normalize_optimizer_state_dtypes(runtime.optimizer)
    runtime.strategy.optimizer_step(runtime.optimizer)
    runtime.scheduler.step()

    after = _distributed_parameter_summary(runtime.model)
    distributed_numeric = {
        "grad_norm": None if grad_norm is None else float(grad_norm.item()),
        "parameter_groups": _parameter_summary_deltas(before, after),
        "parameter_probes": _resolve_parameter_probes(probes),
    }
    return {
        "source_scenario": scenario_id,
        "accumulation_contract": "equivalent_repeated_frozen_microbatch",
        "configured_gradient_accumulation_steps": configured_accumulation,
        "production_sequence": [
            "unscale",
            "clip_grad_norm",
            "normalize_optimizer_state_dtypes",
            "optimizer_step",
            "scheduler_step",
        ],
        "optimizer": type(runtime.optimizer).__name__,
        "optimizer_state_schema": _distributed_optimizer_state_schema(
            runtime.optimizer
        ),
        "scheduler": {
            "type": type(runtime.scheduler).__name__,
            "last_epoch_before": scheduler_epoch_before,
            "last_epoch_after": int(runtime.scheduler.last_epoch),
            "lr_before": list(lr_before),
            "lr_after": [float(value) for value in runtime.scheduler.get_last_lr()],
        },
        "distributed_numeric": distributed_numeric,
    }


def _distributed_parameter_summary(
    model: torch.nn.Module,
) -> dict[str, dict[str, Any]]:
    reduction_device = _distributed_reduction_device(model)
    fields = (
        "parameter_tensors",
        "parameter_elements",
        "finite_elements",
        "sum",
        "absolute_sum",
        "squared_sum",
    )
    summaries: dict[str, dict[str, torch.Tensor]] = defaultdict(
        lambda: {
            field: torch.zeros((), device=reduction_device, dtype=torch.float64)
            for field in fields
        }
    )
    maxima: dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros((), device=reduction_device, dtype=torch.float32)
    )
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        local = _local_tensor(parameter.detach()).float()
        group = _gradient_group(name)
        finite = torch.isfinite(local)
        finite_values = local.masked_fill(~finite, 0.0)
        summaries[group]["parameter_tensors"] += 1
        summaries[group]["parameter_elements"] += local.numel()
        summaries[group]["finite_elements"] += finite.sum().to(reduction_device)
        summaries[group]["sum"] += finite_values.sum().double().to(reduction_device)
        summaries[group]["absolute_sum"] += (
            finite_values.abs().sum().double().to(reduction_device)
        )
        summaries[group]["squared_sum"] += (
            finite_values.square().sum().double().to(reduction_device)
        )
        if finite_values.numel():
            maxima[group] = torch.maximum(
                maxima[group],
                finite_values.abs().max().to(reduction_device),
            )

    resolved: dict[str, dict[str, Any]] = {}
    for group in sorted(set(summaries) | set(maxima)):
        values = summaries[group]
        for value in values.values():
            if dist.is_initialized():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
        maximum = maxima[group]
        if dist.is_initialized():
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        resolved[group] = {
            "parameter_tensors": int(values["parameter_tensors"].item()),
            "parameter_elements": int(values["parameter_elements"].item()),
            "finite_elements": int(values["finite_elements"].item()),
            "sum": float(values["sum"].item()),
            "absolute_sum": float(values["absolute_sum"].item()),
            "l2_norm": float(torch.sqrt(values["squared_sum"]).item()),
            "max_abs": float(maximum.item()),
        }
    return resolved


def _distributed_reduction_device(model: torch.nn.Module) -> torch.device:
    if dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return next(model.parameters()).device


def _parameter_summary_deltas(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if before.keys() != after.keys():
        raise AssertionError(
            "Optimizer step changed the populated trainable parameter groups."
        )
    result: dict[str, dict[str, Any]] = {}
    numeric_fields = ("sum", "absolute_sum", "l2_norm", "max_abs")
    for group in sorted(before):
        before_group = before[group]
        after_group = after[group]
        for field in (
            "parameter_tensors",
            "parameter_elements",
            "finite_elements",
        ):
            if before_group[field] != after_group[field]:
                raise AssertionError(
                    f"Optimizer step changed {group}.{field}: "
                    f"{before_group[field]} -> {after_group[field]}."
                )
        result[group] = {
            "parameter_tensors": before_group["parameter_tensors"],
            "parameter_elements": before_group["parameter_elements"],
            "finite_elements": after_group["finite_elements"],
            "before": {field: before_group[field] for field in numeric_fields},
            "after": {field: after_group[field] for field in numeric_fields},
            "delta": {
                field: float(after_group[field]) - float(before_group[field])
                for field in numeric_fields
            },
        }
    return result


def _select_local_parameter_probes(
    model: torch.nn.Module,
) -> dict[str, tuple[str, torch.Tensor, int, float, float]]:
    candidates: dict[str, tuple[str, torch.Tensor, int, float, float]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        local_grad = _local_tensor(parameter.grad.detach()).float().reshape(-1)
        if not local_grad.numel():
            continue
        absolute_grad = local_grad.abs()
        max_abs, index = absolute_grad.max(dim=0)
        max_value = float(max_abs.item())
        group = _gradient_group(name)
        previous = candidates.get(group)
        if previous is not None and max_value <= previous[4]:
            continue
        flat_parameter = _local_tensor(parameter.detach()).reshape(-1)
        resolved_index = int(index.item())
        candidates[group] = (
            name,
            parameter,
            resolved_index,
            float(flat_parameter[resolved_index].float().item()),
            max_value,
        )
    return candidates


def _resolve_parameter_probes(
    probes: dict[str, tuple[str, torch.Tensor, int, float, float]],
) -> list[dict[str, Any]]:
    local_records: list[dict[str, Any]] = []
    rank = dist.get_rank() if dist.is_initialized() else 0
    for group, (name, parameter, index, before, gradient_abs) in sorted(probes.items()):
        local = _local_tensor(parameter.detach()).reshape(-1)
        after = float(local[index].float().item())
        local_records.append(
            {
                "rank": rank,
                "group": group,
                "parameter": name,
                "local_flat_index": index,
                "gradient_abs": gradient_abs,
                "before": before,
                "after": after,
                "delta": after - before,
            }
        )
    if not dist.is_initialized():
        return local_records
    records_by_rank: list[list[dict[str, Any]] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(records_by_rank, local_records)
    return [
        record
        for rank_records in records_by_rank
        if rank_records is not None
        for record in rank_records
    ]


def _distributed_optimizer_state_schema(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    state_keys: Counter[str] = Counter()
    tensor_dtypes: Counter[str] = Counter()
    tensor_shapes: Counter[str] = Counter()
    scalar_types: Counter[str] = Counter()
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            state_keys[str(key)] += 1
            if isinstance(value, torch.Tensor):
                local = _local_tensor(value)
                tensor_dtypes[str(local.dtype)] += 1
                tensor_shapes[str(tuple(local.shape))] += 1
            else:
                scalar_types[type(value).__name__] += 1
    local = {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "state_entries": len(optimizer.state),
        "state_keys": dict(sorted(state_keys.items())),
        "tensor_dtypes": dict(sorted(tensor_dtypes.items())),
        "tensor_shapes": dict(sorted(tensor_shapes.items())),
        "scalar_types": dict(sorted(scalar_types.items())),
    }
    if not dist.is_initialized():
        return {"per_rank": [local]}
    per_rank: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(per_rank, local)
    return {"per_rank": [item for item in per_rank if item is not None]}


def _assert_optimizer_step(
    report: dict[str, Any],
    *,
    method: DualExpertMethodSpec,
) -> None:
    schemas = report["optimizer_state_schema"]["per_rank"]
    if not schemas or any(int(item["state_entries"]) <= 0 for item in schemas):
        raise AssertionError(
            f"{method.asset_id} optimizer step did not materialize AdamW state."
        )
    groups = report["distributed_numeric"]["parameter_groups"]
    required_groups = {"action_expert", "video_backbone"}
    if method.mode_token:
        required_groups.add("generalist_mode_token")
    missing = sorted(required_groups - groups.keys())
    if missing:
        raise AssertionError(
            f"{method.asset_id} optimizer step lacks parameter groups {missing}."
        )
    unchanged = [
        group
        for group in sorted(required_groups)
        if all(
            float(groups[group]["delta"][field]) == 0.0
            for field in ("sum", "absolute_sum", "l2_norm", "max_abs")
        )
    ]
    if unchanged:
        raise AssertionError(
            f"{method.asset_id} optimizer step did not change groups {unchanged}."
        )


def _execute_resume_update(
    *,
    runtime: TrainingRuntime,
    fixture_path: Path,
    method: DualExpertMethodSpec,
    seed: int,
    scenario_id: str,
) -> dict[str, Any]:
    _seed_everything(seed)
    runtime.strategy.zero_grad(runtime.optimizer)
    torch.cuda.reset_peak_memory_stats(runtime.strategy.device)
    batch = load_latent_batch_fixture(fixture_path)
    device_batch = runtime.step_executor.batch_adapter.move_to_device(
        batch,
        runtime.strategy.device,
    )
    runtime.model.train()
    runtime.strategy.set_gradient_sync(runtime.model, enabled=True)
    with runtime.strategy.autocast_context():
        result = runtime.step_executor.forward_train(device_batch)
    runtime.strategy.backward(result.loss)
    scenario = training_scenarios_for(method)[0]
    training_report = _training_scenario_report(
        runtime=runtime,
        result=result,
        batch=batch,
        method=method,
        scenario_id=scenario_id,
        fixture_id=scenario.fixture_id,
        seed=seed,
    )
    _assert_training_scenario(
        training_report,
        method=method,
        scenario=scenario,
    )
    optimizer_report = _execute_characterization_optimizer_step(
        runtime=runtime,
        scenario_id=scenario_id,
    )
    _assert_optimizer_step(
        optimizer_report,
        method=method,
    )
    runtime.train_state.global_step += 1
    runtime.train_state.seen_batches += 1
    runtime.train_state.optimizer_step += 1
    runtime.strategy.zero_grad(runtime.optimizer)
    report = {
        "scenario": training_report,
        "optimizer_step": optimizer_report,
        "train_state": _stable_train_state(runtime.train_state),
    }
    del result
    del device_batch
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _distributed_runtime_state_digest(runtime: TrainingRuntime) -> dict[str, Any]:
    return {
        "model": _distributed_module_digest(runtime.model),
        "optimizer": _distributed_optimizer_digest(
            runtime.model,
            runtime.optimizer,
        ),
        "scheduler": _stable_json_value(runtime.scheduler.state_dict()),
        "strategy": _stable_json_value(runtime.strategy.state_dict()),
        "train_state": _stable_train_state(runtime.train_state),
    }


def _distributed_module_digest(model: torch.nn.Module) -> dict[str, Any]:
    digest = hashlib.sha256()
    component_digests: dict[str, Any] = {}
    component_counts: Counter[str] = Counter()
    component_bytes: Counter[str] = Counter()
    tensor_count = 0
    total_bytes = 0
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in sorted(values, key=lambda item: item[0]):
            component = _gradient_group(name) if kind == "parameter" else "buffers"
            component_digest = component_digests.setdefault(
                component,
                hashlib.sha256(),
            )
            tensor_count += 1
            component_counts[component] += 1
            tensor_bytes = _update_tensor_digest(
                digest,
                key=f"{kind}:{name}",
                tensor=value,
                additional_digests=(component_digest,),
            )
            component_bytes[component] += tensor_bytes
            total_bytes += tensor_bytes
    return _gather_digest_record(
        {
            "sha256": digest.hexdigest(),
            "tensor_count": tensor_count,
            "total_bytes": total_bytes,
            "components": {
                component: {
                    "sha256": component_digests[component].hexdigest(),
                    "tensor_count": component_counts[component],
                    "total_bytes": component_bytes[component],
                }
                for component in sorted(component_digests)
            },
        }
    )


def _distributed_optimizer_digest(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    tensor_count = 0
    total_bytes = 0
    state_records: list[tuple[str, dict[str, Any]]] = []
    for fallback_index, (parameter, state) in enumerate(optimizer.state.items()):
        name = parameter_names.get(id(parameter), f"<parameter:{fallback_index}>")
        state_records.append((name, state))
    for parameter_name, state in sorted(state_records, key=lambda item: item[0]):
        for key, value in sorted(state.items(), key=lambda item: str(item[0])):
            record_key = f"state:{parameter_name}:{key}"
            if isinstance(value, torch.Tensor):
                tensor_count += 1
                total_bytes += _update_tensor_digest(
                    digest,
                    key=record_key,
                    tensor=value,
                )
            else:
                _update_json_digest(
                    digest,
                    key=record_key,
                    value=value,
                )
    parameter_groups = []
    for group in optimizer.param_groups:
        parameter_groups.append(
            {
                str(key): (
                    [parameter_names.get(id(item), "<unnamed>") for item in value]
                    if key == "params"
                    else _stable_json_value(value)
                )
                for key, value in sorted(group.items(), key=lambda item: str(item[0]))
            }
        )
    _update_json_digest(
        digest,
        key="parameter_groups",
        value=parameter_groups,
    )
    return _gather_digest_record(
        {
            "sha256": digest.hexdigest(),
            "tensor_count": tensor_count,
            "total_bytes": total_bytes,
            "state_entries": len(optimizer.state),
        }
    )


def _update_tensor_digest(
    digest,
    *,
    key: str,
    tensor: torch.Tensor,
    additional_digests: tuple[Any, ...] = (),
) -> int:
    local = _local_tensor(tensor.detach())
    metadata = {
        "key": key,
        "dtype": str(local.dtype),
        "shape": [int(value) for value in local.shape],
        "requires_grad": bool(tensor.requires_grad),
    }
    digest_targets = (digest, *additional_digests)
    encoded_metadata = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    for target in digest_targets:
        target.update(encoded_metadata)
        target.update(b"\0")
    if local.numel() == 0:
        return 0
    flat = local.reshape(-1)
    elements_per_chunk = max(
        1,
        STATE_DIGEST_CHUNK_BYTES // max(1, int(local.element_size())),
    )
    total_bytes = 0
    for start in range(0, int(flat.numel()), elements_per_chunk):
        chunk = flat[start : start + elements_per_chunk].contiguous()
        raw = chunk.view(torch.uint8).cpu().numpy().tobytes()
        for target in digest_targets:
            target.update(raw)
        total_bytes += len(raw)
    return total_bytes


def _update_json_digest(digest, *, key: str, value: Any) -> None:
    payload = {
        "key": key,
        "value": _stable_json_value(value),
    }
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\0")


def _gather_digest_record(local: dict[str, Any]) -> dict[str, Any]:
    record = {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        **local,
    }
    if not dist.is_initialized():
        return {"per_rank": [record]}
    gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, record)
    return {"per_rank": [item for item in gathered if item is not None]}


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_train_state(train_state: TrainState) -> dict[str, Any]:
    payload = train_state.state_dict()
    payload.pop("last_checkpoint_path", None)
    payload.pop("resume_source", None)
    return _stable_json_value(payload)


def _checkpoint_artifact_report(checkpoint_dir: Path) -> dict[str, Any]:
    required_files = (
        "full_training_state.pt",
        "model_state.pt",
        "resolved_config.yaml",
        "train_state.json",
        ".checkpoint_payload_complete",
        ".checkpoint_complete",
    )
    missing = [name for name in required_files if not (checkpoint_dir / name).is_file()]
    if missing:
        raise AssertionError(
            f"Full-state checkpoint is incomplete; missing artifacts {missing}."
        )
    return {
        "step": int(checkpoint_dir.name.rsplit("_", maxsplit=1)[-1]),
        "files": {
            name: {
                "size_bytes": int((checkpoint_dir / name).stat().st_size),
                "nonempty": (checkpoint_dir / name).stat().st_size > 0,
            }
            for name in required_files
        },
    }


def _runtime_state_digest_differences(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    return compare_characterization_reports(
        expected,
        actual,
        tolerance=EXACT_COMPARISON_TOLERANCE,
        path="runtime_state",
    )


def _runtime_state_contract_differences(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    return compare_characterization_reports(
        _state_digest_contract(expected),
        _state_digest_contract(actual),
        tolerance=EXACT_COMPARISON_TOLERANCE,
        path="runtime_state_contract",
    )


def _resume_update_differences(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    def tolerance_for_path(path: str) -> ComparisonTolerance | None:
        if (
            ".optimizer_step.distributed_numeric.parameter_groups." in path
            and ".delta." in path
        ):
            return RESUME_DISTRIBUTED_AGGREGATE_DELTA_TOLERANCE
        if ".optimizer_step.distributed_numeric.parameter_groups." in path:
            return RESUME_DISTRIBUTED_AGGREGATE_TOLERANCE
        if (
            ".gradients." in path
            or ".optimizer_step.distributed_numeric.parameter_probes" in path
            or path.endswith(".optimizer_step.distributed_numeric.grad_norm")
        ):
            return RESUME_GRADIENT_TOLERANCE
        return None

    return compare_characterization_reports(
        _resume_comparison_projection(expected),
        _resume_comparison_projection(actual),
        tolerance=EXACT_COMPARISON_TOLERANCE,
        tolerance_for_path=tolerance_for_path,
        path="continuation",
    )


def _resume_comparison_projection(value: Any) -> Any:
    if isinstance(value, dict):
        projected = {
            key: _resume_comparison_projection(item)
            for key, item in value.items()
            if key
            not in {
                "cuda_peak_memory_bytes",
                "nonzero_elements",
                # Probe selection intentionally follows the largest local
                # gradient. Low-bit GPU reduction differences can change the
                # winner without changing the bounded whole-group result.
                "parameter_probes",
            }
        }
        nonzero_elements = value.get("nonzero_elements")
        parameter_elements = value.get("parameter_elements")
        if (
            isinstance(nonzero_elements, int)
            and isinstance(parameter_elements, int)
            and parameter_elements > 0
        ):
            projected["nonzero_density"] = round(
                nonzero_elements / parameter_elements,
                6,
            )
        return projected
    if isinstance(value, list):
        return [_resume_comparison_projection(item) for item in value]
    return value


def _resume_update_report_contract(
    report: dict[str, Any],
    *,
    preserve_output_values: bool,
) -> dict[str, Any]:
    scenario = report["scenario"]
    optimizer_step = report["optimizer_step"]
    scenario_contract = {
        key: value
        for key, value in scenario.items()
        if key not in {"cuda_peak_memory_bytes", "gradients"}
    }
    if not preserve_output_values:
        scenario_contract["outputs"] = {
            name: {
                key: fingerprint[key]
                for key in (
                    "dtype",
                    "finite_count",
                    "local_shape",
                    "numel",
                    "shape",
                )
            }
            for name, fingerprint in scenario["outputs"].items()
        }
    return {
        "scenario": scenario_contract,
        "gradient_contract": {
            group: {
                key: value
                for key, value in summary.items()
                if key
                in {
                    "parameter_tensors",
                    "parameter_elements",
                    "finite_elements",
                }
            }
            for group, summary in scenario["gradients"].items()
        },
        "optimizer_step": {
            key: value
            for key, value in optimizer_step.items()
            if key != "distributed_numeric"
        },
        "train_state": report["train_state"],
    }


def _state_digest_contract(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": _digest_shape_contract(state["model"]),
        "optimizer": _digest_shape_contract(state["optimizer"]),
        "scheduler": state["scheduler"],
        "strategy": state["strategy"],
        "train_state": state["train_state"],
    }


def _digest_shape_contract(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "per_rank": [
            {
                **{
                    key: item[key]
                    for key in (
                        "rank",
                        "tensor_count",
                        "total_bytes",
                        "state_entries",
                    )
                    if key in item
                },
                **(
                    {
                        "components": {
                            name: {
                                "tensor_count": component["tensor_count"],
                                "total_bytes": component["total_bytes"],
                            }
                            for name, component in item["components"].items()
                        }
                    }
                    if "components" in item
                    else {}
                ),
            }
            for item in value["per_rank"]
        ]
    }


def _gradient_group(name: str) -> str:
    if "generalist_mode" in name:
        return "generalist_mode_token"
    if ".action_block." in name or "action_expert" in name:
        return "action_expert"
    if ".video_block." in name or "visual_tower.core" in name:
        return "video_backbone"
    if "action_decoder" in name:
        return "action_decoder"
    return "other_trainable"


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None
    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _assert_training_scenario(report: dict[str, Any], *, method, scenario) -> None:
    metrics = report["metrics"]
    loss = float(metrics["loss"])
    if not np.isfinite(loss) or loss <= 0.0:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} produced invalid loss {loss}."
        )
    output_finite = all(
        int(fingerprint["finite_count"]) == int(fingerprint["numel"])
        for fingerprint in report["outputs"].values()
    )
    if not output_finite:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} produced non-finite outputs."
        )
    gradients = report["gradients"]
    nonfinite_groups = [
        name
        for name, group in gradients.items()
        if int(group["finite_elements"]) != int(group["parameter_elements"])
    ]
    if nonfinite_groups:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} produced non-finite "
            f"gradients in groups {nonfinite_groups}."
        )
    required_groups = {"action_expert", "video_backbone"}
    if method.mode_token:
        required_groups.add("generalist_mode_token")
    missing_gradient_groups = sorted(
        group
        for group in required_groups
        if group not in gradients or int(gradients[group]["nonzero_elements"]) <= 0
    )
    if missing_gradient_groups:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} produced no nonzero "
            f"gradients for required groups {missing_gradient_groups}."
        )
    if scenario.mode is None:
        return
    if report["mode"] != scenario.mode.value:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} exercised mode "
            f"{report['mode']!r}, expected {scenario.mode.value!r}."
        )
    if report["source"] != scenario.source:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} exercised source "
            f"{report['source']!r}, expected {scenario.source!r}."
        )
    if scenario.mode == GJDTrainingMode.JOINT:
        return
    sample = report["input"]
    expected_layout = {
        "history_frames": 1,
        "loss_frame_start": 1,
        "singleton_chunk_frame": 0,
        "chunk_origin_frame": 1,
    }
    mismatched_layout = {
        field: (sample.get(field), expected)
        for field, expected in expected_layout.items()
        if sample.get(field) != expected
    }
    if mismatched_layout:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} violated the GJD "
            f"target-only t0 contract: {mismatched_layout}."
        )
    sampled_chunk_size = int(sample["sampled_chunk_size"])
    if not 1 <= sampled_chunk_size <= 4:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} sampled conditional "
            f"chunk size {sampled_chunk_size}, expected 1..4."
        )
    video_shape = sample["video_latents"]["shape"]
    action_shape = sample["actions"]["shape"]
    if int(sample["loss_frame_end"]) != int(video_shape[2]):
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} loss range does not "
            "cover the complete target-only latent sequence."
        )
    if (
        int(action_shape[1])
        != int(video_shape[2]) * DEFAULT_INFERENCE_CONTRACT.action_per_frame
    ):
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} does not preserve "
            f"{DEFAULT_INFERENCE_CONTRACT.action_per_frame} actions per latent "
            "frame."
        )
    action_mask = sample.get("action_mask")
    if action_mask is None or action_mask["shape"] != action_shape:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} requires an action mask "
            "with the same shape as actions."
        )
    if int(action_mask["finite_count"]) != int(action_mask["numel"]):
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} has a non-finite action mask."
        )
    if float(action_mask["min"]) < 0.0 or float(action_mask["max"]) > 1.0:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} has action-mask values "
            "outside [0, 1]."
        )
    if float(sample["leading_action_mask_sum"]) != 0.0:
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} supervises the dummy "
            "four-action group aligned with t0."
        )
    expected_future_mask_sum = int(action_mask["numel"]) - (
        int(action_shape[0])
        * DEFAULT_INFERENCE_CONTRACT.action_per_frame
        * int(action_shape[2])
    )
    if float(sample["future_action_mask_sum"]) != float(expected_future_mask_sum):
        raise AssertionError(
            f"{method.asset_id}/{scenario.scenario_id} does not mark every "
            "post-t0 action element as valid."
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real-checkpoint DualExpert characterization worker."
    )
    parser.add_argument(
        "--phase",
        choices=("training", "inference", "cache_rollover", "resume"),
        required=True,
    )
    parser.add_argument("--asset-id", choices=CHARACTERIZATION_ASSET_IDS, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional staged model_state.pt override for the selected asset.",
    )
    parser.add_argument(
        "--base-model-root",
        type=Path,
        help="Optional staged LingBot base-model directory.",
    )
    parser.add_argument(
        "--video-transformer-root",
        type=Path,
        help="Optional staged video-only transformer directory.",
    )
    parser.add_argument(
        "--checkpoint-config",
        type=Path,
        help="Optional staged resolved_config.yaml override for the selected asset.",
    )
    parser.add_argument(
        "--allow-checkpoint-provenance-mismatch",
        action="store_true",
        help="Run a historical checkpoint even when its saved contract is stale.",
    )
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=WORKER_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    method = METHOD_BY_ASSET_ID[resolve_characterization_asset_id(args.asset_id)]
    assets = load_characterization_assets(args.assets)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint override does not exist: {checkpoint}")
        resolved_config = (
            assets.checkpoint_config_for(method.asset_id)
            if args.checkpoint_config is None
            else args.checkpoint_config.expanduser().resolve()
        )
        if not resolved_config.is_file():
            raise FileNotFoundError(
                f"Checkpoint config override does not exist: {resolved_config}"
            )
        assets = replace(
            assets,
            checkpoints={
                **assets.checkpoints,
                method.asset_id: replace(
                    assets.checkpoints[method.asset_id],
                    model_state=checkpoint,
                    resolved_config=resolved_config,
                ),
            },
        )
    static_asset_overrides: dict[str, Path] = {}
    for field_name, value in (
        ("base_model_root", args.base_model_root),
        ("video_transformer_root", args.video_transformer_root),
    ):
        if value is None:
            continue
        resolved = value.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Static asset override does not exist: {resolved}")
        static_asset_overrides[field_name] = resolved
    if static_asset_overrides:
        assets = replace(assets, **static_asset_overrides)
    if args.phase in {"training", "resume"}:
        if args.phase == "resume":
            run_full_state_resume_characterization(
                method=method,
                assets=assets,
                fixture_root=args.fixture_root.expanduser().resolve(),
                output_path=args.output.expanduser().resolve(),
                seed=args.seed,
                allow_provenance_mismatch=(args.allow_checkpoint_provenance_mismatch),
            )
            return
        run_training_characterization(
            method=method,
            assets=assets,
            fixture_root=args.fixture_root.expanduser().resolve(),
            output_path=args.output.expanduser().resolve(),
            seed=args.seed,
            allow_provenance_mismatch=args.allow_checkpoint_provenance_mismatch,
        )
    else:
        run_inference_characterization(
            method=method,
            assets=assets,
            fixture_root=args.fixture_root.expanduser().resolve(),
            output_path=args.output.expanduser().resolve(),
            seed=args.seed,
            allow_provenance_mismatch=args.allow_checkpoint_provenance_mismatch,
            chunk_count=(
                CACHE_ROLLOVER_CHARACTERIZATION_CHUNKS
                if args.phase == "cache_rollover"
                else INFERENCE_CHARACTERIZATION_CHUNKS
            ),
            require_cache_rollover=args.phase == "cache_rollover",
        )


if __name__ == "__main__":
    main()
