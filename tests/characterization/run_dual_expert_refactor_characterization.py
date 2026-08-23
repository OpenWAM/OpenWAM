from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import open_wam
from open_wam.configs import canonical_config_stem

from .dual_expert_refactor_artifacts import (
    EXACT_COMPARISON_TOLERANCE,
    ComparisonTolerance,
    compare_characterization_reports,
    sha256_file,
    write_json_atomic,
)
from .dual_expert_refactor_contract import (
    CHARACTERIZATION_ASSET_IDS,
    CACHE_ROLLOVER_ASSET_IDS,
    EXACT_CHECKPOINT_METHODS,
    FULL_STATE_RESUME_ASSET_ID,
    METHOD_BY_ASSET_ID,
    load_characterization_assets,
    resolve_characterization_asset_id,
)
from .dual_expert_refactor_end_to_end import (
    run_libero_rollout,
    run_training_cli_smoke,
)
from .dual_expert_refactor_fixtures import build_all_characterization_fixtures
from .dual_expert_refactor_provenance import (
    apply_checkpoint_provenance_policy,
    build_checkpoint_source_contract_config,
    canonical_checkpoint_contract_value,
    checkpoint_provenance_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASES = ("training", "inference")
INFRASTRUCTURE_PHASES = ("cache_rollover", "resume")
ALL_PHASES = DEFAULT_PHASES + INFRASTRUCTURE_PHASES
VOLATILE_REPORT_KEYS = frozenset(
    {
        "checkpoint",
        "resolved_config_path",
        "cuda_peak_memory_bytes",
    }
)
NONZERO_GRADIENT_DENSITY_PRECISION = 6
TENSOR_FINGERPRINT_REDUCTION_KEYS = frozenset({"l1", "l2", "mean", "std"})
CONTENT_PROBE_CHUNK_BYTES = 64 * 1024
CONTENT_PROBE_COUNT = 17
DISTRIBUTED_AGGREGATE_TOLERANCE = ComparisonTolerance(
    absolute=0.25,
    relative=0.0,
)
# Delta fields subtract independently reduced before/after aggregates, so their
# worst-case absolute error is the sum of both aggregate error bounds.
DISTRIBUTED_AGGREGATE_DELTA_TOLERANCE = ComparisonTolerance(
    absolute=2 * DISTRIBUTED_AGGREGATE_TOLERANCE.absolute,
    relative=0.0,
)
# Forward tensors and losses remain exact, but changing NCCL host transport can
# shift BF16 FSDP gradient aggregates slightly through reduction order. Keep
# this bound just above the characterized 0.66% repeated-run delta.
DISTRIBUTED_GRADIENT_TOLERANCE = ComparisonTolerance(
    absolute=5e-4,
    relative=7e-3,
)
RESUME_POST_UPDATE_METRIC_TOLERANCE = ComparisonTolerance(
    # Removing an unused parameter changes FSDP's flat-parameter reduction
    # boundaries. Bound the resulting second-update drift to one BF16 quantum
    # at the observed continuation-loss scale.
    absolute=2**-13,
    relative=0.0,
)

# The retired VisualTower decoder contributed exactly these two sharded FP32
# tensors to the old full-state model contract. This signature is deliberately
# narrow: it canonicalizes only the known pruning delta while preserving the
# six retained frontend tensors that shared the old ``other_trainable`` group.
_LEGACY_RETIRED_DECODER_LOCAL_TENSOR_COUNT = 2
_LEGACY_RETIRED_DECODER_LOCAL_BYTES = 9_440_256
_LEGACY_OTHER_TRAINABLE_SIGNATURE = {
    "tensor_count": 8,
    "total_bytes": 10_070_160,
}

# Golden report schema v1 predates the public architecture/config rename. Only
# symbolic metadata is normalized here. Tensor fingerprints, shapes, probes,
# losses, gradients, optimizer deltas, and rollout values remain unchanged.
_REPORT_KEY_ALIASES = {
    "legacy_split_cache_ready": "block_restore_ready",
    "legacy_split_cache_required": "block_restore_required",
    "legacy_split_cache_restored_this_call": "block_restore_performed",
    "requires_legacy_block_restore": "requires_block_restore",
    "method_family": "architecture",
    "mot_action_cond_tokens": "dual_expert_action_cond_tokens",
    "mot_action_context_invalid_tokens": (
        "dual_expert_action_context_invalid_tokens"
    ),
    "mot_action_only_rollout": "dual_expert_action_only_rollout",
    "mot_attention_focus": "dual_expert_attention_focus",
    "mot_cache_debug": "dual_expert_cache_debug",
    "mot_diagnostic_zero_current_action_noise": (
        "dual_expert_diagnostic_zero_current_action_noise"
    ),
    "mot_diagnostic_zero_current_video_noise": (
        "dual_expert_diagnostic_zero_current_video_noise"
    ),
    "mot_first_step_bootstrap": "dual_expert_first_step_bootstrap",
    "mot_generalist_mode_text_token": "dual_expert_generalist_mode_text_token",
    "mot_generalist_mode_text_token_count": (
        "dual_expert_generalist_mode_text_token_count"
    ),
    "mot_generalist_rollout_mode": "dual_expert_generalist_rollout_mode",
    "mot_gjd_action_route": "dual_expert_gjd_action_route",
    "mot_history_anchor_frames": "dual_expert_history_anchor_frames",
    "mot_history_frames": "dual_expert_history_frames",
    "mot_infer_artifacts": "dual_expert_infer_artifacts",
    "mot_invalid_startup_action_tokens": (
        "dual_expert_invalid_startup_action_tokens"
    ),
    "mot_packed_history_debug": "dual_expert_packed_history_debug",
    "mot_video_prefix_frames": "dual_expert_video_prefix_frames",
    "parallel_sequence_contract": "sequence_contract",
    "mot_generalist_training_mode_probs": "generalist_denoising_mode_probs",
    "policy_variant.parallel_sequence_contract": (
        "policy_variant.sequence_contract"
    ),
    "policy_variant.mot_generalist_training_mode_probs": (
        "policy_variant.generalist_denoising_mode_probs"
    ),
}
_REPORT_STRING_ALIASES = {
    "legacy_split_cache": "split_cache",
    "native_packed_coupling": "packed_coupling",
    "split_cache_non_joint": "split_cache",
    "MoTRuntimeState": "DualExpertRuntimeState",
    "MoTActionCache": "DualExpertActionCache",
    "MoTActionLayerCache": "DualExpertActionLayerCache",
    "MoTVideoCache": "DualExpertVideoCache",
    "MoTVideoLayerCache": "DualExpertVideoLayerCache",
    **_REPORT_KEY_ALIASES,
}
def _assert_checkout_import_provenance(
    *,
    package_file: Path | None = None,
) -> None:
    expected_package_root = (REPO_ROOT / "src" / "open_wam").resolve()
    imported_package_file = Path(
        open_wam.__file__ if package_file is None else package_file
    ).resolve()
    imported_package_root = imported_package_file.parent
    if imported_package_root == expected_package_root:
        return
    raise RuntimeError(
        "Characterization checkout/import mismatch: runner checkout "
        f"{REPO_ROOT} expects open_wam under {expected_package_root}, but it was "
        f"imported from {imported_package_root}. Activate/install this checkout "
        f"or set PYTHONPATH={REPO_ROOT / 'src'}:{REPO_ROOT}."
    )


def record_characterization(args: argparse.Namespace) -> None:
    assets = load_characterization_assets(args.assets)
    fixture_root = args.fixture_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = _selected_asset_ids(args.asset_id)
    phases = tuple(args.phase or DEFAULT_PHASES)
    _validate_phase_asset_selection(selected=selected, phases=phases)
    _preflight_checkpoint_provenance(
        assets=assets,
        selected=selected,
        allow_mismatch=args.allow_checkpoint_provenance_mismatch,
    )

    base_model_root = assets.base_model_root
    video_transformer_root = assets.video_transformer_root
    stage_root = (
        None if args.stage_root is None else args.stage_root.expanduser().resolve()
    )
    if stage_root is not None:
        stage_root.mkdir(parents=True, exist_ok=True)
        base_model_root = _stage_directory(
            assets.base_model_root,
            stage_root / "static" / "base_model",
            refresh=args.refresh_stage,
        )
        video_transformer_root = _stage_directory(
            assets.video_transformer_root,
            stage_root / "static" / "video_transformer",
            refresh=args.refresh_stage,
        )

    completed: list[dict[str, Any]] = []
    for asset_id in selected:
        checkpoint = assets.checkpoint_for(asset_id)
        checkpoint_config = assets.checkpoint_config_for(asset_id)
        if stage_root is not None:
            checkpoint = _stage_file(
                checkpoint,
                stage_root / "checkpoints" / asset_id / "model_state.pt",
                refresh=args.refresh_stage,
            )
            checkpoint_config = _stage_file(
                checkpoint_config,
                stage_root / "checkpoints" / asset_id / "resolved_config.yaml",
                refresh=args.refresh_stage,
            )
        for phase in phases:
            report_path = output_root / f"{asset_id}.{phase}.json"
            command, environment = _worker_command(
                phase=phase,
                asset_id=asset_id,
                assets_path=args.assets.expanduser().resolve(),
                fixture_root=fixture_root,
                report_path=report_path,
                checkpoint=checkpoint,
                checkpoint_config=checkpoint_config,
                base_model_root=base_model_root,
                video_transformer_root=video_transformer_root,
                training_world_size=args.training_world_size,
                cuda_devices=args.cuda_devices,
                fsdp_cpu_offload=args.fsdp_cpu_offload,
                disable_nccl_shm=args.disable_nccl_shm,
                allow_provenance_mismatch=args.allow_checkpoint_provenance_mismatch,
            )
            subprocess.run(command, check=True, env=environment)
            completed.append(
                {
                    "asset_id": asset_id,
                    "phase": phase,
                    "report": str(report_path),
                }
            )

    write_json_atomic(
        output_root / "run_summary.json",
        {
            "schema_version": 1,
            "assets": str(args.assets.expanduser().resolve()),
            "fixture_root": str(fixture_root),
            "stage_root": None if stage_root is None else str(stage_root),
            "training_world_size": args.training_world_size,
            "cuda_devices": args.cuda_devices,
            "fsdp_cpu_offload": args.fsdp_cpu_offload,
            "disable_nccl_shm": args.disable_nccl_shm,
            "allow_checkpoint_provenance_mismatch": (
                args.allow_checkpoint_provenance_mismatch
            ),
            "completed": completed,
        },
    )
def verify_characterization(args: argparse.Namespace) -> None:
    actual_root = args.actual_root.expanduser().resolve()
    golden_root = args.golden_root.expanduser().resolve()
    selected = _selected_asset_ids(args.asset_id)
    phases = tuple(args.phase or DEFAULT_PHASES)
    _validate_phase_asset_selection(selected=selected, phases=phases)
    differences: list[str] = []
    for asset_id in selected:
        for phase in phases:
            filename = f"{asset_id}.{phase}.json"
            expected = _load_json(golden_root / filename)
            actual = _load_json(actual_root / filename)
            differences.extend(
                compare_characterization_reports(
                    _comparison_projection(expected),
                    _comparison_projection(actual),
                    tolerance=EXACT_COMPARISON_TOLERANCE,
                    tolerance_for_path=_numeric_tolerance_resolver(
                        ComparisonTolerance(
                            absolute=args.gradient_absolute_tolerance,
                            relative=args.gradient_relative_tolerance,
                        )
                    ),
                    path=filename,
                )
            )
    if differences:
        preview = "\n".join(f"- {item}" for item in differences[:100])
        remainder = len(differences) - min(100, len(differences))
        suffix = "" if remainder <= 0 else f"\n- ... {remainder} more differences"
        raise AssertionError(
            f"DualExpert characterization changed in {len(differences)} places:\n"
            f"{preview}{suffix}"
        )
    print(f"Verified {len(selected) * len(phases)} DualExpert characterization reports.")


def initialize_characterization_goldens(args: argparse.Namespace) -> None:
    actual_root = args.actual_root.expanduser().resolve()
    selected = _selected_asset_ids(args.asset_id)
    phases = tuple(args.phase or DEFAULT_PHASES)
    _validate_phase_asset_selection(selected=selected, phases=phases)
    _initialize_golden_files(
        (
            (
                actual_root / f"{asset_id}.{phase}.json",
                f"{asset_id}.{phase}.json",
            )
            for asset_id in selected
            for phase in phases
        ),
        golden_root=args.golden_root,
    )
    print(
        "Initialized "
        f"{len(selected) * len(phases)} immutable characterization goldens."
    )


def freeze_fixtures(args: argparse.Namespace) -> None:
    generated = build_all_characterization_fixtures(
        assets=load_characterization_assets(args.assets),
        output_root=args.output_root,
        seed=args.seed,
    )
    for fixture_id, path in sorted(generated.items()):
        print(f"{fixture_id}: {path}")


def replay_fixtures(args: argparse.Namespace) -> None:
    expected_root = args.fixture_root.expanduser().resolve()
    if args.output_root is None:
        with tempfile.TemporaryDirectory(
            prefix="open_wam_dual_expert_fixture_replay_"
        ) as temporary:
            _run_fixture_replay(
                assets=args.assets,
                expected_root=expected_root,
                output_root=Path(temporary),
                seed=args.seed,
            )
        return
    _run_fixture_replay(
        assets=args.assets,
        expected_root=expected_root,
        output_root=args.output_root.expanduser().resolve(),
        seed=args.seed,
    )


def training_cli_smoke(args: argparse.Namespace) -> None:
    assets = load_characterization_assets(args.assets)
    asset_id = resolve_characterization_asset_id(args.asset_id)
    selected = (asset_id,)
    _preflight_checkpoint_provenance(
        assets=assets,
        selected=selected,
        allow_mismatch=args.allow_checkpoint_provenance_mismatch,
    )
    report = run_training_cli_smoke(
        method=METHOD_BY_ASSET_ID[asset_id],
        assets=assets,
        output_root=args.output_root.expanduser().resolve(),
        cuda_devices=args.cuda_devices,
        world_size=args.training_world_size,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
        disable_nccl_shm=args.disable_nccl_shm,
    )
    print(
        f"Training CLI smoke passed for {args.asset_id}: step={report['final_step']}."
    )


def record_libero_rollouts(args: argparse.Namespace) -> None:
    assets = load_characterization_assets(args.assets)
    selected = _selected_asset_ids(args.asset_id)
    _preflight_checkpoint_provenance(
        assets=assets,
        selected=selected,
        allow_mismatch=args.allow_checkpoint_provenance_mismatch,
    )
    devices = tuple(
        item.strip() for item in args.cuda_devices.split(",") if item.strip()
    )
    if not devices:
        raise ValueError("--cuda-devices must contain at least one device.")
    output_root = args.output_root.expanduser().resolve()
    libero_repo_root = (
        None
        if args.libero_repo_root is None
        else args.libero_repo_root.expanduser().resolve()
    )

    assignments = [
        tuple(selected[index :: len(devices)]) for index in range(len(devices))
    ]

    def run_device_queue(
        device: str,
        asset_ids: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for asset_id in asset_ids:
            reports.append(
                run_libero_rollout(
                    method=METHOD_BY_ASSET_ID[asset_id],
                    assets=assets,
                    output_root=output_root,
                    cuda_device=device,
                    task_id=args.task_id,
                    episode_idx=args.episode_idx,
                    seed=args.seed,
                    libero_repo_root=libero_repo_root,
                    max_timestep=args.max_timestep,
                    max_chunks=args.max_chunks,
                )
            )
        return reports

    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_device_queue, device, assignment)
            for device, assignment in zip(devices, assignments, strict=True)
            if assignment
        ]
        for future in futures:
            reports.extend(future.result())
    summary_path = output_root / "rollout_summary.json"
    reports_by_asset: dict[str, dict[str, Any]] = {}
    asset_manifests = {str(args.assets.expanduser().resolve())}
    if summary_path.is_file():
        previous_summary = _load_json(summary_path)
        previous_reports = (
            previous_summary.get("reports", [])
            if isinstance(previous_summary, dict)
            else []
        )
        if isinstance(previous_reports, list):
            reports_by_asset.update(
                {
                    str(report["asset_id"]): report
                    for report in previous_reports
                    if isinstance(report, dict) and "asset_id" in report
                }
            )
        if isinstance(previous_summary, dict):
            previous_manifests = previous_summary.get("asset_manifests")
            if isinstance(previous_manifests, list):
                asset_manifests.update(str(item) for item in previous_manifests)
            elif isinstance(previous_summary.get("assets"), str):
                asset_manifests.add(str(previous_summary["assets"]))
    reports_by_asset.update({str(report["asset_id"]): report for report in reports})
    merged_reports = [
        reports_by_asset[asset_id] for asset_id in sorted(reports_by_asset)
    ]
    write_json_atomic(
        summary_path,
        {
            "schema_version": 1,
            "asset_manifests": sorted(asset_manifests),
            "task_id": args.task_id,
            "episode_idx": args.episode_idx,
            "seed": args.seed,
            "reports": merged_reports,
        },
    )
def verify_libero_rollouts(args: argparse.Namespace) -> None:
    selected = _selected_asset_ids(args.asset_id)
    differences: list[str] = []
    actual_root = args.actual_root.expanduser().resolve()
    golden_root = args.golden_root.expanduser().resolve()
    for asset_id in selected:
        expected = _load_json(golden_root / f"{asset_id}.rollout.json")
        actual = _load_json(actual_root / asset_id / "rollout_report.json")
        differences.extend(
            compare_characterization_reports(
                _comparison_projection(expected),
                _comparison_projection(actual),
                tolerance=EXACT_COMPARISON_TOLERANCE,
                path=f"{asset_id}.rollout.json",
            )
        )
    if differences:
        preview = "\n".join(f"- {item}" for item in differences[:100])
        raise AssertionError(f"LIBERO rollout characterization changed:\n{preview}")
    print(f"Verified {len(selected)} LIBERO rollout reports exactly.")


def initialize_libero_rollout_goldens(args: argparse.Namespace) -> None:
    actual_root = args.actual_root.expanduser().resolve()
    selected = _selected_asset_ids(args.asset_id)
    _initialize_golden_files(
        (
            (
                actual_root / asset_id / "rollout_report.json",
                f"{asset_id}.rollout.json",
            )
            for asset_id in selected
        ),
        golden_root=args.golden_root,
    )
    print(f"Initialized {len(selected)} immutable LIBERO rollout goldens.")


def _run_fixture_replay(
    *,
    assets: Path,
    expected_root: Path,
    output_root: Path,
    seed: int,
) -> None:
    if output_root == expected_root:
        raise ValueError("Fixture replay output must differ from the frozen root.")
    build_all_characterization_fixtures(
        assets=load_characterization_assets(assets),
        output_root=output_root,
        seed=seed,
    )
    differences = compare_fixture_directories(expected_root, output_root)
    if differences:
        preview = "\n".join(f"- {item}" for item in differences[:100])
        raise AssertionError(
            f"Deterministic fixture replay diverged from the frozen data:\n{preview}"
        )
    print(
        "Replayed "
        f"{len(_fixture_file_hashes(expected_root))} fixture files byte-for-byte."
    )


def compare_fixture_directories(
    expected_root: Path,
    actual_root: Path,
) -> list[str]:
    expected = _fixture_file_hashes(expected_root)
    actual = _fixture_file_hashes(actual_root)
    differences: list[str] = []
    for relative_path in sorted(expected.keys() - actual.keys()):
        differences.append(f"missing fixture file {relative_path}")
    for relative_path in sorted(actual.keys() - expected.keys()):
        differences.append(f"unexpected fixture file {relative_path}")
    for relative_path in sorted(expected.keys() & actual.keys()):
        if expected[relative_path] != actual[relative_path]:
            differences.append(
                f"{relative_path}: expected sha256 {expected[relative_path]}, "
                f"got {actual[relative_path]}"
            )
    return differences


def _fixture_file_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Fixture root does not exist: {root}")
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    }


def _preflight_checkpoint_provenance(
    *,
    assets,
    selected: tuple[str, ...],
    allow_mismatch: bool,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for asset_id in selected:
        method = METHOD_BY_ASSET_ID[asset_id]
        report = apply_checkpoint_provenance_policy(
            checkpoint_provenance_report(
                method=method,
                expected_config=build_checkpoint_source_contract_config(method),
                resolved_config_path=assets.checkpoint_config_for(asset_id),
            ),
            accepted_origin_mismatch_fields=(
                assets.checkpoints[asset_id].accepted_origin_mismatch_fields
            ),
        )
        reports[asset_id] = report
    mismatched = {
        asset_id: report
        for asset_id, report in reports.items()
        if not bool(report["accepted_for_characterization"])
    }
    if mismatched and not allow_mismatch:
        details: list[str] = []
        for asset_id, report in mismatched.items():
            fields = ", ".join(str(item["field"]) for item in report["mismatches"][:6])
            remainder = len(report["mismatches"]) - min(
                6,
                len(report["mismatches"]),
            )
            suffix = "" if remainder <= 0 else f", plus {remainder} more"
            details.append(f"{asset_id}: {fields}{suffix}")
        raise AssertionError(
            "Checkpoint source-contract preflight failed before staging:\n- "
            + "\n- ".join(details)
        )
    return reports


def _worker_command(
    *,
    phase: str,
    asset_id: str,
    assets_path: Path,
    fixture_root: Path,
    report_path: Path,
    checkpoint: Path,
    checkpoint_config: Path,
    base_model_root: Path,
    video_transformer_root: Path,
    training_world_size: int,
    cuda_devices: str,
    fsdp_cpu_offload: bool,
    disable_nccl_shm: bool,
    allow_provenance_mismatch: bool,
) -> tuple[list[str], dict[str, str]]:
    distributed_phase = phase in {"training", "resume"}
    worker_args = [
        "-m",
        "tests.characterization.dual_expert_refactor_worker",
        "--phase",
        phase,
        "--asset-id",
        asset_id,
        "--assets",
        str(assets_path),
        "--fixture-root",
        str(fixture_root),
        "--output",
        str(report_path),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-config",
        str(checkpoint_config),
        "--base-model-root",
        str(base_model_root),
        "--video-transformer-root",
        str(video_transformer_root),
    ]
    if allow_provenance_mismatch:
        worker_args.append("--allow-checkpoint-provenance-mismatch")
    if distributed_phase:
        torchrun = shutil.which("torchrun")
        if torchrun is None:
            interpreter_sibling = Path(sys.executable).with_name("torchrun")
            if interpreter_sibling.is_file():
                torchrun = str(interpreter_sibling)
        if torchrun is None:
            raise FileNotFoundError(
                "Could not find `torchrun` in PATH or beside the active Python "
                f"interpreter {sys.executable}."
            )
        command = [
            torchrun,
            "--standalone",
            f"--nproc-per-node={training_world_size}",
            *worker_args,
        ]
    else:
        command = [sys.executable, *worker_args]

    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": (
                cuda_devices
                if distributed_phase
                else cuda_devices.split(",", maxsplit=1)[0]
            ),
            "OPEN_WAM_FSDP_CPU_OFFLOAD": "1" if fsdp_cpu_offload else "0",
            "OPEN_WAM_ENABLE_FIXED128_ROLLOUT_CONTEXT": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCHINDUCTOR_COMPILE_THREADS": environment.get(
                "TORCHINDUCTOR_COMPILE_THREADS",
                "1",
            ),
        }
    )
    if disable_nccl_shm:
        environment["NCCL_SHM_DISABLE"] = "1"
        environment["NCCL_CUMEM_HOST_ENABLE"] = "0"
    return command, environment


def _initialize_golden_files(
    sources: Iterable[tuple[Path, str]],
    *,
    golden_root: Path,
) -> None:
    """Create immutable golden files without replacing an existing baseline."""

    resolved_root = golden_root.expanduser().resolve()
    planned = [
        (source.expanduser().resolve(), resolved_root / filename)
        for source, filename in sources
    ]
    existing = [destination for _, destination in planned if destination.exists()]
    if existing:
        formatted = "\n- ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to replace existing characterization goldens:\n- "
            f"{formatted}\nUse a new versioned golden root."
        )
    missing = [source for source, _ in planned if not source.is_file()]
    if missing:
        formatted = "\n- ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Cannot initialize goldens from missing reports:\n- {formatted}"
        )

    resolved_root.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    created: list[Path] = []
    try:
        for source, destination in planned:
            temporary = destination.with_name(
                f".{destination.name}.tmp.{os.getpid()}"
            )
            shutil.copy2(source, temporary)
            staged.append((temporary, destination))
        for temporary, destination in staged:
            os.link(temporary, destination)
            created.append(destination)
    except Exception:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


def _stage_file(source: Path, destination: Path, *, refresh: bool) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        return source
    source_stat = source.stat()
    marker = destination.with_name(
        f".{destination.name}.open_wam_characterization_source.json"
    )
    expected_marker = {
        "source": str(source),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_ctime_ns": source_stat.st_ctime_ns,
        "source_content_probe_sha256": _file_content_probe_sha256(source),
    }
    if (
        not refresh
        and destination.is_file()
        and marker.is_file()
        and destination.stat().st_size == source_stat.st_size
        and _load_json(marker) == expected_marker
    ):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source_stat.st_size:
            raise OSError(f"Staged checkpoint size mismatch: {source} -> {temporary}.")
        temporary.replace(destination)
        write_json_atomic(marker, expected_marker)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _stage_directory(source: Path, destination: Path, *, refresh: bool) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        return source
    marker = destination / ".open_wam_characterization_source.json"
    expected_marker = _directory_source_marker(source)
    if (
        not refresh
        and destination.is_dir()
        and marker.is_file()
        and _load_json(marker) == expected_marker
    ):
        return destination
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, temporary, symlinks=True)
        write_json_atomic(temporary / marker.name, expected_marker)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _directory_source_marker(source: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_size_bytes = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        stat = path.lstat()
        if path.is_symlink():
            entry = (
                "symlink",
                relative,
                os.readlink(path),
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        elif path.is_file():
            file_count += 1
            total_size_bytes += stat.st_size
            entry = (
                "file",
                relative,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                _file_content_probe_sha256(path),
            )
        elif path.is_dir():
            entry = ("directory", relative, stat.st_mtime_ns, stat.st_ctime_ns)
        else:
            entry = (
                "other",
                relative,
                stat.st_mode,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        digest.update(json.dumps(entry, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return {
        "source": str(source),
        "tree_metadata_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_size_bytes": total_size_bytes,
    }


def _file_content_probe_sha256(path: Path) -> str:
    """Hash small files fully and deterministic windows from large model files."""

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(size.to_bytes(8, byteorder="little", signed=False))
    if size <= CONTENT_PROBE_CHUNK_BYTES * CONTENT_PROBE_COUNT:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    maximum_offset = size - CONTENT_PROBE_CHUNK_BYTES
    offsets = {
        round(maximum_offset * index / (CONTENT_PROBE_COUNT - 1))
        for index in range(CONTENT_PROBE_COUNT)
    }
    with path.open("rb") as handle:
        for offset in sorted(offsets):
            handle.seek(offset)
            digest.update(offset.to_bytes(8, byteorder="little", signed=False))
            digest.update(handle.read(CONTENT_PROBE_CHUNK_BYTES))
    return digest.hexdigest()


def _comparison_projection(
    value: Any,
    *,
    _path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, dict):
        value = _canonicalize_pruned_resume_model_rank(value, path=_path)
        tensor_fingerprint = _is_tensor_fingerprint(value)
        projected: dict[str, Any] = {}
        for raw_key, item in value.items():
            if (
                raw_key in VOLATILE_REPORT_KEYS
                or raw_key == "nonzero_elements"
                or (
                    _path == ("backend", "route")
                    and raw_key
                    in {
                        # Schema v1 serialized the retired runtime selector and
                        # duplicated its effective coupling. Schema v2 reports
                        # the public program instead. Coupling + route kind are
                        # the common behavioral contract compared below.
                        "program",
                        "resolved_current_block_coupling",
                        "runtime_mode",
                    }
                )
                or (tensor_fingerprint and raw_key in TENSOR_FINGERPRINT_REDUCTION_KEYS)
                or (
                    raw_key == "size_bytes"
                    and len(_path) >= 3
                    and _path[-3:-1] == ("checkpoint_artifacts", "files")
                )
            ):
                continue
            raw_key_text = str(raw_key)
            key = _REPORT_KEY_ALIASES.get(raw_key_text, raw_key_text)
            if key.startswith("mot_generalist/"):
                key = f"dual_expert_generalist/{key.removeprefix('mot_generalist/')}"
            if key in projected:
                raise ValueError(
                    "Characterization report contains both sides of metadata "
                    f"alias {raw_key!r} -> {key!r} at {_path!r}."
                )
            projected[key] = _comparison_projection(
                item,
                _path=(*_path, key),
            )
        nonzero_elements = value.get("nonzero_elements")
        parameter_elements = value.get("parameter_elements")
        if (
            isinstance(nonzero_elements, int)
            and isinstance(parameter_elements, int)
            and parameter_elements > 0
        ):
            projected["nonzero_density"] = round(
                nonzero_elements / parameter_elements,
                NONZERO_GRADIENT_DENSITY_PRECISION,
            )
        return projected
    if isinstance(value, list):
        return [
            _comparison_projection(item, _path=(*_path, str(index)))
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        if (
            len(_path) >= 3
            and _path[-3] == "checkpoint_provenance"
            and _path[-2] in {"actual", "expected"}
        ):
            return canonical_checkpoint_contract_value(_path[-1], value)
        if _path and _path[-1] == "config_name":
            return canonical_config_stem(value)
        if (
            _path
            and _path[-1] in {"architecture", "policy_variant", "variant"}
            and value == "mot"
        ):
            return "dual_expert"
        return _REPORT_STRING_ALIASES.get(value, value)
    return value


def _canonicalize_pruned_resume_model_rank(
    value: dict[str, Any],
    *,
    path: tuple[str, ...],
) -> dict[str, Any]:
    """Remove only the characterized dead-decoder delta from legacy reports."""

    if len(path) < 3 or path[-3:-1] != ("model", "per_rank"):
        return value
    components = value.get("components")
    if not isinstance(components, dict):
        return value
    other = components.get("other_trainable")
    if other != _LEGACY_OTHER_TRAINABLE_SIGNATURE:
        return value
    tensor_count = value.get("tensor_count")
    total_bytes = value.get("total_bytes")
    if not isinstance(tensor_count, int) or not isinstance(total_bytes, int):
        return value

    projected = dict(value)
    projected_components = dict(components)
    projected_components["other_trainable"] = {
        "tensor_count": (
            other["tensor_count"] - _LEGACY_RETIRED_DECODER_LOCAL_TENSOR_COUNT
        ),
        "total_bytes": (
            other["total_bytes"] - _LEGACY_RETIRED_DECODER_LOCAL_BYTES
        ),
    }
    projected["components"] = projected_components
    projected["tensor_count"] = (
        tensor_count - _LEGACY_RETIRED_DECODER_LOCAL_TENSOR_COUNT
    )
    projected["total_bytes"] = total_bytes - _LEGACY_RETIRED_DECODER_LOCAL_BYTES
    return projected


def _is_tensor_fingerprint(value: dict[str, Any]) -> bool:
    return {
        "content_sha256",
        "dtype",
        "finite_count",
        "local_shape",
        "numel",
        "probe_indices",
        "probe_values",
        "shape",
    }.issubset(value)


def _numeric_tolerance_resolver(
    gradient_tolerance: ComparisonTolerance,
):
    def resolve(path: str) -> ComparisonTolerance | None:
        if (
            ".resume.json.uninterrupted_update.scenario.metrics." in path
            or ".resume.json.resumed_update.scenario.metrics." in path
        ):
            return RESUME_POST_UPDATE_METRIC_TOLERANCE
        if (
            ".optimizer_step.distributed_numeric.parameter_groups." in path
            and ".delta." in path
        ):
            return DISTRIBUTED_AGGREGATE_DELTA_TOLERANCE
        if ".optimizer_step.distributed_numeric.parameter_groups." in path:
            return DISTRIBUTED_AGGREGATE_TOLERANCE
        if (
            ".gradients." in path
            or ".optimizer_step.distributed_numeric.parameter_probes" in path
            or path.endswith(".optimizer_step.distributed_numeric.grad_norm")
        ):
            return gradient_tolerance
        return None

    return resolve


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selected_asset_ids(raw_ids: list[str] | None) -> tuple[str, ...]:
    if not raw_ids:
        return tuple(method.asset_id for method in EXACT_CHECKPOINT_METHODS)
    resolved: list[str] = []
    unknown: list[str] = []
    for asset_id in raw_ids:
        try:
            resolved.append(resolve_characterization_asset_id(asset_id))
        except ValueError:
            unknown.append(asset_id)
    if unknown:
        raise ValueError(f"Unknown characterization asset ids: {sorted(unknown)}")
    return tuple(dict.fromkeys(resolved))


def _validate_phase_asset_selection(
    *,
    selected: tuple[str, ...],
    phases: tuple[str, ...],
) -> None:
    unknown_phases = sorted(set(phases) - set(ALL_PHASES))
    if unknown_phases:
        raise ValueError(f"Unknown characterization phases: {unknown_phases}")
    if "cache_rollover" in phases:
        unsupported = sorted(set(selected) - set(CACHE_ROLLOVER_ASSET_IDS))
        if unsupported:
            raise ValueError(
                "Cache-rollover characterization only supports shared-runtime "
                f"sentinels {CACHE_ROLLOVER_ASSET_IDS}; got {unsupported}."
            )
    if "resume" in phases and selected != (FULL_STATE_RESUME_ASSET_ID,):
        raise ValueError(
            "Full-state resume characterization requires exactly "
            f"{FULL_STATE_RESUME_ASSET_ID!r}; got {selected}."
        )


def _add_selection_args(
    parser: argparse.ArgumentParser,
    *,
    include_phase: bool = True,
) -> None:
    parser.add_argument(
        "--asset-id",
        action="append",
        choices=CHARACTERIZATION_ASSET_IDS,
        help=(
            "Repeat to select assets; defaults to the available exact-checkpoint "
            "matrix (VNA, ANV, and vanilla GJD excluded)."
        ),
    )
    if include_phase:
        parser.add_argument(
            "--phase",
            action="append",
            choices=ALL_PHASES,
            help="Repeat to select phases; defaults to training and inference.",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record and verify strict real-checkpoint DualExpert characterization."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixtures = subparsers.add_parser("fixtures")
    fixtures.add_argument("--assets", type=Path, required=True)
    fixtures.add_argument("--output-root", type=Path, required=True)
    fixtures.add_argument("--seed", type=int, default=20260727)
    fixtures.set_defaults(handler=freeze_fixtures)

    replay = subparsers.add_parser("replay-fixtures")
    replay.add_argument("--assets", type=Path, required=True)
    replay.add_argument("--fixture-root", type=Path, required=True)
    replay.add_argument(
        "--output-root",
        type=Path,
        help="Keep replayed files here; defaults to a temporary directory.",
    )
    replay.add_argument("--seed", type=int, default=20260727)
    replay.set_defaults(handler=replay_fixtures)

    record = subparsers.add_parser("record")
    record.add_argument("--assets", type=Path, required=True)
    record.add_argument("--fixture-root", type=Path, required=True)
    record.add_argument("--output-root", type=Path, required=True)
    record.add_argument("--stage-root", type=Path)
    record.add_argument("--refresh-stage", action="store_true")
    record.add_argument("--training-world-size", type=int, default=4)
    record.add_argument("--cuda-devices", default="0,1,2,3")
    record.add_argument(
        "--fsdp-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    record.add_argument("--disable-nccl-shm", action="store_true")
    record.add_argument(
        "--allow-checkpoint-provenance-mismatch",
        action="store_true",
        help=(
            "Permit exploratory reports from checkpoints whose saved training "
            "contract differs from the strict matrix."
        ),
    )
    _add_selection_args(record)
    record.set_defaults(handler=record_characterization)

    cli_smoke = subparsers.add_parser("training-cli-smoke")
    cli_smoke.add_argument("--assets", type=Path, required=True)
    cli_smoke.add_argument("--output-root", type=Path, required=True)
    cli_smoke.add_argument(
        "--asset-id",
        choices=tuple(
            dict.fromkeys(
                identity
                for method in EXACT_CHECKPOINT_METHODS
                for identity in (method.public_id, method.asset_id)
            )
        ),
        required=True,
    )
    cli_smoke.add_argument("--training-world-size", type=int, default=4)
    cli_smoke.add_argument("--cuda-devices", default="0,1,2,3")
    cli_smoke.add_argument(
        "--fsdp-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    cli_smoke.add_argument(
        "--allow-checkpoint-provenance-mismatch",
        action="store_true",
    )
    cli_smoke.add_argument("--disable-nccl-shm", action="store_true")
    cli_smoke.set_defaults(handler=training_cli_smoke)

    rollout = subparsers.add_parser("libero-rollout")
    rollout.add_argument("--assets", type=Path, required=True)
    rollout.add_argument("--output-root", type=Path, required=True)
    rollout.add_argument("--cuda-devices", default="0,1,2,3")
    rollout.add_argument("--libero-repo-root", type=Path)
    rollout.add_argument("--task-id", type=int, default=0)
    rollout.add_argument("--episode-idx", type=int, default=0)
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--max-timestep", type=int)
    rollout.add_argument("--max-chunks", type=int)
    rollout.add_argument(
        "--allow-checkpoint-provenance-mismatch",
        action="store_true",
    )
    _add_selection_args(rollout, include_phase=False)
    rollout.set_defaults(handler=record_libero_rollouts)

    verify_rollout = subparsers.add_parser("verify-libero-rollout")
    verify_rollout.add_argument("--actual-root", type=Path, required=True)
    verify_rollout.add_argument("--golden-root", type=Path, required=True)
    _add_selection_args(verify_rollout, include_phase=False)
    verify_rollout.set_defaults(handler=verify_libero_rollouts)

    initialize_rollout = subparsers.add_parser("initialize-libero-goldens")
    initialize_rollout.add_argument("--actual-root", type=Path, required=True)
    initialize_rollout.add_argument("--golden-root", type=Path, required=True)
    _add_selection_args(initialize_rollout, include_phase=False)
    initialize_rollout.set_defaults(handler=initialize_libero_rollout_goldens)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--actual-root", type=Path, required=True)
    verify.add_argument("--golden-root", type=Path, required=True)
    verify.add_argument(
        "--gradient-absolute-tolerance",
        "--absolute-tolerance",
        dest="gradient_absolute_tolerance",
        type=float,
        default=DISTRIBUTED_GRADIENT_TOLERANCE.absolute,
    )
    verify.add_argument(
        "--gradient-relative-tolerance",
        "--relative-tolerance",
        dest="gradient_relative_tolerance",
        type=float,
        default=DISTRIBUTED_GRADIENT_TOLERANCE.relative,
    )
    _add_selection_args(verify)
    verify.set_defaults(handler=verify_characterization)

    initialize = subparsers.add_parser("initialize-goldens")
    initialize.add_argument("--actual-root", type=Path, required=True)
    initialize.add_argument("--golden-root", type=Path, required=True)
    _add_selection_args(initialize)
    initialize.set_defaults(handler=initialize_characterization_goldens)
    return parser.parse_args()


def main() -> None:
    _assert_checkout_import_provenance()
    args = _parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
