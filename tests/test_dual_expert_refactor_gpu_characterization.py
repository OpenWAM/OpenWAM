from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.gpu
@pytest.mark.data
@pytest.mark.slow
@pytest.mark.integration
def test_real_checkpoint_dual_expert_characterization_matches_recorded_baseline(
    tmp_path: Path,
) -> None:
    """Opt-in real-data/GPU gate for refactors of the core DualExpert runtime."""

    if os.getenv("OPEN_WAM_RUN_DUAL_EXPERT_GPU_CHARACTERIZATION") != "1":
        pytest.skip("Set OPEN_WAM_RUN_DUAL_EXPERT_GPU_CHARACTERIZATION=1 to run.")
    assets = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSETS")
    fixture_root = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_FIXTURES")
    golden_root = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_GOLDENS")
    asset_id = os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSET_ID")
    if not asset_id:
        pytest.skip(
            "Set OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSET_ID explicitly; "
            "the test does not launch the full exact-checkpoint matrix by accident."
        )

    output_root = tmp_path / "reports"
    record_command = [
        sys.executable,
        "-m",
        "tests.characterization.run_dual_expert_refactor_characterization",
        "record",
        "--assets",
        str(assets),
        "--fixture-root",
        str(fixture_root),
        "--output-root",
        str(output_root),
        "--asset-id",
        asset_id,
        "--cuda-devices",
        os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_CUDA_DEVICES", "0,1,2,3"),
    ]
    stage_root = os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_STAGE_ROOT")
    if stage_root:
        record_command.extend(["--stage-root", stage_root])
    if os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_DISABLE_NCCL_SHM") == "1":
        record_command.append("--disable-nccl-shm")
    subprocess.run(record_command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.characterization.run_dual_expert_refactor_characterization",
            "verify",
            "--actual-root",
            str(output_root),
            "--golden-root",
            str(golden_root),
            "--asset-id",
            asset_id,
        ],
        check=True,
    )


@pytest.mark.gpu
@pytest.mark.data
@pytest.mark.slow
@pytest.mark.integration
def test_shared_dual_expert_cache_rollover_matches_recorded_baseline(
    tmp_path: Path,
) -> None:
    if os.getenv("OPEN_WAM_RUN_DUAL_EXPERT_CACHE_ROLLOVER") != "1":
        pytest.skip("Set OPEN_WAM_RUN_DUAL_EXPERT_CACHE_ROLLOVER=1 to run.")
    assets = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSETS")
    fixture_root = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_FIXTURES")
    golden_root = _required_env_path("OPEN_WAM_DUAL_EXPERT_INFRASTRUCTURE_GOLDENS")
    output_root = tmp_path / "cache_rollover"
    command = [
        sys.executable,
        "-m",
        "tests.characterization.run_dual_expert_refactor_characterization",
        "record",
        "--assets",
        str(assets),
        "--fixture-root",
        str(fixture_root),
        "--output-root",
        str(output_root),
        "--phase",
        "cache_rollover",
        "--asset-id",
        "dual_expert_joint",
        "--asset-id",
        "gjd_mode_token",
        "--cuda-devices",
        os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_CUDA_DEVICES", "0,1,2,3"),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.characterization.run_dual_expert_refactor_characterization",
            "verify",
            "--actual-root",
            str(output_root),
            "--golden-root",
            str(golden_root),
            "--phase",
            "cache_rollover",
            "--asset-id",
            "dual_expert_joint",
            "--asset-id",
            "gjd_mode_token",
        ],
        check=True,
    )


@pytest.mark.gpu
@pytest.mark.data
@pytest.mark.slow
@pytest.mark.integration
def test_mode_token_gjd_full_state_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    if os.getenv("OPEN_WAM_RUN_DUAL_EXPERT_FULL_STATE_RESUME") != "1":
        pytest.skip("Set OPEN_WAM_RUN_DUAL_EXPERT_FULL_STATE_RESUME=1 to run.")
    assets = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSETS")
    fixture_root = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_FIXTURES")
    golden_root = _required_env_path("OPEN_WAM_DUAL_EXPERT_INFRASTRUCTURE_GOLDENS")
    output_root = tmp_path / "full_state_resume"
    command = [
        sys.executable,
        "-m",
        "tests.characterization.run_dual_expert_refactor_characterization",
        "record",
        "--assets",
        str(assets),
        "--fixture-root",
        str(fixture_root),
        "--output-root",
        str(output_root),
        "--phase",
        "resume",
        "--asset-id",
        "gjd_mode_token",
        "--training-world-size",
        "4",
        "--cuda-devices",
        os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_CUDA_DEVICES", "0,1,2,3"),
        "--no-fsdp-cpu-offload",
    ]
    if os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_DISABLE_NCCL_SHM") == "1":
        command.append("--disable-nccl-shm")
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.characterization.run_dual_expert_refactor_characterization",
            "verify",
            "--actual-root",
            str(output_root),
            "--golden-root",
            str(golden_root),
            "--phase",
            "resume",
            "--asset-id",
            "gjd_mode_token",
        ],
        check=True,
    )


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.integration
def test_real_data_fixture_replay_is_byte_identical(tmp_path: Path) -> None:
    if os.getenv("OPEN_WAM_RUN_DUAL_EXPERT_DATA_REPLAY") != "1":
        pytest.skip("Set OPEN_WAM_RUN_DUAL_EXPERT_DATA_REPLAY=1 to run.")
    assets = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSETS")
    fixture_root = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_FIXTURES")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.characterization.run_dual_expert_refactor_characterization",
            "replay-fixtures",
            "--assets",
            str(assets),
            "--fixture-root",
            str(fixture_root),
            "--output-root",
            str(tmp_path / "replayed"),
        ],
        check=True,
    )


@pytest.mark.gpu
@pytest.mark.data
@pytest.mark.slow
@pytest.mark.integration
def test_real_training_cli_executes_one_optimizer_update(tmp_path: Path) -> None:
    if os.getenv("OPEN_WAM_RUN_DUAL_EXPERT_TRAINING_CLI_SMOKE") != "1":
        pytest.skip("Set OPEN_WAM_RUN_DUAL_EXPERT_TRAINING_CLI_SMOKE=1 to run.")
    assets = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSETS")
    asset_id = _required_env_value("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSET_ID")
    command = [
        sys.executable,
        "-m",
        "tests.characterization.run_dual_expert_refactor_characterization",
        "training-cli-smoke",
        "--assets",
        str(assets),
        "--output-root",
        str(tmp_path / "training_cli"),
        "--asset-id",
        asset_id,
        "--cuda-devices",
        os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_CUDA_DEVICES", "0,1,2,3"),
    ]
    if os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_NO_FSDP_CPU_OFFLOAD") == "1":
        command.append("--no-fsdp-cpu-offload")
    if os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_DISABLE_NCCL_SHM") == "1":
        command.append("--disable-nccl-shm")
    subprocess.run(command, check=True)


@pytest.mark.gpu
@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.integration
def test_real_libero_cli_runs_three_stateful_chunks(tmp_path: Path) -> None:
    if os.getenv("OPEN_WAM_RUN_DUAL_EXPERT_LIBERO_CLI_SMOKE") != "1":
        pytest.skip("Set OPEN_WAM_RUN_DUAL_EXPERT_LIBERO_CLI_SMOKE=1 to run.")
    assets = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSETS")
    asset_id = _required_env_value("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSET_ID")
    command = [
        sys.executable,
        "-m",
        "tests.characterization.run_dual_expert_refactor_characterization",
        "libero-rollout",
        "--assets",
        str(assets),
        "--output-root",
        str(tmp_path / "libero_cli"),
        "--asset-id",
        asset_id,
        "--cuda-devices",
        os.getenv("OPEN_WAM_DUAL_EXPERT_LIBERO_CUDA_DEVICE", "0"),
        "--task-id",
        os.getenv("OPEN_WAM_DUAL_EXPERT_LIBERO_TASK_ID", "0"),
        "--episode-idx",
        os.getenv("OPEN_WAM_DUAL_EXPERT_LIBERO_EPISODE_IDX", "0"),
        "--seed",
        os.getenv("OPEN_WAM_DUAL_EXPERT_LIBERO_SEED", "0"),
        "--max-timestep",
        "64",
        "--max-chunks",
        "3",
    ]
    libero_repo_root = os.getenv("OPEN_WAM_LIBERO_REPO_ROOT")
    if libero_repo_root:
        command.extend(["--libero-repo-root", libero_repo_root])
    subprocess.run(command, check=True)

    report_path = tmp_path / "libero_cli" / asset_id / "rollout_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert 1 <= int(report["chunk_count"]) <= 3
    assert len(report["inference_chunks"]) == int(report["chunk_count"])


@pytest.mark.gpu
@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.integration
def test_real_libero_rollout_matrix_matches_recorded_baseline(
    tmp_path: Path,
) -> None:
    if os.getenv("OPEN_WAM_RUN_DUAL_EXPERT_LIBERO_MATRIX") != "1":
        pytest.skip("Set OPEN_WAM_RUN_DUAL_EXPERT_LIBERO_MATRIX=1 to run.")
    assets = _required_env_path("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_ASSETS")
    command = [
        sys.executable,
        "-m",
        "tests.characterization.run_dual_expert_refactor_characterization",
        "libero-rollout",
        "--assets",
        str(assets),
        "--output-root",
        str(tmp_path / "libero_matrix"),
        "--cuda-devices",
        os.getenv("OPEN_WAM_DUAL_EXPERT_CHARACTERIZATION_CUDA_DEVICES", "0,1,2,3"),
        "--task-id",
        os.getenv("OPEN_WAM_DUAL_EXPERT_LIBERO_TASK_ID", "0"),
        "--episode-idx",
        os.getenv("OPEN_WAM_DUAL_EXPERT_LIBERO_EPISODE_IDX", "0"),
        "--seed",
        os.getenv("OPEN_WAM_DUAL_EXPERT_LIBERO_SEED", "0"),
    ]
    libero_repo_root = os.getenv("OPEN_WAM_LIBERO_REPO_ROOT")
    if libero_repo_root:
        command.extend(["--libero-repo-root", libero_repo_root])
    subprocess.run(command, check=True)

    summary = tmp_path / "libero_matrix" / "rollout_summary.json"
    assert summary.is_file()
    golden_root = os.getenv("OPEN_WAM_DUAL_EXPERT_LIBERO_GOLDENS")
    if golden_root:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.characterization.run_dual_expert_refactor_characterization",
                "verify-libero-rollout",
                "--actual-root",
                str(tmp_path / "libero_matrix"),
                "--golden-root",
                golden_root,
            ],
            check=True,
        )


def _required_env_path(name: str) -> Path:
    raw = os.getenv(name)
    if not raw:
        pytest.skip(f"Set {name} to run real-checkpoint characterization.")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _required_env_value(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"Set {name} to run this characterization gate.")
    return value
