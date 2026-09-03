from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.characterization.policy_video_action_golden import (
    verify_checked_in_trace_fixture,
    verify_rollout_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = (
    REPO_ROOT
    / "tests"
    / "characterization"
    / "goldens"
    / "vta_external_idm_48of50_task0_ep0_seed0.json"
)


def test_vta_external_idm_golden_fixture_is_self_consistent() -> None:
    verify_checked_in_trace_fixture(GOLDEN_PATH)
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["source_evaluation"] == {
        "successes": 48,
        "episodes": 50,
        "video_producer": "M5 VTA step 9700",
        "action_consumer": "M5 GJD pure IDM CF60/real40 step 20000",
    }


def test_rollout_golden_rejects_action_drift(tmp_path: Path) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    fixture_path = (
        GOLDEN_PATH.parent / golden["scopes"]["first_chunk"]["action_trace"]["fixture"]
    )
    rows = fixture_path.read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[0])
    changed["action"][0] += 1e-6
    rows[0] = json.dumps(changed)
    drifted_path = tmp_path / "drifted.jsonl"
    drifted_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(golden["scopes"]["first_chunk"]["report"]),
        encoding="utf-8",
    )
    load_report_path = tmp_path / "load_report.json"
    load_report_path.write_text(
        json.dumps(golden["load_report"]),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="action_trace.sha256"):
        verify_rollout_artifacts(
            golden_path=GOLDEN_PATH,
            scope="first_chunk",
            report_path=report_path,
            action_trace_path=drifted_path,
            load_report_path=load_report_path,
        )


def test_transformer_dir_override_prefers_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPEN_WAM_TEST_TRANSFORMER_DIR", raising=False)
    checkpoint = tmp_path / "checkpoint_step_1"
    bundled = checkpoint / "transformer"
    bundled.mkdir(parents=True)
    assert (
        _resolve_transformer_dir_override(
            checkpoint,
            env_name="OPEN_WAM_TEST_TRANSFORMER_DIR",
        )
        == bundled.resolve()
    )

    explicit = tmp_path / "external_transformer"
    explicit.mkdir()
    monkeypatch.setenv("OPEN_WAM_TEST_TRANSFORMER_DIR", str(explicit))
    assert (
        _resolve_transformer_dir_override(
            checkpoint,
            env_name="OPEN_WAM_TEST_TRANSFORMER_DIR",
        )
        == explicit.resolve()
    )


@pytest.mark.gpu
@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.integration
def test_real_vta_external_idm_rollout_matches_48of50_golden(
    tmp_path: Path,
) -> None:
    """Opt-in exact rollout gate for the frozen 48/50 composition route."""

    if os.getenv("OPEN_WAM_RUN_VTA_IDM_GOLDEN") != "1":
        pytest.skip("Set OPEN_WAM_RUN_VTA_IDM_GOLDEN=1 to run.")

    scope = os.getenv("OPEN_WAM_VTA_IDM_GOLDEN_SCOPE", "first_chunk")
    if scope not in {"first_chunk", "full_rollout"}:
        raise ValueError(
            "OPEN_WAM_VTA_IDM_GOLDEN_SCOPE must be first_chunk or full_rollout."
        )
    producer_config = _required_env_path("OPEN_WAM_VTA_IDM_PRODUCER_CONFIG")
    producer_checkpoint = _required_env_path("OPEN_WAM_VTA_IDM_PRODUCER_CHECKPOINT")
    consumer_config = _required_env_path("OPEN_WAM_VTA_IDM_CONSUMER_CONFIG")
    consumer_checkpoint = _required_env_path("OPEN_WAM_VTA_IDM_CONSUMER_CHECKPOINT")
    dataset_root = _required_env_path("OPEN_WAM_VTA_IDM_DATASET_ROOT")
    base_model_root = _required_env_path("OPEN_WAM_VTA_IDM_BASE_MODEL_ROOT")
    libero_repo_root = _required_env_path("OPEN_WAM_LIBERO_REPO_ROOT")
    empty_text_embedding = os.getenv("OPEN_WAM_VTA_IDM_EMPTY_TEXT_EMBEDDING")
    python_executable = Path(
        os.getenv("OPEN_WAM_VTA_IDM_PYTHON", sys.executable)
    ).expanduser()
    if not python_executable.is_file():
        raise FileNotFoundError(
            f"OPEN_WAM_VTA_IDM_PYTHON does not exist: {python_executable}"
        )

    output_root = tmp_path / "vta_idm_golden"
    producer_device = os.getenv("OPEN_WAM_VTA_IDM_PRODUCER_DEVICE", "cuda:0")
    consumer_device = os.getenv("OPEN_WAM_VTA_IDM_CONSUMER_DEVICE", "cuda:1")
    producer_overrides = [
        f"data.local_root={dataset_root}",
        f"backbone.pretrained_model_name_or_path={base_model_root}",
    ]
    consumer_overrides = list(producer_overrides)
    producer_transformer_dir = _resolve_transformer_dir_override(
        producer_checkpoint,
        env_name="OPEN_WAM_VTA_IDM_PRODUCER_TRANSFORMER_DIR",
    )
    consumer_transformer_dir = _resolve_transformer_dir_override(
        consumer_checkpoint,
        env_name="OPEN_WAM_VTA_IDM_CONSUMER_TRANSFORMER_DIR",
    )
    if producer_transformer_dir is not None:
        producer_overrides.append(
            f"backbone.transformer_subdir={producer_transformer_dir}"
        )
    if consumer_transformer_dir is not None:
        consumer_overrides.append(
            f"backbone.transformer_subdir={consumer_transformer_dir}"
        )
    if empty_text_embedding:
        producer_overrides.append(
            f"data.empty_text_embedding_path={empty_text_embedding}"
        )
        consumer_overrides.append(
            f"data.empty_text_embedding_path={empty_text_embedding}"
        )

    command = [
        str(python_executable),
        str(REPO_ROOT / "scripts" / "run_libero_policy_video_action_visualization.py"),
        "--cfg",
        str(producer_config),
        "--checkpoint",
        str(producer_checkpoint),
    ]
    for override in producer_overrides:
        command.extend(("--set", override))
    command.extend(
        [
            "--action-route",
            "generated_video_then_action",
            "--action-consumer-cfg",
            str(consumer_config),
            "--action-consumer-checkpoint",
            str(consumer_checkpoint),
        ]
    )
    for override in consumer_overrides:
        command.extend(("--action-consumer-set", override))
    command.extend(
        [
            "--benchmark",
            "libero_10",
            "--task-id",
            "0",
            "--episode-idx",
            "0",
            "--max-timestep",
            "800",
            "--max-chunks",
            "1" if scope == "first_chunk" else "50",
            "--frontend-encode-mode",
            "lingbot_streaming_vae",
            "--startup-model-obs-frames",
            "1",
            "--startup-env-init-steps",
            "5",
            "--dual-expert-inference-window-size",
            "30",
            "--runtime-device",
            producer_device,
            "--action-device",
            producer_device,
            "--frontend-device",
            producer_device,
            "--decode-device",
            producer_device,
            "--action-consumer-runtime-device",
            consumer_device,
            "--action-consumer-action-device",
            consumer_device,
            "--action-consumer-frontend-device",
            consumer_device,
            "--output-dir",
            str(output_root),
            "--suffix",
            f"golden_{scope}",
            "--seed",
            "0",
            "--allow-deprecated-libero-config",
        ]
    )

    env = os.environ.copy()
    env["LIBERO_REPO_ROOT"] = str(libero_repo_root)
    env["MUJOCO_GL"] = "osmesa"
    current_pythonpath = env.get("PYTHONPATH")
    source_root = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else source_root
    )
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)

    report_path = _find_episode_artifact(output_root, suffix=".json")
    action_trace_path = _find_episode_artifact(output_root, suffix="_actions.jsonl")
    load_report_path = _find_episode_artifact(output_root, suffix="_load_report.json")
    verify_rollout_artifacts(
        golden_path=GOLDEN_PATH,
        scope=scope,
        report_path=report_path,
        action_trace_path=action_trace_path,
        load_report_path=load_report_path,
    )


@pytest.mark.gpu
@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.integration
def test_real_vta_native_and_two_model_rollouts_are_bitwise_equal(
    tmp_path: Path,
) -> None:
    """Opt-in golden for native VTA versus two same-checkpoint model instances."""

    if os.getenv("OPEN_WAM_RUN_VTA_COMPOSITION_PARITY") != "1":
        pytest.skip("Set OPEN_WAM_RUN_VTA_COMPOSITION_PARITY=1 to run.")

    config = _required_env_path("OPEN_WAM_VTA_COMPOSITION_CONFIG")
    checkpoint = _required_env_path("OPEN_WAM_VTA_COMPOSITION_CHECKPOINT")
    dataset_root = _required_env_path("OPEN_WAM_VTA_COMPOSITION_DATASET_ROOT")
    base_model_root = _required_env_path("OPEN_WAM_VTA_COMPOSITION_BASE_MODEL_ROOT")
    libero_repo_root = _required_env_path("OPEN_WAM_LIBERO_REPO_ROOT")
    python_executable = Path(
        os.getenv("OPEN_WAM_VTA_COMPOSITION_PYTHON", sys.executable)
    ).expanduser()
    if not python_executable.is_file():
        raise FileNotFoundError(
            f"OPEN_WAM_VTA_COMPOSITION_PYTHON does not exist: {python_executable}"
        )

    producer_device = os.getenv("OPEN_WAM_VTA_COMPOSITION_PRODUCER_DEVICE", "cuda:0")
    consumer_device = os.getenv("OPEN_WAM_VTA_COMPOSITION_CONSUMER_DEVICE", "cuda:1")
    task_id = "0"
    episode_idx = "0"
    common = [
        str(python_executable),
        str(REPO_ROOT / "scripts" / "run_libero_dual_expert_visualization.py"),
        "--cfg",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--set",
        f"data.local_root={dataset_root}",
        "--set",
        f"backbone.pretrained_model_name_or_path={base_model_root}",
    ]
    empty_text_embedding = os.getenv("OPEN_WAM_VTA_COMPOSITION_EMPTY_TEXT_EMBEDDING")
    if empty_text_embedding:
        common.extend(
            ("--set", f"data.empty_text_embedding_path={empty_text_embedding}")
        )
    transformer_dir = _resolve_transformer_dir_override(
        checkpoint,
        env_name="OPEN_WAM_VTA_COMPOSITION_TRANSFORMER_DIR",
    )
    if transformer_dir is not None:
        common.extend(("--set", f"backbone.transformer_subdir={transformer_dir}"))
    common.extend(
        (
            "--benchmark",
            "libero_10",
            "--task-id",
            task_id,
            "--episode-idx",
            episode_idx,
            "--max-timestep",
            "100",
            "--max-chunks",
            "2",
            "--frontend-encode-mode",
            "lingbot_streaming_vae",
            "--dual-expert-inference-window-size",
            "30",
            "--startup-model-obs-frames",
            "1",
            "--startup-env-init-steps",
            "5",
            "--runtime-device",
            producer_device,
            "--action-device",
            producer_device,
            "--frontend-device",
            producer_device,
            "--decode-device",
            producer_device,
            "--seed",
            episode_idx,
            "--allow-deprecated-libero-config",
        )
    )

    env = os.environ.copy()
    env["LIBERO_REPO_ROOT"] = str(libero_repo_root)
    env.setdefault("MUJOCO_GL", "egl")
    source_root = str(REPO_ROOT / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else source_root
    )
    outputs: dict[str, Path] = {}
    cases = (
        ("native", "joint", ()),
        (
            "two_model",
            "generated_video_then_action",
            (
                "--action-consumer-cfg",
                str(config),
                "--action-consumer-checkpoint",
                str(checkpoint),
                "--action-consumer-runtime-device",
                consumer_device,
                "--action-consumer-action-device",
                consumer_device,
                "--action-consumer-frontend-device",
                consumer_device,
                "--action-consumer-set",
                f"data.local_root={dataset_root}",
                "--action-consumer-set",
                f"backbone.pretrained_model_name_or_path={base_model_root}",
                *(
                    (
                        "--action-consumer-set",
                        f"data.empty_text_embedding_path={empty_text_embedding}",
                    )
                    if empty_text_embedding
                    else ()
                ),
                *(
                    (
                        "--action-consumer-set",
                        f"backbone.transformer_subdir={transformer_dir}",
                    )
                    if transformer_dir is not None
                    else ()
                ),
            ),
        ),
    )
    for label, route, extra_args in cases:
        output_root = tmp_path / label
        command = [
            *common,
            "--action-route",
            route,
            *extra_args,
            "--output-dir",
            str(output_root),
            "--suffix",
            f"vta_{label}",
        ]
        subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)
        outputs[f"{label}.actions"] = _find_episode_artifact(
            output_root,
            suffix="_actions.jsonl",
        )
        outputs[f"{label}.video"] = _find_episode_artifact(
            output_root,
            suffix=".mp4",
        )
        outputs[f"{label}.chunks"] = _find_episode_artifact(
            output_root,
            suffix="_chunks.json",
        )

    assert (
        outputs["native.actions"].read_bytes()
        == outputs["two_model.actions"].read_bytes()
    )
    assert (
        outputs["native.video"].read_bytes() == outputs["two_model.video"].read_bytes()
    )
    native_chunks = [
        chunk
        for chunk in json.loads(outputs["native.chunks"].read_text(encoding="utf-8"))
        if chunk.get("phase") == "infer"
    ]
    composed_chunks = [
        chunk
        for chunk in json.loads(outputs["two_model.chunks"].read_text(encoding="utf-8"))
        if chunk.get("phase") == "infer"
    ]
    assert len(native_chunks) == len(composed_chunks) == 2
    for native_chunk, composed_chunk in zip(
        native_chunks,
        composed_chunks,
        strict=True,
    ):
        assert native_chunk["policy_debug"]["predicted_latents"]["device"] == (
            producer_device
        )
        assert composed_chunk["policy_debug"]["predicted_latents"]["device"] == (
            consumer_device
        )
    assert [
        _without_device_metadata(chunk["policy_debug"]) for chunk in native_chunks
    ] == [_without_device_metadata(chunk["policy_debug"]) for chunk in composed_chunks]


def _find_episode_artifact(root: Path, *, suffix: str) -> Path:
    paths = [
        path
        for path in root.rglob(f"*{suffix}")
        if not (
            suffix == ".json"
            and path.name.endswith(("_chunks.json", "_load_report.json"))
        )
    ]
    if len(paths) != 1:
        raise AssertionError(
            f"Expected one rollout artifact ending in {suffix!r}, found {paths}."
        )
    return paths[0]


def _without_device_metadata(value):
    """Remove runtime placement labels while preserving semantic debug values."""

    if isinstance(value, dict):
        return {
            key: _without_device_metadata(item)
            for key, item in value.items()
            if key != "device"
        }
    if isinstance(value, list):
        return [_without_device_metadata(item) for item in value]
    return value


def _resolve_transformer_dir_override(
    checkpoint: Path,
    *,
    env_name: str,
) -> Path | None:
    explicit = _optional_env_path(env_name)
    if explicit is not None:
        if not explicit.is_dir():
            raise NotADirectoryError(f"{env_name} is not a directory: {explicit}")
        return explicit
    bundled = (
        checkpoint / "transformer"
        if checkpoint.is_dir()
        else checkpoint.parent / "transformer"
    )
    return bundled.resolve() if bundled.is_dir() else None


def _optional_env_path(name: str) -> Path | None:
    raw = os.getenv(name)
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _required_env_path(name: str) -> Path:
    raw = os.getenv(name)
    if not raw:
        pytest.skip(f"Set {name} to run the real rollout golden.")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path
