"""Guard orchestration ownership without freezing implementation import lists."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / "src" / "open_wam"


@pytest.mark.parametrize(
    "relative",
    (
        "dual_expert/runtime_routes.py",
        "dual_expert/split_cache_inference.py",
        "dual_expert/packed_inference.py",
        "dual_expert/attention_cached.py",
        "dual_expert/cache_execution.py",
        "dual_expert/cache_state.py",
        "dual_expert/runtime.py",
        "dual_expert/attention.py",
        "dual_expert/contracts.py",
        "parallel_stream/staged_rollout.py",
        "parallel_stream/packed_rollout.py",
        "parallel_stream/reference_runtime.py",
    ),
)
def test_retired_inference_routes_have_no_compatibility_module(relative):
    assert not (ROOT / "models" / "policy_variants" / relative).exists()


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
def test_architecture_adapter_delegates_denoising(architecture):
    source = (
        ROOT / "models" / "policy_variants" / architecture / "inference.py"
    ).read_text()
    tree = ast.parse(source)
    names = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert names.count("denoise_video_action") == 1
    assert "FlowMatchScheduler" not in names
    assert "denoise_stages" not in names
    other = "parallel_stream" if architecture == "dual_expert" else "dual_expert"
    assert f"policy_variants.{other}" not in source


def test_shared_execution_does_not_import_architecture_implementations():
    for name in (
        "denoising",
        "video_action_inference",
        "denoising_cache",
        "video_action_layout",
        "video_action_state",
        "observed_history",
    ):
        source = (ROOT / "models" / "common" / f"{name}.py").read_text()
        for architecture in ("dual_expert", "parallel_stream"):
            assert f"policy_variants.{architecture}" not in source


def test_realtime_replans_commit_observations_without_selecting_an_executor():
    engine = (ROOT / "runtime" / "rollout_engine.py").read_text()
    planner = (ROOT / "runtime" / "policy_planner.py").read_text()
    assert "runner.reconcile_observed_history(" in planner
    assert "reconcile_observed_history(" not in engine
    for retired in (
        "runtime_routes",
        "snapshot_sequence_visual_runtime",
        "dual_expert_skip_observation_history_update",
        "dual_expert_action_cache_rewind_to_frame",
    ):
        assert retired not in engine + planner
