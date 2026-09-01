from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

import open_wam.training.checkpoints as checkpoints_module
from open_wam.configs import load_experiment_config
from open_wam.configs.enums import CheckpointMode
from open_wam.training.checkpoint_storage import _wait_for_file
from open_wam.training.checkpoints import CheckpointManager
from open_wam.training.state import TrainState


REPO_ROOT = Path(__file__).resolve().parents[1]


def _full_training_state(model_state: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "model_state_dict": model_state,
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
        "strategy_state_dict": None,
        "train_state": TrainState().state_dict(),
    }


def _manager(
    tmp_path: Path,
    *,
    distributed_timeout_seconds: int = 1800,
    export_runtime_backbone: bool = False,
    max_checkpoints_to_keep: int | None = None,
    checkpoint_mode: CheckpointMode = CheckpointMode.MODEL_ONLY,
) -> CheckpointManager:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            distributed_timeout_seconds=distributed_timeout_seconds,
        ),
    )
    return CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=checkpoint_mode,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        export_runtime_backbone=export_runtime_backbone,
    )


def _stub_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        checkpoints_module,
        "get_model_state_dict",
        lambda model, options: {"weight": torch.ones(1)},
    )


def _save_model_only(
    manager: CheckpointManager,
    *,
    model: torch.nn.Module | None = None,
) -> Path:
    return manager.save(
        step=1,
        model=model if model is not None else torch.nn.Linear(2, 2),
        optimizer=None,
        scheduler=None,
        train_state=TrainState(optimizer_step=1),
    )


def test_wait_for_file_gives_error_marker_precedence(tmp_path: Path) -> None:
    expected = tmp_path / "complete"
    failed = tmp_path / "error"
    expected.touch()
    failed.touch()

    with pytest.raises(RuntimeError, match="Peer signalled checkpoint failure"):
        _wait_for_file(
            expected,
            timeout_seconds=1.0,
            poll_seconds=0.0,
            error_marker=failed,
        )


def test_wait_for_file_timeout_names_the_requested_file(tmp_path: Path) -> None:
    expected = tmp_path / "payload"

    with pytest.raises(TimeoutError, match=f"waiting for file: {expected}"):
        _wait_for_file(expected, timeout_seconds=0.0, poll_seconds=0.0)


def test_checkpoint_retry_clears_stale_error_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    checkpoint_dir = manager.checkpoint_dir_for_step(1)
    checkpoint_dir.mkdir(parents=True)
    error_marker = checkpoint_dir / ".checkpoint_error"
    error_marker.write_text("old failure\n", encoding="utf-8")
    _stub_model_state(monkeypatch)

    saved_dir = _save_model_only(manager)

    assert saved_dir == checkpoint_dir
    assert not error_marker.exists()
    assert (checkpoint_dir / ".checkpoint_payload_complete").is_file()
    assert (checkpoint_dir / ".checkpoint_complete").is_file()


def test_checkpoint_peer_waits_use_configured_distributed_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path, distributed_timeout_seconds=37)
    waits: list[tuple[Path, float, Path | None]] = []
    barriers = 0

    def record_barrier() -> None:
        nonlocal barriers
        barriers += 1

    def record_wait(
        path: Path,
        *,
        timeout_seconds: float,
        poll_seconds: float = 2.0,
        error_marker: Path | None = None,
    ) -> None:
        del poll_seconds
        waits.append((path, timeout_seconds, error_marker))

    monkeypatch.setattr(checkpoints_module, "_is_rank_zero", lambda: False)
    monkeypatch.setattr(checkpoints_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoints_module.dist, "barrier", record_barrier)
    monkeypatch.setattr(checkpoints_module, "_wait_for_file", record_wait)
    monkeypatch.setattr(
        checkpoints_module,
        "get_model_state_dict",
        lambda model, options: {},
    )
    monkeypatch.setattr(checkpoints_module, "_release_unused_device_memory", lambda: None)

    _save_model_only(manager)

    assert barriers == 2
    assert [path.name for path, _, _ in waits] == [
        ".checkpoint_payload_complete",
        ".checkpoint_complete",
    ]
    assert [timeout for _, timeout, _ in waits] == [37.0, 37.0]
    assert {marker.name for _, _, marker in waits if marker is not None} == {
        ".checkpoint_error"
    }


@pytest.mark.parametrize(
    "failure_stage",
    ["payload", "export", "prune", "completion"],
)
def test_rank_zero_finalization_failure_invalidates_checkpoint(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        tmp_path,
        export_runtime_backbone=failure_stage == "export",
        max_checkpoints_to_keep=1,
    )
    _stub_model_state(monkeypatch)

    def fail(stage: str):
        raise RuntimeError(f"{stage} failed")

    if failure_stage == "payload":
        monkeypatch.setattr(
            manager,
            "_write_model_state_checkpoint",
            lambda checkpoint_dir, model_state_dict: fail("payload"),
        )
    elif failure_stage == "export":
        monkeypatch.setattr(
            manager,
            "_export_runtime_backbone",
            lambda checkpoint_dir, model: fail("export"),
        )
    elif failure_stage == "prune":
        monkeypatch.setattr(
            manager,
            "_prune_old_checkpoints",
            lambda **kwargs: fail("prune"),
        )
    else:
        original_write_text = Path.write_text

        def fail_completion_marker(
            path: Path,
            *args: object,
            **kwargs: object,
        ) -> int:
            if path.name == ".checkpoint_complete":
                raise OSError("completion marker failed")
            return original_write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_completion_marker)

    with pytest.raises((OSError, RuntimeError), match=failure_stage):
        _save_model_only(manager)

    checkpoint_dir = manager.checkpoint_dir_for_step(1)
    assert (checkpoint_dir / ".checkpoint_error").is_file()
    assert not (checkpoint_dir / ".checkpoint_complete").exists()


def test_optimizer_resume_does_not_retry_out_of_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        tmp_path,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint_dir = manager.save(
        step=1,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_state=TrainState(optimizer_step=1),
    )

    def raise_oom(*args, **kwargs) -> None:
        del args, kwargs
        raise torch.OutOfMemoryError("optimizer restore OOM")

    monkeypatch.setattr(checkpoints_module, "set_optimizer_state_dict", raise_oom)
    monkeypatch.setattr(
        checkpoints_module,
        "_densify_optimizer_state_dict",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("OOM must not enter the densify fallback")
        ),
    )

    resumed_model = torch.nn.Linear(2, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    with pytest.raises(torch.OutOfMemoryError, match="optimizer restore OOM"):
        manager.load(
            path=checkpoint_dir,
            model=resumed_model,
            optimizer=resumed_optimizer,
        )


def test_distributed_non_strict_load_filters_retired_checkpoint_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    checkpoint_path = tmp_path / "full_training_state.pt"
    checkpoint_path.touch()
    model = torch.nn.Linear(2, 2)
    loaded_keys: list[str] = []

    monkeypatch.setattr(checkpoints_module, "_is_rank_zero", lambda: True)
    monkeypatch.setattr(checkpoints_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        checkpoints_module.dist,
        "broadcast_object_list",
        lambda values, src: None,
    )
    monkeypatch.setattr(
        checkpoints_module,
        "_load_tensor_artifact",
        lambda path, map_location: _full_training_state(
            {
                "weight": torch.ones_like(model.weight),
                "bias": torch.ones_like(model.bias),
                "retired_component.proj.weight": torch.ones(1),
            }
        ),
    )
    monkeypatch.setattr(
        checkpoints_module,
        "set_model_state_dict",
        lambda model, state_dict, options: loaded_keys.extend(state_dict),
    )

    with pytest.warns(RuntimeWarning, match="retired_component.proj.weight"):
        manager.load(path=checkpoint_path, model=model)

    assert loaded_keys == ["weight", "bias"]


def test_distributed_load_preserves_activation_checkpoint_canonical_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    checkpoint_path = tmp_path / "full_training_state.pt"
    checkpoint_path.touch()
    model = torch.nn.Sequential(
        checkpoint_wrapper(torch.nn.Linear(2, 2), preserve_rng_state=False)
    )
    canonical_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    loaded_keys: list[str] = []

    monkeypatch.setattr(checkpoints_module, "_is_rank_zero", lambda: True)
    monkeypatch.setattr(checkpoints_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        checkpoints_module.dist,
        "broadcast_object_list",
        lambda values, src: None,
    )
    monkeypatch.setattr(
        checkpoints_module,
        "_load_tensor_artifact",
        lambda path, map_location: _full_training_state(canonical_state),
    )
    monkeypatch.setattr(
        checkpoints_module,
        "set_model_state_dict",
        lambda model, state_dict, options: loaded_keys.extend(state_dict),
    )

    manager.load(path=checkpoint_path, model=model)

    assert loaded_keys == list(canonical_state)
