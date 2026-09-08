"""Full-state resume must not reinterpret a token-budget logical batch cursor."""
from __future__ import annotations

import copy
from dataclasses import replace
import pytest
import torch

from open_wam.configs import BatchingConfig, BatchingMode, CheckpointMode, ExperimentConfig
import open_wam.training.checkpoints as checkpoints_module
from open_wam.training.checkpoints import CheckpointManager
from open_wam.training.state import TrainState


def config(*, max_tokens=150000, batch_size=6, accumulation=1, mode=BatchingMode.PACKED, pad=1):
    base = ExperimentConfig()
    return replace(
        base,
        data=replace(base.data, train_batch_size=batch_size,
                     batching=BatchingConfig(mode=mode, max_tokens=max_tokens, pad_to_multiple_of=pad)),
        training=replace(base.training, gradient_accumulation_steps=accumulation),
        trainer=replace(base.trainer, devices=1),
    )


def manager(root, cfg):
    return CheckpointManager(root_dir=root, config=cfg, checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE)


def model_and_optimizer():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return model, optimizer, scheduler


def update(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    model(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10).square().mean().backward()
    optimizer.step()
    scheduler.step()


def save_checkpoint(root, cfg=None):
    cfg = config() if cfg is None else cfg
    model, optimizer, scheduler = model_and_optimizer()
    update(model, optimizer, scheduler)
    update(model, optimizer, scheduler)
    state = TrainState(global_step=2, optimizer_step=2, epoch_index=3,
                       next_batch_index=2, seen_batches=11, run_name="token-resume")
    directory = manager(root, cfg).save(step=2, model=model, optimizer=optimizer,
                                        scheduler=scheduler, train_state=state,
                                        strategy_state={"test": True})
    return directory, model, optimizer, scheduler, state


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            assert_tree_equal(first, second)
    else:
        assert actual == expected


def assert_rejected_before_mutation(checkpoint, destination):
    model, optimizer, scheduler = model_and_optimizer()
    update(model, optimizer, scheduler)
    before = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    with pytest.raises(ValueError, match="training_batch_contract.*initialize_weights_from"):
        destination.load(path=checkpoint, model=model, optimizer=optimizer, scheduler=scheduler)
    assert_tree_equal((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()), before)


def test_same_token_contract_resumes_real_weights_optimizer_scheduler_and_cursor(tmp_path):
    directory, source_model, source_optimizer, source_scheduler, source_state = save_checkpoint(tmp_path / "source")
    payload = torch.load(directory / "full_training_state.pt", weights_only=True)
    assert payload["training_batch_contract"] == {
        "version": 1, "train_batch_size": 6, "gradient_accumulation_steps": 1,
        "mode": "packed", "max_tokens": 150000, "pad_to_multiple_of": 1, "world_size": 1,
    }
    assert all(type(value) is int for key, value in payload["training_batch_contract"].items() if key != "mode")
    target_model, target_optimizer, target_scheduler = model_and_optimizer()
    restored, loaded = manager(tmp_path / "target", config()).load(
        path=directory / "full_training_state.pt", model=target_model,
        optimizer=target_optimizer, scheduler=target_scheduler,
    )
    for field in ("global_step", "optimizer_step", "epoch_index", "next_batch_index", "seen_batches", "run_name"):
        assert getattr(restored, field) == getattr(source_state, field)
    assert loaded["strategy_state_dict"] == {"test": True}
    assert_tree_equal(target_model.state_dict(), source_model.state_dict())
    assert_tree_equal(target_optimizer.state_dict(), source_optimizer.state_dict())
    assert_tree_equal(target_scheduler.state_dict(), source_scheduler.state_dict())
    update(source_model, source_optimizer, source_scheduler)
    update(target_model, target_optimizer, target_scheduler)
    assert_tree_equal(target_model.state_dict(), source_model.state_dict())


@pytest.mark.parametrize("overrides", [
    {"max_tokens": 160000}, {"max_tokens": None}, {"batch_size": 2},
    {"accumulation": 2}, {"pad": 2}, {"max_tokens": None, "mode": BatchingMode.STRICT},
])
def test_token_contract_changes_fail_before_model_or_optimizer_mutation(tmp_path, overrides):
    directory, *_ = save_checkpoint(tmp_path / "source")
    assert_rejected_before_mutation(directory / "full_training_state.pt", manager(tmp_path / "target", config(**overrides)))


@pytest.mark.parametrize("field, value", [
    ("world_size", 4), ("version", 2), ("train_batch_size", True),
    ("max_tokens", "150000"), ("gradient_accumulation_steps", 1.0),
])
def test_mismatched_or_malformed_stored_contract_fails_before_mutation(tmp_path, field, value):
    directory, *_ = save_checkpoint(tmp_path / "source")
    checkpoint = directory / "full_training_state.pt"
    payload = torch.load(checkpoint, weights_only=True)
    payload["training_batch_contract"][field] = value
    torch.save(payload, checkpoint)
    assert_rejected_before_mutation(checkpoint, manager(tmp_path / "target", config()))


def test_legacy_checkpoint_without_contract_cannot_resume_into_token_budget(tmp_path):
    directory, *_ = save_checkpoint(tmp_path / "source", config(max_tokens=None))
    checkpoint = directory / "full_training_state.pt"
    assert "training_batch_contract" not in torch.load(checkpoint, weights_only=True)
    assert_rejected_before_mutation(checkpoint, manager(tmp_path / "target", config()))


def test_legacy_non_budget_resume_keeps_existing_behavior(tmp_path):
    directory, source_model, *_ = save_checkpoint(tmp_path / "source", config(max_tokens=None))
    model, optimizer, scheduler = model_and_optimizer()
    # No new compatibility restriction is imposed on non-budgeted legacy runs.
    destination = manager(tmp_path / "target", config(max_tokens=None, batch_size=2, accumulation=2))
    restored, _ = destination.load(path=directory / "full_training_state.pt", model=model,
                                   optimizer=optimizer, scheduler=scheduler)
    assert restored.global_step == 2
    assert_tree_equal(model.state_dict(), source_model.state_dict())


def test_model_only_initialization_allows_changing_token_budget_schedule(tmp_path):
    directory, source_model, *_ = save_checkpoint(tmp_path / "source")
    model, _, _ = model_and_optimizer()
    destination = manager(tmp_path / "target", config(max_tokens=None, mode=BatchingMode.STRICT, batch_size=1, accumulation=6))
    destination.initialize_weights(path=directory / "model_state.pt", model=model)
    assert_tree_equal(model.state_dict(), source_model.state_dict())


def test_token_contract_world_size_uses_actual_group_not_device_hint(monkeypatch):
    monkeypatch.setattr(checkpoints_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoints_module.dist, "get_world_size", lambda: 4)
    assert config().trainer.devices == 1
    assert checkpoints_module._training_batch_contract(config())["world_size"] == 4


def test_token_contract_uses_world_size_one_without_an_initialized_group(monkeypatch):
    monkeypatch.setattr(checkpoints_module.dist, "is_initialized", lambda: False)
    cfg = config()
    cfg = replace(cfg, trainer=replace(cfg.trainer, devices=4))
    assert checkpoints_module._training_batch_contract(cfg)["world_size"] == 1


def test_contract_failure_enters_existing_collective_error_gate_before_loading(tmp_path, monkeypatch):
    directory, *_ = save_checkpoint(tmp_path / "source")
    errors = []
    original_gate = checkpoints_module._raise_checkpoint_validation_error

    def gate(error):
        errors.append(error)
        return original_gate(error)

    monkeypatch.setattr(checkpoints_module, "_raise_checkpoint_validation_error", gate)
    assert_rejected_before_mutation(directory / "full_training_state.pt", manager(tmp_path / "target", config(max_tokens=170000)))
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


def test_token_checkpoint_still_rejects_partial_logical_accumulation(tmp_path):
    destination = manager(tmp_path / "target", config(accumulation=2))
    model, optimizer, scheduler = model_and_optimizer()
    with pytest.raises(ValueError, match="optimizer boundary"):
        destination.save(step=0, model=model, optimizer=optimizer, scheduler=scheduler,
                         train_state=TrainState(global_step=1))
    assert not (tmp_path / "target" / "checkpoint_step_0").exists()
