"""Metric-accounting invariants for variable-length training batches."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import BatchingMode
from open_wam.training.runtime import TrainingRuntime, _validation_metric_weight
from open_wam.training.state import TrainState


def _validation_batch(mode: BatchingMode, values: list[float]):
    return SimpleNamespace(
        batching_mode=mode,
        sequence_lengths=tuple(4 + index for index in range(len(values))),
        actions=torch.tensor(values)[:, None, None],
    )


def _validation_runtime(batches: list):
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.val_loader = batches
    runtime.model = SimpleNamespace(eval=lambda: None)
    runtime.strategy = SimpleNamespace(
        device=torch.device("cpu"), autocast_context=lambda: nullcontext()
    )
    runtime.train_state = TrainState(optimizer_step=17)
    logged = []
    runtime.log_sink = SimpleNamespace(
        log_metrics=lambda **kwargs: logged.append(kwargs)
    )
    runtime.step_executor = SimpleNamespace(
        batch_adapter=SimpleNamespace(move_to_device=lambda batch, device: batch),
        forward_train=lambda batch: SimpleNamespace(
            metrics={"loss": batch.actions.mean()}
        ),
    )
    return runtime, logged


@pytest.mark.parametrize(
    "mode", [BatchingMode.BUCKET, BatchingMode.PADDED, BatchingMode.PACKED]
)
def test_variable_validation_weights_original_samples_not_batches(mode):
    runtime, logged = _validation_runtime(
        [_validation_batch(mode, [1.0, 3.0]), _validation_batch(mode, [8.0])]
    )

    assert runtime._run_validation(limit_batches=None)

    assert logged[0]["metrics"]["loss"] == pytest.approx(4.0)


def test_strict_validation_preserves_legacy_batch_mean():
    runtime, logged = _validation_runtime(
        [
            _validation_batch(BatchingMode.STRICT, [1.0, 3.0]),
            _validation_batch(BatchingMode.STRICT, [8.0]),
        ]
    )

    assert runtime._run_validation(limit_batches=None)

    assert logged[0]["metrics"]["loss"] == pytest.approx(5.0)


def test_variable_validation_reduces_sample_sums_and_counts_across_ranks(monkeypatch):
    runtime, logged = _validation_runtime(
        [_validation_batch(BatchingMode.PACKED, [1.0, 3.0])]
    )
    # This rank contributes sum=4/count=2. The peer has one sample with loss=8.
    reductions = []

    def reduce_sum(tensor, op):
        del op
        reductions.append(tensor.item())
        tensor.add_(1.0 if len(reductions) == 1 else 8.0)

    monkeypatch.setattr("open_wam.training.runtime.dist.is_initialized", lambda: True)
    monkeypatch.setattr("open_wam.training.runtime.dist.all_reduce", reduce_sum)

    assert runtime._run_validation(limit_batches=None)

    assert reductions == [2.0, 4.0]
    assert logged[0]["metrics"]["loss"] == pytest.approx(4.0)


def test_variable_validation_honors_limit_without_counting_unvisited_samples():
    runtime, logged = _validation_runtime(
        [
            _validation_batch(BatchingMode.PADDED, [1.0, 3.0]),
            _validation_batch(BatchingMode.PADDED, [8.0]),
        ]
    )

    assert runtime._run_validation(limit_batches=1)

    assert logged[0]["metrics"]["loss"] == pytest.approx(2.0)


@pytest.mark.parametrize("lengths", [(), (4, 0), (4, -1), (4,)])
def test_variable_validation_rejects_invalid_original_sample_counts(lengths):
    batch = _validation_batch(BatchingMode.PADDED, [1.0, 3.0])
    batch.sequence_lengths = lengths
    with pytest.raises(ValueError, match="sequence_lengths"):
        _validation_metric_weight(batch)
