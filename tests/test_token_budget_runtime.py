"""Logical-batch optimization and real CPU distributed token-budget contracts."""

from __future__ import annotations

import gc
import multiprocessing
import time
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from queue import Empty
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from open_wam.configs import BatchingConfig
from open_wam.data.latent_batching import LatentBatchCollator
from open_wam.data.latent_contracts import (
    LatentWAMSample,
    move_latent_wam_batch_to_device,
)
from open_wam.training.runtime import TrainingRuntime
from open_wam.training.state import TrainState
from open_wam.training.strategies import _set_gradient_sync_recursive


def _metadata_token_costs(batch):
    return tuple(row["synthetic_token_cost"] for row in batch.metadata)


def _sample(sample_id, cost):
    frames = 2 + sample_id % 5
    return LatentWAMSample(
        video_latents=torch.full((1, frames, 1, 1), float(sample_id)),
        actions=torch.full(
            (frames, 1), (sample_id % 4 - 1.5) / 3.0, dtype=torch.float64
        ),
        action_mask=torch.ones(frames, 1, dtype=torch.float64),
        state=torch.tensor(
            [[(sample_id + 1) / 5.0, (sample_id % 3 - 1) / 4.0]],
            dtype=torch.float64,
        ),
        state_mask=torch.ones(1, 2),
        condition_latents=torch.full((1, 1, 1, 1), float(sample_id)),
        text_context=torch.full((2 + sample_id % 2, 3), float(sample_id)),
        task_text=f"synthetic sample {sample_id}",
        metadata={
            "sample_id": sample_id,
            "synthetic_token_cost": cost,
            "sampled_chunk_size": 1,
            "sampled_window_size": 4,
            "history_frames": 1,
        },
    )


def _batch(sample_ids, costs):
    collator = LatentBatchCollator(
        BatchingConfig(mode="packed", pad_to_multiple_of=4)
    )
    return collator(
        [_sample(index, cost) for index, cost in zip(sample_ids, costs, strict=True)]
    )


def _model():
    model = torch.nn.Linear(2, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.25, -0.4]], dtype=torch.float64))
        model.bias.fill_(0.1)
    return model


class _RecordingStrategy:
    device = torch.device("cpu")

    def __init__(self):
        self.sync_enabled = True
        self.sync_calls = []
        self.gradients = []
        self.optimizer_steps = 0

    def set_gradient_sync(self, model, *, enabled):
        self.sync_enabled = enabled
        self.sync_calls.append(enabled)
        _set_gradient_sync_recursive(model, enabled)

    def autocast_context(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def unscale_(self, optimizer):
        del optimizer

    def optimizer_step(self, optimizer):
        self.gradients.append(
            [
                parameter.grad.detach().clone()
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
        )
        self.optimizer_steps += 1
        optimizer.step()

    def zero_grad(self, optimizer):
        optimizer.zero_grad(set_to_none=True)


@dataclass
class _ForwardResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class _RecordingExecutor:
    def __init__(self, model, strategy, *, expect_trimmed):
        self.model = model
        self.strategy = strategy
        self.expect_trimmed = expect_trimmed
        self.batch_adapter = SimpleNamespace(
            move_to_device=move_latent_wam_batch_to_device
        )
        self.forward_sync = []
        self.sample_groups = []
        self.previous_graph_refs = []

    def forward_train(self, batch):
        gc.collect()
        assert all(reference() is None for reference in self.previous_graph_refs), (
            "A prior physical microbatch graph/output survived into the next forward."
        )
        self.forward_sync.append(self.strategy.sync_enabled)
        self.sample_groups.append(tuple(row["sample_id"] for row in batch.metadata))
        if self.expect_trimmed:
            assert batch.video_latents.shape[2] == max(batch.sequence_lengths)
            assert batch.actions.shape[1] == max(batch.tensor_lengths["actions"])
            assert batch.text_context.shape[1] == max(
                batch.tensor_lengths["text_context"]
            )
        for index, row in enumerate(batch.metadata):
            assert batch.state[index, 0, 0] == (row["sample_id"] + 1) / 5.0
            assert row["sampled_chunk_size"] == 1
            assert row["sampled_window_size"] == 4
        prediction = self.model(batch.state[:, 0]).squeeze(-1)
        loss = (prediction - batch.actions[:, 0, 0]).square().mean()
        result = _ForwardResult(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "sample_stat": batch.state[:, 0, 0].mean(),
            },
        )
        self.previous_graph_refs = [
            weakref.ref(result),
            weakref.ref(loss),
            weakref.ref(prediction),
        ]
        return result


def _runtime(*, accumulation, max_tokens, model=None):
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.model = _model() if model is None else model
    runtime.strategy = _RecordingStrategy()
    runtime.step_executor = _RecordingExecutor(
        runtime.model, runtime.strategy, expect_trimmed=max_tokens is not None
    )
    runtime.optimizer = torch.optim.SGD(
        runtime.model.parameters(), lr=0.2, momentum=0.9
    )
    runtime.scheduler = torch.optim.lr_scheduler.StepLR(
        runtime.optimizer, step_size=1, gamma=0.5
    )
    runtime.train_state = TrainState(run_name="synthetic-token-budget")
    runtime.config = SimpleNamespace(
        data=SimpleNamespace(batching=SimpleNamespace(max_tokens=max_tokens)),
        training=SimpleNamespace(
            gradient_accumulation_steps=accumulation, max_grad_norm=None
        ),
        trainer=SimpleNamespace(log_every_n_steps=1, save_interval=None),
    )
    runtime.logged = []
    runtime.log_sink = SimpleNamespace(
        log_metrics=lambda **payload: runtime.logged.append(payload)
    )
    runtime._accumulated_train_metrics = {}
    runtime.last_token_batch_plan = None
    runtime.token_cost_fn = _metadata_token_costs
    return runtime


def _run_one_update(*, batch_size, accumulation, max_tokens):
    runtime = _runtime(accumulation=accumulation, max_tokens=max_tokens)
    costs = (1, 3, 4, 2, 1, 3)
    for start in range(0, 6, batch_size):
        runtime._train_micro_step(
            _batch(range(start, start + batch_size), costs[start : start + batch_size])
        )
    return runtime


@pytest.mark.unit
@pytest.mark.parametrize("batch_size,accumulation", [(6, 1), (2, 3)])
def test_token_budget_preserves_reference_gradient_optimizer_metrics_and_sync(
    batch_size, accumulation
):
    reference = _run_one_update(batch_size=1, accumulation=6, max_tokens=None)
    actual = _run_one_update(
        batch_size=batch_size, accumulation=accumulation, max_tokens=4
    )

    assert actual.train_state.global_step == actual.train_state.seen_batches == accumulation
    assert actual.train_state.optimizer_step == actual.strategy.optimizer_steps == 1
    assert actual.scheduler.last_epoch == 1
    assert actual.scheduler.state_dict() == reference.scheduler.state_dict()
    for expected, observed in zip(
        reference.strategy.gradients[0], actual.strategy.gradients[0], strict=True
    ):
        torch.testing.assert_close(observed, expected, atol=1e-14, rtol=1e-14)
    for expected, observed in zip(
        reference.model.parameters(), actual.model.parameters(), strict=True
    ):
        torch.testing.assert_close(observed, expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(
            actual.optimizer.state[observed]["momentum_buffer"],
            reference.optimizer.state[expected]["momentum_buffer"],
            atol=1e-14,
            rtol=1e-14,
        )
    assert len(actual.logged) == 1
    for name in ("loss", "sample_stat"):
        assert actual.logged[0]["metrics"][name] == pytest.approx(
            reference.logged[0]["metrics"][name], abs=1e-7
        )
    assert actual.logged[0]["metrics"]["sample_stat"] == pytest.approx(0.7)
    forward_sync = actual.step_executor.forward_sync
    assert forward_sync == [False] * (len(forward_sync) - 1) + [True]
    assert actual.strategy.sync_calls == forward_sync + [True]
    assert tuple(
        sample for group in actual.step_executor.sample_groups for sample in group
    ) == tuple(range(6))
    assert len(actual.step_executor.sample_groups) == 4
    assert actual.last_token_batch_plan["logical_samples"] == batch_size
    assert actual.last_token_batch_plan["max_tokens"] == 4
    assert max(actual.last_token_batch_plan["physical_token_counts"]) <= 4
    gc.collect()
    assert all(
        reference() is None
        for reference in actual.step_executor.previous_graph_refs
    )


@pytest.mark.unit
def test_unequal_physical_group_metrics_are_sample_weighted_and_cursor_is_logical():
    runtime = _runtime(accumulation=2, max_tokens=4)
    runtime._train_micro_step(_batch(range(6), (1, 3, 4, 2, 1, 3)))
    assert runtime.step_executor.sample_groups == [(0, 1), (2,), (3, 4), (5,)]
    assert runtime.step_executor.forward_sync == [False] * 4
    assert runtime.strategy.optimizer_steps == 0
    assert runtime.scheduler.last_epoch == 0
    assert runtime.train_state.global_step == runtime.train_state.seen_batches == 1
    assert runtime.train_state.next_batch_index == 0
    assert runtime.logged == []
    assert sum(runtime._accumulated_train_metrics["sample_stat"]).item() == pytest.approx(
        0.7 / 2
    )
    assert runtime.last_token_batch_plan == {
        "token_costs": (1, 3, 4, 2, 1, 3),
        "groups": ((0, 1), (2,), (3, 4), (5,)),
        "physical_token_counts": (4, 4, 3, 3),
        "logical_samples": 6,
        "max_tokens": 4,
    }


@pytest.mark.unit
def test_missing_sample_cost_fails_before_forward_or_training_state_mutation():
    runtime = _runtime(accumulation=1, max_tokens=4)
    runtime.token_cost_fn = lambda batch: (1,) * (len(batch.sequence_lengths) - 1)
    parameters_before = [parameter.detach().clone() for parameter in runtime.model.parameters()]
    with pytest.raises(ValueError, match="Token-budget admission failed.*every original sample"):
        runtime._train_micro_step(_batch(range(3), (1, 1, 1)))
    assert runtime.step_executor.sample_groups == []
    assert runtime.strategy.optimizer_steps == 0
    assert runtime.train_state.global_step == runtime.train_state.seen_batches == 0
    assert runtime.train_state.optimizer_step == runtime.scheduler.last_epoch == 0
    assert runtime.last_token_batch_plan is None
    assert runtime.logged == []
    for before, after in zip(parameters_before, runtime.model.parameters(), strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


def _distributed_worker(rank, rendezvous, scenario, results, execution_event):
    """Top-level spawn entrypoint; return compact results without environment dumps."""
    results.put({"phase": "ready", "rank": rank, "monotonic": time.monotonic()})
    # Imports happen before this entrypoint under spawn. Do not let the first
    # imported worker consume its Gloo timeout while its peer is still importing.
    # The parent owns a 180-second startup deadline and terminates its workers
    # on failure; this extra bound also prevents an orphaned event waiter.
    if not execution_event.wait(timeout=240):
        results.put({
            "phase": "startup_error", "rank": rank,
            "error": "Execution event was not released within 240 seconds.",
        })
        return
    execution_started = time.monotonic()
    payload = {"phase": "result", "rank": rank}
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=20),
        )
        model = DistributedDataParallel(_model())
        runtime = _runtime(accumulation=1, max_tokens=6, model=model)
        costs = (1, 1, 1) if rank == 0 else (4, 4, 4)
        if scenario == "over_budget" and rank == 1:
            costs = (4, 7, 4)
        batch = _batch(range(rank * 3, rank * 3 + 3), costs)
        if scenario == "over_budget":
            try:
                runtime._train_micro_step(batch)
            except ValueError as error:
                payload.update({
                    "admission_error": str(error),
                    "forward_calls": len(runtime.step_executor.forward_sync),
                    "optimizer_steps": runtime.strategy.optimizer_steps,
                    "global_step": runtime.train_state.global_step,
                })
            else:
                payload["error"] = "Over-budget admission succeeded."
        else:
            runtime._train_micro_step(batch)
            payload.update({
                "gradients": [value.tolist() for value in runtime.strategy.gradients[0]],
                "parameters": [value.detach().tolist() for value in model.parameters()],
                "metrics": runtime.logged[0]["metrics"],
                "forward_sync": runtime.step_executor.forward_sync,
                "plan": runtime.last_token_batch_plan,
                "global_step": runtime.train_state.global_step,
                "optimizer_steps": runtime.strategy.optimizer_steps,
                "scheduler_steps": runtime.scheduler.last_epoch,
            })
    except BaseException as error:
        payload["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except BaseException as error:
            payload.setdefault("error", f"Process-group cleanup: {type(error).__name__}: {error}")
    payload["execution_seconds"] = time.monotonic() - execution_started
    results.put(payload)


def _phase_diagnostic(phase, ready_ranks, processes):
    exitcodes = {rank: process.exitcode for rank, process in enumerate(processes)}
    return f"phase={phase}; ready_ranks={sorted(ready_ranks)}; exitcodes={exitcodes}"


def _receive_worker_phase(
    results, processes, *, message_phase, phase, deadline, budget_seconds, ready_ranks
):
    """Poll a bounded phase and distinguish delayed imports from collective work."""
    received = {}
    expected_ranks = set(range(len(processes)))
    while set(received) != expected_ranks:
        remaining = deadline - time.monotonic()
        ready = set(received) if message_phase == "ready" else set(ready_ranks)
        diagnostic = _phase_diagnostic(phase, ready, processes)
        if remaining <= 0:
            pytest.fail(f"Gloo smoke exceeded its {budget_seconds}-second budget; {diagnostic}.")
        try:
            message = results.get(timeout=min(0.5, remaining))
        except Empty:
            pending = expected_ranks - set(received)
            if any(processes[rank].exitcode is not None for rank in pending):
                # A worker can flush its last message and exit between the
                # timed get and the exitcode observation. Drain once before
                # treating that normal shutdown as a missing-message failure.
                try:
                    message = results.get_nowait()
                except Empty:
                    diagnostic = _phase_diagnostic(phase, ready, processes)
                    pytest.fail(f"Gloo worker exited before its {message_phase} message; {diagnostic}.")
            else:
                continue
        rank = message.get("rank")
        if message.get("error"):
            pytest.fail(f"Gloo worker {rank} failed: {message['error']}; {diagnostic}.")
        if rank not in expected_ranks or rank in received or message.get("phase") != message_phase:
            pytest.fail(f"Unexpected Gloo worker phase/rank message; {diagnostic}.")
        received[rank] = message
    return received


def _run_two_rank_smoke(tmp_path, scenario):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    execution_event = context.Event()
    rendezvous = (tmp_path / f"gloo-{scenario}").as_uri()
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, rendezvous, scenario, results, execution_event),
        )
        for rank in range(2)
    ]
    startup_started = {}
    ready = {}
    startup_deadline = time.monotonic() + 180
    try:
        for rank, process in enumerate(processes):
            startup_started[rank] = time.monotonic()
            try:
                process.start()
            except Exception as error:
                pytest.fail(
                    f"Could not spawn Gloo worker {rank}: {type(error).__name__}: {error}; "
                    f"{_phase_diagnostic('startup', ready, processes)}."
                )
        ready = _receive_worker_phase(
            results, processes, message_phase="ready", phase="startup",
            deadline=startup_deadline, budget_seconds=180, ready_ranks=(),
        )
        startup_seconds = {
            rank: message["monotonic"] - startup_started[rank]
            for rank, message in ready.items()
        }
        # Only a fully imported pair may enter Gloo. Execution and interpreter
        # shutdown together retain the original 50-second deadline.
        execution_deadline = time.monotonic() + 50
        execution_event.set()
        received = _receive_worker_phase(
            results, processes, message_phase="result", phase="execution",
            deadline=execution_deadline, budget_seconds=50, ready_ranks=ready,
        )
        for process in processes:
            process.join(timeout=max(0, execution_deadline - time.monotonic()))
            diagnostic = _phase_diagnostic("execution_exit", ready, processes)
            assert not process.is_alive(), f"Gloo worker did not exit within 50 seconds; {diagnostic}."
            assert process.exitcode == 0, f"Gloo worker exited unsuccessfully; {diagnostic}."
        ordered = [received[rank] for rank in sorted(received)]
        for result in ordered:
            result["startup_seconds"] = startup_seconds[result["rank"]]
        timings = "; ".join(
            f"rank {result['rank']} startup={result['startup_seconds']:.3f}s "
            f"execution={result['execution_seconds']:.3f}s"
            for result in ordered
        )
        print(f"Gloo smoke[{scenario}] phase=complete: {timings}", flush=True)
        return ordered
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        results.close()
        results.join_thread()


@pytest.mark.unit
@pytest.mark.parametrize("phase,budget", [("startup", 180), ("execution", 50)])
def test_phase_timeout_diagnostic_identifies_separate_budget_and_worker_status(phase, budget):
    processes = [SimpleNamespace(exitcode=None), SimpleNamespace(exitcode=None)]
    with pytest.raises(pytest.fail.Exception, match=rf"{budget}-second budget; phase={phase}.*exitcodes"):
        _receive_worker_phase(
            None, processes,
            message_phase="ready" if phase == "startup" else "result",
            phase=phase, deadline=time.monotonic() - 1, budget_seconds=budget,
            ready_ranks=(0, 1) if phase == "execution" else (),
        )


@pytest.mark.unit
def test_startup_reports_early_worker_exit_without_waiting_for_full_budget():
    def empty_queue(**kwargs):
        del kwargs
        raise Empty

    processes = [SimpleNamespace(exitcode=None), SimpleNamespace(exitcode=7)]
    with pytest.raises(pytest.fail.Exception, match=r"exited before its ready message.*phase=startup.*1: 7"):
        _receive_worker_phase(
            SimpleNamespace(get=empty_queue, get_nowait=empty_queue), processes,
            message_phase="ready", phase="startup", deadline=time.monotonic() + 180,
            budget_seconds=180, ready_ranks=(),
        )


@pytest.mark.unit
def test_execution_drains_message_flushed_between_timeout_and_worker_exit():
    processes = [SimpleNamespace(exitcode=None), SimpleNamespace(exitcode=None)]
    messages = [
        {"phase": "result", "rank": 0, "execution_seconds": 0.2},
        {"phase": "result", "rank": 1, "execution_seconds": 0.3},
    ]
    drains = []

    def timed_get(**kwargs):
        del kwargs
        if len(messages) == 2:
            return messages.pop(0)
        # Simulate a get timeout immediately followed by a flushed result and
        # normal child exit, before the parent examines the process exitcode.
        processes[1].exitcode = 0
        raise Empty

    def final_drain():
        drains.append(1)
        return messages.pop(0)

    received = _receive_worker_phase(
        SimpleNamespace(get=timed_get, get_nowait=final_drain), processes,
        message_phase="result", phase="execution", deadline=time.monotonic() + 50,
        budget_seconds=50, ready_ranks=(0, 1),
    )
    assert set(received) == {0, 1}
    assert received[1]["execution_seconds"] == 0.3
    assert drains == [1]
    assert messages == []


@pytest.mark.unit
def test_startup_requires_both_ready_messages_and_preserves_worker_timestamps():
    messages = iter([
        {"phase": "ready", "rank": 1, "monotonic": 15.0},
        {"phase": "ready", "rank": 0, "monotonic": 17.0},
    ])
    ready = _receive_worker_phase(
        SimpleNamespace(get=lambda **kwargs: next(messages)),
        [SimpleNamespace(exitcode=None), SimpleNamespace(exitcode=None)],
        message_phase="ready", phase="startup", deadline=time.monotonic() + 180,
        budget_seconds=180, ready_ranks=(),
    )
    assert set(ready) == {0, 1}
    assert ready[0]["monotonic"] == 17.0
    assert ready[1]["monotonic"] == 15.0


@pytest.mark.integration
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Requires a PyTorch build with the CPU Gloo distributed backend.",
)
def test_two_rank_gloo_agrees_schedule_and_ddp_matches_global_sample_gradient(tmp_path):
    results = _run_two_rank_smoke(tmp_path, "unequal_groups")
    model = _model()
    batch = _batch(range(6), (1, 1, 1, 4, 4, 4))
    loss = (model(batch.state[:, 0]).squeeze(-1) - batch.actions[:, 0, 0]).square().mean()
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    expected_parameters = [
        parameter.detach() - 0.2 * gradient
        for parameter, gradient in zip(model.parameters(), gradients, strict=True)
    ]
    for result in results:
        assert 0 <= result["startup_seconds"] <= 180
        assert 0 <= result["execution_seconds"] <= 50
        assert result["plan"]["groups"] == ((0,), (1,), (2,))
        assert result["plan"]["physical_token_counts"] == (
            (1, 1, 1) if result["rank"] == 0 else (4, 4, 4)
        )
        assert result["forward_sync"] == [False, False, True]
        assert result["global_step"] == result["optimizer_steps"] == 1
        assert result["scheduler_steps"] == 1
        for actual, expected in zip(result["gradients"], gradients, strict=True):
            torch.testing.assert_close(
                torch.tensor(actual, dtype=torch.float64), expected, atol=1e-12, rtol=1e-12
            )
        for actual, expected in zip(
            result["parameters"], expected_parameters, strict=True
        ):
            torch.testing.assert_close(
                torch.tensor(actual, dtype=torch.float64), expected, atol=1e-12, rtol=1e-12
            )
        assert result["metrics"]["loss"] == pytest.approx(loss.item(), abs=1e-7)
        assert result["metrics"]["sample_stat"] == pytest.approx(0.7, abs=1e-7)


@pytest.mark.integration
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Requires a PyTorch build with the CPU Gloo distributed backend.",
)
def test_two_rank_gloo_rejects_one_rank_over_budget_before_any_forward(tmp_path):
    results = _run_two_rank_smoke(tmp_path, "over_budget")
    for result in results:
        assert 0 <= result["startup_seconds"] <= 180
        assert 0 <= result["execution_seconds"] <= 50
        assert "Token-budget admission failed" in result["admission_error"]
        assert "rank 1" in result["admission_error"]
        assert "exceeding max_tokens=6" in result["admission_error"]
        assert result["forward_calls"] == result["optimizer_steps"] == result["global_step"] == 0
