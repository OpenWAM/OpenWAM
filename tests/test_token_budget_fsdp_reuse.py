"""Opt-in same-instance FSDP transition: B1 -> B1 -> mixed -> mixed.

Run ``OPEN_WAM_RUN_GPU_SANITY=1 pytest -x -s -m gpu
tests/test_token_budget_fsdp_reuse.py``. The test owns its two-worker torchrun
launcher; do not launch pytest itself under torchrun. The original four FSDP
cases are unchanged. This separate matrix uses FP32 and activation checkpointing
with the original strict tolerances: existing single-device BF16 SDPA/Flex bounds
do not establish a multi-update FSDP optimizer-parity tolerance.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


_SPEC = importlib.util.spec_from_file_location(
    "_fsdp_reuse_gate_support", Path(__file__).with_name("test_token_budget_fsdp.py")
)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)

pytestmark = [pytest.mark.gpu, pytest.mark.integration]
_STEPS = 4
_MOMENTUM = 0.9
_NOISY_CONDITION_PROB = 0.5


def _step_samples(program, rank, step):
    return [
        replace(
            sample,
            video_latents=sample.video_latents + step * 0.01,
            actions=sample.actions + step * 0.02,
            metadata={**sample.metadata, "stream_step": step},
        )
        for sample in gate._samples(program, rank)
    ]


def _snapshot_parameters(model):
    return {name: gate._local_snapshot(p) for name, p in model.named_parameters()}


def _run_trajectory(config, strategy, rank, stream, context, phase):
    from torch.distributed.fsdp import FSDPModule
    from open_wam.models.policy_variants.dual_expert import packed_training

    torch.manual_seed(57)
    pipeline = gate.build_variant_pipeline_from_config(config)
    pipeline.policy_variant.initialize_for_training(pipeline.visual_tower)
    phase("sharding_begin")
    pipeline = strategy.prepare_model(pipeline)
    topology = pipeline.module_topology()
    assert isinstance(pipeline, FSDPModule)
    assert len(topology.fsdp_atomic_modules) == 2 and not topology.fsdp_block_stacks
    assert all(isinstance(block, FSDPModule) for block in topology.fsdp_atomic_modules)
    phase("sharding_end", atomic_blocks=2)
    executor = gate.PipelineTrainStepExecutor(
        pipeline=pipeline,
        batch_adapter=gate.LatentBatchAdapter(),
        training_config=config.training,
    )
    runtime = gate.TrainingRuntime.__new__(gate.TrainingRuntime)
    runtime.config = config
    runtime.model = pipeline
    runtime.strategy = strategy
    runtime.train_state = gate.TrainState()
    runtime._accumulated_train_metrics = {}
    runtime.last_token_batch_plan = None
    # Momentum makes reuse of optimizer state observable, not merely object ID.
    runtime.optimizer = torch.optim.SGD(
        pipeline.parameters(), lr=0.001, momentum=_MOMENTUM
    )
    runtime.scheduler = torch.optim.lr_scheduler.StepLR(
        runtime.optimizer, step_size=1, gamma=0.5
    )
    logged = []
    runtime.log_sink = SimpleNamespace(log_metrics=lambda **item: logged.append(item))
    original_backward = strategy.backward
    original_optimizer_step = strategy.optimizer_step
    original_sync = strategy.set_gradient_sync
    original_clip = strategy.clip_grad_norm_
    original_video_artifacts = packed_training.build_video_flow_match_train_artifacts
    observation = {}
    history = []
    identities = (id(pipeline), id(runtime.optimizer), id(runtime.scheduler))

    def video_artifacts(*args, **kwargs):
        # Observe the real decision after the original builder has consumed its
        # RNG. Do not sample, force a branch, or modify any forwarded argument.
        artifacts = original_video_artifacts(*args, **kwargs)
        assert kwargs["noisy_condition_prob"] == _NOISY_CONDITION_PROB
        observation["augmentation_decisions"].append(
            bool((artifacts.condition_timesteps.detach() > 0).any().item())
        )
        return artifacts

    def forward(batch):
        call = len(observation["physical_sizes"]) + 1
        size = len(batch.sequence_lengths)
        observation["physical_sizes"].append(size)
        phase("forward_begin", physical_call=call, samples=size)
        result = executor.forward_train(batch)
        torch.cuda.synchronize(strategy.device)
        phase("forward_end", physical_call=call, samples=size)
        for output in result.output.sample_outputs:
            observation["predictions"].append(
                output.decoder_output.action_pred.detach().cpu().clone()
            )
            observation["losses"].append(output.decoder_output.loss.detach().cpu().clone())
        return result

    def backward(loss):
        call = len(observation["physical_sizes"])
        phase("backward_begin", physical_call=call)
        original_backward(loss)
        torch.cuda.synchronize(strategy.device)
        phase("backward_end", physical_call=call)

    def optimizer_step(optimizer):
        phase("optimizer_begin")
        observation["optimizer_calls"] += 1
        observation["gradients"] = {
            name: gate._local_snapshot(p.grad)
            for name, p in pipeline.named_parameters()
            if p.grad is not None
        }
        original_optimizer_step(optimizer)
        torch.cuda.synchronize(strategy.device)
        observation["momentum"] = {
            name: gate._local_snapshot(optimizer.state[p]["momentum_buffer"])
            for name, p in pipeline.named_parameters()
            if "momentum_buffer" in optimizer.state[p]
        }
        phase("optimizer_end")

    def set_sync(model, *, enabled):
        observation["sync_calls"].append(enabled)
        original_sync(model, enabled=enabled)

    def clip_grad_norm(parameters, max_grad_norm):
        observation["clip_calls"].append(max_grad_norm)
        phase("clip_grad_norm_begin", max_grad_norm=max_grad_norm)
        norm = original_clip(parameters, max_grad_norm)
        torch.cuda.synchronize(strategy.device)
        observation["gradient_norm"] = float(norm.item())
        phase("clip_grad_norm_end", gradient_norm=observation["gradient_norm"])
        return norm

    runtime.step_executor = SimpleNamespace(
        batch_adapter=executor.batch_adapter, forward_train=forward
    )
    strategy.backward = backward
    strategy.optimizer_step = optimizer_step
    strategy.set_gradient_sync = set_sync
    strategy.clip_grad_norm_ = clip_grad_norm
    packed_training.build_video_flow_match_train_artifacts = video_artifacts
    try:
        for step in range(_STEPS):
            schedule = "mixed" if stream == "candidate" and step >= 2 else "single_control"
            context.update(step=step, schedule=schedule)
            assert identities == (id(runtime.model), id(runtime.optimizer), id(runtime.scheduler))
            legacy = config.policy_variant.program is gate.VideoActionProgram.VIDEO_THEN_ACTION
            max_tokens = config.data.batching.max_tokens if schedule == "mixed" else (92 if legacy else 84)
            runtime.config = replace(
                config,
                data=replace(config.data, batching=replace(config.data.batching, max_tokens=max_tokens)),
            )
            runtime.token_cost_fn = gate.build_training_token_cost_fn(runtime.config)
            batch = gate.LatentBatchCollator(runtime.config.data.batching)(
                _step_samples(config.policy_variant.program, rank, step)
            )
            before = _snapshot_parameters(pipeline)
            learning_rate = runtime.optimizer.param_groups[0]["lr"]
            observation = {
                "step": step, "schedule": schedule, "learning_rate": learning_rate,
                "predictions": [], "losses": [], "physical_sizes": [],
                "sync_calls": [], "clip_calls": [], "optimizer_calls": 0,
                "augmentation_decisions": [],
            }
            torch.manual_seed(911 + rank + step * 1000)
            phase("logical_begin", logical_samples=3, max_tokens=max_tokens)
            runtime._train_micro_step(batch)
            phase("logical_end")
            expected_groups = (
                ((0,), (1,), (2,)) if schedule == "single_control"
                else (((0, 1), (2,)) if rank == 0 else ((0,), (1, 2)))
            )
            plan = runtime.last_token_batch_plan
            assert plan["groups"] == expected_groups
            assert observation["physical_sizes"] == list(map(len, expected_groups))
            assert max(plan["physical_token_counts"]) <= max_tokens
            assert observation["sync_calls"] == [False] * (len(expected_groups) - 1) + [True, True]
            assert observation["optimizer_calls"] == 1 and observation["clip_calls"] == [2.0]
            assert runtime.train_state.global_step == runtime.train_state.seen_batches == step + 1
            assert runtime.train_state.optimizer_step == runtime.scheduler.last_epoch == step + 1
            assert len(logged) == step + 1
            assert runtime.optimizer.param_groups[0]["lr"] == 0.001 * 0.5 ** (step + 1)
            assert len(observation["predictions"]) == len(observation["losses"]) == 3
            assert len(observation["augmentation_decisions"]) == 3
            gradients, momentum = observation["gradients"], observation["momentum"]
            assert len(gradients) == len(momentum) == 138
            assert gradients.keys() == momentum.keys()
            assert all(torch.isfinite(g).all() for g in gradients.values())
            assert any(torch.count_nonzero(g) for g in gradients.values())
            assert all(torch.isfinite(x).all() for x in observation["losses"])
            after = _snapshot_parameters(pipeline)
            assert any(not torch.equal(before[name], after[name]) for name in after)
            for name, old in before.items():
                expected = old
                if name in gradients:
                    expected_momentum = gradients[name]
                    if history:
                        expected_momentum = _MOMENTUM * history[-1]["momentum"][name] + expected_momentum
                    torch.testing.assert_close(momentum[name], expected_momentum, atol=3e-6, rtol=3e-4)
                    expected = old - learning_rate * expected_momentum
                torch.testing.assert_close(after[name], expected, atol=2e-6, rtol=2e-5)
            observation.update(
                parameters=after, plan=plan, logged_loss=logged[-1]["metrics"]["loss"],
                optimizer_step=runtime.train_state.optimizer_step,
                scheduler_step=runtime.scheduler.last_epoch,
                same_instance_confirmed=True,
            )
            history.append(observation)
    finally:
        strategy.backward = original_backward
        strategy.optimizer_step = original_optimizer_step
        strategy.set_gradient_sync = original_sync
        strategy.clip_grad_norm_ = original_clip
        packed_training.build_video_flow_match_train_artifacts = original_video_artifacts
    assert len(history) == _STEPS
    return history


def _compare_trajectories(reference, actual, context, phase):
    evidence = []
    for expected, observed in zip(reference, actual, strict=True):
        context.update(step=observed["step"], schedule=observed["schedule"])
        phase("parity_begin")
        for key in (
            "step", "learning_rate", "optimizer_step", "scheduler_step",
            "same_instance_confirmed", "augmentation_decisions",
        ):
            assert observed[key] == expected[key]
        for key in ("predictions", "losses"):
            for lhs, rhs in zip(observed[key], expected[key], strict=True):
                torch.testing.assert_close(lhs, rhs, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            torch.tensor(observed["logged_loss"]), torch.tensor(expected["logged_loss"]),
            atol=2e-6, rtol=2e-5,
        )
        for key, atol, rtol in (
            ("gradients", 3e-6, 3e-4), ("momentum", 3e-6, 3e-4), ("parameters", 2e-6, 2e-5),
        ):
            assert observed[key].keys() == expected[key].keys()
            for name, target in expected[key].items():
                torch.testing.assert_close(
                    observed[key][name], target, atol=atol, rtol=rtol,
                    msg=lambda detail, key=key, name=name: f"{key}/{name}: {detail}",
                )
        phase("parity_end", checked_gradients=len(observed["gradients"]))
        evidence.append({
            "step": observed["step"], "schedule": observed["schedule"],
            "reference_physical_sizes": expected["physical_sizes"],
            "candidate_physical_sizes": observed["physical_sizes"],
            "reference_plan": expected["plan"], "candidate_plan": observed["plan"],
            "learning_rate": observed["learning_rate"],
            "optimizer_step": observed["optimizer_step"],
            "scheduler_step": observed["scheduler_step"],
            "checked_gradients": len(observed["gradients"]),
            "same_instance_confirmed": observed["same_instance_confirmed"],
            "augmentation_decisions": observed["augmentation_decisions"],
        })
    return evidence


def _augmentation_coverage(decisions_by_rank):
    """Require actual stochastic branch coverage, never silently accept one branch."""
    assert len(decisions_by_rank) == 2
    for trajectory in decisions_by_rank:
        assert len(trajectory) == _STEPS
        assert all(len(step) == 3 for step in trajectory)
        assert all(type(value) is bool for step in trajectory for value in step)
    flattened = [
        [value for step in trajectory for value in step]
        for trajectory in decisions_by_rank
    ]
    assert all(set(values) == {False, True} for values in flattened), (
        "Fixed seeds must exercise both clean and noisy conditions on each rank."
    )
    differences = sum(
        first != second
        for first, second in zip(*flattened, strict=True)
    )
    assert differences > 0, (
        "Fixed seeds must exercise different augmentation decisions across ranks."
    )
    return {
        "clean_counts": [values.count(False) for values in flattened],
        "noisy_counts": [values.count(True) for values in flattened],
        "different_logical_samples": differences,
    }


def _worker_main(args):
    rank = int(os.environ["RANK"])
    context = {
        "rank": rank, "program": args.program, "ac": True, "precision": "32-true",
        "noisy_condition_prob": _NOISY_CONDITION_PROB,
    }

    def phase(name, **extra):
        gate._write_phase({**context, "phase": name, "monotonic": time.monotonic(), **extra})

    phase("worker_ready", pid=os.getpid(), pgid=os.getpgrp(), ppid=os.getppid())
    if args.startup_release is not None:
        deadline = time.monotonic() + gate._STARTUP_SECONDS + 30
        while not args.startup_release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("phase=startup: launcher did not release ready workers")
            time.sleep(0.05)
    started = time.monotonic()

    def expired():
        phase("execution_watchdog", timeout_seconds=gate._EXECUTION_SECONDS)
        os._exit(124)

    watchdog = threading.Timer(gate._EXECUTION_SECONDS, expired)
    watchdog.daemon = True
    watchdog.start()
    strategy, original_full_tensor = None, None
    full_tensor_calls = []
    try:
        assert args.activation_checkpointing == 1
        assert int(os.environ["WORLD_SIZE"]) == 2
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available():
            raise RuntimeError("The same-instance gate requires two CUDA devices and NCCL")
        from torch.distributed.tensor import DTensor

        original_full_tensor = DTensor.full_tensor

        def full_tensor(value, *positional, **keywords):
            details = {key: context.get(key) for key in ("stream", "step", "schedule")}
            details["shape"] = list(value.shape)
            full_tensor_calls.append(details)
            phase("full_tensor_begin", shape=details["shape"])
            result = original_full_tensor(value, *positional, **keywords)
            phase("full_tensor_end", shape=details["shape"])
            return result

        DTensor.full_tensor = full_tensor
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = gate._config(gate.VideoActionProgram(args.program), True)
        config = replace(
            config,
            policy_variant=replace(
                config.policy_variant, noisy_video_condition_prob=_NOISY_CONDITION_PROB
            ),
        )
        phase("nccl_init_begin", timeout_seconds=gate._NCCL_SECONDS)
        strategy = gate.DistributedStrategy(
            accelerator=config.trainer.accelerator, precision=config.trainer.precision,
            kind=config.trainer.strategy, distributed_timeout_seconds=gate._NCCL_SECONDS,
            launch_context=gate.DistributedLaunchContext.from_env(),
        )
        phase("nccl_init_end")
        context["stream"] = "reference"
        reference = _run_trajectory(config, strategy, rank, "reference", context, phase)
        gc.collect()
        torch.cuda.empty_cache()
        context.update(stream="candidate", step=None, schedule=None)
        actual = _run_trajectory(config, strategy, rank, "candidate", context, phase)
        evidence = _compare_trajectories(reference, actual, context, phase)
        # This fixed-count diagnostic collective is outside every sample loop
        # and after all numerical parity assertions on both trajectories.
        phase("augmentation_coverage_begin")
        decisions_by_rank = [None, None]
        dist.all_gather_object(
            decisions_by_rank,
            [row["augmentation_decisions"] for row in evidence],
        )
        coverage = _augmentation_coverage(decisions_by_rank)
        phase("augmentation_coverage_end", **coverage)
        summary = {
            **context, "kind": "fsdp_same_instance_reuse", "passed": True,
            "checked_gradients": 138, "trajectory": evidence,
            "single_physical_sizes": reference[0]["physical_sizes"],
            "mixed_physical_sizes": actual[-1]["physical_sizes"],
            "full_tensor_calls": full_tensor_calls,
            "augmentation_decisions_by_rank": decisions_by_rank,
            "augmentation_coverage": coverage,
            "execution_seconds": time.monotonic() - started,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f"rank{rank}.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle)
        phase("complete", execution_seconds=summary["execution_seconds"])
    except BaseException as error:
        phase("error", error_type=type(error).__name__, message=str(error)[:2000])
        raise
    finally:
        if original_full_tensor is not None:
            DTensor.full_tensor = original_full_tensor
        if strategy is not None:
            strategy.close()
        watchdog.cancel()


@pytest.mark.parametrize("program", [gate.VideoActionProgram.VIDEO_THEN_ACTION, gate.VideoActionProgram.JOINT])
def test_same_fsdp_instance_preserves_four_update_transition(program, tmp_path):
    if os.environ.get("OPEN_WAM_RUN_GPU_SANITY") != "1":
        pytest.skip("Set OPEN_WAM_RUN_GPU_SANITY=1 to run the same-instance FSDP gate")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available():
        pytest.skip("The same-instance FSDP gate requires two CUDA GPUs and NCCL")
    gate._launch_case(program, True, tmp_path, worker_script=Path(__file__))
    for rank in range(2):
        with (tmp_path / f"rank{rank}.json").open(encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["kind"] == "fsdp_same_instance_reuse" and report["precision"] == "32-true"
        assert report["ac"] is True and len(report["trajectory"]) == _STEPS
        assert report["noisy_condition_prob"] == _NOISY_CONDITION_PROB
        assert report["augmentation_coverage"] == _augmentation_coverage(
            report["augmentation_decisions_by_rank"]
        )
        assert [row["schedule"] for row in report["trajectory"]] == ["single_control", "single_control", "mixed", "mixed"]
        for step, row in enumerate(report["trajectory"]):
            assert row["step"] == step and row["same_instance_confirmed"]
            assert row["checked_gradients"] == 138
            assert row["optimizer_step"] == row["scheduler_step"] == step + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--program", choices=["video_then_action", "joint"], required=True)
    parser.add_argument("--activation-checkpointing", type=int, choices=[1], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-release", type=Path)
    _worker_main(parser.parse_args())
