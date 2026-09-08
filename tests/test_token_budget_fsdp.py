"""Opt-in two-CUDA-rank FSDP2 gate for unequal physical sample counts.

Run with ``OPEN_WAM_RUN_GPU_SANITY=1 pytest -x -s -m gpu
tests/test_token_budget_fsdp.py``. Pytest owns the two-worker torchrun launcher;
do not launch pytest itself under torchrun. No checkpoint or dataset is needed.

For an externally supervised single case, the same file accepts ``torchrun
--standalone --nproc_per_node=2 tests/test_token_budget_fsdp.py --worker
--program video_then_action --activation-checkpointing 1 --output-dir DIR``.
The output directory must be fresh. Direct workers retain an execution watchdog,
but only the pytest launcher separates Python import/startup from execution time.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

# The direct torchrun entrypoint does not import tests/conftest.py.
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from open_wam.configs import (  # noqa: E402
    ActionSchemaConfig,
    BatchingConfig,
    BatchingMode,
    DualExpertActionDecoderConfig,
    DualExpertPolicyConfig,
    ExperimentConfig,
    HistoryStreamVisibility,
    InferenceConfig,
    ProprioContextMode,
    RobotWinDataConfig,
    TrainerConfig,
    TrainingConfig,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.data.latent_batching import LatentBatchCollator  # noqa: E402
from open_wam.data.latent_contracts import LatentWAMSample  # noqa: E402
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig  # noqa: E402
from open_wam.pipelines import build_variant_pipeline_from_config  # noqa: E402
from open_wam.pipelines.token_cost import build_training_token_cost_fn  # noqa: E402
from open_wam.training.launch import DistributedLaunchContext  # noqa: E402
from open_wam.training.runtime import TrainingRuntime  # noqa: E402
from open_wam.training.state import TrainState  # noqa: E402
from open_wam.training.step_executor import (  # noqa: E402
    LatentBatchAdapter,
    PipelineTrainStepExecutor,
)
from open_wam.training.strategies import DistributedStrategy  # noqa: E402


pytestmark = [pytest.mark.gpu, pytest.mark.integration]
_PHASE_PREFIX = "TOKEN_FSDP_PHASE "
_MAX_PHASE_BYTES = 4096
_STARTUP_SECONDS = 300
_EXECUTION_SECONDS = 360
_NCCL_SECONDS = 120


def _encode_phase(item):
    """Keep each complete log record within the Linux pipe's atomic limit."""
    payload = dict(item)

    def encode():
        body = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        return (_PHASE_PREFIX + body + "\n").encode("ascii")

    encoded = encode()
    if len(encoded) <= _MAX_PHASE_BYTES:
        return encoded
    message = payload.get("message")
    if not isinstance(message, str):
        raise ValueError("FSDP phase exceeds the atomic log limit without a truncatable message")
    payload["message_truncated"] = True
    # Search by encoded byte length, not Unicode character count: JSON escapes
    # can expand one character to twelve bytes. Never slice serialized JSON.
    low, high, best = 0, len(message), None
    while low <= high:
        midpoint = (low + high) // 2
        payload["message"] = message[:midpoint] + " [truncated]"
        candidate = encode()
        if len(candidate) <= _MAX_PHASE_BYTES:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    if best is None:
        raise ValueError("FSDP phase metadata exceeds the atomic log limit")
    return best


def _write_phase(item):
    """One syscall including the newline; short writes fail instead of splitting."""
    payload = _encode_phase(item)
    written = os.write(1, payload)
    if written != len(payload):
        raise RuntimeError(f"Short atomic FSDP phase write: {written}/{len(payload)} bytes")


def _parse_phase_line(line):
    """Read all framed objects without treating markers inside JSON as framing."""
    decoder = json.JSONDecoder()
    cursor = 0
    records = []
    while True:
        marker = line.find(_PHASE_PREFIX, cursor)
        if marker < 0:
            return records
        start = marker + len(_PHASE_PREFIX)
        while start < len(line) and line[start].isspace():
            start += 1
        try:
            item, end = decoder.raw_decode(line, start)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed FSDP phase JSON at column {start}") from error
        if (
            not isinstance(item, dict)
            or "rank" not in item
            or not isinstance(item.get("phase"), str)
            or not item["phase"]
        ):
            raise ValueError(f"Malformed FSDP phase record at column {start}")
        records.append(item)
        # raw_decode consumes strings (including escaped quotes and embedded
        # protocol prefixes) before we search for the next actual record.
        cursor = end


def _config(program: VideoActionProgram, activation_checkpointing: bool):
    """The real-pipeline parity fixture, enlarged only to CUDA's head size 32."""
    legacy = program is VideoActionProgram.VIDEO_THEN_ACTION
    return ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            train_batch_size=3,
            val_batch_size=1,
            batching=BatchingConfig(
                mode=BatchingMode.PACKED,
                pad_to_multiple_of=8,
                max_tokens=112 if legacy else 96,
            ),
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            attention_head_dim=32,
            ffn_dim=256,
            text_dim=16,
            freq_dim=8,
            train_attn_mode="flex",
            load_reference_core_weights=False,
            load_text_conditioning=False,
            load_wan_vae_frontend=False,
        ),
        policy_variant=DualExpertPolicyConfig(
            hidden_size=128,
            action_hidden_size=128,
            action_ffn_dim=256,
            program=program,
            num_action_layers=2,
            use_activation_checkpointing=activation_checkpointing,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            history_stream_visibility=(
                HistoryStreamVisibility.VIDEO_ONLY
                if legacy
                else HistoryStreamVisibility.FULL
            ),
            sequence_contract=(
                VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
                if legacy
                else VideoActionSequenceContract.DEFAULT
            ),
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=128, action_dim=4, action_horizon=4
        ),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            gradient_accumulation_steps=1,
            max_grad_norm=2.0,
            text_condition_dropout_prob=0.0,
            enabled_objectives=("action", "latent"),
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
        ),
        inference=InferenceConfig(frame_chunk_size=2),
        trainer=TrainerConfig(
            batch_adapter="latents",
            accelerator="gpu",
            devices=2,
            strategy="fsdp",
            precision="32-true",
            distributed_timeout_seconds=_NCCL_SECONDS,
        ),
    )


def _samples(program, rank):
    generator = torch.Generator().manual_seed(1234 + rank)
    # The same two lengths produce opposite B2/B1 arrangements across ranks.
    frame_counts = (4, 4, 7) if rank == 0 else (7, 4, 4)
    samples = []
    for index, frames in enumerate(frame_counts):
        text_tokens = 5 if frames == 4 else 7
        samples.append(
            LatentWAMSample(
                video_latents=torch.randn(48, frames, 4, 4, generator=generator),
                actions=torch.randn(2 * frames, 4, generator=generator),
                action_mask=torch.ones(2 * frames, 4),
                state=torch.randn(1, 4, generator=generator),
                state_mask=torch.ones(1, 4),
                condition_latents=torch.randn(
                    48,
                    1 if program is VideoActionProgram.VIDEO_THEN_ACTION else frames,
                    4,
                    4,
                    generator=generator,
                ),
                proprio_context_frames=torch.randn(frames, 4, generator=generator),
                proprio_context_frames_mask=torch.ones(frames, 4),
                text_context=torch.randn(text_tokens, 16, generator=generator),
                negative_text_context=torch.zeros(text_tokens, 16),
                task_text=f"synthetic rank {rank} sample {index}",
                metadata={
                    "sample_index": index,
                    "sampled_chunk_size": 2,
                    "sampled_window_size": 8,
                    "history_frames": 2,
                    "action_tokens_per_frame": 2,
                },
            )
        )
    return samples


def _local_snapshot(value):
    local = value.to_local() if hasattr(value, "to_local") else value
    return local.detach().float().cpu().clone()


def _run_schedule(config, strategy, rank, schedule, phase):
    from torch.distributed.fsdp import FSDPModule

    legacy = config.policy_variant.program is VideoActionProgram.VIDEO_THEN_ACTION
    if schedule == "single_control":
        config = replace(
            config,
            data=replace(
                config.data,
                batching=replace(
                    config.data.batching, max_tokens=92 if legacy else 84
                ),
            ),
        )
    torch.manual_seed(57)
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.policy_variant.initialize_for_training(pipeline.visual_tower)
    phase("sharding_begin")
    pipeline = strategy.prepare_model(pipeline)
    topology = pipeline.module_topology()
    assert isinstance(pipeline, FSDPModule)
    assert len(topology.fsdp_atomic_modules) == 2
    assert all(isinstance(block, FSDPModule) for block in topology.fsdp_atomic_modules)
    assert not topology.fsdp_block_stacks
    assert not pipeline.visual_tower.core.blocks
    assert not pipeline.policy_variant.action_expert.blocks
    phase("sharding_end", atomic_blocks=len(topology.fsdp_atomic_modules))
    batch = LatentBatchCollator(config.data.batching)(
        _samples(config.policy_variant.program, rank)
    )
    executor = PipelineTrainStepExecutor(
        pipeline=pipeline,
        batch_adapter=LatentBatchAdapter(),
        training_config=config.training,
    )
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.config = config
    runtime.model = pipeline
    runtime.strategy = strategy
    runtime.token_cost_fn = build_training_token_cost_fn(config)
    runtime.train_state = TrainState()
    runtime._accumulated_train_metrics = {}
    runtime.last_token_batch_plan = None
    runtime.optimizer = torch.optim.SGD(pipeline.parameters(), lr=0.001)
    runtime.scheduler = torch.optim.lr_scheduler.StepLR(
        runtime.optimizer, step_size=1, gamma=0.5
    )
    logged = []
    runtime.log_sink = SimpleNamespace(log_metrics=lambda **item: logged.append(item))
    initial = {name: _local_snapshot(p) for name, p in pipeline.named_parameters()}
    predictions, losses, physical_sizes, sync_calls, gradients = [], [], [], [], {}
    physical_call = 0
    original_backward = strategy.backward
    original_step = strategy.optimizer_step
    original_sync = strategy.set_gradient_sync
    original_clip = strategy.clip_grad_norm_
    optimizer_calls = []
    clip_calls = []

    def forward(physical_batch):
        nonlocal physical_call
        physical_call += 1
        size = len(physical_batch.sequence_lengths)
        physical_sizes.append(size)
        phase("forward_begin", physical_call=physical_call, samples=size)
        result = executor.forward_train(physical_batch)
        torch.cuda.synchronize(strategy.device)
        phase("forward_end", physical_call=physical_call, samples=size)
        for output in result.output.sample_outputs:
            predictions.append(output.decoder_output.action_pred.detach().cpu().clone())
            losses.append(output.decoder_output.loss.detach().cpu().clone())
        return result

    def backward(loss):
        phase("backward_begin", physical_call=physical_call)
        original_backward(loss)
        torch.cuda.synchronize(strategy.device)
        phase("backward_end", physical_call=physical_call)

    def optimizer_step(optimizer):
        phase("optimizer_begin")
        optimizer_calls.append(1)
        gradients.update(
            (name, _local_snapshot(p.grad))
            for name, p in pipeline.named_parameters()
            if p.grad is not None
        )
        original_step(optimizer)
        torch.cuda.synchronize(strategy.device)
        phase("optimizer_end")

    def set_sync(model, *, enabled):
        sync_calls.append(enabled)
        original_sync(model, enabled=enabled)

    def clip_grad_norm(parameters, max_grad_norm):
        clip_calls.append(max_grad_norm)
        phase("clip_grad_norm_begin", max_grad_norm=max_grad_norm)
        norm = original_clip(parameters, max_grad_norm)
        torch.cuda.synchronize(strategy.device)
        phase("clip_grad_norm_end", gradient_norm=float(norm.item()))
        return norm

    runtime.step_executor = SimpleNamespace(
        batch_adapter=executor.batch_adapter, forward_train=forward
    )
    strategy.backward = backward
    strategy.optimizer_step = optimizer_step
    strategy.set_gradient_sync = set_sync
    strategy.clip_grad_norm_ = clip_grad_norm
    try:
        torch.manual_seed(911 + rank)
        phase("logical_begin", logical_samples=3, max_tokens=config.data.batching.max_tokens)
        runtime._train_micro_step(batch)
        phase("logical_end")
    finally:
        strategy.backward = original_backward
        strategy.optimizer_step = original_step
        strategy.set_gradient_sync = original_sync
        strategy.clip_grad_norm_ = original_clip
    expected_groups = (
        ((0,), (1,), (2,))
        if schedule == "single_control"
        else (((0, 1), (2,)) if rank == 0 else ((0,), (1, 2)))
    )
    plan = runtime.last_token_batch_plan
    assert plan["groups"] == expected_groups
    assert physical_sizes == list(map(len, expected_groups))
    assert max(plan["physical_token_counts"]) <= config.data.batching.max_tokens
    assert sync_calls == [False] * (len(expected_groups) - 1) + [True, True]
    assert runtime.train_state.global_step == runtime.train_state.seen_batches == 1
    assert runtime.train_state.optimizer_step == runtime.scheduler.last_epoch == 1
    assert optimizer_calls == [1]
    assert clip_calls == [2.0]
    assert len(logged) == 1 and len(losses) == len(predictions) == 3
    assert len(gradients) == 138, f"Expected 138 active parameter gradients, got {len(gradients)}"
    assert all(torch.isfinite(g).all() for g in gradients.values())
    assert any(torch.count_nonzero(g) for g in gradients.values())
    assert all(torch.isfinite(loss).all() for loss in losses)
    after = {name: _local_snapshot(p) for name, p in pipeline.named_parameters()}
    assert any(not torch.equal(initial[name], after[name]) for name in after)
    for name, expected in initial.items():
        if name in gradients:
            expected = expected - 0.001 * gradients[name]
        torch.testing.assert_close(after[name], expected, atol=2e-6, rtol=2e-5)
    result = {
        "predictions": predictions,
        "losses": losses,
        "gradients": gradients,
        "parameters": after,
        "logged_loss": logged[0]["metrics"]["loss"],
        "plan": plan,
        "physical_sizes": physical_sizes,
    }
    return result


def _worker_main(args):
    rank = int(os.environ["RANK"])
    context = {"rank": rank, "program": args.program, "ac": bool(args.activation_checkpointing)}

    def phase(name, **extra):
        _write_phase({**context, "phase": name, "monotonic": time.monotonic(), **extra})

    phase("worker_ready", pid=os.getpid(), pgid=os.getpgrp(), ppid=os.getppid())
    if args.startup_release is not None:
        deadline = time.monotonic() + _STARTUP_SECONDS + 30
        while not args.startup_release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("phase=startup: launcher did not release ready workers")
            time.sleep(0.05)
    started = time.monotonic()

    def expired():
        phase("execution_watchdog", timeout_seconds=_EXECUTION_SECONDS)
        os._exit(124)

    watchdog = threading.Timer(_EXECUTION_SECONDS, expired)
    watchdog.daemon = True
    watchdog.start()
    strategy = None
    original_full_tensor = None
    full_tensor_calls = []
    try:
        if int(os.environ["WORLD_SIZE"]) != 2:
            raise ValueError("This gate requires exactly two workers")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("This gate requires two visible CUDA devices")
        if not dist.is_nccl_available():
            raise RuntimeError("NCCL is required; this gate does not fall back to Gloo")
        from torch.distributed.tensor import DTensor

        original_full_tensor = DTensor.full_tensor

        def full_tensor(value, *positional, **keywords):
            shape = list(value.shape)
            full_tensor_calls.append({"schedule": context.get("schedule"), "shape": shape})
            phase("full_tensor_begin", shape=shape)
            result = original_full_tensor(value, *positional, **keywords)
            phase("full_tensor_end", shape=shape)
            return result

        DTensor.full_tensor = full_tensor
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = _config(VideoActionProgram(args.program), bool(args.activation_checkpointing))
        phase("nccl_init_begin", timeout_seconds=_NCCL_SECONDS)
        strategy = DistributedStrategy(
            accelerator=config.trainer.accelerator,
            precision=config.trainer.precision,
            kind=config.trainer.strategy,
            distributed_timeout_seconds=_NCCL_SECONDS,
            launch_context=DistributedLaunchContext.from_env(),
        )
        phase("nccl_init_end")
        context["schedule"] = "single_control"
        reference = _run_schedule(config, strategy, rank, "single_control", phase)
        gc.collect()
        torch.cuda.empty_cache()
        context["schedule"] = "mixed"
        actual = _run_schedule(config, strategy, rank, "mixed", phase)
        phase("parity_begin")
        for observed, expected in zip(actual["predictions"], reference["predictions"], strict=True):
            torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-5)
        for observed, expected in zip(actual["losses"], reference["losses"], strict=True):
            torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            torch.tensor(actual["logged_loss"]), torch.tensor(reference["logged_loss"]),
            atol=2e-6, rtol=2e-5,
        )
        for key, atol, rtol in (("gradients", 3e-6, 3e-4), ("parameters", 2e-6, 2e-5)):
            assert actual[key].keys() == reference[key].keys()
            for name, expected in reference[key].items():
                torch.testing.assert_close(
                    actual[key][name], expected, atol=atol, rtol=rtol,
                    msg=lambda detail, name=name: f"{name}: {detail}",
                )
        phase("parity_end", checked_gradients=len(actual["gradients"]))
        summary = {
            **context, "passed": True, "checked_gradients": len(actual["gradients"]),
            "single_physical_sizes": reference["physical_sizes"],
            "mixed_physical_sizes": actual["physical_sizes"],
            "single_plan": reference["plan"], "mixed_plan": actual["plan"],
            "full_tensor_calls": full_tensor_calls,
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


def _owned_worker_alive(pid, expected_pgid):
    try:
        actual_pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    if actual_pgid != expected_pgid:
        raise RuntimeError(f"Refusing cleanup of worker pid={pid}: process group changed")
    # Exited Linux workers can briefly remain as zombies while their launcher
    # reaps them. They no longer execute or own CUDA allocations.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except FileNotFoundError:
        state = None
    return state != "Z"


def _discover_launcher_children(process, workers):
    """Remember Linux children even while they are still importing Python.

    A worker that has not emitted its ready message may already have its own
    session. Inspect only this launcher's direct-child list, never a global
    process listing or a user-name/process-name match.
    """
    try:
        child_ids = Path(
            f"/proc/{process.pid}/task/{process.pid}/children"
        ).read_text().split()
    except FileNotFoundError:
        return
    for value in child_ids:
        pid = int(value)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
            if int(fields[1]) != process.pid:
                continue
            pgid = os.getpgid(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if pgid not in (process.pid, pid):
            raise RuntimeError(f"Unexpected process group for launcher child pid={pid}")
        workers[pid] = pgid


def _cleanup_workers(process, workers):
    """Clean only verified children, including torchrun's separate sessions."""
    _discover_launcher_children(process, workers)

    def signal_owned(sig):
        for pid, pgid in workers.items():
            if not _owned_worker_alive(pid, pgid):
                continue
            try:
                if pgid == pid:
                    os.killpg(pgid, sig)
                else:
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    signal_owned(signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None and not any(
            _owned_worker_alive(pid, pgid) for pid, pgid in workers.items()
        ):
            break
        time.sleep(0.05)
    signal_owned(signal.SIGKILL)
    process.wait(timeout=5)
    deadline = time.monotonic() + 5
    remaining = []
    while time.monotonic() < deadline:
        remaining = [pid for pid, pgid in workers.items() if _owned_worker_alive(pid, pgid)]
        if not remaining:
            break
        time.sleep(0.05)
    if remaining:
        raise RuntimeError(f"Owned FSDP workers survived cleanup: pids={remaining}")


def _launch_case(program, activation_checkpointing, output_dir, *, worker_script=None):
    release = output_dir / "startup.release"
    command = [
        sys.executable, "-B", "-u", "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", "--nproc_per_node=2", "--max_restarts=0",
        str(Path(__file__ if worker_script is None else worker_script).resolve()),
        "--worker", "--program", program.value,
        "--activation-checkpointing", str(int(activation_checkpointing)),
        "--output-dir", str(output_dir), "--startup-release", str(release),
    ]
    environment = os.environ.copy()
    environment.update(
        TORCH_DISTRIBUTED_DEBUG="DETAIL", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
        PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
    )
    started = time.monotonic()
    print(_PHASE_PREFIX + json.dumps({
        "rank": "launcher", "program": program.value,
        "ac": activation_checkpointing, "phase": "launcher_start",
        "monotonic": started, "startup_timeout_seconds": _STARTUP_SECONDS,
    }), flush=True)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True, env=environment,
    )
    lines = queue.Queue()

    def read_output():
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    ready, latest, workers, tail = {}, {}, {}, deque(maxlen=20)
    execution_started = None
    output_ended = False
    try:
        while not output_ended:
            _discover_launcher_children(process, workers)
            now = time.monotonic()
            stage = "startup" if execution_started is None else "execution"
            deadline = started + _STARTUP_SECONDS if execution_started is None else execution_started + _EXECUTION_SECONDS
            if now >= deadline:
                raise AssertionError(
                    f"phase={stage} timeout; ready_ranks={sorted(ready)}; "
                    f"exitcode={process.poll()}; latest_phases={latest}; tail={list(tail)}"
                )
            try:
                line = lines.get(timeout=min(0.25, deadline - now))
            except queue.Empty:
                continue
            if line is None:
                output_ended = True
                continue
            print(line, end="", flush=True)
            tail.append(line.rstrip())
            for item in _parse_phase_line(line):
                latest[item["rank"]] = item["phase"]
                if item["phase"] == "worker_ready":
                    pid, pgid = item["pid"], item["pgid"]
                    assert item["rank"] in (0, 1) and item["rank"] not in ready
                    assert isinstance(pid, int) and pid > 1 and pid != os.getpid()
                    assert item["ppid"] == process.pid
                    assert pgid in (process.pid, pid) and os.getpgid(pid) == pgid
                    workers[pid] = pgid
                    ready[item["rank"]] = time.monotonic() - started
                    if set(ready) == {0, 1} and execution_started is None:
                        execution_started = time.monotonic()
                        release.touch(exist_ok=False)
                        print(_PHASE_PREFIX + json.dumps({
                            "rank": "launcher", "phase": "workers_released",
                            "monotonic": execution_started,
                            "startup_seconds": ready,
                            "execution_timeout_seconds": _EXECUTION_SECONDS,
                        }), flush=True)
        remaining = (
            started + _STARTUP_SECONDS if execution_started is None
            else execution_started + _EXECUTION_SECONDS
        ) - time.monotonic()
        code = process.wait(timeout=max(0.01, remaining))
        assert code == 0, (
            f"Worker launcher failed: exitcode={code}; ready_ranks={sorted(ready)}; "
            f"latest_phases={latest}; tail={list(tail)}"
        )
        assert set(ready) == {0, 1}
        reports = []
        for rank in range(2):
            with (output_dir / f"rank{rank}.json").open(encoding="utf-8") as handle:
                report = json.load(handle)
            assert report["passed"] and report["checked_gradients"] == 138
            assert report["single_physical_sizes"] == [1, 1, 1]
            assert report["mixed_physical_sizes"] == ([2, 1] if rank == 0 else [1, 2])
            report["startup_seconds"] = ready[rank]
            reports.append(report)
        print("TOKEN_FSDP_RESULT " + json.dumps(reports), flush=True)
    finally:
        try:
            _cleanup_workers(process, workers)
        finally:
            reader.join(timeout=5)
            if reader.is_alive():
                raise RuntimeError("Owned torchrun output pipe remained open after bounded cleanup")
            if process.stdout is not None:
                process.stdout.close()


@pytest.mark.parametrize("program", [VideoActionProgram.VIDEO_THEN_ACTION, VideoActionProgram.JOINT])
@pytest.mark.parametrize("activation_checkpointing", [False, True], ids=["ac_off", "ac_on"])
def test_token_budget_real_fsdp_matches_single_sample_control(program, activation_checkpointing, tmp_path):
    if os.environ.get("OPEN_WAM_RUN_GPU_SANITY") != "1":
        pytest.skip("Set OPEN_WAM_RUN_GPU_SANITY=1 to run the two-GPU FSDP gate")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("The FSDP integration gate requires two visible CUDA GPUs")
    if not dist.is_nccl_available():
        pytest.skip("The FSDP integration gate requires NCCL")
    _launch_case(program, activation_checkpointing, tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--program", choices=["video_then_action", "joint"], required=True)
    parser.add_argument("--activation-checkpointing", type=int, choices=[0, 1], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-release", type=Path)
    _worker_main(parser.parse_args())
