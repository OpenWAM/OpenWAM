from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from open_wam.configs import AuxiliaryValidationTaskConfig, TrainingConfig
from open_wam.configs.enums import BatchAdapterName, CheckpointMode
from open_wam.data import (
    LatentWAMSample,
    WAMBatch,
    collate_latent_wam_samples,
    move_latent_wam_batch_to_device,
)
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.training import TrainingRuntime
from open_wam.training.checkpoints import CheckpointManager
from open_wam.training.loop_policies import StepLoopPolicy
from open_wam.training.runtime import (
    AuxiliaryValidationDataset,
    _normalize_optimizer_state_dtypes,
    _resolve_auxiliary_validation_source,
    _validate_mixed_dynamics_source_sampling,
)
from open_wam.training.state import TrainState
from open_wam.training.step_executor import (
    LatentBatchAdapter,
    ViewBatchAdapter,
    resolve_sample_loss_weight,
)
from open_wam.utils.config_loader import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_normalize_optimizer_state_prefers_gradient_dtype_for_mixed_precision_resume() -> (
    None
):
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.ones_like(parameter)
    optimizer.state[parameter]["step"] = torch.tensor(1.0)
    optimizer.state[parameter]["exp_avg"] = torch.zeros(2, dtype=torch.float32)
    optimizer.state[parameter]["exp_avg_sq"] = torch.zeros(2, dtype=torch.float32)

    _normalize_optimizer_state_dtypes(optimizer)

    assert optimizer.state[parameter]["step"].dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.bfloat16
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.bfloat16


def test_normalize_optimizer_state_handles_wrapped_parameter_keys() -> None:
    class WrappedParameter:
        grad = torch.ones(2, dtype=torch.bfloat16)
        dtype = torch.float32

    parameter = WrappedParameter()
    optimizer = SimpleNamespace(
        state={
            parameter: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(2, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(2, dtype=torch.float32),
            }
        }
    )

    _normalize_optimizer_state_dtypes(optimizer)  # type: ignore[arg-type]

    assert optimizer.state[parameter]["step"].dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.bfloat16
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.bfloat16


def test_view_batch_adapter_repeats_invalid_video_tail_before_online_frontend() -> None:
    view = torch.arange(2 * 6, dtype=torch.float32).view(2, 6, 1, 1, 1)
    batch = WAMBatch(
        views={"cam": view},
        actions=torch.zeros(2, 0, 1),
        action_mask=torch.zeros(2, 0, 1),
        state=torch.zeros(2, 0, 1),
        state_mask=torch.zeros(2, 0, 1),
        metadata=(
            {"valid_video_frames": 4},
            {"valid_video_frames": 6},
        ),
    )

    prepared = ViewBatchAdapter().prepare(batch)
    repaired = prepared.views["cam"]

    assert torch.equal(repaired[0, :4], view[0, :4])
    assert torch.equal(repaired[0, 4:], view[0, 3:4].expand_as(repaired[0, 4:]))
    assert torch.equal(repaired[1], view[1])
    assert torch.equal(batch.views["cam"], view)


def test_train_micro_step_normalizes_optimizer_state_after_gradients() -> None:
    class WrappedParameter:
        grad = None
        dtype = torch.float32

    parameter = WrappedParameter()
    optimizer = SimpleNamespace(
        state={
            parameter: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(2, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(2, dtype=torch.float32),
            }
        }
    )
    step_called = False

    class Strategy:
        device = torch.device("cpu")

        def set_gradient_sync(self, model, *, enabled: bool) -> None:
            del model, enabled

        def autocast_context(self):
            return nullcontext()

        def backward(self, loss: torch.Tensor) -> None:
            del loss
            parameter.grad = torch.ones(2, dtype=torch.bfloat16)

        def unscale_(self, optimizer_arg) -> None:
            del optimizer_arg

        def optimizer_step(self, optimizer_arg) -> None:
            nonlocal step_called
            assert optimizer_arg.state[parameter]["exp_avg"].dtype == torch.bfloat16
            assert optimizer_arg.state[parameter]["exp_avg_sq"].dtype == torch.bfloat16
            step_called = True

        def zero_grad(self, optimizer_arg) -> None:
            del optimizer_arg
            parameter.grad = None

    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.step_executor = SimpleNamespace(
        batch_adapter=SimpleNamespace(move_to_device=lambda batch, device: batch),
        forward_train=lambda batch: SimpleNamespace(
            loss=torch.tensor(1.0, requires_grad=True), metrics={}
        ),
    )
    runtime.strategy = Strategy()
    runtime.optimizer = optimizer
    runtime.scheduler = SimpleNamespace(step=lambda: None, get_last_lr=lambda: [1e-4])
    runtime.model = SimpleNamespace(train=lambda: None)
    runtime.train_state = TrainState(run_name="dtype-normalize-test")
    runtime.config = SimpleNamespace(
        training=SimpleNamespace(gradient_accumulation_steps=1, max_grad_norm=None),
        trainer=SimpleNamespace(log_every_n_steps=1, save_interval=None),
    )
    runtime.log_sink = SimpleNamespace(log_metrics=lambda **kwargs: None)
    runtime._accumulated_train_metrics = {}

    runtime._train_micro_step(batch={})

    assert step_called is True
    assert runtime.train_state.optimizer_step == 1


def _write_temp_config(
    tmp_path: Path, *, source_name: str, output_name: str, mutate
) -> Path:
    source_path = REPO_ROOT / "configs/experiments" / source_name
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = output_name
    mutate(raw)
    config_path = tmp_path / f"{output_name}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return config_path


def _build_step_runtime_config(
    config_path: Path,
    *,
    tmp_path: Path,
    batch_adapter: str = "views",
):
    config = load_experiment_config(config_path)
    return replace(
        config,
        training=replace(
            config.training,
            num_steps=1,
        ),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter=batch_adapter,
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path),
            limit_train_batches=1,
            limit_val_batches=1,
            enable_checkpointing=False,
        ),
    )


@pytest.mark.parametrize(
    "config_name",
    [
        "parallel_stream_robotwin_smoke.yaml",
        "video_sequence_policy_robotwin_smoke.yaml",
        "mot_robotwin_smoke.yaml",
    ],
)
def test_composable_runtime_trains_shared_core_method_smokes(
    tmp_path: Path, config_name: str
) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_step_loop_reshuffles_distributed_sampler_each_loader_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = TensorDataset(torch.arange(1))
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    seen_epochs: list[int] = []
    original_set_epoch = sampler.set_epoch

    def record_set_epoch(epoch: int) -> None:
        seen_epochs.append(epoch)
        original_set_epoch(epoch)

    monkeypatch.setattr(sampler, "set_epoch", record_set_epoch)
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = loader
    runtime.train_state = TrainState(run_name="step-loop-sampler-test")
    runtime._run_validation = lambda *, limit_batches: None
    runtime._save_checkpoint = lambda *, final: None

    def train_one_batch(batch) -> None:
        del batch
        runtime.train_state.global_step += 1
        runtime.train_state.seen_batches += 1
        runtime.train_state.optimizer_step += 1

    runtime._train_micro_step = train_one_batch

    TrainingRuntime._run_step_loop(runtime, StepLoopPolicy(max_steps=3))

    assert seen_epochs == [0, 1, 2]
    assert runtime.train_state.epoch_index == 3


def test_epoch_loop_resume_cursor_skips_seen_batches_within_current_epoch() -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = range(10)
    runtime.train_state = TrainState(
        seen_batches=23, resume_source="/tmp/checkpoint_step_2/full_training_state.pt"
    )
    runtime.config = SimpleNamespace(trainer=SimpleNamespace(limit_train_batches=None))

    assert runtime._current_epoch_resume_batch_index() == 3

    runtime.config = SimpleNamespace(trainer=SimpleNamespace(limit_train_batches=7))

    assert runtime._current_epoch_resume_batch_index() == 2

    runtime.train_state.resume_source = None

    assert runtime._current_epoch_resume_batch_index() == 0


def test_step_loop_resume_cursor_skips_seen_batches_within_current_loader_pass() -> (
    None
):
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = range(5)
    runtime.train_state = TrainState(
        seen_batches=2,
        resume_source="/tmp/checkpoint_step_2/full_training_state.pt",
    )
    runtime.config = SimpleNamespace(trainer=SimpleNamespace(limit_train_batches=None))
    runtime.strategy = SimpleNamespace(is_main_process=True)
    logged_events: list[tuple[str, dict[str, int]]] = []
    runtime.log_sink = SimpleNamespace(
        log_event=lambda *, name, payload: logged_events.append((name, payload)),
    )
    runtime._run_validation = lambda *, limit_batches: None
    runtime._save_checkpoint = lambda *, final: None
    processed_batches: list[int] = []

    def train_one_batch(batch) -> None:
        processed_batches.append(int(batch))
        runtime.train_state.global_step += 1
        runtime.train_state.seen_batches += 1
        runtime.train_state.optimizer_step += 1

    runtime._train_micro_step = train_one_batch

    TrainingRuntime._run_step_loop(runtime, StepLoopPolicy(max_steps=2))

    assert processed_batches == [2, 3]
    assert logged_events == [
        (
            "resume_step_loop_cursor",
            {"epoch_index": 0, "skip_batches": 2, "seen_batches": 2},
        )
    ]


def test_step_loop_interval_checkpoint_runs_after_micro_step_returns() -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = [0, 1]
    runtime.train_state = TrainState(run_name="interval-checkpoint-test")
    runtime.config = SimpleNamespace(
        trainer=SimpleNamespace(
            limit_train_batches=None,
            save_interval=1,
            validation_interval=1,
        )
    )
    runtime.strategy = SimpleNamespace(is_main_process=True)

    in_micro_step = False
    processed_batches: list[int] = []
    events: list[tuple[str, int]] = []
    checkpoint_calls: list[tuple[bool, bool, int]] = []

    def run_validation(*, limit_batches) -> None:
        del limit_batches
        events.append(("validation", runtime.train_state.optimizer_step))

    def train_one_batch(batch) -> None:
        nonlocal in_micro_step
        in_micro_step = True
        processed_batches.append(int(batch))
        runtime.train_state.global_step += 1
        runtime.train_state.seen_batches += 1
        if int(batch) == 1:
            runtime.train_state.optimizer_step += 1
        in_micro_step = False

    def save_checkpoint(*, final: bool) -> None:
        checkpoint_calls.append(
            (final, in_micro_step, runtime.train_state.optimizer_step)
        )
        events.append(("checkpoint", runtime.train_state.optimizer_step))

    runtime._run_all_validation = run_validation
    runtime._train_micro_step = train_one_batch
    runtime._save_checkpoint = save_checkpoint

    TrainingRuntime._run_step_loop(runtime, StepLoopPolicy(max_steps=1))

    assert processed_batches == [0, 1]
    assert events[:2] == [("validation", 1), ("checkpoint", 1)]
    assert checkpoint_calls[0] == (False, False, 1)


def test_sample_loss_weight_can_scale_by_valid_action_steps() -> None:
    actions = torch.zeros(1, 6, 7)
    action_mask = torch.zeros_like(actions)
    action_mask[0, :6] = 1.0
    batch = PolicyTrainBatch(
        actions=actions,
        action_mask=action_mask,
        extra={"metadata": ({"dataset_mean_valid_action_steps": 4.0},)},
    )

    weight = resolve_sample_loss_weight(
        training_config=TrainingConfig(sample_loss_weight_mode="valid_action_steps"),
        batch=batch,
    )
    sqrt_weight = resolve_sample_loss_weight(
        training_config=TrainingConfig(
            sample_loss_weight_mode="sqrt_valid_action_steps"
        ),
        batch=batch,
    )

    assert weight.item() == pytest.approx(1.5)
    assert sqrt_weight.item() == pytest.approx(1.5**0.5)


def test_sample_loss_weight_rejects_reduced_multi_sample_batches() -> None:
    actions = torch.zeros(2, 6, 7)
    action_mask = torch.ones_like(actions)
    batch = PolicyTrainBatch(
        actions=actions,
        action_mask=action_mask,
        extra={
            "metadata": (
                {"dataset_mean_valid_action_steps": 6.0},
                {"dataset_mean_valid_action_steps": 6.0},
            )
        },
    )

    with pytest.raises(ValueError, match="train_batch_size=1"):
        resolve_sample_loss_weight(
            training_config=TrainingConfig(
                sample_loss_weight_mode="valid_action_steps"
            ),
            batch=batch,
        )


def test_mixed_dynamics_source_sampling_runtime_guard_rejects_non_uniform_weights() -> (
    None
):
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    )
    config = replace(
        config,
        trainer=replace(config.trainer, batch_adapter=BatchAdapterName.LATENTS),
        data=replace(
            config.data,
            sample_construction=replace(
                config.data.sample_construction,
                sample_weight_mode="valid_action_steps",
            ),
        ),
    )

    with pytest.raises(ValueError, match="sample_weight_mode"):
        _validate_mixed_dynamics_source_sampling(config)


def test_mixed_dynamics_source_sampling_runtime_guard_rejects_views_adapter() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    )

    with pytest.raises(ValueError, match="batch_adapter=latents"):
        _validate_mixed_dynamics_source_sampling(config)


def test_latent_batch_adapter_preserves_condition_latents() -> None:
    samples = [
        LatentWAMSample(
            video_latents=torch.full((48, 4, 2, 2), float(index)),
            condition_latents=torch.full((48, 1, 2, 2), float(index + 10)),
            actions=torch.zeros(16, 7),
            action_mask=torch.ones(16, 7),
            metadata={"sample": index},
        )
        for index in range(2)
    ]

    batch = collate_latent_wam_samples(samples)
    assert batch.condition_latents is not None
    torch.testing.assert_close(
        batch.condition_latents[:, 0, 0, 0, 0], torch.tensor([10.0, 11.0])
    )

    moved = move_latent_wam_batch_to_device(batch, torch.device("cpu"))
    assert moved.condition_latents is not None
    prepared = LatentBatchAdapter().prepare(moved)

    assert prepared.policy_batch.extra["condition_latents"] is moved.condition_latents


def test_latent_batch_adapter_preserves_proprio_context_state() -> None:
    samples = [
        LatentWAMSample(
            video_latents=torch.full((48, 4, 2, 2), float(index)),
            actions=torch.zeros(16, 7),
            action_mask=torch.ones(16, 7),
            proprio_context_state=torch.full((3, 8), float(index + 20)),
            proprio_context_state_mask=torch.ones(3, 8),
            metadata={"sample": index},
        )
        for index in range(2)
    ]

    batch = collate_latent_wam_samples(samples)
    assert batch.proprio_context_state is not None
    assert batch.proprio_context_state_mask is not None
    torch.testing.assert_close(
        batch.proprio_context_state[:, 0, 0], torch.tensor([20.0, 21.0])
    )
    torch.testing.assert_close(batch.proprio_context_state_mask[:, 0, 0], torch.ones(2))

    moved = move_latent_wam_batch_to_device(batch, torch.device("cpu"))
    assert moved.proprio_context_state is not None
    assert moved.proprio_context_state_mask is not None
    prepared = LatentBatchAdapter().prepare(moved)

    assert (
        prepared.policy_batch.extra["proprio_context_state"]
        is moved.proprio_context_state
    )
    assert (
        prepared.policy_batch.extra["proprio_context_state_mask"]
        is moved.proprio_context_state_mask
    )


def test_auxiliary_validation_dataset_forces_generalist_metadata_and_drops_text() -> (
    None
):
    sample = LatentWAMSample(
        video_latents=torch.zeros(2, 3),
        actions=torch.zeros(4, 7),
        task_text="put the mug on the plate",
        text_context=torch.ones(1, 2),
        negative_text_context=torch.zeros(1, 2),
        metadata={"existing": "kept"},
    )
    task = AuxiliaryValidationTaskConfig(
        name="fdm_val",
        mode_override="action_conditioned_video",
        report_prefix="val_fdm",
    )

    wrapped = AuxiliaryValidationDataset([sample], task=task)
    forced = wrapped[0]

    assert forced.task_text is None
    assert torch.equal(forced.text_context, torch.zeros(1, 2))
    assert forced.metadata["existing"] == "kept"
    assert (
        forced.metadata["generalist_training_mode_override"]
        == "action_conditioned_video"
    )
    assert forced.metadata["generalist_drop_text_conditioning"] is True
    assert forced.metadata["generalist_training_source"] == "auxiliary_validation"
    assert forced.metadata["generalist_training_bucket"] == "fdm_val"
    assert sample.task_text == "put the mug on the plate"


def test_auxiliary_validation_source_can_select_pure_counterfactual_dataset() -> None:
    class MixedDataset(Dataset):
        def __init__(self) -> None:
            self.real_dataset = TensorDataset(torch.ones(1, 1))
            self.counterfactual_dataset = TensorDataset(torch.zeros(1, 1))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            return self.real_dataset[index]

    mixed = MixedDataset()
    task = AuxiliaryValidationTaskConfig(
        name="fdm_val", source="counterfactual_dynamics"
    )

    selected, resolved_source = _resolve_auxiliary_validation_source(mixed, task=task)

    assert selected is mixed.counterfactual_dataset
    assert resolved_source == "counterfactual_dynamics"
    with pytest.raises(ValueError, match="does not expose"):
        _resolve_auxiliary_validation_source(TensorDataset(torch.ones(1, 1)), task=task)


def test_auxiliary_validation_source_can_fallback_when_counterfactual_is_unavailable() -> (
    None
):
    dataset = TensorDataset(torch.ones(1, 1))
    task = AuxiliaryValidationTaskConfig(
        name="fdm_val", source="counterfactual_dynamics_if_available"
    )

    selected, resolved_source = _resolve_auxiliary_validation_source(dataset, task=task)

    assert selected is dataset
    assert resolved_source == "dataset"


def test_training_runtime_runs_primary_and_auxiliary_validation_phases() -> None:
    task = AuxiliaryValidationTaskConfig(
        name="fdm_val",
        mode_override="action_conditioned_video",
        max_batches=2,
        report_prefix="val_fdm",
    )
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.val_loader = [1]
    runtime.auxiliary_validation_runs = (
        SimpleNamespace(config=task, loader=[2, 4, 6]),
    )
    runtime.model = SimpleNamespace(eval=lambda: None)
    runtime.strategy = SimpleNamespace(
        device=torch.device("cpu"),
        autocast_context=lambda: nullcontext(),
    )
    runtime.train_state = TrainState(optimizer_step=7)
    logged: list[tuple[str, int, dict[str, float]]] = []
    runtime.log_sink = SimpleNamespace(
        log_metrics=lambda *, step, phase, metrics: logged.append(
            (phase, step, metrics)
        ),
    )

    class Adapter:
        def move_to_device(self, batch, device):
            del device
            return batch

    class Executor:
        batch_adapter = Adapter()

        def forward_train(self, batch):
            value = torch.tensor(float(batch))
            return SimpleNamespace(
                loss=value,
                metrics={
                    "loss": value,
                    "joint_denoise/action_loss_active": torch.tensor(0.0),
                    "joint_denoise/latent_loss_active": torch.tensor(1.0),
                    "joint_denoise/action_conditioned_video/count": torch.tensor(1.0),
                },
            )

    runtime.step_executor = Executor()

    runtime._run_all_validation(limit_batches=1)

    assert logged[0] == (
        "val",
        7,
        {
            "loss": 1.0,
            "joint_denoise/action_loss_active": 0.0,
            "joint_denoise/latent_loss_active": 1.0,
            "joint_denoise/action_conditioned_video/count": 1.0,
        },
    )
    assert logged[1][0] == "val_fdm"
    assert logged[1][1] == 7
    assert logged[1][2]["loss"] == pytest.approx(3.0)
    assert logged[1][2]["count"] == 2.0
    assert logged[1][2]["action_loss_active"] == 0.0
    assert logged[1][2]["latent_loss_active"] == 1.0
    assert logged[1][2]["mode_fraction"] == 1.0


def test_validation_metrics_reduce_sums_and_counts_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.val_loader = [1, 3]
    runtime.model = SimpleNamespace(eval=lambda: None)
    runtime.strategy = SimpleNamespace(
        device=torch.device("cpu"),
        autocast_context=lambda: nullcontext(),
    )
    runtime.train_state = TrainState(optimizer_step=5)
    logged: list[tuple[int, str, dict[str, float]]] = []
    runtime.log_sink = SimpleNamespace(
        log_metrics=lambda *, step, phase, metrics: logged.append(
            (step, phase, metrics)
        ),
    )
    runtime.step_executor = SimpleNamespace(
        batch_adapter=SimpleNamespace(move_to_device=lambda batch, device: batch),
        forward_train=lambda batch: SimpleNamespace(
            loss=torch.tensor(float(batch)),
            metrics={"loss": torch.tensor(float(batch))},
        ),
    )

    def fake_all_reduce(tensor: torch.Tensor, op) -> None:
        del op
        if tensor.item() == pytest.approx(2.0):
            tensor.add_(2.0)
        elif tensor.item() == pytest.approx(4.0):
            tensor.add_(8.0)

    monkeypatch.setattr("open_wam.training.runtime.dist.is_initialized", lambda: True)
    monkeypatch.setattr("open_wam.training.runtime.dist.all_reduce", fake_all_reduce)

    assert runtime._run_validation(limit_batches=None) is True

    assert logged == [(5, "val", {"loss": pytest.approx(3.0)})]


def test_step_loop_runs_validation_interval_without_duplicate_final_validation() -> (
    None
):
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = range(4)
    runtime.train_state = TrainState(run_name="validation-interval")
    runtime.config = SimpleNamespace(
        trainer=SimpleNamespace(limit_train_batches=None, validation_interval=2)
    )
    runtime.strategy = SimpleNamespace(is_main_process=True)
    validation_steps: list[int] = []

    def record_validation(*, limit_batches) -> bool:
        del limit_batches
        validation_steps.append(runtime.train_state.optimizer_step)
        return True

    runtime._run_validation = record_validation
    runtime._save_checkpoint = lambda *, final: None

    def train_one_batch(batch) -> None:
        del batch
        runtime.train_state.global_step += 1
        runtime.train_state.seen_batches += 1
        runtime.train_state.optimizer_step += 1

    runtime._train_micro_step = train_one_batch

    TrainingRuntime._run_step_loop(
        runtime, StepLoopPolicy(max_steps=4, limit_val_batches=1)
    )

    assert validation_steps == [2, 4]


def test_composable_runtime_trains_causal_video_prediction_smoke(
    tmp_path: Path,
) -> None:
    config_path = (
        REPO_ROOT / "configs/experiments/causal_video_prediction_robotwin_smoke.yaml"
    )
    config = _build_step_runtime_config(
        config_path, tmp_path=tmp_path, batch_adapter="latents"
    )

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_training_runtime_initializes_mot_variant_before_strategy_wrap(
    tmp_path: Path,
) -> None:
    config_path = REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml"
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    pipeline = runtime.strategy.unwrap_model(runtime.model)

    assert pipeline.policy_variant._action_expert_initialized is True


def test_generalist_checkpoint_writes_yaml_safe_enum_dict_keys(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    checkpoint_dir = manager.checkpoint_dir_for_step(1)
    checkpoint_dir.mkdir(parents=True)

    manager._write_resolved_config(checkpoint_dir)

    resolved_text = (checkpoint_dir / "resolved_config.yaml").read_text(
        encoding="utf-8"
    )
    assert "mot_generalist_training_mode_probs:" in resolved_text
    assert "joint:" in resolved_text


def test_model_only_checkpoint_does_not_collect_optimizer_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    calls = {"optimizer_state": 0}

    def fake_model_state_dict(model, options):
        del model, options
        return {"weight": torch.ones(1)}

    def fail_optimizer_state_dict(*args, **kwargs):
        del args, kwargs
        calls["optimizer_state"] += 1
        raise AssertionError("model_only checkpoints must not collect optimizer state")

    monkeypatch.setattr(
        "open_wam.training.checkpoints.get_model_state_dict", fake_model_state_dict
    )
    monkeypatch.setattr(
        "open_wam.training.checkpoints.get_optimizer_state_dict",
        fail_optimizer_state_dict,
    )

    checkpoint_dir = manager.save(
        step=1,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_state=TrainState(),
    )

    assert calls["optimizer_state"] == 0
    assert (checkpoint_dir / "model_state.pt").exists()
    assert not (checkpoint_dir / "full_training_state.pt").exists()
    assert (checkpoint_dir / ".checkpoint_complete").exists()


def test_full_state_resume_preserves_sparse_adamw_state(tmp_path: Path) -> None:
    class SparseOptimizerModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Parameter(torch.tensor([1.0]))
            self.later = torch.nn.Parameter(torch.tensor([2.0]))

    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    model = SparseOptimizerModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.first.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert model.later not in optimizer.state

    checkpoint_dir = manager.save(
        step=1,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_state=TrainState(global_step=1, optimizer_step=1),
    )

    resumed_model = SparseOptimizerModel()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_state, _ = manager.load(
        path=checkpoint_dir,
        model=resumed_model,
        optimizer=resumed_optimizer,
    )

    assert resumed_state.optimizer_step == 1
    assert resumed_model.later not in resumed_optimizer.state
    assert len(resumed_optimizer.state) == len(optimizer.state) == 1

    for candidate_model, candidate_optimizer in (
        (model, optimizer),
        (resumed_model, resumed_optimizer),
    ):
        candidate_model.later.square().sum().backward()
        candidate_optimizer.step()
        candidate_optimizer.zero_grad(set_to_none=True)

    torch.testing.assert_close(resumed_model.first, model.first, rtol=0, atol=0)
    torch.testing.assert_close(resumed_model.later, model.later, rtol=0, atol=0)
    for original_parameter, resumed_parameter in (
        (model.first, resumed_model.first),
        (model.later, resumed_model.later),
    ):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                resumed_optimizer.state[resumed_parameter][key],
                optimizer.state[original_parameter][key],
                rtol=0,
                atol=0,
            )


def test_checkpoint_manager_prunes_old_checkpoints_after_successful_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
        max_checkpoints_to_keep=3,
    )
    model = torch.nn.Linear(2, 2)

    def fake_model_state_dict(model, options):
        del model, options
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(
        "open_wam.training.checkpoints.get_model_state_dict", fake_model_state_dict
    )

    for step in (100, 200, 300, 400, 500):
        train_state = TrainState(optimizer_step=step)
        manager.save(
            step=step,
            model=model,
            optimizer=None,
            scheduler=None,
            train_state=train_state,
        )

    remaining = sorted(
        path.name for path in (tmp_path / "checkpoints").glob("checkpoint_step_*")
    )
    assert remaining == [
        "checkpoint_step_300",
        "checkpoint_step_400",
        "checkpoint_step_500",
    ]


def test_model_only_checkpoint_loads_sibling_train_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    checkpoint_dir = manager.checkpoint_dir_for_step(500)
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": {"weight": torch.ones(1)}},
        checkpoint_dir / "model_state.pt",
    )
    (checkpoint_dir / "train_state.json").write_text(
        json.dumps(
            {"global_step": 10000, "optimizer_step": 500, "seen_batches": 10000}
        ),
        encoding="utf-8",
    )

    loaded_keys: list[str] = []

    def fake_set_model_state_dict(model, state_dict, options):
        del model, options
        loaded_keys.extend(state_dict.keys())

    monkeypatch.setattr(
        "open_wam.training.checkpoints.set_model_state_dict", fake_set_model_state_dict
    )

    train_state, payload = manager.load(
        path=checkpoint_dir / "model_state.pt", model=torch.nn.Linear(1, 1)
    )

    assert loaded_keys == ["weight"]
    assert "optimizer_state_dict" not in payload
    assert train_state.global_step == 10000
    assert train_state.optimizer_step == 500
    assert train_state.seen_batches == 0
    assert train_state.resume_source == str(checkpoint_dir / "model_state.pt")


def test_full_state_checkpoint_resume_prefers_sibling_full_state(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    checkpoint_dir = manager.checkpoint_dir_for_step(500)
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": {"weight": torch.ones(1)}},
        checkpoint_dir / "model_state.pt",
    )
    torch.save(
        {
            "model_state_dict": {"weight": torch.ones(1)},
            "train_state": TrainState(global_step=500, optimizer_step=500).state_dict(),
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "strategy_state_dict": None,
        },
        checkpoint_dir / "full_training_state.pt",
    )

    with pytest.warns(RuntimeWarning, match="Promoting model_state.pt resume path"):
        resolved = manager.resolve_checkpoint_path(checkpoint_dir / "model_state.pt")

    assert resolved == checkpoint_dir / "full_training_state.pt"


def test_checkpoint_latest_ignores_unmarked_partial_when_markers_exist(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    complete = manager.checkpoint_dir_for_step(100)
    complete.mkdir(parents=True)
    torch.save(
        {"model_state_dict": {"weight": torch.ones(1)}}, complete / "model_state.pt"
    )
    (complete / ".checkpoint_complete").write_text("ok\n", encoding="utf-8")
    partial = manager.checkpoint_dir_for_step(200)
    partial.mkdir(parents=True)
    torch.save(
        {"model_state_dict": {"weight": torch.ones(1)}}, partial / "model_state.pt"
    )

    assert manager.find_latest_checkpoint(tmp_path / "checkpoints") == complete


def test_checkpoint_latest_preserves_legacy_unmarked_dirs(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    for step in (100, 200):
        checkpoint_dir = manager.checkpoint_dir_for_step(step)
        checkpoint_dir.mkdir(parents=True)
        torch.save(
            {"model_state_dict": {"weight": torch.ones(1)}},
            checkpoint_dir / "model_state.pt",
        )

    assert manager.find_latest_checkpoint(
        tmp_path / "checkpoints"
    ) == manager.checkpoint_dir_for_step(200)


def test_final_checkpoint_skips_when_interval_checkpoint_already_saved(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            enable_checkpointing=False,
            save_interval=5,
        ),
    )
    runtime = SimpleNamespace(
        config=config,
        train_state=TrainState(optimizer_step=5),
        checkpoint_manager=SimpleNamespace(
            checkpoint_dir_for_step=lambda step: (
                tmp_path / "checkpoints" / f"checkpoint_step_{step}"
            ),
            save=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("duplicate final checkpoint")
            ),
        ),
    )
    runtime.train_state.last_checkpoint_path = str(
        tmp_path / "checkpoints" / "checkpoint_step_5"
    )

    TrainingRuntime._save_checkpoint(runtime, final=True)


def test_save_interval_zero_disables_final_checkpoint(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            enable_checkpointing=False,
            save_interval=0,
        ),
    )
    runtime = SimpleNamespace(
        config=config,
        train_state=TrainState(optimizer_step=5),
        checkpoint_manager=SimpleNamespace(
            checkpoint_dir_for_step=lambda step: (
                tmp_path / "checkpoints" / f"checkpoint_step_{step}"
            ),
            save=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("checkpoint should be disabled")
            ),
        ),
    )

    TrainingRuntime._save_checkpoint(runtime, final=True)


def test_composable_runtime_logs_checkpoints_and_resume(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    config = replace(
        config,
        training=replace(
            config.training,
            num_steps=1,
        ),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter="latents",
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path),
            enable_checkpointing=True,
            save_interval=1,
            enable_jsonl_logging=True,
            metrics_filename="metrics.jsonl",
        ),
    )

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    output_dir = tmp_path / config.name
    checkpoint_root = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"

    assert final_state.optimizer_step == 1
    assert metrics_path.exists()
    assert checkpoint_root.exists()

    resumed_config = replace(
        config,
        training=replace(config.training, num_steps=2),
        trainer=replace(config.trainer, resume_from=str(checkpoint_root)),
    )
    resumed_runtime = TrainingRuntime.from_config(resumed_config)
    resumed_state = resumed_runtime.run()

    assert resumed_state.optimizer_step == 2


def test_composable_runtime_exports_runtime_backbone(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    config = replace(
        config,
        training=replace(config.training, num_steps=1),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter="latents",
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path),
            enable_checkpointing=True,
            save_interval=1,
            checkpoint_mode="model_only",
            export_runtime_backbone=True,
        ),
    )

    runtime = TrainingRuntime.from_config(config)
    runtime.run()

    checkpoint_dir = (
        tmp_path / config.name / "checkpoints" / "checkpoint_step_1" / "transformer"
    )
    assert (checkpoint_dir / "diffusion_pytorch_model.safetensors").exists()
    assert (checkpoint_dir / "config.json").exists()


def test_composable_runtime_disable_checkpointing_suppresses_export_runtime_backbone(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    config = replace(
        config,
        training=replace(config.training, num_steps=1),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter="latents",
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            save_interval=None,
            checkpoint_mode="model_only",
            export_runtime_backbone=True,
        ),
    )

    runtime = TrainingRuntime.from_config(config)
    runtime.run()

    checkpoint_root = tmp_path / config.name / "checkpoints"
    assert not list(checkpoint_root.glob("checkpoint_step_*"))


def test_composable_runtime_ddp_strategy_degrades_cleanly_to_single_process(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    )
    config = replace(
        config,
        training=replace(config.training, num_steps=1),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter="latents",
            loop_policy="steps",
            strategy="ddp",
            default_root_dir=str(tmp_path),
        ),
    )

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_composable_runtime_trains_post_latent_with_core_layer_visual_readout(
    tmp_path: Path,
) -> None:
    def _mutate(raw) -> None:
        raw["policy_variant"]["visual_readout"] = {
            "source_family": "core_layer_tokens",
            "layer_index": 0,
        }
        raw["backbone"]["num_layers"] = 2

    config_path = _write_temp_config(
        tmp_path,
        source_name="post_latent_robotwin.yaml",
        output_name="post_latent_core_layer_runtime",
        mutate=_mutate,
    )
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_composable_runtime_trains_post_decoded_with_multi_layer_visual_readout(
    tmp_path: Path,
) -> None:
    def _mutate(raw) -> None:
        raw["policy_variant"]["visual_readout"] = {
            "source_family": "core_multi_layer_tokens",
            "layer_indices": [0, 1],
            "fusion_mode": "concat_project",
        }
        raw["backbone"]["num_layers"] = 2

    config_path = _write_temp_config(
        tmp_path,
        source_name="post_decoded_robotwin.yaml",
        output_name="post_decoded_multi_layer_runtime",
        mutate=_mutate,
    )
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_composable_runtime_trains_method4_with_implicit_video_conditioned_decoder(
    tmp_path: Path,
) -> None:
    post_latent_path = (
        REPO_ROOT / "configs/experiments/post_latent_robotwin_video_conditioned.yaml"
    )
    post_latent_config = _build_step_runtime_config(post_latent_path, tmp_path=tmp_path)
    post_latent_runtime = TrainingRuntime.from_config(post_latent_config)
    post_latent_state = post_latent_runtime.run()
    assert post_latent_state.optimizer_step == 1

    post_decoded_path = (
        REPO_ROOT / "configs/experiments/post_decoded_robotwin_video_conditioned.yaml"
    )
    post_decoded_config = _build_step_runtime_config(
        post_decoded_path, tmp_path=tmp_path
    )
    post_decoded_runtime = TrainingRuntime.from_config(post_decoded_config)
    post_decoded_state = post_decoded_runtime.run()
    assert post_decoded_state.optimizer_step == 1


def test_composable_runtime_trains_method4_current_frame_regression_mode(
    tmp_path: Path,
) -> None:
    config_path = (
        REPO_ROOT
        / "configs/experiments/post_decoded_robotwin_current_frame_regression.yaml"
    )
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_composable_runtime_trains_video_sequence_policy_with_core_layer_visual_readout(
    tmp_path: Path,
) -> None:
    def _mutate(raw) -> None:
        raw["policy_variant"]["name"] = "video_sequence_policy"
        raw["policy_variant"]["attach_site"] = "post_visual_core"
        raw["policy_variant"]["visual_readout"] = {
            "source_family": "core_layer_tokens",
            "layer_index": 0,
        }
        raw["policy_variant"].pop("pooling_mode", None)
        raw["policy_variant"].pop("query_count", None)
        raw["policy_variant"].pop("use_state_projection", None)
        raw["action_decoder"]["name"] = "vpp_decoder"
        raw["action_decoder"]["num_sampling_steps"] = 2
        raw["action_decoder"]["rollout_chunk_steps"] = 2
        raw["backbone"]["num_layers"] = 2

    config_path = _write_temp_config(
        tmp_path,
        source_name="post_latent_robotwin.yaml",
        output_name="video_sequence_policy_core_layer_runtime",
        mutate=_mutate,
    )
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1
