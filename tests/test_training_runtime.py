from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import open_wam.training.checkpoints as checkpoints_module
import open_wam.training.data_loading as data_loading_module
import open_wam.training.runtime as runtime_module
from open_wam.configs import (
    EXPERIMENT_CONFIG_SCHEMA_VERSION,
    AuxiliaryValidationTaskConfig,
    DynamicsObjective,
    TrainingConfig,
    load_experiment_config,
)
from open_wam.configs.enums import (
    BatchAdapterName,
    CheckpointMode,
    SampleOrderMode,
    TrainingComponentSelector,
)
from open_wam.contracts import DYNAMICS_ROUTING_MODE_METADATA_KEY
from open_wam.data import (
    LatentWAMSample,
    WAMBatch,
    collate_latent_wam_samples,
    move_latent_wam_batch_to_device,
)
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.runtime.runtime_backbone_manifest import (
    RUNTIME_BACKBONE_MANIFEST_FILENAME,
    load_runtime_backbone_manifest,
)
from open_wam.training import TrainingRuntime
from open_wam.training.auxiliary_validation import (
    AuxiliaryValidationDataset,
    _resolve_auxiliary_validation_source,
    build_auxiliary_validation_runs,
)
from open_wam.training.checkpoint_export import resolve_runtime_backbone_export_keys
from open_wam.training.checkpoints import CheckpointManager
from open_wam.training.data_loading import (
    _validate_dynamics_source_sampling,
    build_runtime_dataloaders,
    preflight_runtime_dataset_artifacts,
)
from open_wam.training.loop_policies import EpochLoopPolicy, StepLoopPolicy
from open_wam.training.optim import _normalize_optimizer_state_dtypes
from open_wam.training.state import TrainState
from open_wam.training.step_executor import (
    LatentBatchAdapter,
    ViewBatchAdapter,
    resolve_sample_loss_weight,
)
from open_wam.utils.config_overrides import apply_config_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_compatibility_aliases_keep_canonical_owner_identity() -> None:
    assert runtime_module.AuxiliaryValidationDataset is AuxiliaryValidationDataset
    assert (
        runtime_module._normalize_optimizer_state_dtypes
        is _normalize_optimizer_state_dtypes
    )
    assert (
        runtime_module._resolve_auxiliary_validation_source
        is _resolve_auxiliary_validation_source
    )
    assert (
        runtime_module._validate_dynamics_source_sampling
        is _validate_dynamics_source_sampling
    )


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
        "dual_expert_robotwin_smoke.yaml",
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
    runtime.config = SimpleNamespace(
        trainer=SimpleNamespace(
            limit_train_batches=None,
            save_interval=None,
            validation_interval=None,
        )
    )
    runtime._run_all_validation = lambda *, limit_batches: None
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
    assert runtime.train_state.next_batch_index == 0


def test_step_loop_loader_boundary_checkpoint_resumes_next_sampler_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = TensorDataset(torch.arange(2))
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    seen_epochs: list[int] = []
    original_set_epoch = sampler.set_epoch

    def record_set_epoch(epoch: int) -> None:
        seen_epochs.append(epoch)
        original_set_epoch(epoch)

    monkeypatch.setattr(sampler, "set_epoch", record_set_epoch)

    def make_runtime(train_state: TrainState) -> TrainingRuntime:
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.train_loader = loader
        runtime.train_state = train_state
        runtime.config = SimpleNamespace(
            trainer=SimpleNamespace(
                limit_train_batches=None,
                save_interval=2,
                validation_interval=None,
            )
        )
        runtime.strategy = SimpleNamespace(is_main_process=True)
        runtime._run_all_validation = lambda *, limit_batches: None
        runtime.log_sink = SimpleNamespace(log_event=lambda **kwargs: None)
        return runtime

    source = make_runtime(TrainState(run_name="loader-boundary-source"))
    saved_states: list[TrainState] = []

    def train_one_batch(batch) -> None:
        del batch
        source.train_state.global_step += 1
        source.train_state.seen_batches += 1
        source.train_state.optimizer_step += 1

    source._train_micro_step = train_one_batch
    source._save_checkpoint = lambda *, final: (
        saved_states.append(TrainState.from_state_dict(source.train_state.state_dict()))
        if not final
        else None
    )

    TrainingRuntime._run_step_loop(source, StepLoopPolicy(max_steps=2))

    assert seen_epochs == [0]
    assert len(saved_states) == 1
    assert saved_states[0].epoch_index == 1
    assert saved_states[0].next_batch_index == 0

    resumed_state = saved_states[0]
    resumed_state.resume_source = "/tmp/checkpoint_step_2/full_training_state.pt"
    resumed = make_runtime(resumed_state)

    def train_resumed_batch(batch) -> None:
        del batch
        resumed.train_state.global_step += 1
        resumed.train_state.seen_batches += 1
        resumed.train_state.optimizer_step += 1

    resumed._train_micro_step = train_resumed_batch
    resumed._save_checkpoint = lambda *, final: None
    seen_epochs.clear()

    TrainingRuntime._run_step_loop(resumed, StepLoopPolicy(max_steps=3))

    assert seen_epochs == [1]
    assert resumed.train_state.epoch_index == 1
    assert resumed.train_state.next_batch_index == 1


def test_full_state_checkpointing_rejects_unsized_train_loader() -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = iter((0, 1))
    runtime.config = SimpleNamespace(
        trainer=SimpleNamespace(
            checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
            enable_checkpointing=True,
            save_interval=None,
        )
    )

    with pytest.raises(ValueError, match="requires a sized train dataloader"):
        runtime.run()


def test_full_state_resume_rejects_unsized_train_loader_before_loading() -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = iter((0, 1))
    runtime.checkpoint_manager = SimpleNamespace(
        load=lambda **kwargs: pytest.fail("unsupported resume must fail before loading")
    )

    with pytest.raises(ValueError, match="requires a sized train dataloader"):
        runtime.resume("full_training_state.pt")


def test_epoch_loop_resume_cursor_skips_seen_batches_within_current_epoch() -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = range(10)
    runtime.train_state = TrainState(
        seen_batches=23,
        next_batch_index=3,
        resume_source="/tmp/checkpoint_step_2/full_training_state.pt",
    )
    runtime.config = SimpleNamespace(trainer=SimpleNamespace(limit_train_batches=None))

    assert runtime._next_train_batch_index(loader_pass_batches=10) == 3

    runtime.config = SimpleNamespace(trainer=SimpleNamespace(limit_train_batches=7))

    assert runtime._next_train_batch_index(loader_pass_batches=7) == 3

    runtime.train_state.resume_source = None

    assert runtime._next_train_batch_index(loader_pass_batches=7) == 3


def test_step_loop_resume_cursor_skips_seen_batches_within_current_loader_pass() -> (
    None
):
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = range(5)
    runtime.train_state = TrainState(
        seen_batches=2,
        next_batch_index=2,
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


def test_dynamics_source_sampling_runtime_guard_rejects_non_uniform_weights() -> None:
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
                sample_order_mode=SampleOrderMode.REPLACEMENT,
                sample_weight_mode="valid_action_steps",
            ),
        ),
    )

    with pytest.raises(ValueError, match="sample_weight_mode"):
        _validate_dynamics_source_sampling(config)


def test_dynamics_source_sampling_runtime_guard_rejects_epoch_order() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    config = replace(
        config,
        data=replace(
            config.data,
            sample_construction=replace(
                config.data.sample_construction,
                sample_order_mode=SampleOrderMode.EPOCH_ORDER,
            ),
        ),
    )

    with pytest.raises(ValueError, match="sample_order_mode.*replacement"):
        _validate_dynamics_source_sampling(config)


def test_dynamics_source_sampling_runtime_guard_rejects_views_adapter() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    )

    with pytest.raises(ValueError, match="batch_adapter=latents"):
        _validate_dynamics_source_sampling(config)


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


def test_auxiliary_validation_source_uses_source_view_protocol() -> None:
    class RoutedDataset(Dataset):
        def __init__(self) -> None:
            self.counterfactual_dataset = TensorDataset(torch.zeros(1, 1))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            return self.counterfactual_dataset[index]

        def has_route(self, *, source: str, mode: str) -> bool:
            del mode
            return source == "counterfactual_dynamics"

        def build_source_view(self, **kwargs):
            assert kwargs == {
                "source": "counterfactual_dynamics",
                "mode": "joint",
                "bucket_name": "fdm_val",
                "spread_indices": True,
            }
            return self.counterfactual_dataset

    routed = RoutedDataset()
    task = AuxiliaryValidationTaskConfig(
        name="fdm_val", source="counterfactual_dynamics"
    )

    selected, resolved_source = _resolve_auxiliary_validation_source(routed, task=task)

    assert selected is routed.counterfactual_dataset
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


@pytest.mark.parametrize(
    "fixed_mode",
    [
        "action_conditioned_video",
        "video_conditioned_action",
    ],
)
def test_single_route_gjd_validation_preserves_explicit_conditional_probes(
    fixed_mode: str,
) -> None:
    class RoutedDataset(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            del index
            return LatentWAMSample(
                video_latents=torch.zeros(2, 3),
                actions=torch.zeros(4, 7),
            )

        def has_route(self, *, source: str, mode: str) -> bool:
            del mode
            return source == "real_demo"

        def build_source_view(self, **kwargs):
            return self

    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    config = apply_config_overrides(
        config,
        {
            "data.dynamics_routing.routes": [
                {"source": "real_demo", "mode": fixed_mode, "weight": 1.0}
            ]
        },
    )
    loader = DataLoader(RoutedDataset(), batch_size=1)

    runs = build_auxiliary_validation_runs(
        config,
        SimpleNamespace(distributed=False, world_size=1, rank=0),
        train_loader=loader,
        val_loader=loader,
    )

    assert [run.config.name for run in runs] == ["fdm_val", "idm_val"]


@pytest.mark.parametrize(
    ("program", "expected_mode"),
    [
        ("forward_dynamics", "action_conditioned_video"),
        ("inverse_dynamics", "video_conditioned_action"),
    ],
)
def test_strict_dynamics_validation_inherits_program_mode(
    program: str,
    expected_mode: str,
) -> None:
    class RecordingMixture(Dataset):
        def __init__(self) -> None:
            self.source_view_kwargs: dict[str, object] | None = None

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            del index
            return LatentWAMSample(
                video_latents=torch.zeros(2, 3),
                actions=torch.zeros(4, 7),
                task_text="task",
                text_context=torch.ones(1, 2),
                negative_text_context=torch.zeros(1, 2),
            )

        def has_route(self, *, source: str, mode: str) -> bool:
            del mode
            return source == "real_demo"

        def build_source_view(self, **kwargs):
            self.source_view_kwargs = kwargs
            return self

    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    route = {
        "source": "real_demo",
        "mode": expected_mode,
        "weight": 1.0,
    }
    config = apply_config_overrides(
        config,
        {
            "policy_variant.program": program,
            "data.dynamics_routing.routes": [route],
            "validation.auxiliary_tasks": [
                {
                    "name": "strict_val",
                    "source": "real_demo",
                    "max_batches": 1,
                    "report_prefix": "val_strict",
                }
            ],
        },
    )
    dataset = RecordingMixture()
    loader = DataLoader(dataset, batch_size=1)

    (run,) = build_auxiliary_validation_runs(
        config,
        SimpleNamespace(distributed=False, world_size=1, rank=0),
        train_loader=loader,
        val_loader=loader,
    )

    assert run.config.mode_override == DynamicsObjective(expected_mode)
    assert dataset.source_view_kwargs == {
        "source": "real_demo",
        "mode": expected_mode,
        "bucket_name": "strict_val",
        "spread_indices": True,
    }
    sample = run.loader.dataset[0]
    assert sample.task_text is None
    assert sample.metadata[DYNAMICS_ROUTING_MODE_METADATA_KEY] == expected_mode


def test_strict_dynamics_dataloader_does_not_build_planning_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OneSampleDataset(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> LatentWAMSample:
            del index
            return LatentWAMSample(
                video_latents=torch.zeros(2, 2, 2, 2),
                actions=torch.zeros(8, 7),
                action_mask=torch.ones(8, 7),
            )

    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    config = replace(config, data=replace(config.data, num_workers=0))
    routed = OneSampleDataset()

    monkeypatch.setattr(
        data_loading_module,
        "build_train_val_latent_datasets",
        lambda _: pytest.fail("conditional-only routes must not build planning data"),
    )

    def build_routed(**kwargs):
        assert kwargs["train_dataset"] is None
        assert kwargs["val_dataset"] is None
        return routed, routed

    monkeypatch.setattr(
        data_loading_module,
        "build_dynamics_routing_datasets",
        build_routed,
    )

    train_loader, val_loader = build_runtime_dataloaders(
        config,
        SimpleNamespace(distributed=False, world_size=1, rank=0),
    )

    assert train_loader.dataset is routed
    assert val_loader.dataset is routed


def test_strict_dynamics_preflight_checks_only_encoded_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    observed: list[tuple[object, tuple[object, ...]]] = []
    monkeypatch.setattr(
        data_loading_module,
        "preflight_dataset_artifacts",
        lambda _: pytest.fail(
            "conditional-only routes must not preflight planning data"
        ),
    )
    monkeypatch.setattr(
        data_loading_module,
        "preflight_encoded_dynamics_artifact",
        lambda root, *, sources, config_path: observed.append((root, sources)) or (),
    )

    statuses = preflight_runtime_dataset_artifacts(config)

    assert statuses == ()
    assert len(observed) == 2
    assert all(len(sources) == 2 for _, sources in observed)


@pytest.mark.parametrize(
    "task_overrides",
    [
        {"enabled": False, "max_batches": 1},
        {"enabled": True, "max_batches": 0},
    ],
)
def test_disabled_strict_dynamics_probe_is_ignored(
    task_overrides: dict[str, object],
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    config = apply_config_overrides(
        config,
        {
            "validation.auxiliary_tasks": [
                {
                    "name": "disabled_idm_probe",
                    "source": "real_demo",
                    "mode_override": "video_conditioned_action",
                    "report_prefix": "val_disabled_idm",
                    **task_overrides,
                }
            ]
        },
    )
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)

    runs = build_auxiliary_validation_runs(
        config,
        SimpleNamespace(distributed=False, world_size=1, rank=0),
        train_loader=loader,
        val_loader=loader,
    )

    assert runs == ()


@pytest.mark.parametrize(
    "dynamics_metric_namespace",
    ("joint_denoise", "dual_expert_generalist"),
)
def test_training_runtime_runs_primary_and_auxiliary_validation_phases(
    dynamics_metric_namespace: str,
) -> None:
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
    runtime.dynamics_metric_namespace = dynamics_metric_namespace
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
                    f"{dynamics_metric_namespace}/action_loss_active": torch.tensor(
                        0.0
                    ),
                    f"{dynamics_metric_namespace}/latent_loss_active": torch.tensor(
                        1.0
                    ),
                    f"{dynamics_metric_namespace}/action_conditioned_video/count": torch.tensor(
                        1.0
                    ),
                },
            )

    runtime.step_executor = Executor()

    runtime._run_all_validation(limit_batches=1)

    assert logged[0] == (
        "val",
        7,
        {
            "loss": 1.0,
            f"{dynamics_metric_namespace}/action_loss_active": 0.0,
            f"{dynamics_metric_namespace}/latent_loss_active": 1.0,
            f"{dynamics_metric_namespace}/action_conditioned_video/count": 1.0,
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


def test_training_runtime_initializes_dual_expert_variant_before_strategy_wrap(
    tmp_path: Path,
) -> None:
    config_path = REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    pipeline = runtime.strategy.unwrap_model(runtime.model)

    assert pipeline.policy_variant._action_expert_initialized is True


def test_generalist_checkpoint_writes_yaml_safe_enum_dict_keys(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
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
    assert f"schema_version: {EXPERIMENT_CONFIG_SCHEMA_VERSION}" in resolved_text
    assert "routes:" in resolved_text
    assert "source: real_demo" in resolved_text
    assert "mode: joint" in resolved_text
    assert "python/object" not in resolved_text


def test_model_only_checkpoint_does_not_collect_optimizer_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
    assert (checkpoint_dir / "full_training_state.pt").exists()
    assert (checkpoint_dir / "model_state.pt").exists()

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


def test_full_state_resume_notifies_checkpoint_lifecycle(tmp_path: Path) -> None:
    class LifecycleLinear(torch.nn.Linear):
        loaded_keys: frozenset[str] | None = None
        missing_keys: frozenset[str] | None = None

        def on_checkpoint_loaded(
            self,
            *,
            loaded_state_keys: frozenset[str],
            missing_state_keys: frozenset[str],
        ) -> None:
            self.loaded_keys = loaded_state_keys
            self.missing_keys = missing_state_keys

    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    model = LifecycleLinear(2, 2)
    checkpoint_path = tmp_path / "full_training_state.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "strategy_state_dict": None,
            "train_state": TrainState().state_dict(),
        },
        checkpoint_path,
    )

    manager.load(path=checkpoint_path, model=model)

    assert model.loaded_keys == frozenset({"weight", "bias"})
    assert model.missing_keys == frozenset()


def test_checkpoint_manager_prunes_old_checkpoints_after_successful_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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


def test_model_only_checkpoint_initializes_weights_without_train_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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

    resolved = manager.initialize_weights(
        path=checkpoint_dir / "model_state.pt", model=torch.nn.Linear(1, 1)
    )

    assert loaded_keys == ["weight"]
    assert resolved == (checkpoint_dir / "model_state.pt").resolve()


def test_weight_initialization_normalizes_pipeline_prefix_and_notifies_lifecycle(
    tmp_path: Path,
) -> None:
    class LifecycleLinear(torch.nn.Linear):
        loaded_keys: frozenset[str] | None = None
        missing_keys: frozenset[str] | None = None

        def on_checkpoint_loaded(
            self,
            *,
            loaded_state_keys: frozenset[str],
            missing_state_keys: frozenset[str],
        ) -> None:
            self.loaded_keys = loaded_state_keys
            self.missing_keys = missing_state_keys

    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    source = torch.nn.Linear(2, 2)
    checkpoint_path = tmp_path / "prefixed_state.pt"
    torch.save(
        {
            "state_dict": {
                f"pipeline.{key}": value.detach().clone()
                for key, value in source.state_dict().items()
            }
        },
        checkpoint_path,
    )
    target = LifecycleLinear(2, 2)

    manager.initialize_weights(path=checkpoint_path, model=target)

    for key, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], expected)
    assert target.loaded_keys == frozenset({"weight", "bias"})
    assert target.missing_keys == frozenset()


def test_weight_initialization_rejects_checkpoint_without_matching_parameters(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    checkpoint_path = tmp_path / "model_state.pt"
    torch.save(
        {"model_state_dict": {"unrelated.weight": torch.ones(1)}},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="no parameters matching"):
        manager.initialize_weights(
            path=checkpoint_path,
            model=torch.nn.Linear(1, 1),
        )


def test_distributed_weight_initialization_uses_rank_zero_broadcast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    checkpoint_path = tmp_path / "model_state.pt"
    torch.save({"model_state_dict": {"weight": torch.ones(1)}}, checkpoint_path)
    observed_broadcast_flags: list[bool] = []

    def fake_set_model_state_dict(model, state_dict, options):
        del model, state_dict
        observed_broadcast_flags.append(options.broadcast_from_rank0)

    monkeypatch.setattr(checkpoints_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoints_module, "_is_rank_zero", lambda: True)
    monkeypatch.setattr(
        checkpoints_module.dist,
        "broadcast_object_list",
        lambda objects, src: None,
    )
    monkeypatch.setattr(
        checkpoints_module,
        "set_model_state_dict",
        fake_set_model_state_dict,
    )

    manager.initialize_weights(path=checkpoint_path, model=torch.nn.Linear(1, 1))

    assert observed_broadcast_flags == [True]


def test_distributed_weight_initialization_reads_only_on_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    checkpoint_path = tmp_path / "model_state.pt"
    checkpoint_path.touch()

    monkeypatch.setattr(checkpoints_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoints_module, "_is_rank_zero", lambda: False)
    monkeypatch.setattr(
        checkpoints_module,
        "_load_tensor_artifact",
        lambda *args, **kwargs: pytest.fail("nonzero rank read the checkpoint"),
    )

    broadcast_count = 0

    def fake_broadcast(objects, src):
        nonlocal broadcast_count
        del src
        broadcast_count += 1
        if broadcast_count == 2:
            objects[0] = (frozenset({"weight"}), frozenset({"bias"}))

    monkeypatch.setattr(
        checkpoints_module.dist,
        "broadcast_object_list",
        fake_broadcast,
    )
    monkeypatch.setattr(
        checkpoints_module,
        "set_model_state_dict",
        lambda model, state_dict, options: None,
    )

    manager.initialize_weights(path=checkpoint_path, model=torch.nn.Linear(1, 1))


def test_full_state_resume_rejects_explicit_model_only_file(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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

    with pytest.raises(FileNotFoundError, match="full_training_state.pt"):
        manager.resolve_checkpoint_path(checkpoint_dir / "model_state.pt")

    assert manager.resolve_checkpoint_path(checkpoint_dir) == (
        checkpoint_dir / "full_training_state.pt"
    ).resolve()


def test_full_state_resume_rejects_model_only_payload_with_full_state_filename(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    checkpoint_path = tmp_path / "full_training_state.pt"
    model = torch.nn.Linear(1, 1)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="not a full training-state checkpoint"):
        manager.load(path=checkpoint_path, model=model)


@pytest.mark.parametrize(
    "missing_cursor_key",
    [
        "global_step",
        "optimizer_step",
        "epoch_index",
        "next_batch_index",
        "seen_batches",
    ],
)
def test_full_state_resume_requires_complete_train_cursor(
    tmp_path: Path,
    missing_cursor_key: str,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    checkpoint_path = tmp_path / "full_training_state.pt"
    model = torch.nn.Linear(1, 1)
    train_state = TrainState().state_dict()
    del train_state[missing_cursor_key]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "strategy_state_dict": None,
            "train_state": train_state,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match=missing_cursor_key):
        manager.load(path=checkpoint_path, model=model)


def test_full_state_save_rejects_partial_gradient_accumulation(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    config = replace(
        config,
        training=replace(config.training, gradient_accumulation_steps=4),
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )

    with pytest.raises(ValueError, match="optimizer boundary"):
        manager.save(
            step=0,
            model=torch.nn.Linear(1, 1),
            optimizer=None,
            scheduler=None,
            train_state=TrainState(global_step=3),
        )

    assert not (tmp_path / "checkpoints" / "checkpoint_step_0").exists()


def test_full_state_resume_rejects_partial_gradient_accumulation(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    config = replace(
        config,
        training=replace(config.training, gradient_accumulation_steps=4),
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    checkpoint_path = tmp_path / "full_training_state.pt"
    model = torch.nn.Linear(1, 1)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "strategy_state_dict": None,
            "train_state": TrainState(global_step=3).state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="partially accumulated"):
        manager.load(path=checkpoint_path, model=model)


def test_full_state_resume_requires_optimizer_state_when_optimizer_is_present(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    checkpoint_path = tmp_path / "full_training_state.pt"
    model = torch.nn.Linear(1, 1)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "strategy_state_dict": None,
            "train_state": TrainState().state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="no optimizer state"):
        manager.load(
            path=checkpoint_path,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
        )


def test_full_state_resume_requires_all_current_model_keys(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    checkpoint_path = tmp_path / "full_training_state.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.ones(1, 1)},
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "strategy_state_dict": None,
            "train_state": TrainState().state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="missing 1 current model key"):
        manager.load(path=checkpoint_path, model=torch.nn.Linear(1, 1))


def test_full_state_resume_requires_scheduler_state_when_scheduler_is_present(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
    )
    checkpoint_path = tmp_path / "full_training_state.pt"
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "strategy_state_dict": None,
            "train_state": TrainState().state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="no scheduler state"):
        manager.load(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: 1.0,
            ),
        )


@pytest.mark.parametrize(
    ("load_error", "expected_error_type"),
    [
        (OSError("corrupt checkpoint"), RuntimeError),
        (torch.OutOfMemoryError("checkpoint load OOM"), torch.OutOfMemoryError),
    ],
)
def test_distributed_initialization_broadcasts_deserialization_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_error: Exception,
    expected_error_type: type[Exception],
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
    )
    checkpoint_path = tmp_path / "model_state.pt"
    checkpoint_path.touch()
    broadcasts: list[object] = []

    monkeypatch.setattr(checkpoints_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoints_module, "_is_rank_zero", lambda: True)
    monkeypatch.setattr(
        checkpoints_module,
        "_load_tensor_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(load_error),
    )
    monkeypatch.setattr(
        checkpoints_module.dist,
        "broadcast_object_list",
        lambda objects, src: broadcasts.append(objects[0]),
    )

    with pytest.raises(expected_error_type, match=str(load_error)):
        manager.initialize_weights(path=checkpoint_path, model=torch.nn.Linear(1, 1))

    assert broadcasts == [(type(load_error).__name__, str(load_error))]


def test_checkpoint_latest_ignores_unmarked_partial_when_markers_exist(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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


def test_final_checkpoint_rejects_partial_accumulation_before_deduplication(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    config = replace(
        config,
        training=replace(config.training, gradient_accumulation_steps=4),
        trainer=replace(
            config.trainer,
            checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
            enable_checkpointing=True,
        ),
    )
    checkpoint_dir = tmp_path / "checkpoints" / "checkpoint_step_5"
    runtime = SimpleNamespace(
        config=config,
        train_state=TrainState(
            global_step=3,
            optimizer_step=5,
            last_checkpoint_path=str(checkpoint_dir),
        ),
        checkpoint_manager=SimpleNamespace(
            checkpoint_dir_for_step=lambda step: checkpoint_dir,
        ),
    )

    with pytest.raises(ValueError, match="optimizer boundary"):
        TrainingRuntime._save_checkpoint(runtime, final=True)


def test_epoch_loop_final_save_rejects_partial_gradient_accumulation(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    config = replace(
        config,
        training=replace(config.training, gradient_accumulation_steps=4),
        trainer=replace(
            config.trainer,
            checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
            enable_checkpointing=True,
            save_interval=None,
        ),
    )
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.config = config
    runtime.train_loader = range(3)
    runtime.train_state = TrainState(run_name="partial-epoch-final-save")
    runtime.strategy = SimpleNamespace(is_main_process=True)
    runtime._run_all_validation = lambda *, limit_batches: None
    runtime.checkpoint_manager = SimpleNamespace(
        checkpoint_dir_for_step=lambda step: (
            tmp_path / "checkpoints" / f"checkpoint_step_{step}"
        )
    )

    def train_one_batch(batch) -> None:
        del batch
        runtime.train_state.global_step += 1
        runtime.train_state.seen_batches += 1

    runtime._train_micro_step = train_one_batch

    with pytest.raises(ValueError, match="optimizer boundary"):
        TrainingRuntime._run_epoch_loop(runtime, EpochLoopPolicy(max_epochs=1))

    assert runtime.train_state.epoch_index == 1
    assert runtime.train_state.next_batch_index == 0


def test_save_interval_zero_disables_final_checkpoint(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
    assert (checkpoint_dir / RUNTIME_BACKBONE_MANIFEST_FILENAME).exists()
    manifest = load_runtime_backbone_manifest(checkpoint_dir)
    assert manifest is not None
    assert manifest.components == ("visual_tower.runtime_backbone",)


def test_composable_runtime_exports_only_selected_backbone_components(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
            runtime_backbone_export_components=(
                TrainingComponentSelector.VISUAL_TOWER_SHARED_VIDEO_BACKBONE,
            ),
        ),
    )

    TrainingRuntime.from_config(config).run()

    transformer_dir = (
        tmp_path / config.name / "checkpoints" / "checkpoint_step_1" / "transformer"
    )
    manifest = load_runtime_backbone_manifest(transformer_dir)
    assert manifest is not None
    assert manifest.components == (
        TrainingComponentSelector.VISUAL_TOWER_SHARED_VIDEO_BACKBONE,
    )
    with safe_open(
        transformer_dir / "diffusion_pytorch_model.safetensors",
        framework="pt",
    ) as handle:
        exported_keys = set(handle.keys())
    assert exported_keys == set(manifest.state_keys)
    assert "patch_embedding_mlp.weight" in exported_keys
    assert "action_embedder.weight" not in exported_keys
    assert not any(key.startswith("action_time_conditioner.") for key in exported_keys)


def test_scoped_runtime_backbone_export_accepts_rank_zero_only_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            export_runtime_backbone=True,
            runtime_backbone_export_components=(
                TrainingComponentSelector.VISUAL_TOWER_SHARED_VIDEO_BACKBONE,
            ),
        ),
    )
    pipeline = build_variant_pipeline_from_config(config)
    backbone = pipeline.visual_tower.get_runtime_backbone(
        action_dim=int(pipeline.visual_tower.action_dim)
    )
    export_keys = resolve_runtime_backbone_export_keys(
        backbone=backbone,
        topology=pipeline.module_topology(),
        selectors=config.trainer.runtime_backbone_export_components,
    )
    assert export_keys
    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
        export_runtime_backbone=True,
        runtime_backbone_export_keys=export_keys,
    )
    monkeypatch.setattr(checkpoints_module, "_is_rank_zero", lambda: False)
    monkeypatch.setattr(
        checkpoints_module,
        "get_model_state_dict",
        lambda model, options: {},
    )

    manager._export_runtime_backbone(tmp_path, pipeline)

    assert not (tmp_path / "transformer").exists()


def test_scoped_export_keys_survive_activation_checkpoint_wrapping(
    tmp_path: Path,
) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    config = replace(
        config,
        backbone=replace(
            config.backbone,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            attention_head_dim=4,
            ffn_dim=32,
            text_dim=8,
            freq_dim=8,
            pretrained_model_name_or_path=None,
            load_reference_core_weights=False,
        ),
        trainer=replace(
            config.trainer,
            checkpoint_mode=CheckpointMode.MODEL_ONLY,
            export_runtime_backbone=True,
            runtime_backbone_export_components=(
                TrainingComponentSelector.VISUAL_TOWER_SHARED_VIDEO_BACKBONE,
            ),
        ),
    )
    pipeline = build_variant_pipeline_from_config(config)
    backbone = pipeline.visual_tower.get_runtime_backbone(
        action_dim=int(pipeline.visual_tower.action_dim)
    )
    export_keys = resolve_runtime_backbone_export_keys(
        backbone=backbone,
        topology=pipeline.module_topology(),
        selectors=config.trainer.runtime_backbone_export_components,
    )
    assert export_keys is not None
    expected_block_keys = {key for key in export_keys if key.startswith("blocks.0.")}
    assert expected_block_keys

    backbone.blocks[0] = checkpoint_wrapper(
        backbone.blocks[0],
        preserve_rng_state=False,
    )
    assert any(
        "_checkpoint_wrapped_module" in name for name, _ in backbone.named_parameters()
    )

    manager = CheckpointManager(
        root_dir=tmp_path / "checkpoints",
        config=config,
        checkpoint_mode=CheckpointMode.MODEL_ONLY,
        export_runtime_backbone=True,
        runtime_backbone_export_keys=export_keys,
    )
    checkpoint_dir = manager.save(
        step=1,
        model=pipeline,
        optimizer=None,
        scheduler=None,
        train_state=TrainState(run_name="scoped_export"),
    )
    with safe_open(
        checkpoint_dir / "transformer" / "diffusion_pytorch_model.safetensors",
        framework="pt",
    ) as handle:
        exported_keys = set(handle.keys())

    assert expected_block_keys <= exported_keys
    assert exported_keys == set(export_keys)


def test_composable_runtime_disable_checkpointing_suppresses_export_runtime_backbone(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
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
