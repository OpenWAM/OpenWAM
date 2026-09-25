"""Training strategies own storage precision across policies and full resumes."""

import random

import pytest
import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.fsdp import FSDPModule

from open_wam.configs import (
    ActionSchemaConfig,
    CausalVideoPredictionPolicyConfig,
    CausalVideoProgram,
    CheckpointMode,
    DualExpertActionDecoderConfig,
    DualExpertPolicyConfig,
    ExperimentConfig,
    ParallelStreamActionDecoderConfig,
    ParallelStreamPolicyConfig,
    RobotWinDataConfig,
    SampleConstructionConfig,
    SharedVideoTransformerConfig,
    StrategyName,
    TrainerConfig,
    TrainingConfig,
    VideoOnlyActionDecoderConfig,
)
from open_wam.data.latent_contracts import LatentWAMSample, collate_latent_wam_samples
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training.checkpoints import CheckpointManager
from open_wam.training.controls import apply_training_component_controls
from open_wam.training.optim import build_optimizer, build_scheduler
from open_wam.training.state import TrainState
from open_wam.training.step_executor import (
    LatentBatchAdapter,
    PipelineTrainStepExecutor,
)
from open_wam.training.strategies import build_training_strategy


def _config(architecture, program, device, strategy):
    video_only = architecture == "causal_video"
    common = dict(
        hidden_size=128,
        program=program,
        sequence_contract="legacy_prefix_single_frame_perchunk_proprio",
        proprio_context_mode="per_chunk_additive",
    )
    if architecture == "parallel_stream":
        policy = ParallelStreamPolicyConfig(**common, action_per_frame=2)
        decoder = ParallelStreamActionDecoderConfig(
            hidden_size=128, action_dim=4, action_horizon=8
        )
    elif architecture == "dual_expert":
        policy = DualExpertPolicyConfig(**common, num_action_layers=1)
        decoder = DualExpertActionDecoderConfig(
            hidden_size=128, action_dim=4, action_horizon=8
        )
    else:
        policy = CausalVideoPredictionPolicyConfig(
            hidden_size=128, program=program, noisy_video_condition_prob=0.5
        )
        decoder = VideoOnlyActionDecoderConfig(hidden_size=128, action_dim=4)
    return ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            train_batch_size=1,
            sample_construction=SampleConstructionConfig(
                mode="uniform_segment",
                chunk_size=2,
                window_size=8,
                require_full_segment=True,
                condition_source_frame_offset=-1,
            ),
            action_schema=ActionSchemaConfig(
                action_dim=4,
                action_horizon=0 if video_only else 8,
                state_dim=4,
                state_horizon=0 if video_only else 1,
            ),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=128,
            num_layers=1,
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
        policy_variant=policy,
        action_decoder=decoder,
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            learning_rate=1e-5,
            warmup_steps=2,
            enabled_objectives=("latent",) if video_only else ("action", "latent"),
            action_loss_weight=0.0 if video_only else 1.0,
            trainable_components=(
                ("visual_tower.shared_video_backbone",) if video_only else ("all",)
            ),
        ),
        trainer=TrainerConfig(
            accelerator="cpu" if device == "cpu" else "gpu",
            precision="32-true" if device == "cpu" else "bf16-mixed",
            strategy=strategy,
        ),
    )


def _batch(config):
    schema = config.data.action_schema
    return collate_latent_wam_samples(
        [
            LatentWAMSample(
                video_latents=torch.randn(48, 4, 4, 4),
                condition_latents=torch.randn(48, 1, 4, 4),
                actions=torch.randn(schema.action_horizon, 4),
                action_mask=torch.ones(schema.action_horizon, 4),
                state=torch.randn(schema.state_horizon, 4),
                proprio_context_frames=torch.randn(4, 4),
                text_context=torch.randn(3, 16),
                negative_text_context=torch.zeros(3, 16),
                task_text="move object",
                metadata={
                    "sampled_chunk_size": 2,
                    "sampled_window_size": 8,
                    "action_tokens_per_frame": 2,
                    "frame_shift": 0,
                },
            )
        ]
    )


def _model_and_optimizer(config, strategy):
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.visual_tower.get_runtime_backbone(
        action_dim=config.data.action_schema.action_dim
    )
    pipeline.policy_variant.initialize_for_training(pipeline.visual_tower)
    apply_training_component_controls(pipeline, config.training)
    model = strategy.prepare_model(pipeline)
    optimizer = build_optimizer(model, config.training)
    return model, optimizer, build_scheduler(optimizer, config.training)


def _assert_optimizer_storage(optimizer):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert parameter.dtype == torch.float32
    for state in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            assert state[name].dtype == torch.float32
            assert torch.isfinite(state[name]).all()


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize(
    "strategy_name", (StrategyName.SINGLE_DEVICE, StrategyName.FSDP)
)
@pytest.mark.parametrize(
    "architecture,program",
    [
        (architecture, program)
        for architecture in ("parallel_stream", "dual_expert")
        for program in ("video_then_action", "joint")
    ]
    + [("causal_video", CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO)],
)
def test_training_preserves_precision_through_update_and_resume(
    tmp_path, monkeypatch, device, strategy_name, architecture, program
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise BF16 mixed precision.")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPEN_WAM_FSDP_CPU_OFFLOAD", "0")
    config = _config(architecture, program, device, strategy_name)
    strategy = build_training_strategy(config.trainer)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    try:
        torch.manual_seed(719)
        model, optimizer, scheduler = _model_and_optimizer(config, strategy)
        if strategy_name == StrategyName.FSDP:
            assert isinstance(model, FSDPModule)
            assert strategy.world_size == 1
        batch = LatentBatchAdapter().move_to_device(_batch(config), strategy.device)
        parameters = tuple(model.parameters())
        original = {
            name: value.clone()
            for name, value in get_model_state_dict(model, options=options).items()
        }

        def update(current_model, current_optimizer, current_scheduler, seed):
            # Replay the same stochastic inputs for the resumed optimizer update.
            torch.manual_seed(seed)
            random.seed(seed)
            strategy.zero_grad(current_optimizer)
            executor = PipelineTrainStepExecutor(
                pipeline=current_model,
                batch_adapter=LatentBatchAdapter(),
                training_config=config.training,
            )
            with strategy.autocast_context():
                result = executor.forward_train(batch)
            assert torch.isfinite(result.loss)
            _assert_optimizer_storage(current_optimizer)
            strategy.backward(result.loss)
            norm = strategy.clip_grad_norm_(current_model.parameters(), 2.0)
            assert torch.isfinite(norm) and norm > 0
            strategy.optimizer_step(current_optimizer)
            current_scheduler.step()
            _assert_optimizer_storage(current_optimizer)

        update(model, optimizer, scheduler, 811)
        assert tuple(map(id, model.parameters())) == tuple(map(id, parameters))
        first = get_model_state_dict(model, options=options)
        changed = {
            name for name in original if not torch.equal(original[name], first[name])
        }
        assert any("blocks.0." in name for name in changed)
        assert any("proj_out.weight" in name for name in changed)
        assert optimizer.state
        manager = CheckpointManager(
            root_dir=tmp_path,
            config=config,
            checkpoint_mode=CheckpointMode.FULL_TRAINING_STATE,
        )
        checkpoint = manager.save(
            step=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_state=TrainState(global_step=1, optimizer_step=1),
            strategy_state=strategy.state_dict(),
        )
        saved = torch.load(checkpoint / "full_training_state.pt", weights_only=False)
        update(model, optimizer, scheduler, 812)
        expected = get_model_state_dict(model, options=options)
        expected_optimizer = get_optimizer_state_dict(model, optimizer, options=options)

        resumed, restored_optimizer, restored_scheduler = _model_and_optimizer(
            config, strategy
        )
        state, payload = manager.load(
            path=checkpoint,
            model=resumed,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
        )
        strategy.load_state_dict(payload["strategy_state_dict"])
        assert state.optimizer_step == 1
        assert state.global_step == 1
        assert restored_scheduler.state_dict() == saved["scheduler_state_dict"]
        _assert_optimizer_storage(restored_optimizer)
        torch.testing.assert_close(
            get_model_state_dict(resumed, options=options),
            saved["model_state_dict"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            get_optimizer_state_dict(resumed, restored_optimizer, options=options)[
                "state"
            ],
            saved["optimizer_state_dict"]["state"],
            rtol=0,
            atol=0,
        )
        update(resumed, restored_optimizer, restored_scheduler, 812)
        torch.testing.assert_close(
            get_model_state_dict(resumed, options=options),
            expected,
            rtol=1e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            get_optimizer_state_dict(resumed, restored_optimizer, options=options)[
                "state"
            ],
            expected_optimizer["state"],
            rtol=1e-5,
            atol=1e-7,
        )
        assert restored_scheduler.state_dict() == scheduler.state_dict()
    finally:
        strategy.close()
