import torch

from open_wam.models.action_decoders.lingbot_parallel_decoder import LingbotParallelActionDecoder
from open_wam.models.common.flow_matching import FlowMatchScheduler
from open_wam.models.policy_variants.contracts import PolicyInferOutput, PolicyInferState


def _scheduler() -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=1000,
    )
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def _model_action_from_source(source: torch.Tensor) -> torch.Tensor:
    model = source.new_zeros(source.shape[0], source.shape[1], 30)
    model[..., 0:9] = source[..., 0:9]
    model[..., 29] = source[..., 9]
    return model


def test_recovered_osc_metrics_use_flow_clean_action_estimate() -> None:
    position_scale = 0.01
    rotation_scale = 0.1
    source = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2],
                [position_scale, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3],
                [2.0 * position_scale, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.4],
                [3.0 * position_scale, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
            ]
        ],
        dtype=torch.float32,
    )
    clean = _model_action_from_source(source)
    clean_5d = clean.reshape(1, 1, 4, 30).permute(0, 3, 1, 2).unsqueeze(-1)
    scheduler = _scheduler()
    timestep = scheduler.timesteps[250].reshape(1, 1)
    sigma = scheduler.sigma_for_timesteps(timestep).reshape(1, 1, 1, 1, 1).to(clean_5d)
    noise = torch.zeros_like(clean_5d)
    target_flow = noise - clean_5d
    noisy = (1.0 - sigma) * clean_5d + sigma * noise
    decoder = LingbotParallelActionDecoder(
        hidden_size=8,
        action_dim=30,
        action_horizon=4,
        recovered_osc_loss_weight=0.1,
        recovered_osc_position_scale=position_scale,
        recovered_osc_rotation_scale=rotation_scale,
        source_action_channel_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8, 29),
    )
    metrics, osc_loss = decoder._compute_abs_eef_and_recovered_osc_metrics(
        action_pred_5d=target_flow,
        input_dict={
            "action_dict": {
                "noisy_latents": noisy,
                "targets": target_flow,
                "timesteps": timestep,
                "actions_mask": torch.ones_like(clean_5d),
                "loss_mask": torch.ones_like(clean_5d),
            }
        },
        action_scheduler=scheduler,
    )
    assert torch.allclose(metrics["abs_eef_mse"], torch.tensor(0.0))
    assert torch.allclose(metrics["recovered_osc_mse"], torch.tensor(0.0))
    assert torch.allclose(osc_loss, torch.tensor(0.0))


def test_recovered_osc_masks_unsupervised_prefix_transition() -> None:
    source = torch.zeros(1, 3, 10)
    source[..., 3] = 1.0
    source[..., 7] = 1.0
    clean = _model_action_from_source(source)
    clean_5d = clean.reshape(1, 1, 3, 30).permute(0, 3, 1, 2).unsqueeze(-1)
    scheduler = _scheduler()
    timestep = scheduler.timesteps[100].reshape(1, 1)
    sigma = scheduler.sigma_for_timesteps(timestep).reshape(1, 1, 1, 1, 1).to(clean_5d)
    target_flow = -clean_5d
    noisy = (1.0 - sigma) * clean_5d
    loss_mask = torch.ones_like(clean_5d)
    loss_mask[:, :, :, 0:1] = 0.0
    decoder = LingbotParallelActionDecoder(
        hidden_size=8,
        action_dim=30,
        action_horizon=3,
        recovered_osc_loss_weight=0.1,
        source_action_channel_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8, 29),
    )
    metrics, _ = decoder._compute_abs_eef_and_recovered_osc_metrics(
        action_pred_5d=target_flow,
        input_dict={
            "action_dict": {
                "noisy_latents": noisy,
                "targets": target_flow,
                "timesteps": timestep,
                "actions_mask": torch.ones_like(clean_5d),
                "loss_mask": loss_mask,
            }
        },
        action_scheduler=scheduler,
    )
    assert metrics["recovered_osc_transition_count"].item() == 2.0


def test_recovered_osc_loss_has_finite_gradients_for_invalid_early_predictions() -> None:
    clean = torch.zeros(1, 30, 1, 4, 1)
    clean[:, 3, :, :, :] = 1.0
    clean[:, 7, :, :, :] = 1.0
    scheduler = _scheduler()
    timestep = scheduler.timesteps[10].reshape(1, 1)
    sigma = scheduler.sigma_for_timesteps(timestep).reshape(1, 1, 1, 1, 1).to(clean)
    target_flow = -clean
    noisy = (1.0 - sigma) * clean
    pred_flow = (target_flow + torch.randn_like(target_flow) * 0.1).requires_grad_(True)
    decoder = LingbotParallelActionDecoder(
        hidden_size=8,
        action_dim=30,
        action_horizon=4,
        recovered_osc_loss_weight=0.001,
        source_action_channel_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8, 29),
    )
    _, osc_loss = decoder._compute_abs_eef_and_recovered_osc_metrics(
        action_pred_5d=pred_flow,
        input_dict={
            "action_dict": {
                "noisy_latents": noisy,
                "targets": target_flow,
                "timesteps": timestep,
                "actions_mask": torch.ones_like(clean),
                "loss_mask": torch.ones_like(clean),
            }
        },
        action_scheduler=scheduler,
    )
    osc_loss.backward()
    assert torch.isfinite(osc_loss)
    assert torch.isfinite(pred_flow.grad).all()


def test_recovered_osc_loss_has_finite_gradients_for_near_identity_rotation() -> None:
    decoder = LingbotParallelActionDecoder(
        hidden_size=8,
        action_dim=30,
        action_horizon=4,
        recovered_osc_loss_weight=0.001,
        source_action_channel_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8, 29),
    )
    source = torch.zeros(1, 4, 10, dtype=torch.float32)
    source[..., 3] = 1.0
    source[..., 7] = 1.0
    source = source.requires_grad_(True)
    target = source.detach().clone()
    # Keep the relative rotation very close to identity. The previous
    # acos-based conversion could hit a non-finite backward derivative here
    # under bf16/autocast startup training.
    target[:, 1:, 3] = 0.999999
    target[:, 1:, 4] = 0.000001
    target[:, 1:, 6] = -0.000001
    target[:, 1:, 7] = 0.999999
    source_mask = torch.ones_like(target)

    _, osc_loss = decoder._source_recovered_osc_metrics(
        source,
        target,
        source_mask,
        source_mask,
        zero=source.sum() * 0.0,
    )
    osc_loss.backward()

    assert torch.isfinite(osc_loss)
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_recovered_osc_loss_includes_rotation_error() -> None:
    decoder = LingbotParallelActionDecoder(
        hidden_size=8,
        action_dim=30,
        action_horizon=2,
        recovered_osc_rotation_scale=0.1,
        source_action_channel_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8, 29),
    )
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    angle = torch.tensor(0.2)
    rotated_6d = torch.tensor(
        [
            torch.cos(angle).item(),
            torch.sin(angle).item(),
            0.0,
            -torch.sin(angle).item(),
            torch.cos(angle).item(),
            0.0,
        ],
        dtype=torch.float32,
    )
    pred_source = torch.zeros(1, 2, 10)
    target_source = torch.zeros(1, 2, 10)
    pred_source[:, :, 3:9] = identity_6d
    target_source[:, 0, 3:9] = identity_6d
    target_source[:, 1, 3:9] = rotated_6d
    pred_source = pred_source.requires_grad_(True)
    source_mask = torch.ones_like(target_source)

    _, osc_loss = decoder._source_recovered_osc_metrics(
        pred_source,
        target_source,
        source_mask,
        source_mask,
        zero=pred_source.sum() * 0.0,
    )
    osc_loss.backward()

    assert osc_loss.item() > 0.0
    assert pred_source.grad is not None
    assert torch.isfinite(pred_source.grad).all()


def test_lingbot_parallel_infer_keeps_model_actions_and_exposes_raw_aux() -> None:
    decoder = LingbotParallelActionDecoder(hidden_size=8, action_dim=30, action_horizon=4)
    model_actions = torch.randn(1, 4, 30)
    raw_actions = torch.randn(1, 4, 10)
    output = decoder.forward_infer(
        PolicyInferOutput(
            policy_features=model_actions,
            next_state=PolicyInferState(),
            aux={"raw_chunk_action_pred": raw_actions},
        )
    )

    assert output.action_pred.shape == (1, 4, 30)
    torch.testing.assert_close(output.action_pred, model_actions)
    torch.testing.assert_close(output.aux["raw_action_pred"], raw_actions)
    assert output.aux["action_space"] == "model"
    assert output.aux["raw_action_space"] == "raw"
    torch.testing.assert_close(output.aux["model_action_pred"], model_actions)
