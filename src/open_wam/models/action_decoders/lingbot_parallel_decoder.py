from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.configs import JointDenoiseTrainingMode
from open_wam.models.policy_variants.contracts import PolicyInferOutput, PolicyTrainBatch, PolicyTrainOutput
from open_wam.models.common.metric_rollups import add_joint_conditioning_mode_metrics

from .base import ActionDecoder, ActionDecoderInferOutput, ActionDecoderTrainOutput, align_policy_features
from open_wam.models.policy_variants.parallel_stream.reference_runtime import data_seq_to_patch


class LingbotParallelActionDecoder(ActionDecoder):
    """Pass-through decoder and exact LingBot joint loss for the parallel-stream runtime."""

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        action_horizon: int,
        dropout: float = 0.0,
        *,
        recovered_osc_loss_weight: float = 0.0,
        recovered_osc_position_scale: float = 0.010576533139391671,
        recovered_osc_rotation_scale: float = 0.1136411594890211,
        source_action_channel_ids: tuple[int, ...] = (),
        source_action_mean: tuple[float, ...] = (),
        source_action_std: tuple[float, ...] = (),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.dropout = dropout
        self.recovered_osc_loss_weight = float(recovered_osc_loss_weight)
        self.recovered_osc_position_scale = float(recovered_osc_position_scale)
        self.recovered_osc_rotation_scale = float(recovered_osc_rotation_scale)
        self.source_action_channel_ids = tuple(int(index) for index in source_action_channel_ids)
        self.source_action_mean = tuple(float(value) for value in source_action_mean)
        self.source_action_std = tuple(float(value) for value in source_action_std)

    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        latent_pred = policy_output.aux["latent_pred"]
        train_artifacts = policy_output.aux["lingbot_train_artifacts"]
        loss_weights = policy_output.aux.get("loss_weights", {})
        latent_scheduler = train_artifacts.latent_scheduler
        action_scheduler = train_artifacts.action_scheduler
        input_dict = train_artifacts.input_dict
        action_pred = policy_output.policy_features
        configured_latent_loss_weight = float(loss_weights.get("latent", 1.0))
        configured_action_loss_weight = float(loss_weights.get("action", 1.0))

        action_pred_5d = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=input_dict["action_dict"]["targets"].shape[-3],
        )
        latent_pred_5d = data_seq_to_patch(
            policy_output.aux["patch_size"],
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )

        latent_batch_frames, latent_num_frames = input_dict["latent_dict"]["timesteps"].shape
        action_batch_frames, action_num_frames = input_dict["action_dict"]["timesteps"].shape
        latent_scheduler_weight = latent_scheduler.training_weight(input_dict["latent_dict"]["timesteps"].flatten()).reshape(
            latent_batch_frames,
            latent_num_frames,
        )
        action_scheduler_weight = action_scheduler.training_weight(input_dict["action_dict"]["timesteps"].flatten()).reshape(
            action_batch_frames,
            action_num_frames,
        )

        latent_loss = F.mse_loss(
            latent_pred_5d.float(),
            input_dict["latent_dict"]["targets"].float().detach(),
            reduction="none",
        )
        latent_loss = latent_loss * latent_scheduler_weight[:, None, :, None, None]
        latent_loss_mask = input_dict["latent_dict"].get("loss_mask")
        if latent_loss_mask is None:
            latent_loss_mask = torch.ones_like(input_dict["latent_dict"]["targets"])
        latent_loss = latent_loss * latent_loss_mask.float()
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = (
            latent_loss_mask.float().permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).sum(dim=1)
        )
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        action_loss = F.mse_loss(
            action_pred_5d.float(),
            input_dict["action_dict"]["targets"].float().detach(),
            reduction="none",
        )
        action_loss = action_loss * action_scheduler_weight[:, None, :, None, None]
        action_loss_mask = input_dict["action_dict"].get("loss_mask")
        if action_loss_mask is None:
            action_loss_mask = torch.ones_like(input_dict["action_dict"]["targets"])
        effective_action_mask = input_dict["action_dict"]["actions_mask"].float() * action_loss_mask.float()
        action_loss = action_loss * effective_action_mask
        action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_mask = effective_action_mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        abs_osc_metrics, recovered_osc_loss = self._compute_abs_eef_and_recovered_osc_metrics(
            action_pred_5d=action_pred_5d,
            input_dict=input_dict,
            action_scheduler=action_scheduler,
        )

        weighted_latent_loss = latent_loss * configured_latent_loss_weight
        weighted_action_loss = action_loss * configured_action_loss_weight
        weighted_recovered_osc_loss = recovered_osc_loss * self.recovered_osc_loss_weight
        loss = weighted_latent_loss + weighted_action_loss + weighted_recovered_osc_loss
        metrics = {
            "action_mse": action_loss.detach(),
            "latent_mse": latent_loss.detach(),
            "weighted_action_loss": weighted_action_loss.detach(),
            "weighted_latent_loss": weighted_latent_loss.detach(),
            "weighted_recovered_osc_loss": weighted_recovered_osc_loss.detach(),
            "joint_loss": loss.detach(),
            **abs_osc_metrics,
        }
        joint_denoise_mode = input_dict.get("joint_denoise_training_mode")
        if joint_denoise_mode is not None:
            action_loss_active = (
                effective_action_mask.float().sum() > 0
            ).to(dtype=torch.float32)
            latent_loss_active = (
                latent_loss_mask.float().sum() > 0
            ).to(dtype=torch.float32)
            add_joint_conditioning_mode_metrics(
                metrics,
                namespace="joint_denoise",
                mode_value=str(joint_denoise_mode),
                modes=JointDenoiseTrainingMode,
                action_loss=action_loss,
                latent_loss=latent_loss,
                action_loss_active=action_loss_active,
                latent_loss_active=latent_loss_active,
                action_metric_name="action_flow_loss_sum",
                latent_metric_name="latent_flow_loss_sum",
                action_metric_aliases=("action_mse_sum",),
                latent_metric_aliases=("latent_mse_sum",),
            )
        return ActionDecoderTrainOutput(
            action_pred=action_pred,
            loss=loss,
            metrics=metrics,
            aux={"decoder": self.__class__.__name__},
        )

    def _compute_abs_eef_and_recovered_osc_metrics(
        self,
        *,
        action_pred_5d: torch.Tensor,
        input_dict: dict[str, object],
        action_scheduler: object,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        action_dict = input_dict["action_dict"]
        assert isinstance(action_dict, dict)
        zero = action_pred_5d.float().sum() * 0.0
        if action_pred_5d.shape[1] < 10:
            return {}, zero

        noisy_actions = action_dict["noisy_latents"].float()
        target_flow = action_dict["targets"].float().detach()
        timesteps = action_dict["timesteps"]
        sigmas = action_scheduler.sigma_for_timesteps(timesteps).to(
            device=action_pred_5d.device,
            dtype=torch.float32,
        )
        sigma_view = sigmas[:, None, :, None, None]
        pred_clean = noisy_actions - sigma_view * action_pred_5d.float()
        target_clean = noisy_actions - sigma_view * target_flow

        available_mask = action_dict.get("actions_mask")
        if available_mask is None:
            available_mask = torch.ones_like(target_clean)
        loss_mask = action_dict.get("loss_mask")
        if loss_mask is None:
            loss_mask = torch.ones_like(target_clean)
        current_mask = available_mask.float() * loss_mask.float()

        pred_source, target_source, source_available_mask, source_loss_mask = self._extract_source_action_sequences(
            pred_clean=pred_clean,
            target_clean=target_clean,
            available_mask=available_mask.float(),
            current_mask=current_mask,
        )
        if pred_source is None:
            return {}, zero

        pred_source = self._denormalize_source_actions(pred_source)
        target_source = self._denormalize_source_actions(target_source)

        abs_metrics = self._source_abs_eef_metrics(
            pred_source,
            target_source,
            source_loss_mask,
            zero=zero,
        )
        osc_metrics, osc_loss = self._source_recovered_osc_metrics(
            pred_source,
            target_source,
            source_available_mask,
            source_loss_mask,
            zero=zero,
        )
        return {**abs_metrics, **osc_metrics}, osc_loss

    def _extract_source_action_sequences(
        self,
        *,
        pred_clean: torch.Tensor,
        target_clean: torch.Tensor,
        available_mask: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        pred_seq = rearrange(pred_clean, "b c f n 1 -> b (f n) c")
        target_seq = rearrange(target_clean, "b c f n 1 -> b (f n) c")
        available_seq = rearrange(available_mask, "b c f n 1 -> b (f n) c")
        current_seq = rearrange(current_mask, "b c f n 1 -> b (f n) c")
        if self.source_action_channel_ids:
            if max(self.source_action_channel_ids) >= pred_seq.shape[-1]:
                return None, None, None, None
            source_indices = torch.tensor(self.source_action_channel_ids, device=pred_seq.device, dtype=torch.long)
        elif pred_seq.shape[-1] == 10:
            source_indices = torch.arange(10, device=pred_seq.device, dtype=torch.long)
        else:
            return None, None, None, None
        if source_indices.numel() < 10:
            return None, None, None, None
        return (
            pred_seq.index_select(dim=-1, index=source_indices),
            target_seq.index_select(dim=-1, index=source_indices),
            available_seq.index_select(dim=-1, index=source_indices),
            current_seq.index_select(dim=-1, index=source_indices),
        )

    def _denormalize_source_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if len(self.source_action_mean) != actions.shape[-1] or len(self.source_action_std) != actions.shape[-1]:
            return actions
        mean = torch.tensor(self.source_action_mean, device=actions.device, dtype=actions.dtype)
        std = torch.tensor(self.source_action_std, device=actions.device, dtype=actions.dtype)
        return actions * std.clamp_min(1e-6) + mean

    def _source_abs_eef_metrics(
        self,
        pred_source: torch.Tensor,
        target_source: torch.Tensor,
        source_loss_mask: torch.Tensor,
        *,
        zero: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "abs_eef_mse": self._masked_mse(pred_source, target_source, source_loss_mask, zero=zero).detach(),
            "abs_eef_position_mse": self._masked_mse(
                pred_source[..., 0:3],
                target_source[..., 0:3],
                source_loss_mask[..., 0:3],
                zero=zero,
            ).detach(),
            "abs_eef_rotation6d_mse": self._masked_mse(
                pred_source[..., 3:9],
                target_source[..., 3:9],
                source_loss_mask[..., 3:9],
                zero=zero,
            ).detach(),
            "abs_eef_gripper_mse": self._masked_mse(
                pred_source[..., 9:10],
                target_source[..., 9:10],
                source_loss_mask[..., 9:10],
                zero=zero,
            ).detach(),
        }

    def _source_recovered_osc_metrics(
        self,
        pred_source: torch.Tensor,
        target_source: torch.Tensor,
        source_available_mask: torch.Tensor,
        source_loss_mask: torch.Tensor,
        *,
        zero: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if pred_source.shape[1] < 2:
            return {
                "recovered_osc_mse": zero.detach(),
                "recovered_osc_position_mse": zero.detach(),
                "recovered_osc_rotation_mse": zero.detach(),
                "recovered_osc_gripper_mse": zero.detach(),
                "recovered_osc_transition_count": zero.detach(),
            }, zero
        pred_train_osc = self._recover_osc_from_source_sequence(pred_source)
        pred_osc = pred_train_osc.detach()
        target_osc = self._recover_osc_from_source_sequence(target_source.detach())
        available_now = source_available_mask[:, 1:, :10].amin(dim=-1, keepdim=True)
        available_prev = source_available_mask[:, :-1, :10].amin(dim=-1, keepdim=True)
        supervised_now = source_loss_mask[:, 1:, :10].amin(dim=-1, keepdim=True)
        transition_mask = available_now * available_prev * supervised_now
        transition_mask_7d = transition_mask.expand_as(pred_osc).to(dtype=pred_osc.dtype)
        osc_loss = self._masked_mse(pred_train_osc, target_osc, transition_mask_7d, zero=zero)
        full_osc_mse = self._masked_mse(pred_osc, target_osc, transition_mask_7d, zero=zero)
        metrics = {
            "recovered_osc_train_mse": osc_loss.detach(),
            "recovered_osc_mse": full_osc_mse.detach(),
            "recovered_osc_full_mse": full_osc_mse.detach(),
            "recovered_osc_position_mse": self._masked_mse(
                pred_osc[..., 0:3],
                target_osc[..., 0:3],
                transition_mask.expand_as(pred_osc[..., 0:3]).to(dtype=pred_osc.dtype),
                zero=zero,
            ).detach(),
            "recovered_osc_rotation_mse": self._masked_mse(
                pred_osc[..., 3:6],
                target_osc[..., 3:6],
                transition_mask.expand_as(pred_osc[..., 3:6]).to(dtype=pred_osc.dtype),
                zero=zero,
            ).detach(),
            "recovered_osc_gripper_mse": self._masked_mse(
                pred_osc[..., 6:7],
                target_osc[..., 6:7],
                transition_mask.to(dtype=pred_osc.dtype),
                zero=zero,
            ).detach(),
            "recovered_osc_transition_count": transition_mask.sum().detach(),
        }
        return metrics, osc_loss

    def _recover_osc_from_source_sequence(self, source: torch.Tensor) -> torch.Tensor:
        position_delta = (source[:, 1:, 0:3] - source[:, :-1, 0:3]) / self.recovered_osc_position_scale
        current_rotation = self._continuous_6d_to_rotation_matrix_stable(source[:, 1:, 3:9])
        previous_rotation = self._continuous_6d_to_rotation_matrix_stable(source[:, :-1, 3:9])
        relative_rotation = current_rotation @ previous_rotation.transpose(-1, -2)
        rotation_delta = self._rotation_matrix_to_axis_angle_stable(relative_rotation) / self.recovered_osc_rotation_scale
        gripper = source[:, 1:, 9:10]
        return torch.cat([position_delta, rotation_delta, gripper], dim=-1)

    def _continuous_6d_to_rotation_matrix_stable(self, rotation_6d: torch.Tensor) -> torch.Tensor:
        first = self._normalize_vector_stable(rotation_6d[..., 0:3])
        second_raw = rotation_6d[..., 3:6] - (first * rotation_6d[..., 3:6]).sum(dim=-1, keepdim=True) * first
        fallback_seed = torch.zeros_like(first)
        fallback_seed[..., 0] = 1.0
        y_seed = torch.zeros_like(first)
        y_seed[..., 1] = 1.0
        near_x_axis = (first * fallback_seed).sum(dim=-1, keepdim=True).abs() > 0.9
        fallback_seed = torch.where(near_x_axis, y_seed, fallback_seed)
        fallback = torch.cross(first, fallback_seed, dim=-1)
        second_norm = torch.linalg.vector_norm(second_raw, dim=-1, keepdim=True)
        second = self._normalize_vector_stable(torch.where(second_norm > 1e-3, second_raw, fallback))
        third = torch.cross(first, second, dim=-1)
        return torch.stack([first, second, third], dim=-1)

    def _normalize_vector_stable(self, vector: torch.Tensor) -> torch.Tensor:
        return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-3)

    def _rotation_matrix_to_axis_angle_stable(self, matrix: torch.Tensor) -> torch.Tensor:
        trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
        cos_angle = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
        vee = torch.stack(
            [
                matrix[..., 2, 1] - matrix[..., 1, 2],
                matrix[..., 0, 2] - matrix[..., 2, 0],
                matrix[..., 1, 0] - matrix[..., 0, 1],
            ],
            dim=-1,
        )
        sin_angle = 0.5 * torch.linalg.vector_norm(vee, dim=-1)
        angle = torch.atan2(sin_angle, cos_angle)
        scale = torch.where(
            sin_angle > 1e-4,
            angle / (2.0 * sin_angle.clamp_min(1e-6)),
            torch.full_like(angle, 0.5),
        )
        return vee * scale.unsqueeze(-1)

    def _masked_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        denom = mask.sum()
        return ((pred - target).square() * mask).sum() / denom.clamp_min(1e-6) + zero

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: object | None = None,
    ) -> ActionDecoderInferOutput:
        del previous_state
        action_pred = align_policy_features(policy_output.policy_features, self.action_horizon)
        action_pred = self._apply_action_sampler_mask(action_pred)
        aux = {
            "decoder": self.__class__.__name__,
            "action_space": "model",
            "model_action_pred": action_pred,
        }
        raw_chunk_action_pred = policy_output.aux.get("raw_chunk_action_pred")
        if isinstance(raw_chunk_action_pred, torch.Tensor):
            raw_action_pred = align_policy_features(raw_chunk_action_pred, self.action_horizon)
            aux["raw_action_pred"] = raw_action_pred
            aux["raw_chunk_action_pred"] = raw_action_pred
            aux["raw_action_space"] = "raw"
        return ActionDecoderInferOutput(
            action_pred=action_pred,
            aux=aux,
        )
