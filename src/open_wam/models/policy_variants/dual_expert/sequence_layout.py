from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import TrainingConfig
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.contracts import SampleConstructionMetadata
from open_wam.models.common.flow_supervision import build_video_frame_loss_mask
from open_wam.models.common.flow_training import (
    ActionFlowMatchTrainArtifacts,
)
from open_wam.models.visual_tower.grid_ids import build_action_grid_ids

from ..contracts import PolicyTrainBatch


def build_action_grid_ids_for_sequence(
    *,
    batch_size: int,
    seq_len: int,
    action_tokens_per_frame: int,
    device: torch.device,
    frame_shift: int,
) -> torch.Tensor:
    """Build frame-aligned action coordinates shared by train and inference."""

    if seq_len <= 0:
        raise ValueError(f"Expected positive action seq_len, got {seq_len}.")
    if action_tokens_per_frame <= 0 or seq_len % action_tokens_per_frame != 0:
        raise ValueError(
            "DualExpert action grid ids require `seq_len` to divide by `action_tokens_per_frame`, "
            f"got seq_len={seq_len}, action_tokens_per_frame={action_tokens_per_frame}."
        )
    num_frames = seq_len // action_tokens_per_frame
    return build_action_grid_ids(
        num_frames=num_frames,
        action_per_frame=action_tokens_per_frame,
        device=device,
        frame_shift=float(frame_shift),
    )[None].expand(batch_size, -1, -1)


@dataclass(frozen=True, slots=True)
class DualExpertTrainingLayout:
    """Resolve DualExpert sample metadata into train-time sequence geometry and masks."""

    policy_config: DualExpertPolicyConfig
    training_config: TrainingConfig

    def resolve_loss_frame_range(
        self,
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
        start_key: str = "loss_frame_start",
        end_key: str = "loss_frame_end",
        fallback_to_generic: bool = True,
    ) -> tuple[int, int] | None:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(
            batch.extra.get("metadata")
        )
        if sample_metadata is None:
            return None
        return sample_metadata.optional_frame_range(
            observed_num_frames=observed_num_frames,
            start_key=start_key,
            end_key=end_key,
            fallback_to_generic=fallback_to_generic,
            error_label="DualExpert train loss-frame metadata",
        )

    def resolve_history_frames(
        self,
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> int:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(
            batch.extra.get("metadata")
        )
        resolved_history_frames: int | None = None
        if sample_metadata is not None:
            resolved_history_frames = sample_metadata.history_frames
        loss_frame_range = self.resolve_loss_frame_range(
            batch=batch,
            observed_num_frames=observed_num_frames,
        )
        if loss_frame_range is not None:
            resolved_history_frames = int(loss_frame_range[0])
        if resolved_history_frames is None:
            resolved_history_frames = int(self.policy_config.video_prefix_frames)
        if (
            resolved_history_frames <= 0
            or resolved_history_frames >= observed_num_frames
        ):
            raise ValueError(
                "DualExpert training requires at least one history frame and one current frame, "
                f"got resolved_history_frames={resolved_history_frames}, observed_num_frames={observed_num_frames}."
            )
        return resolved_history_frames

    def build_effective_action_mask(
        self,
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> torch.Tensor | None:
        base_mask = batch.action_mask
        loss_frame_range = self.resolve_loss_frame_range(
            batch=batch,
            observed_num_frames=observed_num_frames,
            start_key="action_loss_frame_start",
            end_key="action_loss_frame_end",
        )
        if loss_frame_range is None:
            return base_mask
        if batch.actions.shape[1] % max(1, observed_num_frames) != 0:
            return base_mask
        action_per_frame = batch.actions.shape[1] // max(1, observed_num_frames)
        if action_per_frame <= 0:
            return base_mask
        loss_frame_start, loss_frame_end = loss_frame_range
        effective_mask = (
            torch.ones_like(batch.actions, dtype=torch.float32)
            if base_mask is None
            else base_mask.to(dtype=torch.float32)
        )
        frame_mask = torch.zeros_like(effective_mask)
        frame_mask[
            :, loss_frame_start * action_per_frame : loss_frame_end * action_per_frame
        ] = 1.0
        return effective_mask * frame_mask

    def build_effective_video_loss_mask(
        self,
        *,
        video_latents: torch.Tensor,
        batch: PolicyTrainBatch,
        default_history_frames: int,
        prefix_condition_frames: int = 0,
        target_num_video_frames: int | None = None,
    ) -> torch.Tensor:
        prefix_frames = int(prefix_condition_frames)
        target_frames = (
            int(video_latents.shape[2]) - prefix_frames
            if target_num_video_frames is None
            else int(target_num_video_frames)
        )
        return build_video_frame_loss_mask(
            video_latents,
            sample_metadata=batch.extra.get("metadata"),
            prefix_frame_count=prefix_frames,
            target_frame_count=target_frames,
            default_target_start=(
                0 if prefix_frames > 0 else int(default_history_frames)
            ),
            error_label="DualExpert train loss-frame metadata",
        )

    @staticmethod
    def resolve_action_tokens_per_frame(
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> int | None:
        if observed_num_frames <= 0:
            return None
        if batch.actions.shape[1] % observed_num_frames != 0:
            return None
        action_tokens_per_frame = batch.actions.shape[1] // observed_num_frames
        return action_tokens_per_frame if action_tokens_per_frame > 0 else None

    @staticmethod
    def resolve_sampled_chunk_size(
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> int | None:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(
            batch.extra.get("metadata")
        )
        if sample_metadata is None:
            return None
        return sample_metadata.sampled_chunk_size_for(observed_num_frames)

    @staticmethod
    def resolve_sampled_window_size(*, batch: PolicyTrainBatch) -> int | None:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(
            batch.extra.get("metadata")
        )
        return None if sample_metadata is None else sample_metadata.sampled_window_size

    @staticmethod
    def resolve_frame_shift(*, batch: PolicyTrainBatch) -> int:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(
            batch.extra.get("metadata")
        )
        if sample_metadata is None or sample_metadata.frame_shift is None:
            return 0
        return int(sample_metadata.frame_shift)

    @staticmethod
    def resolve_chunk_origin_frame(
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> int:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(
            batch.extra.get("metadata")
        )
        if sample_metadata is None:
            return 0
        return sample_metadata.chunk_origin_frame_for(
            observed_num_frames=observed_num_frames,
        )

    @staticmethod
    def resolve_singleton_chunk_frame(
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> int | None:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(
            batch.extra.get("metadata")
        )
        if sample_metadata is None:
            return None
        return sample_metadata.singleton_chunk_frame_for(
            observed_num_frames=observed_num_frames,
        )

    def sample_full_segment_geometry(
        self,
        *,
        observed_num_frames: int,
        device: torch.device,
    ) -> tuple[int, int, int]:
        """Draw the parallel-stream-compatible chunk, window, and history geometry."""

        cs_max = max(1, int(self.training_config.chunk_size))
        sampled_chunk_size = int(
            torch.randint(1, cs_max + 1, (1,), device=device).item()
        )
        if int(self.training_config.window_size) >= 4:
            sampled_window_size = int(
                torch.randint(
                    4, int(self.training_config.window_size) + 1, (1,), device=device
                ).item()
            )
        else:
            sampled_window_size = max(1, int(self.training_config.window_size))
        max_history_chunks = max(1, observed_num_frames // sampled_chunk_size - 1)
        history_chunks = int(
            torch.randint(1, max_history_chunks + 1, (1,), device=device).item()
        )
        history_frames = max(
            1,
            min(
                history_chunks * sampled_chunk_size,
                observed_num_frames - sampled_chunk_size,
            ),
        )
        return sampled_chunk_size, sampled_window_size, history_frames

    @staticmethod
    def apply_history_action_condition(
        *,
        train_artifacts: ActionFlowMatchTrainArtifacts,
        actions: torch.Tensor,
        observed_num_frames: int,
        history_frames: int,
    ) -> ActionFlowMatchTrainArtifacts:
        if observed_num_frames <= 0 or actions.shape[1] % observed_num_frames != 0:
            return train_artifacts
        action_tokens_per_frame = actions.shape[1] // observed_num_frames
        if action_tokens_per_frame <= 0:
            return train_artifacts
        history_action_tokens = int(history_frames * action_tokens_per_frame)
        if history_action_tokens <= 0:
            return train_artifacts
        history_action_tokens = min(history_action_tokens, int(actions.shape[1]))
        train_artifacts.noisy_actions[:, :history_action_tokens] = actions[
            :, :history_action_tokens
        ]
        train_artifacts.timesteps[:, :history_action_tokens] = 0.0
        return train_artifacts
