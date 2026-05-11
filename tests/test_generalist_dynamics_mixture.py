from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from open_wam.configs import (
    ActionSchemaConfig,
    GeneralistDynamicsMixtureConfig,
    GenericDataConfig,
    PaddedTargetPolicy,
    SampleConstructionConfig,
    TailPaddingPolicy,
    WindowSamplingMode,
)
from open_wam.configs.variant_semantics import (
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
)
from open_wam.data import (
    EncodedCounterfactualDynamicsLatentDataset,
    GeneralistDynamicsMixtureDataset,
    build_generalist_dynamics_mixture_datasets,
)
from open_wam.data.latent_contracts import LatentWAMSample


class _OneSampleLatentDataset(Dataset[LatentWAMSample]):
    def __init__(self, sample: LatentWAMSample) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> LatentWAMSample:
        del index
        return self.sample


class _RecordingDrawKeyDataset(Dataset[LatentWAMSample]):
    uses_epoch_offset_draw_keys = True

    def __init__(self, *, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LatentWAMSample:
        return LatentWAMSample(
            video_latents=torch.ones(2, 4, 2, 2),
            actions=torch.ones(8, 7),
            action_mask=torch.ones(8, 7),
            metadata={"seen_draw_key": int(index)},
        )


def test_encoded_counterfactual_dataset_concatenates_context_and_future(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert sample.video_latents.shape == (2, 4, 2, 2)
    assert sample.actions.shape == (8, 7)
    assert sample.action_mask is not None
    assert sample.action_mask.sum().item() == 56
    assert torch.equal(sample.actions[:2], torch.zeros(2, 7))
    assert torch.equal(sample.actions[2:6], torch.ones(4, 7))
    assert torch.equal(sample.actions[6:], torch.full((2, 7), 2.0))
    assert sample.task_text is None
    assert sample.text_context is not None
    assert torch.equal(sample.text_context, torch.zeros(3, 4))
    assert sample.metadata["history_frames"] == 2
    assert sample.metadata["loss_frame_start"] == 2
    assert sample.metadata["loss_frame_end"] == 4
    assert sample.metadata["action_loss_frame_start"] == 2
    assert sample.metadata["segment_pre_start_frames"] == 0
    assert sample.metadata["start_padding_mode"] == "none"
    assert sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_frames"] == 1
    assert sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_mask"] == 1.0


def test_encoded_counterfactual_dataset_uses_hierarchical_fixed_segment_semantics(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = replace(
        _data_config(empty_text_path),
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
            segment_frames=6,
            chunk_size=2,
            window_size=4,
            start_padding_frames=1,
            tail_padding_policy=TailPaddingPolicy.ZERO_ORDER_HOLD,
            padded_target_policy=PaddedTargetPolicy.MASK_LOSS,
        ),
    )

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert len(dataset) == 3
    assert sample.video_latents.shape == (2, 6, 2, 2)
    assert sample.actions.shape == (12, 7)
    assert sample.metadata["window_sampling_mode"] == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT
    assert sample.metadata["tail_padding_policy"] == str(TailPaddingPolicy.ZERO_ORDER_HOLD)
    assert sample.metadata["padded_target_policy"] == str(PaddedTargetPolicy.MASK_LOSS)
    assert sample.metadata["loss_frame_start"] == sample.metadata["history_frames"]
    assert sample.metadata["loss_frame_end"] <= sample.metadata["segment_valid_latent_frames"]
    assert sample.metadata["segment_padded_latent_frames"] >= 1
    assert sample.metadata["hierarchical_epoch_sample_count"] == 3


def test_generalist_dynamics_mixture_stamps_forced_mode_and_drops_text(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    real_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        state=torch.zeros(1, 8),
        state_mask=torch.zeros(1, 8),
        task_text="real task",
        text_context=torch.ones(3, 4),
        negative_text_context=torch.zeros(3, 4),
        metadata={"dataset_kind": "real"},
    )
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=_OneSampleLatentDataset(real_sample),
        counterfactual_dataset=counterfactual,
        mixture_config=GeneralistDynamicsMixtureConfig(
            counterfactual_action_conditioned_video_weight=1.0,
            real_joint_weight=0.0,
            real_action_conditioned_video_weight=0.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_video_conditioned_action_weight=0.0,
        ),
        split="train",
    )

    sample = mixture[0]

    assert sample.metadata[GENERALIST_TRAINING_SOURCE_METADATA_KEY] == "counterfactual_dynamics"
    assert sample.metadata[GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY] == "action_conditioned_video"
    assert sample.metadata[GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY] is True
    assert sample.task_text is None
    assert sample.text_context is not None
    assert torch.equal(sample.text_context, torch.zeros(3, 4))


def test_generalist_dynamics_mixture_trims_conditional_history(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    real_sample = LatentWAMSample(
        video_latents=torch.arange(2 * 8 * 2 * 2, dtype=torch.float32).reshape(2, 8, 2, 2),
        actions=torch.arange(16 * 7, dtype=torch.float32).reshape(16, 7),
        action_mask=torch.ones(16, 7),
        task_text="real task",
        text_context=torch.ones(3, 4),
        negative_text_context=torch.zeros(3, 4),
        metadata={
            "dataset_kind": "real",
            "sample_start_frame": 100,
            "observation_start": 100,
            "window_start_frame": 100,
            "observation_frame_indices": list(range(100, 108)),
            "observed_frame_ids": list(range(100, 108)),
            "frame_shift": 100,
            "history_frames": 6,
            "loss_frame_start": 6,
            "loss_frame_end": 8,
            "latent_loss_frame_start": 6,
            "latent_loss_frame_end": 8,
            "action_loss_frame_start": 6,
            "action_loss_frame_end": 8,
            "supervised_start": 6,
            "supervised_end": 8,
            "segment_length_frames": 8,
            "effective_start": 100,
            "effective_end": 108,
            "effective_frame_start": 100,
            "effective_frame_end": 108,
            "logical_frame_start": 100,
            "logical_frame_end": 108,
            "target_frame_start": 106,
            "target_frame_end": 108,
            "subwindow_latent_start": 106,
            "subwindow_latent_end": 108,
            "virtual_latent_start": 106,
            "context_prefix_frames_requested": 6,
            "context_prefix_frames_in_sample": 6,
            "context_prefix_real_frames": 6,
            "context_prefix_truncated_frames": 0,
            "subwindow_action_start": 100,
            "subwindow_action_end": 116,
        },
    )
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=_OneSampleLatentDataset(real_sample),
        counterfactual_dataset=counterfactual,
        mixture_config=GeneralistDynamicsMixtureConfig(
            real_action_conditioned_video_weight=1.0,
            real_joint_weight=0.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_action_conditioned_video_weight=0.0,
            counterfactual_video_conditioned_action_weight=0.0,
            conditional_history_frames=2,
        ),
        split="train",
    )

    sample = mixture[0]

    assert sample.video_latents.shape == (2, 4, 2, 2)
    assert sample.actions.shape == (8, 7)
    assert sample.metadata["history_frames"] == 2
    assert sample.metadata["loss_frame_start"] == 2
    assert sample.metadata["loss_frame_end"] == 4
    assert sample.metadata["latent_loss_frame_start"] == 2
    assert sample.metadata["action_loss_frame_start"] == 2
    assert sample.metadata["sample_start_frame"] == 104
    assert sample.metadata["observation_frame_indices"] == [104, 105, 106, 107]
    assert sample.metadata["frame_shift"] == 104
    assert sample.metadata["effective_frame_start"] == 104
    assert sample.metadata["effective_frame_end"] == 108
    assert sample.metadata["logical_frame_start"] == 104
    assert sample.metadata["target_frame_start"] == 106
    assert sample.metadata["target_frame_end"] == 108
    assert sample.metadata["subwindow_latent_start"] == 106
    assert sample.metadata["subwindow_latent_end"] == 108
    assert sample.metadata["virtual_latent_start"] == 106
    assert sample.metadata["supervised_start"] == 2
    assert sample.metadata["supervised_end"] == 4
    assert sample.metadata["context_prefix_frames_in_sample"] == 2
    assert sample.metadata["context_prefix_real_frames"] == 2
    assert sample.metadata["context_prefix_truncated_frames"] == 4
    assert sample.metadata["subwindow_action_start"] == 108
    assert sample.metadata["valid_action_steps"] == 8
    assert sample.metadata["valid_action_values"] == 56
    assert sample.metadata["generalist_conditional_history_frames"] == 2
    assert sample.metadata["generalist_history_trimmed_frames"] == 4
    assert sample.task_text is None

    view = mixture.build_source_view(
        source="real_demo",
        mode="action_conditioned_video",
        bucket_name="val_fdm",
        drop_text=True,
    )
    view_sample = view[0]

    assert view_sample.video_latents.shape == (2, 4, 2, 2)
    assert view_sample.metadata["history_frames"] == 2
    assert view_sample.metadata[GENERALIST_TRAINING_SOURCE_METADATA_KEY] == "real_demo"
    assert view_sample.metadata[GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY] == "action_conditioned_video"
    assert view_sample.metadata["generalist_training_bucket"] == "val_fdm"


def test_generalist_dynamics_mixture_trims_partially_unavailable_prefix(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    real_sample = LatentWAMSample(
        video_latents=torch.arange(2 * 13 * 2 * 2, dtype=torch.float32).reshape(2, 13, 2, 2),
        actions=torch.arange(26 * 7, dtype=torch.float32).reshape(26, 7),
        action_mask=torch.ones(26, 7),
        task_text="real task",
        text_context=torch.ones(3, 4),
        negative_text_context=torch.zeros(3, 4),
        metadata={
            "dataset_kind": "real",
            "sample_start_frame": 0,
            "observation_start": 0,
            "window_start_frame": 0,
            "observation_frame_indices": list(range(13)),
            "observed_frame_ids": list(range(13)),
            "frame_shift": 0,
            "history_frames": 8,
            "loss_frame_start": 8,
            "loss_frame_end": 13,
            "latent_loss_frame_start": 8,
            "latent_loss_frame_end": 13,
            "action_loss_frame_start": 8,
            "action_loss_frame_end": 13,
            "supervised_start": 5,
            "supervised_end": 13,
            "segment_length_frames": 13,
            "segment_valid_latent_frames": 13,
            "segment_padded_latent_frames": 0,
            "effective_start": 0,
            "effective_end": 13,
            "effective_frame_start": 0,
            "effective_frame_end": 13,
            "logical_frame_start": -3,
            "logical_frame_end": 13,
            "target_frame_start": 5,
            "target_frame_end": 13,
            "subwindow_latent_start": 5,
            "subwindow_latent_end": 13,
            "virtual_latent_start": 5,
            "context_prefix_frames_requested": 8,
            "context_prefix_frames_in_sample": 5,
            "context_prefix_real_frames": 5,
            "context_prefix_truncated_frames": 3,
            "subwindow_action_start": 0,
            "subwindow_action_end": 26,
        },
    )
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=_OneSampleLatentDataset(real_sample),
        counterfactual_dataset=counterfactual,
        mixture_config=GeneralistDynamicsMixtureConfig(
            real_action_conditioned_video_weight=1.0,
            real_joint_weight=0.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_action_conditioned_video_weight=0.0,
            counterfactual_video_conditioned_action_weight=0.0,
            conditional_history_frames=2,
        ),
        split="train",
    )

    sample = mixture[0]

    assert sample.video_latents.shape == (2, 7, 2, 2)
    assert sample.actions.shape == (14, 7)
    assert sample.metadata["history_frames"] == 2
    assert sample.metadata["loss_frame_start"] == 2
    assert sample.metadata["loss_frame_end"] == 7
    assert sample.metadata["sample_start_frame"] == 6
    assert sample.metadata["observation_frame_indices"] == list(range(6, 13))
    assert sample.metadata["frame_shift"] == 6
    assert sample.metadata["effective_frame_start"] == 6
    assert sample.metadata["effective_frame_end"] == 13
    assert sample.metadata["target_frame_start"] == 6
    assert sample.metadata["target_frame_end"] == 13
    assert sample.metadata["subwindow_latent_start"] == 6
    assert sample.metadata["subwindow_latent_end"] == 13
    assert sample.metadata["virtual_latent_start"] == 6
    assert sample.metadata["supervised_start"] == 0
    assert sample.metadata["supervised_end"] == 7
    assert sample.metadata["context_prefix_frames_in_sample"] == 0
    assert sample.metadata["context_prefix_real_frames"] == 0
    assert sample.metadata["context_prefix_truncated_frames"] == 8
    assert sample.metadata["context_prefix_truncated_frames"] <= sample.metadata["context_prefix_frames_requested"]
    assert sample.metadata["generalist_history_trimmed_frames"] == 6


def test_generalist_dynamics_mixture_preserves_epoch_offset_sampler(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    real_dataset = _RecordingDrawKeyDataset(length=5)
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual,
        mixture_config=GeneralistDynamicsMixtureConfig(
            real_joint_weight=1.0,
            real_action_conditioned_video_weight=0.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_action_conditioned_video_weight=0.0,
            counterfactual_video_conditioned_action_weight=0.0,
        ),
        split="train",
    )

    sampler = mixture.build_train_sampler(world_size=1, rank=0)
    assert list(iter(sampler)) == list(range(len(mixture)))
    sampler.set_epoch(1)
    assert list(iter(sampler)) == list(range(len(mixture), 2 * len(mixture)))

    sample = mixture[len(mixture)]

    assert sample.metadata["generalist_source_index"] >= len(real_dataset)
    assert sample.metadata["seen_draw_key"] == sample.metadata["generalist_source_index"]


def test_generalist_dynamics_mixture_trim_start_padding_keeps_action_source_start(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    real_sample = LatentWAMSample(
        video_latents=torch.arange(2 * 6 * 2 * 2, dtype=torch.float32).reshape(2, 6, 2, 2),
        actions=torch.arange(12 * 7, dtype=torch.float32).reshape(12, 7),
        action_mask=torch.ones(12, 7),
        task_text="real task",
        text_context=torch.ones(3, 4),
        negative_text_context=torch.zeros(3, 4),
        metadata={
            "dataset_kind": "real",
            "history_frames": 4,
            "loss_frame_start": 4,
            "loss_frame_end": 6,
            "segment_length_frames": 6,
            "segment_pre_start_frames": 2,
            "subwindow_action_start": 0,
        },
    )
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=_OneSampleLatentDataset(real_sample),
        counterfactual_dataset=counterfactual,
        mixture_config=GeneralistDynamicsMixtureConfig(
            real_action_conditioned_video_weight=1.0,
            real_joint_weight=0.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_action_conditioned_video_weight=0.0,
            counterfactual_video_conditioned_action_weight=0.0,
            conditional_history_frames=2,
        ),
        split="train",
    )

    sample = mixture[0]

    assert sample.metadata["history_frames"] == 2
    assert sample.metadata["segment_pre_start_frames"] == 0
    assert sample.metadata["subwindow_action_start"] == 0
    assert sample.metadata["generalist_history_trimmed_frames"] == 2


def test_build_generalist_dynamics_mixture_datasets_uses_train_and_val_roots(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    data_config = _replace_data_mixture_root(data_config, str(encoded_root))
    real_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        metadata={},
    )

    train_dataset, val_dataset = build_generalist_dynamics_mixture_datasets(
        data_config=data_config,
        train_dataset=_OneSampleLatentDataset(real_sample),
        val_dataset=_OneSampleLatentDataset(real_sample),
    )

    assert isinstance(train_dataset, GeneralistDynamicsMixtureDataset)
    assert isinstance(val_dataset, GeneralistDynamicsMixtureDataset)


def test_build_generalist_dynamics_mixture_requires_val_root(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = replace(
        _data_config(empty_text_path),
        generalist_dynamics_mixture=GeneralistDynamicsMixtureConfig(train_latent_root=str(encoded_root)),
    )
    real_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        metadata={},
    )

    with pytest.raises(ValueError, match="val_latent_root"):
        build_generalist_dynamics_mixture_datasets(
            data_config=data_config,
            train_dataset=_OneSampleLatentDataset(real_sample),
            val_dataset=_OneSampleLatentDataset(real_sample),
        )


def test_build_generalist_dynamics_mixture_allows_explicit_debug_val_fallback(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = replace(
        _data_config(empty_text_path),
        generalist_dynamics_mixture=GeneralistDynamicsMixtureConfig(
            train_latent_root=str(encoded_root),
            allow_train_latent_root_for_val=True,
        ),
    )
    real_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        metadata={},
    )

    train_dataset, val_dataset = build_generalist_dynamics_mixture_datasets(
        data_config=data_config,
        train_dataset=_OneSampleLatentDataset(real_sample),
        val_dataset=_OneSampleLatentDataset(real_sample),
    )

    assert isinstance(train_dataset, GeneralistDynamicsMixtureDataset)
    assert isinstance(val_dataset, GeneralistDynamicsMixtureDataset)


def _data_config(empty_text_path: Path) -> GenericDataConfig:
    return GenericDataConfig(
        dataset_name="libero",
        dataset_type="lerobot_v2_latent_local",
        empty_text_embedding_path=str(empty_text_path),
        action_schema=ActionSchemaConfig(
            action_dim=7,
            action_horizon=8,
            state_dim=8,
            state_horizon=1,
        ),
        sample_construction=SampleConstructionConfig(chunk_size=2, window_size=4),
    )


def _replace_data_mixture_root(data_config: GenericDataConfig, root: str) -> GenericDataConfig:
    return replace(
        data_config,
        generalist_dynamics_mixture=GeneralistDynamicsMixtureConfig(
            train_latent_root=root,
            val_latent_root=root,
        ),
    )


def _write_encoded_counterfactual_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw_root = tmp_path / "raw"
    encoded_root = tmp_path / "encoded"
    (raw_root / "metadata").mkdir(parents=True)
    (raw_root / "contexts").mkdir()
    (raw_root / "samples").mkdir()
    (encoded_root / "metadata").mkdir(parents=True)
    (encoded_root / "contexts").mkdir()
    (encoded_root / "samples").mkdir()
    empty_text_path = tmp_path / "empty_emb.pt"
    torch.save(torch.zeros(3, 4), empty_text_path)

    context_row = {
        "context_id": 0,
        "dataset_episode_index": 1,
        "task_id": 2,
        "task_text": "task",
        "init_state_index": 3,
        "t0_frame": 10,
        "context_start_frame": 8,
        "context_path": "contexts/context_000000.npz",
        "context_latent_path": "contexts/context_000000_latents.pt",
    }
    transition_row = {
        "sample_id": 0,
        "context_id": 0,
        "dataset_episode_index": 1,
        "task_id": 2,
        "task_text": "task",
        "init_state_index": 3,
        "t0_frame": 10,
        "context_start_frame": 8,
        "branch": "axis_pulse_x_neg",
        "branch_family": "axis_pulse",
        "branch_strength": "strong",
        "branch_is_ood": True,
        "sample_path": "samples/sample_000000.npz",
        "target_latent_path": "samples/sample_000000_latents.pt",
    }
    _write_jsonl(raw_root / "metadata" / "contexts.jsonl", [context_row])
    _write_jsonl(raw_root / "metadata" / "transitions.jsonl", [transition_row])
    _write_jsonl(encoded_root / "metadata" / "encoded_contexts.jsonl", [context_row])
    _write_jsonl(encoded_root / "metadata" / "encoded_transitions.jsonl", [transition_row])
    (encoded_root / "manifest.json").write_text(
        json.dumps({"dataset_root": str(raw_root)}),
        encoding="utf-8",
    )
    np.savez(
        raw_root / "contexts" / "context_000000.npz",
        action_context=np.ones((4, 7), dtype=np.float32),
    )
    np.savez(
        raw_root / "samples" / "sample_000000.npz",
        future_actions=np.full((4, 7), 2.0, dtype=np.float32),
    )
    torch.save(
        {"video_latents": torch.ones(2, 2, 2, 2)},
        encoded_root / "contexts" / "context_000000_latents.pt",
    )
    torch.save(
        {"target_video_latents": torch.full((2, 2, 2, 2), 2.0)},
        encoded_root / "samples" / "sample_000000_latents.pt",
    )
    return encoded_root, empty_text_path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
