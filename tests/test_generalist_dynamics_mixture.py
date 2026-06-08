from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import open_wam.data.generalist_dynamics as generalist_dynamics_module
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
    GENERALIST_TRAINING_BUCKET_METADATA_KEY,
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


class _BalancedDrawKeyDataset(_RecordingDrawKeyDataset):
    def __init__(self, *, length: int, balanced_indices: tuple[int, ...]) -> None:
        super().__init__(length=length)
        self._balanced_indices = tuple(int(index) for index in balanced_indices)

    def build_balanced_source_indices(self) -> tuple[int, ...]:
        return self._balanced_indices


def test_counterfactual_balanced_source_indices_spread_tasks_and_branches() -> None:
    rows = []
    for task_id in range(3):
        for repeat in range(2):
            for branch in ("gt", "stop_motion", "scale"):
                rows.append({"task_id": task_id, "branch": branch, "repeat": repeat})

    order = generalist_dynamics_module._balanced_counterfactual_source_indices(rows)

    assert sorted(order) == list(range(len(rows)))
    first_three = [rows[index] for index in order[:3]]
    assert {row["task_id"] for row in first_three} == {0, 1, 2}
    assert {row["branch"] for row in first_three} == {"gt", "stop_motion", "scale"}
    first_nine = [rows[index] for index in order[:9]]
    assert {row["branch"] for row in first_nine} == {"gt", "stop_motion", "scale"}
    assert {row["task_id"] for row in first_nine} == {0, 1, 2}
    assert all(
        sum(1 for row in first_nine if row["branch"] == branch) == 3
        for branch in ("gt", "stop_motion", "scale")
    )


def test_encoded_counterfactual_dataset_uses_target_only_t0_and_future(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert sample.video_latents.shape == (2, 2, 2, 2)
    assert torch.equal(sample.video_latents, torch.full((2, 2, 2, 2), 2.0))
    assert sample.actions.shape == (8, 7)
    assert sample.action_mask is not None
    assert sample.action_mask.sum().item() == 28
    assert torch.equal(sample.actions[:4], torch.zeros(4, 7))
    assert torch.equal(sample.actions[4:], torch.full((4, 7), 2.0))
    assert torch.equal(sample.action_mask[:4], torch.zeros(4, 7))
    assert torch.equal(sample.action_mask[4:], torch.ones(4, 7))
    assert sample.proprio_context_frames is not None
    assert sample.proprio_context_frames_mask is not None
    assert sample.proprio_context_frames.shape == (2, 8)
    assert sample.proprio_context_frames_mask.sum().item() == 0
    assert sample.metadata["proprio_context_source"] == "unavailable_zero_mask"
    assert sample.task_text is None
    assert sample.text_context is not None
    assert torch.equal(sample.text_context, torch.zeros(3, 4))
    assert sample.metadata["generalist_conditional_contract"] == "target_only_t0_observation_plus_future"
    assert sample.metadata["generalist_conditional_training_sequence"] == "target_only"
    assert sample.metadata["generalist_conditional_context_used_for_training"] is False
    assert sample.metadata["generalist_gjd_chunk_contract"] == "t0_singleton"
    assert sample.metadata["generalist_conditional_history_policy"] == "previous_boundary_video_only"
    assert sample.metadata["counterfactual_contract"] == "target_only_t0_observation_plus_future"
    assert sample.metadata["counterfactual_generation_contract"] == "t0_observation_plus_future"
    assert sample.metadata["counterfactual_context_used_for_training"] is False
    assert sample.metadata["counterfactual_contract"] == sample.metadata["generalist_conditional_contract"]
    assert (
        sample.metadata["counterfactual_conditional_history_policy"]
        == sample.metadata["generalist_conditional_history_policy"]
    )
    assert sample.metadata["history_frames"] == 1
    assert sample.metadata["loss_frame_start"] == 1
    assert sample.metadata["loss_frame_end"] == 2
    assert sample.metadata["action_loss_frame_start"] == 1
    assert sample.metadata["chunk_origin_frame"] == 1
    assert sample.metadata["target_observation_frame_in_sample"] == 0
    assert sample.metadata["target_observation_frame_index"] == 40
    assert sample.metadata["first_supervised_future_frame_in_sample"] == 1
    assert sample.metadata["first_supervised_future_frame_index"] == 44
    assert sample.metadata["supervised_future_latent_frames"] == 1
    assert sample.metadata["sampled_chunk_size"] == 2
    assert sample.metadata["counterfactual_gjd_chunk_contract"] == sample.metadata["generalist_gjd_chunk_contract"]
    assert sample.metadata["conditional_history_policy"] == "previous_boundary_video_only"
    assert sample.metadata["singleton_chunk_frame"] == sample.metadata["target_observation_frame_in_sample"]
    t0_chunk_id = _relative_chunk_id(
        sample.metadata["target_observation_frame_in_sample"],
        chunk_origin=sample.metadata["chunk_origin_frame"],
        chunk_size=sample.metadata["sampled_chunk_size"],
        singleton_chunk_frame=sample.metadata["singleton_chunk_frame"],
    )
    first_future_chunk_id = _relative_chunk_id(
        sample.metadata["first_supervised_future_frame_in_sample"],
        chunk_origin=sample.metadata["chunk_origin_frame"],
        chunk_size=sample.metadata["sampled_chunk_size"],
        singleton_chunk_frame=sample.metadata["singleton_chunk_frame"],
    )
    assert first_future_chunk_id == t0_chunk_id + 1
    assert sample.metadata["observation_frame_indices"] == [40, 44]
    assert sample.metadata["source_action_steps"] == 4
    assert sample.metadata["transition_action_steps_required"] == 4
    assert sample.metadata["extra_source_action_steps"] == 0
    assert sample.metadata["segment_pre_start_frames"] == 0
    assert sample.metadata["start_padding_mode"] == "none"
    assert sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_frames"] == 1
    assert sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_mask"] == 0.0


def test_encoded_counterfactual_dataset_uses_saved_observation_state(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path, include_state=True)
    data_config = _data_config(empty_text_path)

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert sample.proprio_context_frames is not None
    assert sample.proprio_context_frames_mask is not None
    assert sample.proprio_context_state is not None
    assert sample.proprio_context_state_mask is not None
    torch.testing.assert_close(sample.proprio_context_frames[:, 0], torch.tensor([20.0, 24.0]))
    torch.testing.assert_close(sample.proprio_context_frames_mask, torch.ones(2, 8))
    torch.testing.assert_close(sample.proprio_context_state, sample.proprio_context_frames)
    torch.testing.assert_close(sample.proprio_context_state_mask, sample.proprio_context_frames_mask)
    torch.testing.assert_close(sample.state[:, 0], torch.tensor([20.0]))
    torch.testing.assert_close(sample.state_mask, torch.ones(1, 8))
    assert sample.metadata["proprio_context_source"] == "observation.state"
    assert sample.metadata["state_source_key"] == "observation.state"
    assert sample.metadata["state_anchor_frame"] == 0
    assert sample.metadata["state_anchor_source_frame"] == 0
    assert sample.metadata["state_anchor_frame_in_sample"] == 0


def test_real_and_counterfactual_conditional_samples_share_target_only_contract(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    cf_sample = counterfactual[0]
    real_sample = LatentWAMSample(
        video_latents=torch.full((2, 2, 2, 2), 3.0),
        actions=torch.full((8, 7), 4.0),
        action_mask=torch.ones(8, 7),
        condition_latents=torch.full((2, 2, 2, 2), 5.0),
        task_text="real task",
        text_context=torch.ones(3, 4),
        negative_text_context=torch.zeros(3, 4),
        metadata={
            "dataset_kind": "real",
            "sample_start_frame": 40,
            "observation_frame_indices": [40, 44],
            "loss_frame_start": 1,
            "loss_frame_end": 2,
            "segment_length_frames": 2,
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
        ),
        split="train",
    )

    real_conditional = mixture[0]

    for key in (
        "history_frames",
        "loss_frame_start",
        "chunk_origin_frame",
        "target_observation_frame_in_sample",
        "first_supervised_future_frame_in_sample",
        "singleton_chunk_frame",
    ):
        assert real_conditional.metadata[key] == cf_sample.metadata[key]
    assert (
        real_conditional.metadata["generalist_conditional_contract"]
        == cf_sample.metadata["generalist_conditional_contract"]
    )
    assert (
        real_conditional.metadata["generalist_conditional_history_policy"]
        == cf_sample.metadata["generalist_conditional_history_policy"]
    )
    assert cf_sample.metadata["counterfactual_contract"] == cf_sample.metadata["generalist_conditional_contract"]
    assert real_conditional.metadata["loss_frame_end"] == cf_sample.metadata["loss_frame_end"]
    assert real_conditional.action_mask is not None
    assert cf_sample.action_mask is not None
    torch.testing.assert_close(real_conditional.action_mask[:4], cf_sample.action_mask[:4])
    torch.testing.assert_close(real_conditional.action_mask[4:], cf_sample.action_mask[4:])

    joint_mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=_OneSampleLatentDataset(real_sample),
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
    real_joint = joint_mixture[0]

    assert real_joint.video_latents.shape == real_sample.video_latents.shape
    assert real_joint.actions.shape == real_sample.actions.shape
    assert real_joint.action_mask is not None
    torch.testing.assert_close(real_joint.action_mask, real_sample.action_mask)
    assert "generalist_conditional_contract" not in real_joint.metadata
    assert real_joint.metadata[GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY] == "joint"


def test_generalist_dynamics_mixture_keeps_multichunk_conditional_sources(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path, target_latent_frames=6)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    raw_cf_sample = counterfactual[0]
    real_sample = LatentWAMSample(
        video_latents=torch.arange(2 * 7 * 2 * 2, dtype=torch.float32).reshape(2, 7, 2, 2),
        actions=torch.arange(14 * 7, dtype=torch.float32).reshape(14, 7),
        action_mask=torch.ones(14, 7),
        task_text="real task",
        text_context=torch.ones(3, 4),
        negative_text_context=torch.zeros(3, 4),
        metadata={
            "dataset_kind": "real",
            "sample_start_frame": 0,
            "observation_frame_indices": list(range(7)),
            "history_frames": 3,
            "latent_loss_frame_start": 0,
            "sampled_chunk_size": 2,
            "sampled_window_size": 8,
            "segment_length_frames": 7,
        },
    )
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=_OneSampleLatentDataset(real_sample),
        counterfactual_dataset=counterfactual,
        mixture_config=GeneralistDynamicsMixtureConfig(
            real_joint_weight=0.0,
            real_action_conditioned_video_weight=1.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_action_conditioned_video_weight=0.0,
            counterfactual_video_conditioned_action_weight=0.0,
        ),
        split="train",
    )

    real_conditional = mixture[0]

    assert raw_cf_sample.video_latents.shape == (2, 6, 2, 2)
    assert real_conditional.video_latents.shape == (2, 5, 2, 2)
    torch.testing.assert_close(real_conditional.video_latents, real_sample.video_latents[:, 2:7])
    assert real_conditional.actions.shape == (10, 7)
    assert real_conditional.metadata["generalist_conditional_source_t0_frame_in_sample"] == 2
    assert real_conditional.metadata["generalist_gjd_chunk_contract"] == "t0_singleton"
    assert real_conditional.metadata["conditional_history_policy"] == "previous_boundary_video_only"
    assert real_conditional.metadata["loss_frame_start"] == 1
    assert real_conditional.metadata["loss_frame_end"] == 5
    assert real_conditional.metadata["supervised_future_latent_frames"] == 4
    assert real_conditional.action_mask is not None
    torch.testing.assert_close(real_conditional.action_mask[:2], torch.zeros(2, 7))
    torch.testing.assert_close(real_conditional.action_mask[2:], torch.ones(8, 7))

    cf_view = mixture.build_source_view(
        source="counterfactual_dynamics",
        mode="action_conditioned_video",
        bucket_name="cf_fdm",
        drop_text=True,
    )
    cf_conditional = cf_view[0]

    assert cf_conditional.video_latents.shape == (2, 6, 2, 2)
    assert cf_conditional.actions.shape == (12, 7)
    assert cf_conditional.metadata["generalist_conditional_contract"] == "target_only_t0_observation_plus_future"
    assert cf_conditional.metadata["generalist_gjd_chunk_contract"] == "t0_singleton"
    assert (
        cf_conditional.metadata["counterfactual_gjd_chunk_contract"]
        == cf_conditional.metadata["generalist_gjd_chunk_contract"]
    )
    assert cf_conditional.metadata["generalist_conditional_history_policy"] == "previous_boundary_video_only"
    assert cf_conditional.metadata["conditional_history_policy"] == "previous_boundary_video_only"
    assert cf_conditional.metadata["loss_frame_start"] == 1
    assert cf_conditional.metadata["loss_frame_end"] == 6
    assert cf_conditional.metadata["supervised_future_latent_frames"] == 5
    assert cf_conditional.action_mask is not None
    torch.testing.assert_close(cf_conditional.action_mask[:2], torch.zeros(2, 7))
    torch.testing.assert_close(cf_conditional.action_mask[2:], torch.ones(10, 7))

    joint_mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=_OneSampleLatentDataset(real_sample),
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
    real_joint = joint_mixture[0]

    assert real_joint.video_latents.shape == real_sample.video_latents.shape
    assert real_joint.actions.shape == real_sample.actions.shape
    assert "conditional_history_policy" not in real_joint.metadata


def test_encoded_counterfactual_dataset_accepts_source_dataset_root_manifest(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    manifest_path = encoded_root / "manifest.json"
    raw_root = json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_root"]
    manifest_path.write_text(json.dumps({"source_dataset_root": raw_root}), encoding="utf-8")
    data_config = _data_config(empty_text_path)

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert sample.metadata["dataset_kind"] == "encoded_counterfactual_dynamics"
    assert sample.metadata["counterfactual_sample_id"] == 0


def test_encoded_counterfactual_dataset_rejects_missing_condition_latents_for_offset(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = replace(
        _data_config(empty_text_path),
        sample_construction=SampleConstructionConfig(
            chunk_size=2,
            window_size=4,
            condition_source_frame_offset=-1,
        ),
    )

    with pytest.raises(ValueError, match="missing explicit condition latents"):
        EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")


def test_encoded_counterfactual_dataset_prefers_encoded_single_frame_condition_latents(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(
        tmp_path,
        include_condition_latents=True,
    )
    data_config = replace(
        _data_config(empty_text_path),
        sample_construction=SampleConstructionConfig(
            chunk_size=2,
            window_size=4,
            condition_source_frame_offset=-1,
        ),
    )

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert sample.condition_latents is not None
    torch.testing.assert_close(sample.condition_latents, torch.full((2, 2, 2, 2), 9.0))
    assert sample.metadata["has_condition_latents"] is True
    assert sample.metadata["condition_source_frame_offset"] == -1
    assert sample.metadata["condition_latents_source"] == "encoded_target_single_frame"


def test_encoded_counterfactual_dataset_randomizes_uniform_segment_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = replace(
        _data_config(empty_text_path),
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.UNIFORM_SEGMENT,
            chunk_size=4,
            window_size=8,
            randomize_geometry=True,
        ),
    )
    draws = iter((2, 7))

    def fake_randint(low: int, high: int) -> int:
        value = next(draws)
        assert low <= value <= high
        return value

    monkeypatch.setattr(generalist_dynamics_module.random, "randint", fake_randint)

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert sample.metadata["sampled_chunk_size"] == 2
    assert sample.metadata["generalist_gjd_chunk_contract"] == "t0_singleton"
    assert sample.metadata["counterfactual_gjd_chunk_contract"] == sample.metadata["generalist_gjd_chunk_contract"]
    assert sample.metadata["singleton_chunk_frame"] == sample.metadata["target_observation_frame_in_sample"]
    assert sample.metadata["sampled_window_size"] == 7


def test_encoded_counterfactual_dataset_keeps_fixed_geometry_when_disabled(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = replace(
        _data_config(empty_text_path),
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.UNIFORM_SEGMENT,
            chunk_size=4,
            window_size=8,
            randomize_geometry=False,
        ),
    )

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert sample.metadata["sampled_chunk_size"] == 2
    assert sample.metadata["generalist_gjd_chunk_contract"] == "t0_singleton"
    assert sample.metadata["counterfactual_gjd_chunk_contract"] == sample.metadata["generalist_gjd_chunk_contract"]
    assert sample.metadata["singleton_chunk_frame"] == sample.metadata["target_observation_frame_in_sample"]
    assert sample.metadata["sampled_window_size"] == 8


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

    assert len(dataset) == 1
    assert sample.video_latents.shape == (2, 6, 2, 2)
    assert sample.actions.shape == (24, 7)
    assert sample.metadata["window_sampling_mode"] == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT
    assert sample.metadata["tail_padding_policy"] == str(TailPaddingPolicy.ZERO_ORDER_HOLD)
    assert sample.metadata["padded_target_policy"] == str(PaddedTargetPolicy.MASK_LOSS)
    assert sample.metadata["loss_frame_start"] == sample.metadata["history_frames"]
    assert sample.metadata["loss_frame_end"] <= sample.metadata["segment_valid_latent_frames"]
    assert sample.metadata["segment_padded_latent_frames"] == 4
    assert sample.metadata["hierarchical_epoch_sample_count"] == 1
    assert sample.metadata["sampled_chunk_size"] == 2
    assert sample.metadata["generalist_gjd_chunk_contract"] == "t0_singleton"
    assert sample.metadata["counterfactual_gjd_chunk_contract"] == sample.metadata["generalist_gjd_chunk_contract"]
    assert sample.metadata["singleton_chunk_frame"] == sample.metadata["target_observation_frame_in_sample"]


def test_encoded_counterfactual_dataset_short_context_uses_actual_history_boundary(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(
        tmp_path,
        context_latent_frames=1,
        target_latent_frames=4,
    )
    data_config = replace(
        _data_config(empty_text_path),
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
            segment_frames=6,
            chunk_size=2,
            window_size=4,
            start_padding_frames=0,
            tail_padding_policy=TailPaddingPolicy.ZERO_ORDER_HOLD,
            padded_target_policy=PaddedTargetPolicy.MASK_LOSS,
        ),
    )

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert sample.video_latents.shape == (2, 6, 2, 2)
    assert torch.equal(sample.video_latents[:, :4], torch.full((2, 4, 2, 2), 2.0))
    assert sample.metadata["history_frames"] == 1
    assert sample.metadata["loss_frame_start"] == 1
    assert sample.metadata["action_loss_frame_start"] == 1
    assert sample.metadata["chunk_origin_frame"] == 1
    assert sample.metadata["target_observation_frame_in_sample"] == 0
    assert sample.metadata["first_supervised_future_frame_in_sample"] == 1
    assert sample.metadata["segment_valid_latent_frames"] == 4
    assert sample.metadata["segment_padded_latent_frames"] == 2
    assert sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_frames"] == 1


def test_encoded_counterfactual_dataset_prefix_state_uses_condition_source_frame_for_shifted_window(
    tmp_path: Path,
) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(
        tmp_path,
        include_state=True,
        include_condition_latents=True,
        context_latent_frames=4,
        target_latent_frames=3,
    )
    data_config = replace(
        _data_config(empty_text_path),
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
            segment_frames=4,
            chunk_size=2,
            window_size=4,
            condition_source_frame_offset=-1,
            start_padding_frames=0,
            tail_padding_policy=TailPaddingPolicy.ZERO_ORDER_HOLD,
            padded_target_policy=PaddedTargetPolicy.MASK_LOSS,
        ),
    )

    dataset = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    sample = dataset[0]

    assert sample.metadata["latent_frame_start"] == 0
    assert sample.metadata["state_anchor_source_frame"] == 0
    assert sample.metadata["state_anchor_frame_in_sample"] == 0
    assert sample.proprio_context_frames is not None
    torch.testing.assert_close(sample.proprio_context_frames[:, 0], torch.tensor([20.0, 24.0, 24.0, 24.0]))
    torch.testing.assert_close(sample.state[:, 0], torch.tensor([20.0]))


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


def test_generalist_dynamics_mixture_projects_real_conditional_to_target_only(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedCounterfactualDynamicsLatentDataset(data_config, encoded_root, split="train")
    real_sample = LatentWAMSample(
        video_latents=torch.arange(2 * 8 * 2 * 2, dtype=torch.float32).reshape(2, 8, 2, 2),
        actions=torch.arange(16 * 7, dtype=torch.float32).reshape(16, 7),
        action_mask=torch.ones(16, 7),
        condition_latents=torch.arange(1000, 1000 + 2 * 8 * 2 * 2, dtype=torch.float32).reshape(2, 8, 2, 2),
        proprio_context_state=torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5),
        proprio_context_state_mask=torch.ones(8, 5),
        proprio_context_frames=torch.arange(2000, 2000 + 8 * 3, dtype=torch.float32).reshape(8, 3),
        proprio_context_frames_mask=torch.ones(8, 3),
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

    assert sample.video_latents.shape == (2, 3, 2, 2)
    assert sample.actions.shape == (6, 7)
    assert sample.action_mask is not None
    assert sample.condition_latents is not None
    assert sample.proprio_context_state is not None
    assert sample.proprio_context_state_mask is not None
    assert sample.proprio_context_frames is not None
    assert sample.proprio_context_frames_mask is not None
    assert sample.condition_latents.shape == (2, 3, 2, 2)
    assert sample.proprio_context_state.shape == (3, 5)
    assert sample.proprio_context_frames.shape == (3, 3)
    torch.testing.assert_close(sample.condition_latents, real_sample.condition_latents[:, 5:])
    torch.testing.assert_close(sample.proprio_context_state, real_sample.proprio_context_state[5:])
    torch.testing.assert_close(sample.proprio_context_state_mask, real_sample.proprio_context_state_mask[5:])
    torch.testing.assert_close(sample.proprio_context_frames, real_sample.proprio_context_frames[5:])
    torch.testing.assert_close(sample.proprio_context_frames_mask, real_sample.proprio_context_frames_mask[5:])
    torch.testing.assert_close(sample.action_mask[:2], torch.zeros(2, 7))
    torch.testing.assert_close(sample.action_mask[2:], torch.ones(4, 7))
    assert sample.metadata["generalist_conditional_contract"] == "target_only_t0_observation_plus_future"
    assert sample.metadata["generalist_conditional_context_used_for_training"] is False
    assert sample.metadata["history_frames"] == 1
    assert sample.metadata["loss_frame_start"] == 1
    assert sample.metadata["loss_frame_end"] == 3
    assert sample.metadata["latent_loss_frame_start"] == 1
    assert sample.metadata["action_loss_frame_start"] == 1
    assert sample.metadata["chunk_origin_frame"] == 1
    assert sample.metadata["singleton_chunk_frame"] == 0
    assert sample.metadata["conditional_history_policy"] == "previous_boundary_video_only"
    assert sample.metadata["target_observation_frame_in_sample"] == 0
    assert sample.metadata["first_supervised_future_frame_in_sample"] == 1
    assert sample.metadata["sample_start_frame"] == 105
    assert sample.metadata["observation_frame_indices"] == [105, 106, 107]
    assert sample.metadata["frame_shift"] == 105
    assert sample.metadata["effective_frame_start"] == 105
    assert sample.metadata["effective_frame_end"] == 108
    assert sample.metadata["logical_frame_start"] == 105
    assert sample.metadata["logical_frame_end"] == 108
    assert sample.metadata["target_frame_start"] == 106
    assert sample.metadata["target_frame_end"] == 109
    assert sample.metadata["subwindow_latent_start"] == 105
    assert sample.metadata["subwindow_latent_end"] == 109
    assert sample.metadata["virtual_latent_start"] == 105
    assert sample.metadata["supervised_start"] == 1
    assert sample.metadata["supervised_end"] == 3
    assert sample.metadata["context_prefix_frames_in_sample"] == 1
    assert sample.metadata["context_prefix_real_frames"] == 1
    assert sample.metadata["context_prefix_truncated_frames"] == 5
    assert sample.metadata["subwindow_action_start"] == 110
    assert sample.metadata["valid_action_steps"] == 4
    assert sample.metadata["valid_action_values"] == 28
    assert sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_mask"] == 0.0
    assert sample.task_text is None

    view = mixture.build_source_view(
        source="real_demo",
        mode="action_conditioned_video",
        bucket_name="val_fdm",
        drop_text=True,
    )
    view_sample = view[0]

    assert view_sample.video_latents.shape == (2, 3, 2, 2)
    assert view_sample.metadata["history_frames"] == 1
    assert view_sample.metadata[GENERALIST_TRAINING_SOURCE_METADATA_KEY] == "real_demo"
    assert view_sample.metadata[GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY] == "action_conditioned_video"
    assert view_sample.metadata["generalist_training_bucket"] == "val_fdm"


def test_generalist_dynamics_mixture_uses_history_boundary_when_loss_start_is_zero(tmp_path: Path) -> None:
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
            "sample_start_frame": 0,
            "observation_frame_indices": list(range(8)),
            "history_frames": 6,
            "loss_frame_start": 0,
            "latent_loss_frame_start": 0,
            "action_loss_frame_start": 0,
            "segment_length_frames": 8,
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
        ),
        split="train",
    )

    sample = mixture[0]

    assert sample.video_latents.shape == (2, 3, 2, 2)
    torch.testing.assert_close(sample.video_latents, real_sample.video_latents[:, 5:])
    assert sample.metadata["generalist_conditional_boundary_source"] == "history_frames"
    assert sample.metadata["generalist_conditional_source_t0_frame_in_sample"] == 5
    assert sample.metadata["sample_start_frame"] == 5
    assert sample.metadata["loss_frame_start"] == 1
    assert sample.metadata["loss_frame_end"] == 3
    assert sample.metadata["subwindow_action_start"] == 10
    assert sample.action_mask is not None
    torch.testing.assert_close(sample.action_mask[:2], torch.zeros(2, 7))
    torch.testing.assert_close(sample.action_mask[2:], torch.ones(4, 7))


def test_generalist_source_view_can_spread_indices_for_short_validation() -> None:
    real_dataset = _RecordingDrawKeyDataset(length=100)
    counterfactual_dataset = _RecordingDrawKeyDataset(length=100)
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual_dataset,
        mixture_config=GeneralistDynamicsMixtureConfig(
            real_joint_weight=0.6,
            real_action_conditioned_video_weight=0.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_action_conditioned_video_weight=0.2,
            counterfactual_video_conditioned_action_weight=0.2,
        ),
        split="val",
    )

    direct_view = mixture.build_source_view(
        source="counterfactual_dynamics",
        mode="action_conditioned_video",
        bucket_name="val_fdm",
        drop_text=False,
    )
    spread_view = mixture.build_source_view(
        source="counterfactual_dynamics",
        mode="action_conditioned_video",
        bucket_name="val_fdm",
        drop_text=False,
        spread_indices=True,
    )

    direct_indices = [direct_view[index].metadata["generalist_source_index"] for index in range(8)]
    spread_indices = [spread_view[index].metadata["generalist_source_index"] for index in range(8)]

    assert direct_indices == list(range(8))
    assert spread_indices != list(range(8))
    assert len(set(spread_indices)) == len(spread_indices)
    assert spread_view[1].metadata["generalist_source_view_index"] == 1
    assert spread_view[1].metadata["generalist_source_view_stride"] > 1


def test_generalist_source_view_prefers_dataset_balanced_indices_for_validation() -> None:
    real_dataset = _RecordingDrawKeyDataset(length=100)
    counterfactual_dataset = _BalancedDrawKeyDataset(
        length=100,
        balanced_indices=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90),
    )
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual_dataset,
        mixture_config=GeneralistDynamicsMixtureConfig(
            real_joint_weight=0.6,
            real_action_conditioned_video_weight=0.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_action_conditioned_video_weight=0.2,
            counterfactual_video_conditioned_action_weight=0.2,
        ),
        split="val",
    )

    view = mixture.build_source_view(
        source="counterfactual_dynamics",
        mode="action_conditioned_video",
        bucket_name="val_fdm",
        drop_text=False,
        spread_indices=True,
    )

    source_indices = [view[index].metadata["generalist_source_index"] for index in range(10)]

    assert source_indices == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert view[1].metadata["generalist_source_view_order"] == "balanced"
    assert view[1].metadata["generalist_source_view_stride"] == 1


def test_generalist_dynamics_mixture_projects_real_conditional_from_partial_prefix(tmp_path: Path) -> None:
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

    assert sample.video_latents.shape == (2, 6, 2, 2)
    assert sample.actions.shape == (12, 7)
    assert sample.action_mask is not None
    torch.testing.assert_close(sample.action_mask[:2], torch.zeros(2, 7))
    torch.testing.assert_close(sample.action_mask[2:], torch.ones(10, 7))
    assert sample.metadata["generalist_conditional_contract"] == "target_only_t0_observation_plus_future"
    assert sample.metadata["history_frames"] == 1
    assert sample.metadata["loss_frame_start"] == 1
    assert sample.metadata["loss_frame_end"] == 6
    assert sample.metadata["chunk_origin_frame"] == 1
    assert sample.metadata["singleton_chunk_frame"] == 0
    assert sample.metadata["conditional_history_policy"] == "previous_boundary_video_only"
    assert sample.metadata["sample_start_frame"] == 7
    assert sample.metadata["observation_frame_indices"] == list(range(7, 13))
    assert sample.metadata["frame_shift"] == 7
    assert sample.metadata["effective_frame_start"] == 7
    assert sample.metadata["effective_frame_end"] == 13
    assert sample.metadata["target_frame_start"] == 8
    assert sample.metadata["target_frame_end"] == 14
    assert sample.metadata["subwindow_latent_start"] == 7
    assert sample.metadata["subwindow_latent_end"] == 14
    assert sample.metadata["virtual_latent_start"] == 7
    assert sample.metadata["supervised_start"] == 1
    assert sample.metadata["supervised_end"] == 6
    assert sample.metadata["context_prefix_frames_in_sample"] == 1
    assert sample.metadata["context_prefix_real_frames"] == 1
    assert sample.metadata["context_prefix_truncated_frames"] == 7
    assert sample.metadata["context_prefix_truncated_frames"] <= sample.metadata["context_prefix_frames_requested"]
    assert sample.metadata["valid_action_steps"] == 10
    assert sample.metadata["valid_action_values"] == 70


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


def test_generalist_dynamics_train_sampler_coordinates_bucket_across_ranks() -> None:
    real_dataset = _RecordingDrawKeyDataset(length=10_000)
    counterfactual_dataset = _RecordingDrawKeyDataset(length=10_000)
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual_dataset,
        mixture_config=GeneralistDynamicsMixtureConfig(
            real_joint_weight=0.6,
            real_action_conditioned_video_weight=0.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_action_conditioned_video_weight=0.2,
            counterfactual_video_conditioned_action_weight=0.2,
        ),
        split="train",
    )

    indices = [
        next(iter(mixture.build_train_sampler(world_size=4, rank=rank)))
        for rank in range(4)
    ]
    samples = [mixture[index] for index in indices]

    assert indices == [0, 1, 2, 3]
    assert len({sample.metadata[GENERALIST_TRAINING_BUCKET_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata[GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata[GENERALIST_TRAINING_SOURCE_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata["seen_draw_key"] for sample in samples}) > 1


def test_generalist_dynamics_train_sampler_coordinates_padded_epoch_tail() -> None:
    real_dataset = _RecordingDrawKeyDataset(length=5)
    counterfactual_dataset = _RecordingDrawKeyDataset(length=5)
    mixture = GeneralistDynamicsMixtureDataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual_dataset,
        mixture_config=GeneralistDynamicsMixtureConfig(
            real_joint_weight=0.6,
            real_action_conditioned_video_weight=0.0,
            real_video_conditioned_action_weight=0.0,
            counterfactual_action_conditioned_video_weight=0.2,
            counterfactual_video_conditioned_action_weight=0.2,
        ),
        split="train",
    )

    samplers = [mixture.build_train_sampler(world_size=4, rank=rank) for rank in range(4)]
    for sampler in samplers:
        sampler.set_epoch(1)
    indices = [next(iter(sampler)) for sampler in samplers]
    samples = [mixture[index] for index in indices]

    assert indices == [8, 9, 10, 11]
    assert len({sample.metadata[GENERALIST_TRAINING_BUCKET_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata[GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata[GENERALIST_TRAINING_SOURCE_METADATA_KEY] for sample in samples}) == 1
    assert all(sample.metadata["generalist_source_index"] >= 5 for sample in samples)
    assert len({sample.metadata["seen_draw_key"] for sample in samples}) > 1


def test_generalist_dynamics_mixture_target_only_start_padding_uses_real_action_offset(tmp_path: Path) -> None:
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

    assert sample.video_latents.shape == (2, 3, 2, 2)
    assert sample.actions.shape == (6, 7)
    assert sample.action_mask is not None
    torch.testing.assert_close(sample.action_mask[:2], torch.zeros(2, 7))
    torch.testing.assert_close(sample.action_mask[2:], torch.ones(4, 7))
    assert sample.metadata["history_frames"] == 1
    assert sample.metadata["loss_frame_start"] == 1
    assert sample.metadata["loss_frame_end"] == 3
    assert sample.metadata["chunk_origin_frame"] == 1
    assert sample.metadata["singleton_chunk_frame"] == 0
    assert sample.metadata["conditional_history_policy"] == "previous_boundary_video_only"
    assert sample.metadata["segment_pre_start_frames"] == 0
    assert sample.metadata["subwindow_action_start"] == 6
    assert sample.metadata["subwindow_action_end"] == 12
    assert sample.metadata["generalist_conditional_source_t0_frame_in_sample"] == 3


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


def _write_encoded_counterfactual_fixture(
    tmp_path: Path,
    *,
    include_state: bool = False,
    include_condition_latents: bool = False,
    context_latent_frames: int = 2,
    target_latent_frames: int = 2,
) -> tuple[Path, Path]:
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
        json.dumps(
            {
                "dataset_root": str(raw_root),
                "condition_latents": bool(include_condition_latents),
                "condition_source_frame_offset": -1 if include_condition_latents else 0,
                "condition_source_frame_policy": "next_latent_source_offset" if include_condition_latents else None,
            }
        ),
        encoding="utf-8",
    )
    action_per_frame = 2
    context_payload = {
        "action_context": np.ones((context_latent_frames * action_per_frame, 7), dtype=np.float32)
    }
    sample_payload = {
        "future_actions": np.full((target_latent_frames * action_per_frame, 7), 2.0, dtype=np.float32)
    }
    if include_state:
        context_payload["observation.state"] = (
            np.arange(10, 15, dtype=np.float32).reshape(5, 1).repeat(8, axis=1)
        )
        sample_payload["observation.state"] = (
            np.arange(20, 25, dtype=np.float32).reshape(5, 1).repeat(8, axis=1)
        )
    np.savez(raw_root / "contexts" / "context_000000.npz", **context_payload)
    np.savez(raw_root / "samples" / "sample_000000.npz", **sample_payload)
    context_latent_payload = {"video_latents": torch.ones(2, context_latent_frames, 2, 2)}
    sample_latent_payload = {"target_video_latents": torch.full((2, target_latent_frames, 2, 2), 2.0)}
    if include_condition_latents:
        context_latent_payload.update(
            {
                "condition_video_latents": torch.full((2, context_latent_frames, 2, 2), 7.0),
                "condition_source_frame_offset": -1,
                "condition_source_frame_policy": "next_latent_source_offset",
            }
        )
        sample_latent_payload.update(
            {
                "target_condition_video_latents": torch.full((2, target_latent_frames, 2, 2), 9.0),
                "condition_source_frame_offset": -1,
                "condition_source_frame_policy": "next_latent_source_offset",
            }
        )
    torch.save(context_latent_payload, encoded_root / "contexts" / "context_000000_latents.pt")
    torch.save(sample_latent_payload, encoded_root / "samples" / "sample_000000_latents.pt")
    return encoded_root, empty_text_path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _relative_chunk_id(
    frame: int,
    *,
    chunk_origin: int,
    chunk_size: int,
    singleton_chunk_frame: int | None = None,
) -> int:
    chunk_id = (int(frame) - int(chunk_origin)) // int(chunk_size)
    if singleton_chunk_frame is None:
        return chunk_id
    singleton_chunk_id = (int(singleton_chunk_frame) - int(chunk_origin)) // int(chunk_size)
    if int(frame) < int(singleton_chunk_frame) and chunk_id == singleton_chunk_id:
        return chunk_id - 1
    return chunk_id
