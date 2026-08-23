from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import open_wam.data.dynamics_routing as dynamics_routing_module
from open_wam.configs import (
    ActionSchemaConfig,
    DynamicsObjective,
    DynamicsRouteConfig,
    DynamicsRoutingConfig,
    DynamicsSource,
    GenericDataConfig,
    SampleConstructionConfig,
    SampleOrderMode,
    WindowSamplingMode,
)
from open_wam.contracts import (
    DYNAMICS_ROUTING_BUCKET_METADATA_KEY,
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY,
    DYNAMICS_ROUTING_MODE_METADATA_KEY,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
)
from open_wam.data import (
    ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1,
    DatasetArtifactPreflightError,
    DynamicsRouteKey,
    DynamicsRoutingDataset,
    EncodedDynamicsLatentDataset,
    build_dynamics_routing_datasets,
    load_encoded_dynamics_artifact,
    migrate_encoded_dynamics_artifact,
    preflight_encoded_dynamics_artifact,
    resolve_dynamics_dataset_plan,
)
from open_wam.data.encoded_dynamics_ordering import (
    build_task_branch_balanced_indices,
)
from open_wam.data.latent_contracts import LatentWAMSample


def _routing_config(
    *,
    real_joint: float = 0.6,
    real_fdm: float = 0.1,
    real_idm: float = 0.1,
    counterfactual_fdm: float = 0.1,
    counterfactual_idm: float = 0.1,
    **kwargs,
) -> DynamicsRoutingConfig:
    route_specs = (
        ("real_demo", "joint", real_joint),
        ("real_demo", "action_conditioned_video", real_fdm),
        ("real_demo", "video_conditioned_action", real_idm),
        ("counterfactual_dynamics", "action_conditioned_video", counterfactual_fdm),
        ("counterfactual_dynamics", "video_conditioned_action", counterfactual_idm),
    )
    return DynamicsRoutingConfig(
        routes=tuple(
            DynamicsRouteConfig(source=source, mode=mode, weight=weight)
            for source, mode, weight in route_specs
            if float(weight) > 0.0
        ),
        **kwargs,
    )


class _OneSampleLatentDataset(Dataset[LatentWAMSample]):
    def __init__(self, sample: LatentWAMSample) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> LatentWAMSample:
        del index
        return self.sample


class _RecordingDrawKeyDataset(Dataset[LatentWAMSample]):
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


def _routing_dataset(
    *,
    real_dataset: Dataset[LatentWAMSample],
    counterfactual_dataset: Dataset[LatentWAMSample] | None,
    routing_config: DynamicsRoutingConfig,
    split: str,
    fixed_mode: DynamicsObjective | str | None = None,
) -> DynamicsRoutingDataset:
    route_datasets: dict[DynamicsRouteKey, Dataset[LatentWAMSample]] = {}
    for route in routing_config.active_routes:
        dataset = (
            real_dataset
            if route.source == DynamicsSource.REAL_DEMO
            else counterfactual_dataset
        )
        if dataset is not None:
            route_datasets[
                DynamicsRouteKey(source=route.source, mode=route.mode)
            ] = dataset
    return DynamicsRoutingDataset(
        route_datasets=route_datasets,
        routing_config=routing_config,
        split=split,
        fixed_mode=fixed_mode,
    )


def test_counterfactual_balanced_source_indices_spread_tasks_and_branches() -> None:
    rows = []
    for task_id in range(3):
        for repeat in range(2):
            for branch in ("gt", "stop_motion", "scale"):
                rows.append({"task_id": task_id, "branch": branch, "repeat": repeat})

    order = build_task_branch_balanced_indices(rows)

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


@pytest.mark.parametrize(
    ("routing_config", "requires_planning", "encoded_sources"),
    [
        (
            _routing_config(
                real_joint=1.0,
                real_fdm=0.0,
                real_idm=0.0,
                counterfactual_fdm=0.0,
                counterfactual_idm=0.0,
            ),
            True,
            (),
        ),
        (
            _routing_config(
                real_joint=0.0,
                real_fdm=1.0,
                real_idm=0.0,
                counterfactual_fdm=0.0,
                counterfactual_idm=0.0,
            ),
            False,
            (DynamicsSource.REAL_DEMO,),
        ),
        (
            _routing_config(
                real_joint=0.0,
                real_fdm=0.0,
                real_idm=0.0,
                counterfactual_fdm=0.0,
                counterfactual_idm=1.0,
            ),
            False,
            (DynamicsSource.COUNTERFACTUAL_DYNAMICS,),
        ),
        (
            _routing_config(),
            True,
            (
                DynamicsSource.REAL_DEMO,
                DynamicsSource.COUNTERFACTUAL_DYNAMICS,
            ),
        ),
    ],
)
def test_dynamics_dataset_plan_requires_only_active_route_inputs(
    routing_config: DynamicsRoutingConfig,
    requires_planning: bool,
    encoded_sources: tuple[DynamicsSource, ...],
) -> None:
    config = replace(
        _data_config(Path("empty_embedding.pt")),
        dynamics_routing=routing_config,
    )

    plan = resolve_dynamics_dataset_plan(config)

    assert plan.requires_planning is requires_planning
    assert plan.encoded_sources == encoded_sources


def test_encoded_dynamics_artifact_exposes_canonical_source_views(
    tmp_path: Path,
) -> None:
    encoded_root, _ = _write_encoded_counterfactual_fixture(tmp_path)

    artifact = load_encoded_dynamics_artifact(encoded_root)

    assert artifact.manifest["artifact_schema"] == (
        ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1
    )
    assert [row["branch"] for row in artifact.rows_for_source("real_demo")] == [
        "gt"
    ]
    assert [
        row["branch"]
        for row in artifact.rows_for_source("counterfactual_dynamics")
    ] == ["axis_pulse_x_neg"]


def test_encoded_dynamics_artifact_rejects_unknown_schema(tmp_path: Path) -> None:
    encoded_root, _ = _write_encoded_counterfactual_fixture(tmp_path)
    manifest_path = encoded_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_schema"] = "open_wam.encoded_dynamics.v999"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported encoded dynamics artifact schema"):
        load_encoded_dynamics_artifact(encoded_root)


def test_encoded_dynamics_artifact_requires_explicit_schema(tmp_path: Path) -> None:
    encoded_root, _ = _write_encoded_counterfactual_fixture(tmp_path)
    manifest_path = encoded_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifact_schema"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is unversioned"):
        load_encoded_dynamics_artifact(encoded_root)


def test_encoded_dynamics_artifact_migration_only_canonicalizes_manifest(
    tmp_path: Path,
) -> None:
    encoded_root, _ = _write_encoded_counterfactual_fixture(tmp_path)
    manifest_path = encoded_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifact_schema"]
    del manifest["reference_branch"]
    del manifest["raw_payload_root"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    latent_path = encoded_root / "samples" / "sample_000000_latents.pt"
    raw_path = tmp_path / "raw" / "samples" / "sample_000000.npz"
    latent_payload = latent_path.read_bytes()
    raw_payload = raw_path.read_bytes()

    artifact = migrate_encoded_dynamics_artifact(encoded_root)

    migrated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {
        key: value
        for key, value in migrated_manifest.items()
        if key not in {"artifact_schema", "reference_branch", "raw_payload_root"}
    } == manifest
    assert migrated_manifest["artifact_schema"] == (
        ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1
    )
    assert migrated_manifest["reference_branch"] == "gt"
    assert migrated_manifest["raw_payload_root"] == "../raw"
    assert artifact.reference_branch == "gt"
    assert latent_path.read_bytes() == latent_payload
    assert raw_path.read_bytes() == raw_payload


def test_encoded_dynamics_preflight_checks_every_indexed_payload(
    tmp_path: Path,
) -> None:
    encoded_root, _ = _write_encoded_counterfactual_fixture(tmp_path)
    index_path = encoded_root / "metadata" / "encoded_transitions.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows.append(
        {
            **rows[0],
            "sample_id": 99,
            "target_latent_path": "samples/missing_latents.pt",
        }
    )
    _write_jsonl(index_path, rows)

    with pytest.raises(
        DatasetArtifactPreflightError,
        match=r"sample_id=99.*missing_latents\.pt",
    ):
        preflight_encoded_dynamics_artifact(
            encoded_root,
            sources=(DynamicsSource.COUNTERFACTUAL_DYNAMICS,),
            config_path="data.dynamics_routing.train_latent_root",
        )


def test_encoded_counterfactual_dataset_uses_target_only_t0_and_future(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)

    dataset = EncodedDynamicsLatentDataset.from_root(data_config, encoded_root, split="train", source=DynamicsSource.COUNTERFACTUAL_DYNAMICS)
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
    assert sample.metadata["context_prefix_frames_in_sample"] == 1
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


@pytest.mark.parametrize("future_action_steps", [124, 128])
def test_encoded_dynamics_accepts_canonical_action_alignments(
    tmp_path: Path,
    future_action_steps: int,
) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(
        tmp_path,
        target_latent_frames=32,
        future_action_steps=future_action_steps,
    )
    dataset = EncodedDynamicsLatentDataset.from_root(
        _data_config(empty_text_path),
        encoded_root,
        split="train",
        source=DynamicsSource.COUNTERFACTUAL_DYNAMICS,
    )

    sample = dataset[0]

    assert sample.metadata["source_action_steps"] == future_action_steps
    assert sample.metadata["transition_action_steps_required"] == 124
    assert sample.metadata["extra_source_action_steps"] == future_action_steps - 124


@pytest.mark.parametrize("future_action_steps", [62, 125, 126, 127])
def test_encoded_dynamics_rejects_malformed_action_alignment(
    tmp_path: Path,
    future_action_steps: int,
) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(
        tmp_path,
        target_latent_frames=32,
        future_action_steps=future_action_steps,
    )
    dataset = EncodedDynamicsLatentDataset.from_root(
        _data_config(empty_text_path),
        encoded_root,
        split="train",
        source=DynamicsSource.COUNTERFACTUAL_DYNAMICS,
    )

    with pytest.raises(ValueError, match="configured temporal geometry"):
        _ = dataset[0]


def test_encoded_counterfactual_dataset_uses_saved_observation_state(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path, include_state=True)
    data_config = _data_config(empty_text_path)

    dataset = EncodedDynamicsLatentDataset.from_root(data_config, encoded_root, split="train", source=DynamicsSource.COUNTERFACTUAL_DYNAMICS)
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
    reference = EncodedDynamicsLatentDataset.from_root(
        data_config,
        encoded_root,
        split="train",
        source=DynamicsSource.REAL_DEMO,
    )
    counterfactual = EncodedDynamicsLatentDataset.from_root(
        data_config,
        encoded_root,
        split="train",
        source=DynamicsSource.COUNTERFACTUAL_DYNAMICS,
    )
    cf_sample = counterfactual[0]
    mixture = _routing_dataset(
        real_dataset=reference,
        counterfactual_dataset=counterfactual,
        routing_config=_routing_config(
            real_fdm=1.0,
            real_joint=0.0,
            real_idm=0.0,
            counterfactual_fdm=1.0,
            counterfactual_idm=0.0,
        ),
        split="train",
    )

    real_view = mixture.build_source_view(
        source="real_demo",
        mode="action_conditioned_video",
        bucket_name="real_fdm",
    )
    real_conditional = real_view[0]

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

def test_dynamics_routing_keeps_multichunk_conditional_sources(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path, target_latent_frames=6)
    data_config = _data_config(empty_text_path)
    reference = EncodedDynamicsLatentDataset.from_root(
        data_config,
        encoded_root,
        split="train",
        source=DynamicsSource.REAL_DEMO,
    )
    counterfactual = EncodedDynamicsLatentDataset.from_root(
        data_config,
        encoded_root,
        split="train",
        source=DynamicsSource.COUNTERFACTUAL_DYNAMICS,
    )
    raw_cf_sample = counterfactual[0]
    mixture = _routing_dataset(
        real_dataset=reference,
        counterfactual_dataset=counterfactual,
        routing_config=_routing_config(
            real_joint=0.0,
            real_fdm=1.0,
            real_idm=0.0,
            counterfactual_fdm=1.0,
            counterfactual_idm=0.0,
        ),
        split="train",
    )

    real_view = mixture.build_source_view(
        source="real_demo",
        mode="action_conditioned_video",
        bucket_name="real_fdm",
    )
    real_conditional = real_view[0]

    assert raw_cf_sample.video_latents.shape == (2, 6, 2, 2)
    assert real_conditional.video_latents.shape == (2, 6, 2, 2)
    assert real_conditional.actions.shape == (24, 7)
    assert real_conditional.metadata["generalist_gjd_chunk_contract"] == "t0_singleton"
    assert real_conditional.metadata["conditional_history_policy"] == "previous_boundary_video_only"
    assert real_conditional.metadata["loss_frame_start"] == 1
    assert real_conditional.metadata["loss_frame_end"] == 6
    assert real_conditional.metadata["supervised_future_latent_frames"] == 5
    assert real_conditional.action_mask is not None
    torch.testing.assert_close(real_conditional.action_mask[:4], torch.zeros(4, 7))
    torch.testing.assert_close(real_conditional.action_mask[4:], torch.ones(20, 7))

    cf_view = mixture.build_source_view(
        source="counterfactual_dynamics",
        mode="action_conditioned_video",
        bucket_name="cf_fdm",
    )
    cf_conditional = cf_view[0]

    assert cf_conditional.video_latents.shape == (2, 6, 2, 2)
    assert cf_conditional.actions.shape == (24, 7)
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
    torch.testing.assert_close(cf_conditional.action_mask[:4], torch.zeros(4, 7))
    torch.testing.assert_close(cf_conditional.action_mask[4:], torch.ones(20, 7))


def test_encoded_dynamics_runtime_rejects_legacy_absolute_root_fields(
    tmp_path: Path,
) -> None:
    encoded_root, _ = _write_encoded_counterfactual_fixture(tmp_path)
    manifest_path = encoded_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_dataset_root"] = manifest.pop("dataset_root")
    del manifest["raw_payload_root"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(KeyError, match="raw_payload_root"):
        load_encoded_dynamics_artifact(encoded_root)

    artifact = migrate_encoded_dynamics_artifact(encoded_root)

    assert artifact.raw_root == (tmp_path / "raw").resolve()
    assert artifact.manifest["raw_payload_root"] == "../raw"


def test_encoded_dynamics_migration_repairs_stale_provenance_with_raw_root(
    tmp_path: Path,
) -> None:
    encoded_root, _ = _write_encoded_counterfactual_fixture(tmp_path)
    manifest_path = encoded_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("raw_payload_root")
    manifest["dataset_root"] = "/missing/source-machine/raw"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Pass --raw-root explicitly"):
        migrate_encoded_dynamics_artifact(encoded_root)

    artifact = migrate_encoded_dynamics_artifact(
        encoded_root,
        raw_root=tmp_path / "raw",
    )

    assert artifact.raw_root == (tmp_path / "raw").resolve()
    assert artifact.manifest["dataset_root"] == "/missing/source-machine/raw"
    assert artifact.manifest["raw_payload_root"] == "../raw"


def test_encoded_dynamics_artifact_is_relocatable_as_one_tree(
    tmp_path: Path,
) -> None:
    source_bundle = tmp_path / "source_bundle"
    encoded_root, _ = _write_encoded_counterfactual_fixture(source_bundle)
    assert load_encoded_dynamics_artifact(encoded_root).raw_root == (
        source_bundle / "raw"
    ).resolve()

    relocated_bundle = tmp_path / "relocated_bundle"
    source_bundle.rename(relocated_bundle)
    assert not source_bundle.exists()
    relocated_encoded_root = relocated_bundle / "encoded"

    statuses = preflight_encoded_dynamics_artifact(
        relocated_encoded_root,
        sources=(
            DynamicsSource.REAL_DEMO,
            DynamicsSource.COUNTERFACTUAL_DYNAMICS,
        ),
        config_path="data.dynamics_routing.train_latent_root",
    )
    dataset = EncodedDynamicsLatentDataset.from_root(
        _data_config(relocated_bundle / "empty_emb.pt"),
        relocated_encoded_root,
        split="train",
        source=DynamicsSource.COUNTERFACTUAL_DYNAMICS,
    )

    assert statuses
    assert dataset.artifact.raw_root == (relocated_bundle / "raw").resolve()
    assert dataset[0].metadata["counterfactual_sample_id"] == 0


def test_encoded_dynamics_target_only_layout_does_not_require_condition_latents(
    tmp_path: Path,
) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = replace(
        _data_config(empty_text_path),
        sample_construction=SampleConstructionConfig(
            chunk_size=2,
            window_size=4,
            condition_source_frame_offset=-1,
        ),
    )

    dataset = EncodedDynamicsLatentDataset.from_root(
        data_config,
        encoded_root,
        split="train",
        source=DynamicsSource.COUNTERFACTUAL_DYNAMICS,
    )
    sample = dataset[0]

    assert sample.condition_latents is None
    assert sample.metadata["condition_latents_source"] == "in_sequence_t0"


def test_encoded_dynamics_ignores_stale_external_condition_latents(
    tmp_path: Path,
) -> None:
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

    dataset = EncodedDynamicsLatentDataset.from_root(data_config, encoded_root, split="train", source=DynamicsSource.COUNTERFACTUAL_DYNAMICS)
    sample = dataset[0]

    assert sample.condition_latents is None
    assert sample.metadata["has_condition_latents"] is False
    assert sample.metadata["condition_source_frame_offset"] is None
    assert sample.metadata["condition_latents_source"] == "in_sequence_t0"


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

    monkeypatch.setattr(dynamics_routing_module.random, "randint", fake_randint)

    dataset = EncodedDynamicsLatentDataset.from_root(data_config, encoded_root, split="train", source=DynamicsSource.COUNTERFACTUAL_DYNAMICS)
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

    dataset = EncodedDynamicsLatentDataset.from_root(data_config, encoded_root, split="train", source=DynamicsSource.COUNTERFACTUAL_DYNAMICS)
    sample = dataset[0]

    assert sample.metadata["sampled_chunk_size"] == 2
    assert sample.metadata["generalist_gjd_chunk_contract"] == "t0_singleton"
    assert sample.metadata["counterfactual_gjd_chunk_contract"] == sample.metadata["generalist_gjd_chunk_contract"]
    assert sample.metadata["singleton_chunk_frame"] == sample.metadata["target_observation_frame_in_sample"]
    assert sample.metadata["sampled_window_size"] == 8


def test_encoded_dynamics_state_is_anchored_to_in_sequence_t0(
    tmp_path: Path,
) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(
        tmp_path,
        include_state=True,
        context_latent_frames=4,
        target_latent_frames=3,
    )
    data_config = replace(
        _data_config(empty_text_path),
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.UNIFORM_SEGMENT,
            sample_order_mode=SampleOrderMode.REPLACEMENT,
            chunk_size=2,
            window_size=4,
            condition_source_frame_offset=-1,
            start_padding_frames=0,
        ),
    )

    dataset = EncodedDynamicsLatentDataset.from_root(data_config, encoded_root, split="train", source=DynamicsSource.COUNTERFACTUAL_DYNAMICS)
    sample = dataset[0]

    assert sample.metadata["latent_frame_start"] == 0
    assert sample.metadata["state_anchor_source_frame"] == 0
    assert sample.metadata["state_anchor_frame_in_sample"] == 0
    assert sample.proprio_context_frames is not None
    torch.testing.assert_close(
        sample.proprio_context_frames[:, 0],
        torch.tensor([20.0, 24.0, 24.0]),
    )
    torch.testing.assert_close(sample.state[:, 0], torch.tensor([20.0]))
    torch.testing.assert_close(sample.actions[:4], torch.zeros(4, 7))
    torch.testing.assert_close(sample.action_mask[:4], torch.zeros(4, 7))
    torch.testing.assert_close(sample.actions[4:8], torch.full((4, 7), 2.0))
    torch.testing.assert_close(sample.action_mask[4:8], torch.ones(4, 7))


def test_dynamics_routing_stamps_forced_mode_and_drops_text(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedDynamicsLatentDataset.from_root(data_config, encoded_root, split="train", source=DynamicsSource.COUNTERFACTUAL_DYNAMICS)
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
    mixture = _routing_dataset(
        real_dataset=_OneSampleLatentDataset(real_sample),
        counterfactual_dataset=counterfactual,
        routing_config=_routing_config(
            counterfactual_fdm=1.0,
            real_joint=0.0,
            real_fdm=0.0,
            real_idm=0.0,
            counterfactual_idm=0.0,
        ),
        split="train",
    )

    sample = mixture[0]

    assert sample.metadata[DYNAMICS_ROUTING_SOURCE_METADATA_KEY] == "counterfactual_dynamics"
    assert sample.metadata[DYNAMICS_ROUTING_MODE_METADATA_KEY] == "action_conditioned_video"
    assert sample.metadata[DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY] is True
    assert sample.task_text is None
    assert sample.text_context is not None
    assert torch.equal(sample.text_context, torch.zeros(3, 4))


@pytest.mark.parametrize(
    ("fixed_mode", "real_bucket", "counterfactual_bucket"),
    [
        (
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            "real_action_conditioned_video",
            "counterfactual_action_conditioned_video",
        ),
        (
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            "real_video_conditioned_action",
            "counterfactual_video_conditioned_action",
        ),
    ],
)
def test_fixed_conditional_mixture_keeps_matching_real_and_cf_sources(
    tmp_path: Path,
    fixed_mode: DynamicsObjective,
    real_bucket: str,
    counterfactual_bucket: str,
) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    counterfactual = EncodedDynamicsLatentDataset.from_root(
        _data_config(empty_text_path),
        encoded_root,
        split="train",
        source=DynamicsSource.COUNTERFACTUAL_DYNAMICS,
    )
    real_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        metadata={},
    )
    mixture = _routing_dataset(
        real_dataset=_OneSampleLatentDataset(real_sample),
        counterfactual_dataset=counterfactual,
        routing_config=DynamicsRoutingConfig(
            routes=(
                DynamicsRouteConfig(
                    source="real_demo", mode=fixed_mode, weight=3.0
                ),
                DynamicsRouteConfig(
                    source="counterfactual_dynamics", mode=fixed_mode, weight=1.0
                ),
            ),
        ),
        split="train",
        fixed_mode=fixed_mode,
    )

    assert [(bucket.name, bucket.weight) for bucket in mixture.buckets] == [
        (real_bucket, 3.0),
        (counterfactual_bucket, 1.0),
    ]
    for index in range(32):
        sample = mixture[index]
        assert sample.metadata[DYNAMICS_ROUTING_MODE_METADATA_KEY] == fixed_mode.value
        assert sample.metadata[DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY] is True


def test_fixed_conditional_mixture_rejects_conflicting_active_routes() -> None:
    real_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        metadata={},
    )

    with pytest.raises(ValueError, match="accepts only matching routes"):
        _routing_dataset(
            real_dataset=_OneSampleLatentDataset(real_sample),
            counterfactual_dataset=None,
            routing_config=_routing_config(
                real_joint=1.0,
                real_fdm=1.0,
                real_idm=0.0,
                counterfactual_fdm=0.0,
                counterfactual_idm=0.0,
            ),
            split="train",
            fixed_mode=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        )


def test_fixed_conditional_real_only_requires_rollout_local_encoded_root(
    tmp_path: Path,
) -> None:
    empty_text_path = tmp_path / "empty_emb.pt"
    torch.save(torch.zeros(3, 4), empty_text_path)
    data_config = replace(
        _data_config(empty_text_path),
        dynamics_routing=_routing_config(
            train_latent_root=None,
            val_latent_root=None,
            real_joint=0.0,
            real_fdm=1.0,
            real_idm=0.0,
            counterfactual_fdm=0.0,
            counterfactual_idm=0.0,
        ),
    )
    real_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        metadata={},
    )

    with pytest.raises(ValueError, match="rollout-local target-only encoded root"):
        build_dynamics_routing_datasets(
            data_config=data_config,
            train_dataset=_OneSampleLatentDataset(real_sample),
            val_dataset=_OneSampleLatentDataset(real_sample),
            fixed_mode=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        )




def test_generalist_source_view_can_spread_indices_for_short_validation() -> None:
    real_dataset = _RecordingDrawKeyDataset(length=100)
    counterfactual_dataset = _RecordingDrawKeyDataset(length=100)
    mixture = _routing_dataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual_dataset,
        routing_config=_routing_config(
            real_joint=0.6,
            real_fdm=0.0,
            real_idm=0.0,
            counterfactual_fdm=0.2,
            counterfactual_idm=0.2,
        ),
        split="val",
    )

    direct_view = mixture.build_source_view(
        source="counterfactual_dynamics",
        mode="action_conditioned_video",
        bucket_name="val_fdm",
    )
    spread_view = mixture.build_source_view(
        source="counterfactual_dynamics",
        mode="action_conditioned_video",
        bucket_name="val_fdm",
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
    mixture = _routing_dataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual_dataset,
        routing_config=_routing_config(
            real_joint=0.6,
            real_fdm=0.0,
            real_idm=0.0,
            counterfactual_fdm=0.2,
            counterfactual_idm=0.2,
        ),
        split="val",
    )

    view = mixture.build_source_view(
        source="counterfactual_dynamics",
        mode="action_conditioned_video",
        bucket_name="val_fdm",
        spread_indices=True,
    )

    source_indices = [view[index].metadata["generalist_source_index"] for index in range(10)]

    assert source_indices == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert view[1].metadata["generalist_source_view_order"] == "balanced"
    assert view[1].metadata["generalist_source_view_stride"] == 1



def test_dynamics_routing_passes_local_replacement_indices(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    counterfactual = EncodedDynamicsLatentDataset.from_root(data_config, encoded_root, split="train", source=DynamicsSource.COUNTERFACTUAL_DYNAMICS)
    real_dataset = _RecordingDrawKeyDataset(length=5)
    mixture = _routing_dataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual,
        routing_config=_routing_config(
            real_joint=1.0,
            real_fdm=0.0,
            real_idm=0.0,
            counterfactual_fdm=0.0,
            counterfactual_idm=0.0,
        ),
        split="train",
    )

    sampler = mixture.build_train_sampler(world_size=1, rank=0)
    assert list(iter(sampler)) == list(range(len(mixture)))
    sampler.set_epoch(1)
    assert list(iter(sampler)) == list(range(len(mixture), 2 * len(mixture)))

    sample = mixture[len(mixture)]

    assert 0 <= sample.metadata["generalist_source_index"] < len(real_dataset)
    assert sample.metadata["seen_draw_key"] == sample.metadata["generalist_source_index"]


def test_dynamics_routing_train_sampler_coordinates_bucket_across_ranks() -> None:
    real_dataset = _RecordingDrawKeyDataset(length=10_000)
    counterfactual_dataset = _RecordingDrawKeyDataset(length=10_000)
    mixture = _routing_dataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual_dataset,
        routing_config=_routing_config(
            real_joint=0.6,
            real_fdm=0.0,
            real_idm=0.0,
            counterfactual_fdm=0.2,
            counterfactual_idm=0.2,
        ),
        split="train",
    )

    indices = [
        next(iter(mixture.build_train_sampler(world_size=4, rank=rank)))
        for rank in range(4)
    ]
    samples = [mixture[index] for index in indices]

    assert indices == [0, 1, 2, 3]
    assert len({sample.metadata[DYNAMICS_ROUTING_BUCKET_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata[DYNAMICS_ROUTING_MODE_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata[DYNAMICS_ROUTING_SOURCE_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata["seen_draw_key"] for sample in samples}) > 1


def test_dynamics_routing_train_sampler_coordinates_padded_epoch_tail() -> None:
    real_dataset = _RecordingDrawKeyDataset(length=5)
    counterfactual_dataset = _RecordingDrawKeyDataset(length=5)
    mixture = _routing_dataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual_dataset,
        routing_config=_routing_config(
            real_joint=0.6,
            real_fdm=0.0,
            real_idm=0.0,
            counterfactual_fdm=0.2,
            counterfactual_idm=0.2,
        ),
        split="train",
    )

    samplers = [mixture.build_train_sampler(world_size=4, rank=rank) for rank in range(4)]
    for sampler in samplers:
        sampler.set_epoch(1)
    indices = [next(iter(sampler)) for sampler in samplers]
    samples = [mixture[index] for index in indices]

    assert indices == [8, 9, 10, 11]
    assert len({sample.metadata[DYNAMICS_ROUTING_BUCKET_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata[DYNAMICS_ROUTING_MODE_METADATA_KEY] for sample in samples}) == 1
    assert len({sample.metadata[DYNAMICS_ROUTING_SOURCE_METADATA_KEY] for sample in samples}) == 1
    assert all(0 <= sample.metadata["generalist_source_index"] < 5 for sample in samples)
    assert len({sample.metadata["seen_draw_key"] for sample in samples}) > 1


def test_dynamics_routing_validation_sampler_coordinates_padded_tail() -> None:
    real_dataset = _RecordingDrawKeyDataset(length=5)
    counterfactual_dataset = _RecordingDrawKeyDataset(length=5)
    mixture = _routing_dataset(
        real_dataset=real_dataset,
        counterfactual_dataset=counterfactual_dataset,
        routing_config=_routing_config(
            real_joint=0.6,
            real_fdm=0.0,
            real_idm=0.0,
            counterfactual_fdm=0.2,
            counterfactual_idm=0.2,
        ),
        split="val",
    )

    rank_orders = [
        list(mixture.build_validation_sampler(world_size=4, rank=rank))
        for rank in range(4)
    ]
    tail_samples = [mixture[indices[1]] for indices in rank_orders]

    assert rank_orders == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert len(
        {
            sample.metadata[DYNAMICS_ROUTING_BUCKET_METADATA_KEY]
            for sample in tail_samples
        }
    ) == 1
    assert len(
        {
            sample.metadata[DYNAMICS_ROUTING_MODE_METADATA_KEY]
            for sample in tail_samples
        }
    ) == 1
    assert len(
        {
            sample.metadata[DYNAMICS_ROUTING_SOURCE_METADATA_KEY]
            for sample in tail_samples
        }
    ) == 1



def test_build_dynamics_routing_datasets_uses_train_and_val_roots(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = _data_config(empty_text_path)
    data_config = _replace_data_mixture_root(data_config, str(encoded_root))
    real_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        metadata={},
    )

    train_dataset, val_dataset = build_dynamics_routing_datasets(
        data_config=data_config,
        train_dataset=_OneSampleLatentDataset(real_sample),
        val_dataset=_OneSampleLatentDataset(real_sample),
    )

    assert isinstance(train_dataset, DynamicsRoutingDataset)
    assert isinstance(val_dataset, DynamicsRoutingDataset)
    encoded_views = tuple(
        dataset.route_dataset(
            source=source,
            mode=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        )
        for dataset in (train_dataset, val_dataset)
        for source in (
            DynamicsSource.REAL_DEMO,
            DynamicsSource.COUNTERFACTUAL_DYNAMICS,
        )
    )
    assert all(
        view.artifact is encoded_views[0].artifact for view in encoded_views[1:]
    )
    assert all(
        view.empty_text_embedding is encoded_views[0].empty_text_embedding
        for view in encoded_views[1:]
    )


def test_build_dynamics_routing_loads_each_unique_artifact_once(
    tmp_path: Path,
) -> None:
    train_root, empty_text_path = _write_encoded_counterfactual_fixture(
        tmp_path / "train"
    )
    val_root, _ = _write_encoded_counterfactual_fixture(tmp_path / "val")
    data_config = replace(
        _data_config(empty_text_path),
        dynamics_routing=_routing_config(
            train_latent_root=str(train_root),
            val_latent_root=str(val_root),
        ),
    )
    planning_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        metadata={},
    )

    train_dataset, val_dataset = build_dynamics_routing_datasets(
        data_config=data_config,
        train_dataset=_OneSampleLatentDataset(planning_sample),
        val_dataset=_OneSampleLatentDataset(planning_sample),
    )
    train_views = tuple(
        train_dataset.route_dataset(
            source=source,
            mode=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        )
        for source in (
            DynamicsSource.REAL_DEMO,
            DynamicsSource.COUNTERFACTUAL_DYNAMICS,
        )
    )
    val_views = tuple(
        val_dataset.route_dataset(
            source=source,
            mode=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        )
        for source in (
            DynamicsSource.REAL_DEMO,
            DynamicsSource.COUNTERFACTUAL_DYNAMICS,
        )
    )

    assert train_views[0].artifact is train_views[1].artifact
    assert val_views[0].artifact is val_views[1].artifact
    assert train_views[0].artifact is not val_views[0].artifact
    assert train_views[0].artifact.root == train_root.resolve()
    assert val_views[0].artifact.root == val_root.resolve()
    assert all(
        view.empty_text_embedding is train_views[0].empty_text_embedding
        for view in (*train_views[1:], *val_views)
    )


def test_build_dynamics_routing_requires_val_root(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = replace(
        _data_config(empty_text_path),
        dynamics_routing=_routing_config(train_latent_root=str(encoded_root)),
    )
    real_sample = LatentWAMSample(
        video_latents=torch.ones(2, 4, 2, 2),
        actions=torch.ones(8, 7),
        action_mask=torch.ones(8, 7),
        metadata={},
    )

    with pytest.raises(ValueError, match="val_latent_root"):
        build_dynamics_routing_datasets(
            data_config=data_config,
            train_dataset=_OneSampleLatentDataset(real_sample),
            val_dataset=_OneSampleLatentDataset(real_sample),
        )


def test_build_dynamics_routing_allows_explicit_debug_val_fallback(tmp_path: Path) -> None:
    encoded_root, empty_text_path = _write_encoded_counterfactual_fixture(tmp_path)
    data_config = replace(
        _data_config(empty_text_path),
        dynamics_routing=_routing_config(
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

    train_dataset, val_dataset = build_dynamics_routing_datasets(
        data_config=data_config,
        train_dataset=_OneSampleLatentDataset(real_sample),
        val_dataset=_OneSampleLatentDataset(real_sample),
    )

    assert isinstance(train_dataset, DynamicsRoutingDataset)
    assert isinstance(val_dataset, DynamicsRoutingDataset)


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
        dynamics_routing=_routing_config(
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
    future_action_steps: int | None = None,
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
    reference_row = {
        **transition_row,
        "sample_id": 1,
        "branch": "gt",
        "branch_family": "demo",
        "branch_strength": "none",
        "branch_is_ood": False,
    }
    _write_jsonl(raw_root / "metadata" / "contexts.jsonl", [context_row])
    _write_jsonl(raw_root / "metadata" / "transitions.jsonl", [transition_row])
    _write_jsonl(encoded_root / "metadata" / "encoded_contexts.jsonl", [context_row])
    _write_jsonl(
        encoded_root / "metadata" / "encoded_transitions.jsonl",
        [transition_row, reference_row],
    )
    (encoded_root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema": ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1,
                "raw_payload_root": "../raw",
                "dataset_root": str(raw_root),
                "reference_branch": "gt",
                "condition_latents": bool(include_condition_latents),
                "condition_source_frame_offset": -1 if include_condition_latents else 0,
                "condition_source_frame_policy": "next_latent_source_offset" if include_condition_latents else None,
            }
        ),
        encoding="utf-8",
    )
    action_per_frame = 4
    resolved_future_action_steps = (
        max(0, target_latent_frames - 1) * action_per_frame
        if future_action_steps is None
        else int(future_action_steps)
    )
    context_payload = {
        "action_context": np.ones((context_latent_frames * action_per_frame, 7), dtype=np.float32)
    }
    sample_payload = {
        "future_actions": np.full(
            (resolved_future_action_steps, 7),
            2.0,
            dtype=np.float32,
        )
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
